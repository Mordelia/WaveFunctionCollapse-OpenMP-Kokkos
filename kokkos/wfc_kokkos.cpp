#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * @file wfc_kokkos.cpp
 * @brief Kokkos-based Wave Function Collapse (WFC) implementation with bit-mask domains.
 *
 * This version keeps the same overlap model and backtracking strategy as the CPU variants,
 * while offloading the propagation fixed-point step to a Kokkos parallel kernel.
 */

/** Bit-mask storage word type. */
using Word = uint64_t;
/** Default host-side wall-clock type used for timing output. */
using Clock = std::chrono::high_resolution_clock;

/** Number of cardinal neighbor directions. */
static constexpr int DIR_COUNT = 4;
/** Direction identifier: up. */
static constexpr int DIR_UP = 0;
/** Direction identifier: down. */
static constexpr int DIR_DOWN = 1;
/** Direction identifier: left. */
static constexpr int DIR_LEFT = 2;
/** Direction identifier: right. */
static constexpr int DIR_RIGHT = 3;
/** Direction X offsets in order [up, down, left, right]. */
static constexpr int DIR_DX[DIR_COUNT] = {0, 0, -1, 1};
/** Direction Y offsets in order [up, down, left, right]. */
static constexpr int DIR_DY[DIR_COUNT] = {-1, 1, 0, 0};

/** Host memory view for packed mask words. */
using HostWordView = Kokkos::View<Word *, Kokkos::HostSpace>;
/** Device memory view for packed mask words. */
using DevWordView = Kokkos::View<Word *, Kokkos::DefaultExecutionSpace>;
/** Device memory view for integer flags/counters. */
using DevIntView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

/**
 * @brief Compute the flat word offset of one cell mask word.
 * @param cellIdx Flattened cell index.
 * @param wordIdx Word index within the cell mask.
 * @param wordsPerCell Number of words in one cell mask.
 * @return Flat offset in the packed mask buffer.
 */
static inline std::size_t cellWordOffset(int cellIdx, int wordIdx, int wordsPerCell)
{
    return (std::size_t)cellIdx * (std::size_t)wordsPerCell + (std::size_t)wordIdx;
}

/**
 * @brief Compute the flat offset in the compatibility table.
 * @param dir Direction index.
 * @param tile Tile index (source/target depending on table convention).
 * @param wordIdx Word index inside one compatibility mask.
 * @param tileCount Number of tiles.
 * @param wordsPerMask Number of words per compatibility mask.
 * @return Flat offset in packed compatibility storage.
 */
KOKKOS_INLINE_FUNCTION static std::size_t compatWordOffset(int dir, int tile, int wordIdx, int tileCount, int wordsPerMask)
{
    return (((std::size_t)dir * (std::size_t)tileCount + (std::size_t)tile) * (std::size_t)wordsPerMask) + (std::size_t)wordIdx;
}

/**
 * @brief Count set bits in one mask word.
 * @param value Input word.
 * @return Number of active bits.
 */
static inline int popcountWord(Word value)
{
    return __builtin_popcountll((unsigned long long)value);
}

/**
 * @brief Count trailing zeros in one mask word (host/device compatible).
 * @param value Input word (must be non-zero for meaningful result).
 * @return Bit index of the least significant set bit.
 */
KOKKOS_INLINE_FUNCTION static int ctzWord(Word value)
{
// For device code: use __ffsll (first set bit position, 1-indexed, returns 0 if none)
// For host code: use __builtin_ctzll (count trailing zeros)
#ifdef __CUDA_ARCH__
    int ffs = __ffsll((long long)value);
    return (ffs == 0) ? 0 : (ffs - 1);
#else
    return __builtin_ctzll((unsigned long long)value);
#endif
}

/**
 * @brief Test whether a tile bit is active in a cell mask.
 * @param masks Packed host mask buffer.
 * @param cellIdx Flattened cell index.
 * @param tileIdx Tile index.
 * @param wordsPerCell Number of words per cell mask.
 * @return True if bit is set.
 */
static inline bool hasBit(const HostWordView &masks, int cellIdx, int tileIdx, int wordsPerCell)
{
    int wordIdx = tileIdx / 64;
    int bitIdx = tileIdx % 64;
    return (masks(cellWordOffset(cellIdx, wordIdx, wordsPerCell)) & (Word(1) << bitIdx)) != 0;
}

/**
 * @brief Set one tile bit in a cell mask.
 * @param masks Packed host mask buffer.
 * @param cellIdx Flattened cell index.
 * @param tileIdx Tile index.
 * @param wordsPerCell Number of words per cell mask.
 */
static inline void setBit(HostWordView &masks, int cellIdx, int tileIdx, int wordsPerCell)
{
    int wordIdx = tileIdx / 64;
    int bitIdx = tileIdx % 64;
    masks(cellWordOffset(cellIdx, wordIdx, wordsPerCell)) |= (Word(1) << bitIdx);
}

/**
 * @brief Clear all bits in a cell mask.
 * @param masks Packed host mask buffer.
 * @param cellIdx Flattened cell index.
 * @param wordsPerCell Number of words per cell mask.
 */
static inline void clearCell(HostWordView &masks, int cellIdx, int wordsPerCell)
{
    for (int w = 0; w < wordsPerCell; ++w)
    {
        masks(cellWordOffset(cellIdx, w, wordsPerCell)) = 0;
    }
}

/**
 * @brief Check whether a cell mask is empty.
 * @param masks Packed host mask buffer.
 * @param cellIdx Flattened cell index.
 * @param wordsPerCell Number of words per cell mask.
 * @return True if no tile remains possible.
 */
static inline bool isZeroCell(const HostWordView &masks, int cellIdx, int wordsPerCell)
{
    for (int w = 0; w < wordsPerCell; ++w)
    {
        if (masks(cellWordOffset(cellIdx, w, wordsPerCell)) != 0)
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Count active tiles in one cell mask.
 * @param masks Packed host mask buffer.
 * @param cellIdx Flattened cell index.
 * @param wordsPerCell Number of words per cell mask.
 * @return Number of active tiles.
 */
static inline int popcountCell(const HostWordView &masks, int cellIdx, int wordsPerCell)
{
    int count = 0;
    for (int w = 0; w < wordsPerCell; ++w)
    {
        count += popcountWord(masks(cellWordOffset(cellIdx, w, wordsPerCell)));
    }
    return count;
}

/**
 * @brief Get the first active tile index in a cell mask.
 * @param masks Packed host mask buffer.
 * @param cellIdx Flattened cell index.
 * @param wordsPerCell Number of words per cell mask.
 * @return Tile index or -1 if empty.
 */
static inline int firstSetTile(const HostWordView &masks, int cellIdx, int wordsPerCell)
{
    for (int w = 0; w < wordsPerCell; ++w)
    {
        Word word = masks(cellWordOffset(cellIdx, w, wordsPerCell));
        if (word != 0)
        {
            return w * 64 + ctzWord(word);
        }
    }
    return -1;
}

/**
 * @brief Load an RGB image and convert it to a palette-index grid.
 * @param path Input image path.
 * @param[out] palette Palette values encoded as 0xRRGGBB.
 * @return 2D palette-index grid with shape [height][width].
 */
std::vector<std::vector<int>> loadImage(const std::string &path, std::vector<uint32_t> &palette)
{
    int w, h, ch;
    unsigned char *data = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!data)
    {
        std::cerr << "[ERROR] Cannot load: " << path << "\n";
        std::exit(1);
    }

    std::map<uint32_t, int> colorIndex;
    palette.clear();
    std::vector<std::vector<int>> grid(h, std::vector<int>(w));

    for (int r = 0; r < h; r++)
    {
        for (int c = 0; c < w; c++)
        {
            int base = (r * w + c) * 3;
            uint32_t color = ((uint32_t)data[base] << 16) |
                             ((uint32_t)data[base + 1] << 8) |
                             (uint32_t)data[base + 2];
            auto it = colorIndex.find(color);
            if (it == colorIndex.end())
            {
                colorIndex[color] = (int)palette.size();
                grid[r][c] = (int)palette.size();
                palette.push_back(color);
            }
            else
            {
                grid[r][c] = it->second;
            }
        }
    }

    stbi_image_free(data);
    std::cout << "[INPUT] Loaded " << path << " (" << w << "x" << h
              << ", " << palette.size() << " colors)\n";
    return grid;
}

/**
 * @brief NxN tile represented as palette indices.
 */
struct Tile
{
    /** Tile edge size. */
    int N;
    /** Tile data as [row][col] indices. */
    std::vector<std::vector<int>> data;

    /** Lexicographic ordering for associative containers. */
    bool operator<(const Tile &o) const { return data < o.data; }
    /** Equality based on tile content. */
    bool operator==(const Tile &o) const { return data == o.data; }
};

/**
 * @brief Rotate a tile by 90 degrees clockwise.
 * @param t Source tile.
 * @return Rotated tile.
 */
Tile rotate90(const Tile &t)
{
    int N = t.N;
    Tile r;
    r.N = N;
    r.data.assign(N, std::vector<int>(N));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            r.data[j][N - 1 - i] = t.data[i][j];
        }
    }
    return r;
}

/**
 * @brief Reflect a tile along the vertical axis (horizontal mirror).
 * @param t Source tile.
 * @return Reflected tile.
 */
Tile reflectH(const Tile &t)
{
    int N = t.N;
    Tile r;
    r.N = N;
    r.data = t.data;
    for (int i = 0; i < N; i++)
    {
        std::reverse(r.data[i].begin(), r.data[i].end());
    }
    return r;
}

/**
 * @brief Extract unique overlapping NxN tiles from the input sample.
 *
 * Extraction uses periodic indexing on the sample and can optionally include
 * rotations/reflections to augment the model.
 *
 * @param sample Input sample as palette-index grid.
 * @param N Tile size.
 * @param[out] tiles Unique tile list.
 * @param[out] freq Frequency of each corresponding tile.
 * @param useSymmetries If true, include symmetric variants of each tile.
 */
void extractTiles(const std::vector<std::vector<int>> &sample, int N,
                  std::vector<Tile> &tiles, std::vector<int> &freq,
                  bool useSymmetries = true)
{
    int rows = (int)sample.size();
    int cols = (int)sample[0].size();

    std::map<std::vector<std::vector<int>>, int> counts;
    auto addTile = [&](const Tile &tile)
    {
        counts[tile.data]++;
    };

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            Tile base;
            base.N = N;
            base.data.resize(N);
            for (int i = 0; i < N; i++)
            {
                base.data[i].resize(N);
                for (int j = 0; j < N; j++)
                {
                    base.data[i][j] = sample[(r + i) % rows][(c + j) % cols];
                }
            }

            if (!useSymmetries)
            {
                addTile(base);
            }
            else
            {
                Tile cur = base;
                for (int rot = 0; rot < 4; rot++)
                {
                    addTile(cur);
                    addTile(reflectH(cur));
                    cur = rotate90(cur);
                }
            }
        }
    }

    tiles.clear();
    freq.clear();
    for (const auto &kv : counts)
    {
        Tile tile;
        tile.N = N;
        tile.data = kv.first;
        tiles.push_back(tile);
        freq.push_back(kv.second);
    }

    std::cout << "[TILES] " << tiles.size() << " unique " << N << "x" << N
              << " tiles extracted" << (useSymmetries ? " (with symmetries)" : "") << "\n";
}

/**
 * @brief Check directional compatibility between two tiles.
 * @param current Tile placed at current cell.
 * @param neighbor Tile placed in the neighbor cell.
 * @param dir Relative neighbor direction.
 * @return True if overlap constraints are satisfied.
 */
bool tilesCompatible(const Tile &current, const Tile &neighbor, int dir)
{
    int N = current.N;
    switch (dir)
    {
    case DIR_UP:
        for (int i = 0; i < N - 1; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (current.data[i][j] != neighbor.data[i + 1][j])
                {
                    return false;
                }
            }
        }
        return true;
    case DIR_DOWN:
        for (int i = 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (current.data[i][j] != neighbor.data[i - 1][j])
                {
                    return false;
                }
            }
        }
        return true;
    case DIR_LEFT:
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N - 1; j++)
            {
                if (current.data[i][j] != neighbor.data[i][j + 1])
                {
                    return false;
                }
            }
        }
        return true;
    case DIR_RIGHT:
        for (int i = 0; i < N; i++)
        {
            for (int j = 1; j < N; j++)
            {
                if (current.data[i][j] != neighbor.data[i][j - 1])
                {
                    return false;
                }
            }
        }
        return true;
    default:
        return false;
    }
}

/**
 * @brief Directional compatibility table stored as packed bit masks.
 *
 * For each pair (dir, neighborTile), the table stores the set of current-cell
 * tile indices that remain valid given that neighbor tile.
 */
struct CompatTable
{
    /** Number of unique tiles. */
    int T = 0;
    /** Number of 64-bit words per compatibility mask. */
    int wordsPerMask = 0;
    /** Host staging storage for packed masks. */
    std::vector<Word> incoming;
    /** Host-side compatibility view. */
    HostWordView hostView;
    /** Device-side compatibility view. */
    DevWordView devView;

    /**
     * @brief Build host and device compatibility tables.
     * @param tiles Unique extracted tiles.
     */
    void build(const std::vector<Tile> &tiles)
    {
        T = (int)tiles.size();
        if (T <= 0)
        {
            std::cerr << "[ERROR] No tiles extracted\n";
            std::exit(1);
        }

        wordsPerMask = (T + 63) / 64;
        incoming.assign((std::size_t)DIR_COUNT * (std::size_t)T * (std::size_t)wordsPerMask, Word(0));

        for (int a = 0; a < T; a++)
        {
            for (int b = 0; b < T; b++)
            {
                for (int dir = 0; dir < DIR_COUNT; dir++)
                {
                    if (tilesCompatible(tiles[a], tiles[b], dir))
                    {
                        int wordIdx = a / 64;
                        int bitIdx = a % 64;
                        incoming[compatWordOffset(dir, b, wordIdx, T, wordsPerMask)] |= (Word(1) << bitIdx);
                    }
                }
            }
        }

        hostView = HostWordView("compat_host", incoming.size());
        for (std::size_t i = 0; i < incoming.size(); i++)
        {
            hostView(i) = incoming[i];
        }
        devView = DevWordView("compat_dev", incoming.size());
        Kokkos::deep_copy(devView, hostView);

        std::cout << "[COMPAT] Table built for " << T << " tiles (" << wordsPerMask << " words per mask)\n";
    }

    /**
     * @brief Fetch one compatibility mask word on device.
     * @param dir Direction index.
     * @param tile Tile index.
     * @param wordIdx Word index in the mask.
     * @return Compatibility word.
     */
    KOKKOS_INLINE_FUNCTION
    Word incomingWord(int dir, int tile, int wordIdx) const
    {
        return devView(compatWordOffset(dir, tile, wordIdx, T, wordsPerMask));
    }
};

/**
 * @brief Mutable WFC state stored on host for control/backtracking.
 */
struct WFCState
{
    /** Flattened packed masks, wordsPerMask words per cell. */
    std::vector<Word> masks;
    /** Cached entropy per cell. */
    std::vector<double> entropy;
    /** Grid height. */
    int rows = 0;
    /** Grid width. */
    int cols = 0;
    /** Number of tile states. */
    int T = 0;
    /** Number of words in one cell mask. */
    int wordsPerMask = 0;
};

/**
 * @brief Entropy-driven WFC solver with Kokkos propagation kernel.
 */
struct WFC
{
    /** Output dimensions, tile size, and number of tile states. */
    int rows, cols, N, T;
    /** Number of output cells (rows*cols). */
    int cells;
    /** Number of words per mask. */
    int wordsPerMask;
    /** Immutable model data. */
    const std::vector<Tile> &tiles;
    const std::vector<int> &freq;
    const CompatTable &compat;

    /** Host mask view for control-side operations. */
    HostWordView currentHost;
    /** Current device mask view (input of propagation iteration). */
    DevWordView currentDev;
    /** Next device mask view (output of propagation iteration). */
    DevWordView nextDev;
    /** Device flag set when at least one cell changed in an iteration. */
    DevIntView changedFlagDev;
    /** Device flag set when at least one contradiction is detected. */
    DevIntView contradictionFlagDev;

    /** Precomputed log-frequency weights. */
    std::vector<double> logFreq;
    /** Global sum(freq[t] * log(freq[t])). */
    double totalLogFreqSum = 0.0;
    /** Global sum(freq[t]). */
    double totalFreqSum = 0.0;
    std::mt19937 rng;
    std::uniform_real_distribution<double> noiseDist{0.0, 1e-6};
    WFCState state;

    /**
     * @brief Backtracking snapshot containing full host-side state.
     */
    struct Snapshot
    {
        /** Flattened masks snapshot. */
        std::vector<Word> masks;
        /** Entropy snapshot. */
        std::vector<double> entropy;
        /** Cell index where collapse happened. */
        int cell = -1;
        /** Chosen tile at collapse time. */
        int chosenTile = -1;
    };

    std::stack<Snapshot> history;
    int maxBacktracks;

    /**
     * @brief Construct a WFC solver instance.
     * @param rows Output height.
     * @param cols Output width.
     * @param N Tile size.
     * @param tiles Unique model tiles.
     * @param freq Tile frequencies for weighted collapse.
     * @param compat Precomputed compatibility table.
     * @param seed RNG seed.
     * @param maxBacktracks Maximum allowed backtracks.
     */
    WFC(int rows, int cols, int N,
        const std::vector<Tile> &tiles,
        const std::vector<int> &freq,
        const CompatTable &compat,
        unsigned int seed = 42,
        int maxBacktracks = 100)
        : rows(rows), cols(cols), N(N), T((int)tiles.size()), cells(rows * cols),
          wordsPerMask(compat.wordsPerMask), tiles(tiles), freq(freq), compat(compat),
          currentHost("currentHost", (std::size_t)cells * (std::size_t)wordsPerMask),
          currentDev("currentDev", (std::size_t)cells * (std::size_t)wordsPerMask),
          nextDev("nextDev", (std::size_t)cells * (std::size_t)wordsPerMask),
          changedFlagDev("changedFlagDev", 1),
          contradictionFlagDev("contradictionFlagDev", 1),
          rng(seed), maxBacktracks(maxBacktracks)
    {
        if (T <= 0 || T > 8192)
        {
            std::cerr << "[ERROR] This implementation expects 1..8192 tiles, got " << T << "\n";
            std::exit(1);
        }

        logFreq.resize(T);
        for (int t = 0; t < T; t++)
        {
            logFreq[t] = std::log((double)freq[t]);
            totalFreqSum += freq[t];
            totalLogFreqSum += freq[t] * logFreq[t];
        }

        initState();
    }

    /**
     * @brief Initialize solver state with full domains in every cell.
     */
    void initState()
    {
        state.rows = rows;
        state.cols = cols;
        state.T = T;
        state.wordsPerMask = wordsPerMask;
        state.masks.assign((std::size_t)cells * (std::size_t)wordsPerMask, Word(0));
        state.entropy.assign(cells, 0.0);

        Word fullWord = ~Word(0);
        for (int cell = 0; cell < cells; cell++)
        {
            for (int w = 0; w < wordsPerMask; w++)
            {
                Word value = fullWord;
                if (w == wordsPerMask - 1 && (T % 64) != 0)
                {
                    value = (Word(1) << (T % 64)) - 1;
                }
                state.masks[cellWordOffset(cell, w, wordsPerMask)] = value;
                currentHost(cellWordOffset(cell, w, wordsPerMask)) = value;
            }
            state.entropy[cell] = entropyForCell(cell);
        }

        Kokkos::deep_copy(currentDev, currentHost);
    }

    /**
     * @brief Compute Shannon entropy from aggregated weighted sums.
     * @param sumW Sum of active weights.
     * @param sumWlogW Sum of active weight*log(weight).
     * @return Shannon entropy value.
     */
    double shannonEntropy(double sumW, double sumWlogW) const
    {
        if (sumW <= 0.0)
        {
            return 0.0;
        }
        return std::log(sumW) - sumWlogW / sumW;
    }

    /**
     * @brief Compute entropy for one cell from a provided mask view.
     * @param cellIdx Flattened cell index.
     * @param masks Host mask view to read from.
     * @return Entropy value.
     */
    double entropyFromMaskCell(int cellIdx, const HostWordView &masks) const
    {
        double sumW = 0.0;
        double sumWlogW = 0.0;
        for (int t = 0; t < T; t++)
        {
            if (hasBit(masks, cellIdx, t, wordsPerMask))
            {
                sumW += freq[t];
                sumWlogW += freq[t] * logFreq[t];
            }
        }
        return shannonEntropy(sumW, sumWlogW);
    }

    /**
     * @brief Compute entropy for one cell from current host masks.
     * @param cellIdx Flattened cell index.
     * @return Entropy value.
     */
    double entropyForCell(int cellIdx) const
    {
        return entropyFromMaskCell(cellIdx, currentHost);
    }

    /**
     * @brief Recompute entropy cache for all cells.
     */
    void recomputeAllEntropies()
    {
        for (int cell = 0; cell < cells; cell++)
        {
            state.entropy[cell] = entropyFromMaskCell(cell, currentHost);
        }
    }

    /**
     * @brief Copy current host masks to device.
     */
    void syncDeviceFromHost()
    {
        Kokkos::deep_copy(currentDev, currentHost);
    }

    /**
     * @brief Copy current device masks to host and refresh entropy cache.
     */
    void syncHostFromDevice()
    {
        Kokkos::deep_copy(currentHost, currentDev);
        for (int cell = 0; cell < cells; cell++)
        {
            std::size_t base = (std::size_t)cell * (std::size_t)wordsPerMask;
            for (int w = 0; w < wordsPerMask; w++)
            {
                state.masks[base + w] = currentHost(base + w);
            }
        }
        recomputeAllEntropies();
    }

    /**
     * @brief Overwrite one cell mask on host and refresh its entropy.
     * @param cellIdx Flattened cell index.
     * @param maskWords New packed mask words.
     */
    void updateCellMaskHost(int cellIdx, const std::vector<Word> &maskWords)
    {
        std::size_t base = (std::size_t)cellIdx * (std::size_t)wordsPerMask;
        for (int w = 0; w < wordsPerMask; w++)
        {
            currentHost(base + w) = maskWords[w];
            state.masks[base + w] = maskWords[w];
        }
        state.entropy[cellIdx] = entropyFromMaskCell(cellIdx, currentHost);
    }

    /**
     * @brief Extract one cell mask as a vector.
     * @param cellIdx Flattened cell index.
     * @return Packed mask words.
     */
    std::vector<Word> cellMaskVector(int cellIdx) const
    {
        std::vector<Word> mask(wordsPerMask, Word(0));
        std::size_t base = (std::size_t)cellIdx * (std::size_t)wordsPerMask;
        for (int w = 0; w < wordsPerMask; w++)
        {
            mask[w] = currentHost(base + w);
        }
        return mask;
    }

    /**
     * @brief Count active tiles for one cell.
     * @param cellIdx Flattened cell index.
     * @return Number of active tile candidates.
     */
    int cellPopcount(int cellIdx) const
    {
        std::size_t base = (std::size_t)cellIdx * (std::size_t)wordsPerMask;
        int count = 0;
        for (int w = 0; w < wordsPerMask; w++)
        {
            count += popcountWord(currentHost(base + w));
        }
        return count;
    }

    /**
     * @brief Select the next cell with minimum entropy.
     * @return Cell index, -1 when complete, or -2 on contradiction.
     */
    int pickLowestEntropy()
    {
        int bestCell = -1;
        double bestEntropy = std::numeric_limits<double>::infinity();

        for (int cell = 0; cell < cells; cell++)
        {
            int count = cellPopcount(cell);
            if (count == 0)
            {
                return -2;
            }
            if (count == 1)
            {
                continue;
            }

            double e = state.entropy[cell] + noiseDist(rng);
            if (e < bestEntropy)
            {
                bestEntropy = e;
                bestCell = cell;
            }
        }

        return bestCell;
    }

    /**
     * @brief Choose one tile for a cell according to frequency weights.
     * @param cellIdx Flattened cell index.
     * @return Chosen tile index, or -1 if no candidate is available.
     */
    int chooseTile(int cellIdx)
    {
        std::vector<int> choices;
        std::vector<int> weights;
        for (int t = 0; t < T; t++)
        {
            if (hasBit(currentHost, cellIdx, t, wordsPerMask))
            {
                choices.push_back(t);
                weights.push_back(freq[t]);
            }
        }

        if (choices.empty())
        {
            return -1;
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return choices[dist(rng)];
    }

    /**
     * @brief Build a backtracking snapshot.
     * @param cellIdx Collapsed cell index.
     * @param chosenTile Chosen tile index.
     * @return Snapshot containing host-side solver state.
     */
    Snapshot makeSnapshot(int cellIdx, int chosenTile) const
    {
        Snapshot snap;
        snap.masks.assign(state.masks.begin(), state.masks.end());
        snap.entropy = state.entropy;
        snap.cell = cellIdx;
        snap.chosenTile = chosenTile;
        return snap;
    }

    /**
     * @brief Restore solver state from a snapshot.
     * @param snap Snapshot to restore.
     */
    void restoreSnapshot(const Snapshot &snap)
    {
        state.masks = snap.masks;
        state.entropy = snap.entropy;
        for (std::size_t i = 0; i < state.masks.size(); i++)
        {
            currentHost(i) = state.masks[i];
        }
        syncDeviceFromHost();
    }

    /**
     * @brief Run propagation iterations on device until convergence or contradiction.
     * @return True if propagation converges without contradiction.
     */
    bool propagateUntilStable()
    {
        syncDeviceFromHost();

        while (true)
        {
            Kokkos::deep_copy(nextDev, currentDev);
            Kokkos::deep_copy(changedFlagDev, 0);
            Kokkos::deep_copy(contradictionFlagDev, 0);

            auto current = currentDev;
            auto next = nextDev;
            auto compatView = compat.devView;
            int localRows = rows;
            int localCols = cols;
            int localCells = cells;
            int localWords = wordsPerMask;
            int localTiles = T;

            // Each iteration computes a synchronized next state from the current state.
            Kokkos::parallel_for(
                "WFCPropagate",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, localCells),
                KOKKOS_LAMBDA(const int cell) {
                    int x = cell % localCols;
                    int y = cell / localCols;
                    std::size_t base = (std::size_t)cell * (std::size_t)localWords;
                    bool changed = false;
                    bool anyNonZero = false;

                    // Capture direction offsets for device code
                    const int dx[4] = {0, 0, -1, 1};
                    const int dy[4] = {-1, 1, 0, 0};

                    for (int w = 0; w < localWords; w++)
                    {
                        Word wordMask = current(base + (std::size_t)w);
                        Word newWord = wordMask;

                        for (int dir = 0; dir < DIR_COUNT; dir++)
                        {
                            int nx = x + dx[dir];
                            int ny = y + dy[dir];
                            if (nx < 0 || ny < 0 || nx >= localCols || ny >= localRows)
                            {
                                continue;
                            }

                            std::size_t neighborBase = ((std::size_t)(ny * localCols + nx)) * (std::size_t)localWords;
                            // Build direction-constrained support for this word from neighbor domain.
                            Word allowedWord = 0;
                            for (int nw = 0; nw < localWords; nw++)
                            {
                                Word neighborWord = current(neighborBase + (std::size_t)nw);
                                while (neighborWord != 0)
                                {
                                    int bit = ctzWord(neighborWord);
                                    int tile = nw * 64 + bit;
                                    std::size_t compBase = compatWordOffset(dir, tile, w, localTiles, localWords);
                                    allowedWord |= compatView(compBase);
                                    neighborWord &= (neighborWord - 1);
                                }
                            }

                            newWord &= allowedWord;
                        }

                        next(base + (std::size_t)w) = newWord;
                        if (newWord != wordMask)
                        {
                            changed = true;
                        }
                        if (newWord != 0)
                        {
                            anyNonZero = true;
                        }
                    }

                    if (!anyNonZero)
                    {
                        Kokkos::atomic_add(&contradictionFlagDev(0), 1);
                    }
                    else if (changed)
                    {
                        Kokkos::atomic_add(&changedFlagDev(0), 1);
                    }
                });

            Kokkos::fence();

            auto changedHost = Kokkos::create_mirror_view(changedFlagDev);
            auto contradictionHost = Kokkos::create_mirror_view(contradictionFlagDev);
            Kokkos::deep_copy(changedHost, changedFlagDev);
            Kokkos::deep_copy(contradictionHost, contradictionFlagDev);

            if (contradictionHost(0) > 0)
            {
                return false;
            }

            std::swap(currentDev, nextDev);

            if (changedHost(0) == 0)
            {
                break;
            }
        }

        syncHostFromDevice();
        return true;
    }

    /**
     * @brief Collapse one cell to a single tile on host and sync to device.
     * @param cellIdx Flattened cell index.
     * @param chosenTile Tile index to keep.
     */
    void applyCollapse(int cellIdx, int chosenTile)
    {
        std::vector<Word> mask(wordsPerMask, Word(0));
        int wordIdx = chosenTile / 64;
        int bitIdx = chosenTile % 64;
        mask[wordIdx] = (Word(1) << bitIdx);
        updateCellMaskHost(cellIdx, mask);
        syncDeviceFromHost();
    }

    /**
     * @brief Backtrack by excluding previous choices until a valid state is found.
     * @param[in,out] backtracks Number of backtracks already performed.
     * @return True if a valid resumed branch is found.
     */
    bool backtrack(int &backtracks)
    {
        while (!history.empty() && backtracks < maxBacktracks)
        {
            Snapshot snap = history.top();
            history.pop();
            backtracks++;

            restoreSnapshot(snap);

            int cellIdx = snap.cell;
            int chosenTile = snap.chosenTile;
            int wordIdx = chosenTile / 64;
            int bitIdx = chosenTile % 64;
            currentHost(cellWordOffset(cellIdx, wordIdx, wordsPerMask)) &= ~(Word(1) << bitIdx);
            state.masks[cellWordOffset(cellIdx, wordIdx, wordsPerMask)] &= ~(Word(1) << bitIdx);
            state.entropy[cellIdx] = entropyFromMaskCell(cellIdx, currentHost);
            syncDeviceFromHost();

            if (isZeroCell(currentHost, cellIdx, wordsPerMask))
            {
                continue;
            }

            std::cout << "[BACKTRACK] #" << backtracks
                      << " excluding tile " << chosenTile
                      << " at (" << (cellIdx / cols) << "," << (cellIdx % cols) << ")\n";

            if (propagateUntilStable())
            {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Execute the WFC loop with propagation and bounded backtracking.
     * @return True if generation succeeds, false on unrecoverable contradiction.
     */
    bool run()
    {
        std::cout << "=== WFC " << rows << "x" << cols << " ===\n";
        int backtracks = 0;

        while (true)
        {
            int idx = pickLowestEntropy();

            if (idx == -2)
            {
                if (!backtrack(backtracks))
                {
                    std::cout << "[FAIL] Contradiction, no more backtrack options ("
                              << backtracks << " done)\n";
                    return false;
                }
                continue;
            }

            if (idx == -1)
            {
                std::cout << "=== All cells collapsed! ===\n";
                return true;
            }

            int chosen = chooseTile(idx);
            if (chosen < 0)
            {
                return false;
            }

            applyCollapse(idx, chosen);
            history.push(makeSnapshot(idx, chosen));

            if (!propagateUntilStable())
            {
                if (!backtrack(backtracks))
                {
                    std::cout << "[FAIL] Contradiction after collapse, no backtrack\n";
                    return false;
                }
            }
        }
    }

    /**
     * @brief Build final output grid by reading one tile per collapsed cell.
     * @return Output palette-index grid [rows][cols].
     */
    std::vector<std::vector<int>> buildOutput() const
    {
        std::vector<std::vector<int>> output(rows, std::vector<int>(cols, -1));
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int idx = r * cols + c;
                int tile = firstSetTile(currentHost, idx, wordsPerMask);
                if (tile >= 0)
                {
                    output[r][c] = tiles[tile].data[0][0];
                }
            }
        }
        return output;
    }

    /**
     * @brief Verify every NxN subgrid belongs to the learned tile set.
     * @param output Generated palette-index grid.
     * @return True if all subgrids are valid.
     */
    bool verify(const std::vector<std::vector<int>> &output) const
    {
        int errors = 0;
        for (int r = 0; r + N <= rows; r++)
        {
            for (int c = 0; c + N <= cols; c++)
            {
                std::vector<std::vector<int>> sub(N, std::vector<int>(N));
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        sub[i][j] = output[r + i][c + j];
                    }
                }

                bool found = false;
                for (const auto &t : tiles)
                {
                    if (t.data == sub)
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    std::cout << "[VERIFY ERROR] at (" << r << "," << c << ")\n";
                    errors++;
                }
            }
        }

        if (!errors)
        {
            std::cout << "[VERIFY] All subgrids valid.\n";
        }
        else
        {
            std::cout << "[VERIFY] " << errors << " invalid subgrids.\n";
        }
        return errors == 0;
    }

    /**
     * @brief Save generated output as a PNG image.
     * @param filename Destination file path.
     * @param output Generated palette-index grid.
     * @param palette RGB palette encoded as 0xRRGGBB.
     * @param scale Pixel scale per output cell.
     */
    void saveImage(const std::string &filename,
                   const std::vector<std::vector<int>> &output,
                   const std::vector<uint32_t> &palette,
                   int scale = 32)
    {
        int imgW = cols * scale;
        int imgH = rows * scale;
        std::vector<unsigned char> pixels(imgW * imgH * 3);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                uint32_t color = (output[r][c] < (int)palette.size()) ? palette[output[r][c]] : 0;
                unsigned char R = (color >> 16) & 0xFF;
                unsigned char G = (color >> 8) & 0xFF;
                unsigned char B = color & 0xFF;
                for (int py = 0; py < scale; py++)
                {
                    for (int px = 0; px < scale; px++)
                    {
                        int i = ((r * scale + py) * imgW + (c * scale + px)) * 3;
                        pixels[i] = R;
                        pixels[i + 1] = G;
                        pixels[i + 2] = B;
                    }
                }
            }
        }

        int ok = stbi_write_png(filename.c_str(), imgW, imgH, 3, pixels.data(), imgW * 3);
        if (ok)
        {
            std::cout << "[IMAGE] Saved " << filename << " (" << imgW << "x" << imgH << ")\n";
        }
        else
        {
            std::cout << "[IMAGE] Failed to save " << filename << "\n";
        }
    }
};

/**
 * @brief Program entry point.
 *
 * Command line:
 * - argv[1]: input image path (optional)
 * - argv[2]: output width in cells (optional)
 * - argv[3]: output height in cells (optional)
 * - argv[4]: tile size N (optional, default 3)
 * - argv[5]: render scale in pixels per cell (optional, default 32)
 *
 * If no image is provided, a built-in 5x5 binary sample is used.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument vector.
 * @return Process exit code.
 */
int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);

    int exitCode = 0;
    {
        using MainClock = std::chrono::high_resolution_clock;
        auto t_start = MainClock::now();

        std::vector<std::vector<int>> sample;
        std::vector<uint32_t> palette;

        if (argc >= 2)
        {
            sample = loadImage(argv[1], palette);
        }
        else
        {
            sample = {
                {1, 0, 1, 1, 1},
                {1, 0, 1, 1, 1},
                {0, 0, 1, 1, 1},
                {0, 1, 1, 1, 1},
                {0, 0, 0, 0, 0}};
            palette = {0x000000, 0xFFFFFF};
            std::cout << "[INPUT] Using built-in 5x5 sample.\n";
        }

        int inputRows = (int)sample.size();
        int inputCols = (int)sample[0].size();
        int outCols = (argc >= 3) ? std::stoi(argv[2]) : inputCols;
        int outRows = (argc >= 4) ? std::stoi(argv[3]) : inputRows;
        int N = (argc >= 5) ? std::stoi(argv[4]) : 3;
        int scale = (argc >= 6) ? std::stoi(argv[5]) : 32;

        std::cout << "[OUTPUT SIZE] " << outCols << "x" << outRows
                  << "  N=" << N << "  scale=" << scale << "\n";

        auto t_extract = MainClock::now();

        std::vector<Tile> tiles;
        std::vector<int> freq;
        extractTiles(sample, N, tiles, freq, /*useSymmetries=*/false);

        auto t_compat = MainClock::now();

        CompatTable compat;
        compat.build(tiles);

        auto t_wfc = MainClock::now();

        unsigned int seed = (unsigned int)
                                std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count();

        WFC wfc(outRows, outCols, N, tiles, freq, compat, seed, /*maxBacktracks=*/200);
        bool success = wfc.run();

        auto t_end = MainClock::now();

        if (success)
        {
            auto output = wfc.buildOutput();
            wfc.verify(output);
            wfc.saveImage("output.png", output, palette, scale);

            int inNonBg = 0;
            for (const auto &row : sample)
            {
                for (int v : row)
                {
                    inNonBg += (v != 0);
                }
            }
            double inDensity = (double)inNonBg / (inputRows * inputCols);

            int outNonBg = 0;
            for (const auto &row : output)
            {
                for (int v : row)
                {
                    outNonBg += (v != 0);
                }
            }
            double outDensity = (double)outNonBg / (outRows * outCols);

            std::cout << "[RESULT] WFC succeeded.\n";
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "[SIMILARITY] Input density:  " << inDensity << "\n";
            std::cout << "[SIMILARITY] Output density: " << outDensity << "\n";
            std::cout << "[SIMILARITY] Difference:     " << std::abs(outDensity - inDensity) << "\n";
        }
        else
        {
            std::cout << "[RESULT] WFC failed.\n";
            exitCode = 1;
        }

        auto elapsed_ms = [](MainClock::time_point a, MainClock::time_point b)
        {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        std::cout << "\n"
                  << std::string(50, '=') << "\n";
        std::cout << "[TIMING] wfc_kokkos\n";
        std::cout << "  Extract tiles:         " << std::fixed << std::setprecision(2)
                  << elapsed_ms(t_start, t_extract) << " ms\n";
        std::cout << "  Build compatibility:   " << elapsed_ms(t_extract, t_compat) << " ms\n";
        std::cout << "  WFC algorithm:         " << elapsed_ms(t_compat, t_wfc) << " ms\n";
        std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
        std::cout << std::string(50, '=') << "\n";
    }

    Kokkos::finalize();
    return exitCode;
}
