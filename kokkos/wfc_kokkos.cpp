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
#include <unordered_set>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// WFC_STATS=1 : compteurs de profilage (campagne "leviers").
// Imprime une ligne [STATS] key=value en fin de programme. Compteurs seuls :
// aucun effet sur l'algorithme ni sur le resultat, desactive par defaut.
// ---------------------------------------------------------------------------
#include <cstdlib>
#include <atomic>
namespace wfcstats {
static const bool on = std::getenv("WFC_STATS") != nullptr;
static unsigned long long propagate_calls = 0, sweeps = 0, pops = 0, pushes = 0,
                          critical_entries = 0, regions = 0, collapses = 0,
                          queue_max = 0;
static std::atomic<unsigned long long> tasks{0};
static double sync_ms = 0.0, work_ms = 0.0;

static inline void bump(unsigned long long &c) { if (on) ++c; }
static inline void note_queue(unsigned long long depth)
{
    if (on && depth > queue_max) queue_max = depth;
}
static void print(const char *code)
{
    if (!on) return;
    std::cout << "[STATS] code=" << code
              << " propagate_calls=" << propagate_calls
              << " sweeps=" << sweeps << " pops=" << pops << " pushes=" << pushes
              << " tasks=" << tasks.load() << " critical_entries=" << critical_entries
              << " regions=" << regions << " collapses=" << collapses
              << " queue_max=" << queue_max
              << " sync_ms=" << sync_ms << " work_ms=" << work_ms << "\n";
}
} // namespace wfcstats

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

/** Compatibility model: full (2N-1)^2 overlap window, same as sequential/OpenMP. */

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
 * @param offset Offset index from CompatTable::offsetIdx (full (2N-1)^2 window).
 * @param tile Tile index (source/target depending on table convention).
 * @param wordIdx Word index inside one compatibility mask.
 * @param tileCount Number of tiles.
 * @param wordsPerMask Number of words per compatibility mask.
 * @return Flat offset in packed compatibility storage.
 */
KOKKOS_INLINE_FUNCTION static std::size_t compatWordOffset(int offset, int tile, int wordIdx, int tileCount, int wordsPerMask)
{
    return (((std::size_t)offset * (std::size_t)tileCount + (std::size_t)tile) * (std::size_t)wordsPerMask) + (std::size_t)wordIdx;
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
 * @brief Check whether two tiles are compatible for a relative offset.
 * Same overlap rule as sequential/OpenMP: every overlapping cell must match.
 * @param current Tile placed at current cell.
 * @param neighbor Tile placed at the neighbor cell.
 * @param dy Row offset from current to neighbor cell.
 * @param dx Column offset from current to neighbor cell.
 * @return True if overlap constraints are satisfied.
 */
bool canOverlap(const Tile &current, const Tile &neighbor, int dy, int dx)
{
    int N = current.N;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int bi = i - dy, bj = j - dx;
            if (bi >= 0 && bi < N && bj >= 0 && bj < N)
            {
                if (current.data[i][j] != neighbor.data[bi][bj])
                {
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * @brief Full-window overlap compatibility table stored as packed bit masks.
 *
 * Same model as sequential/OpenMP: for each offset (dy, dx) in the (2N-1)^2
 * window and each neighbor tile, the table stores the set of current-cell
 * tile indices that remain valid given that neighbor tile.
 */
struct CompatTable
{
    /** Number of unique tiles. */
    int T = 0;
    /** Tile size (tiles are NxN). */
    int N = 0;
    /** Number of offsets, equals (2N-1)^2. */
    int D = 0;
    /** Number of 64-bit words per compatibility mask. */
    int wordsPerMask = 0;
    /** Host staging storage for packed masks. */
    std::vector<Word> incoming;
    /** Host-side compatibility view. */
    HostWordView hostView;
    /** Device-side compatibility view. */
    DevWordView devView;

    /**
     * @brief Encode an offset (dy, dx) into a contiguous index.
     * @param dy Row offset in [-(N-1), N-1].
     * @param dx Column offset in [-(N-1), N-1].
     * @return Linear offset index in [0, D).
     */
    int offsetIdx(int dy, int dx) const
    {
        return (dy + N - 1) * (2 * N - 1) + (dx + N - 1);
    }

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
        N = tiles[0].N;
        D = (2 * N - 1) * (2 * N - 1);

        wordsPerMask = (T + 63) / 64;
        incoming.assign((std::size_t)D * (std::size_t)T * (std::size_t)wordsPerMask, Word(0));

        for (int a = 0; a < T; a++)
        {
            for (int b = 0; b < T; b++)
            {
                for (int dy = -(N - 1); dy <= N - 1; dy++)
                {
                    for (int dx = -(N - 1); dx <= N - 1; dx++)
                    {
                        if (dy == 0 && dx == 0)
                        {
                            continue;
                        }
                        if (canOverlap(tiles[a], tiles[b], dy, dx))
                        {
                            int wordIdx = a / 64;
                            int bitIdx = a % 64;
                            incoming[compatWordOffset(offsetIdx(dy, dx), b, wordIdx, T, wordsPerMask)] |= (Word(1) << bitIdx);
                        }
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
    // Packed sweep flag: low 20 bits count changed cells, high bits count
    // contradictions (one atomic_add target instead of two views + mirrors).
    // Grids here stay far below 2^20 cells, so the two counters never mix.
    /** Device-side packed changed/contradiction flag (size 1). */
    DevIntView flagDev;
    /** Host-side mirror of the packed flag, reused across sweeps. */
    Kokkos::View<int *, Kokkos::HostSpace> flagHost;
    // Host mask edits reach the device lazily: single-cell edits sync
    // immediately, full restores only set this flag and sync once at the
    // next propagate() entry (previously 3 full copies per backtrack pop).
    /** True when host masks changed since the last device sync. */
    bool hostDirty = true;

    /** Precomputed log-frequency weights. */
    std::vector<double> logFreq;
    /** Global sum(freq[t] * log(freq[t])). */
    double totalLogFreqSum = 0.0;
    /** Global sum(freq[t]). */
    double totalFreqSum = 0.0;
    std::mt19937 rng;
    std::uniform_real_distribution<double> noiseDist{0.0, 1e-6};
    // Per-collapse logging is I/O-heavy on big grids; enable with --verbose.
    bool verbose = false;
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
    int lastBacktracks = 0; // backtracks used by the last run() call
    // Sparse checkpoints: snapshot every K collapses (+ always #0), see
    // sequential variant. Memory O(cells^2/K); undoes up to K per pop.
    int collapses = 0;
    static const int kCheckpointEvery = 16;

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
          flagDev("flagDev", 1),
          rng(seed), maxBacktracks(maxBacktracks)
    {
        flagHost = Kokkos::create_mirror_view(flagDev);
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
     * @brief Copy one cell mask from host to device (collapse fast path).
     * @param cellIdx Flattened cell index.
     */
    void syncCellToDevice(int cellIdx)
    {
        std::size_t base = (std::size_t)cellIdx * (std::size_t)wordsPerMask;
        auto devSub = Kokkos::subview(currentDev, std::make_pair(base, base + (std::size_t)wordsPerMask));
        auto hostSub = Kokkos::subview(currentHost, std::make_pair(base, base + (std::size_t)wordsPerMask));
        Kokkos::deep_copy(devSub, hostSub);
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
     * @brief Restore solver state from a snapshot (host side only).
     *
     * Device sync is deferred via hostDirty: the next propagate() entry
     * performs the single full copy.
     *
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
        hostDirty = true;
    }

    /**
     * @brief Run propagation iterations on device until convergence or contradiction.
     *
     * Per-sweep host traffic is one scalar round-trip: `nextDev` is fully
     * overwritten each sweep (no seed copy), and changed/contradiction share
     * one packed atomic flag with a reused mirror.
     *
     * @return True if propagation converges without contradiction.
     */
    bool propagateUntilStable()
    {
        static constexpr int kChangedBits = 20;
        static constexpr int kContraUnit = 1 << kChangedBits;
        wfcstats::bump(wfcstats::propagate_calls);
        auto stats_ms = [](std::chrono::high_resolution_clock::time_point a,
                           std::chrono::high_resolution_clock::time_point b)
        { return std::chrono::duration<double, std::milli>(b - a).count(); };
        if (hostDirty)
        {
            syncDeviceFromHost();
            hostDirty = false;
        }

        while (true)
        {
            wfcstats::bump(wfcstats::sweeps);
            Kokkos::deep_copy(flagDev, 0);

            auto current = currentDev;
            auto next = nextDev;
            auto flagView = flagDev;
            auto compatView = compat.devView;
            int localRows = rows;
            int localCols = cols;
            int localCells = cells;
            int localWords = wordsPerMask;
            int localTiles = T;
            int localTileN = N;

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

                    // Capture tile size for device code
                    const int WN = localTileN;

                    for (int w = 0; w < localWords; w++)
                    {
                        Word wordMask = current(base + (std::size_t)w);
                        Word newWord = wordMask;

                        // Full (2N-1)^2 overlap window, same as sequential/OpenMP.
                        for (int dy = -(WN - 1); dy <= WN - 1; dy++)
                        {
                            for (int dx = -(WN - 1); dx <= WN - 1; dx++)
                            {
                                if (dx == 0 && dy == 0)
                                {
                                    continue;
                                }
                                int nx = x + dx;
                                int ny = y + dy;
                                if (nx < 0 || ny < 0 || nx >= localCols || ny >= localRows)
                                {
                                    continue;
                                }
                                int off = (dy + WN - 1) * (2 * WN - 1) + (dx + WN - 1);

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
                                        std::size_t compBase = compatWordOffset(off, tile, w, localTiles, localWords);
                                        allowedWord |= compatView(compBase);
                                        neighborWord &= (neighborWord - 1);
                                    }
                                }

                                newWord &= allowedWord;
                            }
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
                        Kokkos::atomic_add(&flagView(0), kContraUnit);
                    }
                    else if (changed)
                    {
                        Kokkos::atomic_add(&flagView(0), 1);
                    }
                });

            auto t_kernel0 = std::chrono::high_resolution_clock::now();
            Kokkos::fence();
            auto t_kernel1 = std::chrono::high_resolution_clock::now();

            Kokkos::deep_copy(flagHost, flagDev);
            auto t_sync1 = std::chrono::high_resolution_clock::now();
            // work_ms : attente du noyau (calcul device + lancement).
            // sync_ms : rapatriement device -> host.
            if (wfcstats::on)
            {
                wfcstats::work_ms += stats_ms(t_kernel0, t_kernel1);
                wfcstats::sync_ms += stats_ms(t_kernel1, t_sync1);
            }
            int packed = flagHost(0);

            if (packed >= kContraUnit)
            {
                return false;
            }

            std::swap(currentDev, nextDev);

            if (packed == 0)
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
        syncCellToDevice(cellIdx);
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
            hostDirty = true; // single full sync at propagate() entry

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
                    lastBacktracks = backtracks;
                    return false;
                }
                continue;
            }

            if (idx == -1)
            {
                std::cout << "=== All cells collapsed! ===\n";
                lastBacktracks = backtracks;
                return true;
            }

            int chosen = chooseTile(idx);
            if (chosen < 0)
            {
                return false;
            }

            // Sparse checkpoint of the PRE-collapse masks (see member docs).
            bool checkpoint = (collapses == 0 || collapses % kCheckpointEvery == 0);
            if (checkpoint)
                history.push(makeSnapshot(idx, chosen));
            collapses++;
            wfcstats::bump(wfcstats::collapses);
            applyCollapse(idx, chosen);
            if (verbose)
                std::cout << "[COLLAPSE] (" << (idx / cols) << "," << (idx % cols) << ") -> tile " << chosen << "\n";

            if (!propagateUntilStable())
            {
                if (!backtrack(backtracks))
                {
                    std::cout << "[FAIL] Contradiction after collapse, no backtrack\n";
                    lastBacktracks = backtracks;
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
        // Hash tile set once so each subgrid check is O(N^2), not O(T*N^2).
        std::unordered_set<std::string> tileSet;
        for (const auto &t : tiles)
        {
            std::string key;
            key.reserve((std::size_t)N * (std::size_t)N * 4);
            for (const auto &row : t.data)
            {
                for (int v : row)
                {
                    key += std::to_string(v);
                    key += ',';
                }
            }
            tileSet.insert(key);
        }
        int errors = 0;
        std::string sub;
        for (int r = 0; r + N <= rows; r++)
        {
            for (int c = 0; c + N <= cols; c++)
            {
                sub.clear();
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        sub += std::to_string(output[r + i][c + j]);
                        sub += ',';
                    }
                }

                if (tileSet.find(sub) == tileSet.end())
                {
                    std::cout << "[VERIFY_FAIL] at (" << r << "," << c << ")\n";
                    errors++;
                }
            }
        }

        if (!errors)
        {
            std::cout << "[VERIFY_OK] All subgrids valid.\n";
        }
        else
        {
            std::cout << "[VERIFY_FAIL] " << errors << " invalid subgrids.\n";
        }
        return errors == 0;
    }

    /**
     * @brief Save generated output as a PNG image.
     * @param filename Destination file path.
     * @param output Generated palette-index grid.
     * @param palette RGB palette encoded as 0xRRGGBB.
     * @param scale Pixel scale per output cell.
     * @return True if the file was written.
     */
    bool saveImage(const std::string &filename,
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
        return ok != 0;
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
 * - argv[6]: symmetry flag (optional, default on)
 * - argv[7]: RNG seed (optional, default 42)
 * - argv[8]: max backtracks per attempt (optional, default 200)
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

        std::vector<std::vector<int>> sample;
        std::vector<uint32_t> palette;

        bool verbose = false;
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--verbose")
                verbose = true;

        if (argc >= 2 && std::string(argv[1]) != "--verbose")
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
        bool useSymmetries = true;
        if (argc >= 7)
        {
            std::string flag = argv[6];
            useSymmetries = (flag != "0" && flag != "false" && flag != "False" &&
                             flag != "--no-symmetry");
        }
        unsigned int seed = 42u;
        if (argc >= 8 && std::string(argv[7]) != "--verbose")
            seed = (unsigned int)std::stoul(argv[7]);
        int maxBacktracks = 200;
        if (argc >= 9 && std::string(argv[8]) != "--verbose")
            maxBacktracks = std::stoi(argv[8]);

        if (outCols <= 0 || outRows <= 0 || N <= 0 || scale <= 0 || maxBacktracks <= 0)
        {
            std::cerr << "[ERROR] outW, outH, N, scale and maxBacktracks must be positive "
                         "(got " << outCols << "x" << outRows << " N=" << N
                      << " scale=" << scale << " maxBacktracks=" << maxBacktracks << ")\n";
            Kokkos::finalize();
            return 1;
        }

        std::cout << "[OUTPUT SIZE] " << outCols << "x" << outRows
                  << "  N=" << N << "  scale=" << scale
                  << "  symmetries=" << (useSymmetries ? "on" : "off")
                  << "  seed=" << seed << "\n";

        auto t_start = MainClock::now();

        std::vector<Tile> tiles;
        std::vector<int> freq;
        extractTiles(sample, N, tiles, freq, useSymmetries);

        if (tiles.empty())
        {
            std::cerr << "[ERROR] No tiles extracted\n";
            Kokkos::finalize();
            return 1;
        }

        auto t_extract = MainClock::now();

        CompatTable compat;
        compat.build(tiles);

        auto t_compat = MainClock::now();

        // Up to 3 trajectories; a failed collapse restarts with the next seed.
        static const int kMaxAttempts = 3;
        bool success = false;
        int restarts = 0;
        int lastBacktracks = 0;
        for (int attempt = 0; attempt < kMaxAttempts && !success; ++attempt)
        {
            if (attempt > 0)
            {
                restarts = attempt;
                std::cout << "[RESTART] attempt " << (attempt + 1)
                          << " with seed " << (seed + (unsigned int)attempt) << "\n";
            }
            WFC wfc(outRows, outCols, N, tiles, freq, compat, seed + (unsigned int)attempt, maxBacktracks);
            wfc.verbose = verbose;
            if (!wfc.run())
            {
                lastBacktracks = wfc.lastBacktracks;
                continue;
            }
            lastBacktracks = wfc.lastBacktracks;
            auto output = wfc.buildOutput();
            if (!wfc.verify(output))
                continue;
            if (!wfc.saveImage("output.png", output, palette, scale))
                continue;
            success = true;
        }

        auto t_end = MainClock::now();
        std::cout << "[BACKTRACKS] " << lastBacktracks << " used, " << restarts << " restarts\n";

        if (success)
            std::cout << "[RESULT] WFC succeeded.\n";
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
        std::cout << "  WFC algorithm:         " << elapsed_ms(t_compat, t_end) << " ms\n";
        std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
        wfcstats::print("kokkos");
        std::cout << std::string(50, '=') << "\n";
    }

    Kokkos::finalize();
    return exitCode;
}
