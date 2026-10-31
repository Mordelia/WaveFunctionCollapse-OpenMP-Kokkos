#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <climits>
#include <cstdlib>
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
static double pick_ms = 0.0, collapse_ms = 0.0, backtrack_ms = 0.0,
              build_ms = 0.0, verify_ms = 0.0;
static double hostsync_ms = 0.0, refresh_ms = 0.0, choose_ms = 0.0,
              snap_ms = 0.0, seed_ms = 0.0, launch_ms = 0.0;
static unsigned long long frontier_cells = 0, frontier_max = 0;

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
              << " sync_ms=" << sync_ms << " work_ms=" << work_ms
              << " pick_ms=" << pick_ms << " collapse_ms=" << collapse_ms
              << " backtrack_ms=" << backtrack_ms
              << " build_ms=" << build_ms << " verify_ms=" << verify_ms
              << " hostsync_ms=" << hostsync_ms << " refresh_ms=" << refresh_ms
              << " choose_ms=" << choose_ms << " snap_ms=" << snap_ms
              << " seed_ms=" << seed_ms << " launch_ms=" << launch_ms
              << " frontier_cells=" << frontier_cells
              << " frontier_max=" << frontier_max
              << "\n";
}
} // namespace wfcstats

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

/**
 * @brief Parse a decimal integer argument, exiting cleanly on bad input.
 *
 * std::stoi throws on a non-numeric argument, which aborts the process with
 * SIGABRT; a typo on the command line must be a normal exit(1) instead.
 *
 * @param s Argument text.
 * @param what Option name, used in the error message.
 * @return Parsed value.
 */
static int parseInt(const char *s, const char *what)
{
    char *end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX)
    {
        std::cerr << "[ERROR] Invalid " << what << ": '" << s << "'\n";
        std::exit(1);
    }
    return (int)v;
}

/**
 * @brief Parse an unsigned 32-bit argument (seed), exiting cleanly on bad input.
 * @param s Argument text.
 * @param what Option name, used in the error message.
 * @return Parsed value.
 */
static unsigned int parseUInt(const char *s, const char *what)
{
    char *end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > 0xFFFFFFFFull)
    {
        std::cerr << "[ERROR] Invalid " << what << ": '" << s << "'\n";
        std::exit(1);
    }
    return (unsigned int)v;
}

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
 * @param offset Cardinal direction index (0..3, see CompatTable::DIRS).
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
 * @brief Allowed values for word @p w through direction @p d.
 * @return Mask of values allowed by the neighbour; all ones on the border.
 */
KOKKOS_INLINE_FUNCTION static Word allowedMaskFor(
    int d, int w, int x, int y, int cols, int rows, int words, int tiles,
    const DevWordView &current, const DevWordView &compatView,
    const DevWordView &uniView, const DevWordView &fullView, bool useUni)
{
    const int ny = y + ((d == 0) ? 1 : (d == 1) ? -1 : 0);
    const int nx = x + ((d == 2) ? 1 : (d == 3) ? -1 : 0);
    if (nx < 0 || ny < 0 || nx >= cols || ny >= rows)
    {
        return ~(Word)0;
    }
    const std::size_t neighborBase = ((std::size_t)(ny * cols + nx)) * (std::size_t)words;
    Word allowedWord = 0;
    const Word fullRow = fullView[(std::size_t)d * (std::size_t)words + (std::size_t)w];
    if (useUni)
    {
        for (int nw = 0; nw < words; nw++)
        {
            Word neighborWord = current(neighborBase + (std::size_t)nw);
            if (neighborWord == ~(Word)0)
            {
                allowedWord |= fullRow;
                continue;
            }
            for (int by = 0; by < 8 && neighborWord; by++, neighborWord >>= 8)
            {
                const int v = (int)(neighborWord & 0xFF);
                if (!v)
                    continue;
                allowedWord |= uniView(((((std::size_t)d * words + w) * words + nw) * 8 + by) * 256 + v);
            }
        }
    }
    else
    {
        for (int nw = 0; nw < words; nw++)
        {
            Word neighborWord = current(neighborBase + (std::size_t)nw);
            if (neighborWord == ~(Word)0)
            {
                allowedWord |= fullRow;
                continue;
            }
            while (neighborWord != 0)
            {
                const int bit = ctzWord(neighborWord);
                const int tile = nw * 64 + bit;
                allowedWord |= compatView(compatWordOffset(d, tile, w, tiles, words));
                neighborWord &= (neighborWord - 1);
            }
        }
    }
    return allowedWord;
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
    /** Cardinal directions, in the order used as table index. */
    static constexpr int DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    /** Number of unique tiles. */
    int T = 0;
    /** Tile size (tiles are NxN). */
    int N = 0;
    /** Number of stored offsets: the four cardinal directions. */
    int D = 4;
    /** Number of 64-bit words per compatibility mask. */
    int wordsPerMask = 0;
    /** Host staging storage for packed masks. */
    std::vector<Word> incoming;
    /** Host-side compatibility view. */
    HostWordView hostView;
    /** Device-side compatibility view. */
    DevWordView devView;
    // Union lookup table, eight domain bits at a time.
    // uni[((((dir*W + w) * W + nw) * 8 + byte) * 256 + v)] is the OR of the
    // compatibility rows of the tiles selected by byte v of neighbour word nw,
    // restricted to the cell's word w. The kernel's inner loop otherwise unions
    // popcount(neighbour domain) rows per (word, direction), which is 10 to 100
    // times more arithmetic than this table needs.
    /** Host staging for the union table. */
    std::vector<Word> uniHost;
    /** Device union table. */
    DevWordView uniDev;
    /** True when the union table is built (small masks only). */
    bool useUni = false;
    // Union of every tile's compatibility row per (direction, word): what a
    // neighbour whose word is still full allows. One lookup replaces the eight
    // byte lookups of the union path (and the popcount loop of the compat
    // path), which is the common case early in a run.
    /** Host staging for the full-word rows. */
    std::vector<Word> fullHost;
    /** Device full-word rows, D*wordsPerMask words. */
    DevWordView fullDev;
    /** Largest words-per-mask for which the table stays under ~16 MB. */
    static const int kMaxUniWords = 16;

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

        // Only the four cardinal offsets are stored: for the overlapping model
        // the constraint at (dy,dx) is implied by the chain of unit steps, so
        // the propagation fixed point is identical with 4/(2N-1)^2 of the work.
        wordsPerMask = (T + 63) / 64;
        incoming.assign((std::size_t)D * (std::size_t)T * (std::size_t)wordsPerMask, Word(0));

        for (int a = 0; a < T; a++)
        {
            for (int b = 0; b < T; b++)
            {
                for (int d = 0; d < D; d++)
                {
                    if (canOverlap(tiles[a], tiles[b], DIRS[d][0], DIRS[d][1]))
                    {
                        int wordIdx = a / 64;
                        int bitIdx = a % 64;
                        incoming[compatWordOffset(d, b, wordIdx, T, wordsPerMask)] |= (Word(1) << bitIdx);
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

        useUni = (wordsPerMask <= kMaxUniWords);
        if (useUni)
        {
            const int W = wordsPerMask;
            uniHost.assign((std::size_t)D * W * W * 8 * 256, Word(0));
            for (int d = 0; d < D; d++)
                for (int w = 0; w < W; w++)
                    for (int nw = 0; nw < W; nw++)
                        for (int by = 0; by < 8; by++)
                            for (int v = 1; v < 256; v++)
                            {
                                std::size_t base = ((((std::size_t)d * W + w) * W + nw) * 8 + by) * 256 + v;
                                const int bit = __builtin_ctz((unsigned)v);
                                const int prev = v & (v - 1);
                                const int t = nw * 64 + by * 8 + bit;
                                Word acc = prev ? uniHost[((((std::size_t)d * W + w) * W + nw) * 8 + by) * 256 + prev] : Word(0);
                                if (t < T)
                                    acc |= incoming[compatWordOffset(d, t, w, T, W)];
                                uniHost[base] = acc;
                            }
            uniDev = DevWordView("uni_dev", uniHost.size());
            Kokkos::deep_copy(uniDev, Kokkos::View<const Word *, Kokkos::HostSpace>(uniHost.data(), uniHost.size()));
        }
        fullHost.assign((std::size_t)D * (std::size_t)wordsPerMask, Word(0));
        for (int d = 0; d < D; d++)
            for (int t = 0; t < T; t++)
                for (int w = 0; w < wordsPerMask; w++)
                    fullHost[(std::size_t)d * (std::size_t)wordsPerMask + (std::size_t)w] |=
                        incoming[compatWordOffset(d, t, w, T, wordsPerMask)];
        fullDev = DevWordView("full_dev", fullHost.size());
        Kokkos::deep_copy(fullDev,
                          Kokkos::View<const Word *, Kokkos::HostSpace>(fullHost.data(), fullHost.size()));

        std::cout << "[COMPAT] Table built for " << T << " tiles (" << wordsPerMask
                  << " words per mask, union table " << (useUni ? "on" : "off") << ")\n";
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
 * @brief Smallest power of two greater than or equal to @p n.
 * @param n Requested size (>= 1).
 * @return Power of two.
 */
static inline int nextPow2(int n)
{
    int cap = 1;
    while (cap < n)
    {
        cap <<= 1;
    }
    return cap;
}

/**
 * @brief Indexed min-heap of (entropy, cell) with decrease-key.
 *
 * The std::set it replaces did a red-black-tree erase plus insert for every
 * dirty cell of every propagation (247k equal_range and 276k inserts per
 * 128x128 run, 14 % of the forest profile). Here a cell has at most one entry
 * and its key moves in place. Ordering stays lexicographic on (key, cell) so
 * the pick sequence does not depend on the insertion order.
 */
struct CellHeap
{
    std::vector<std::pair<double, int>> h;
    std::vector<int> pos; // cell -> index in h, -1 when absent

    void init(int cells)
    {
        h.clear();
        h.reserve((size_t)cells);
        pos.assign((size_t)cells, -1);
    }
    bool empty() const { return h.empty(); }
    bool has(int cell) const { return pos[(size_t)cell] >= 0; }
    static bool less(const std::pair<double, int> &a, const std::pair<double, int> &b)
    {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    }
    void swapAt(int i, int j)
    {
        std::swap(h[(size_t)i], h[(size_t)j]);
        pos[(size_t)h[(size_t)i].second] = i;
        pos[(size_t)h[(size_t)j].second] = j;
    }
    void up(int i)
    {
        while (i > 0)
        {
            const int p = (i - 1) / 2;
            if (!less(h[(size_t)i], h[(size_t)p]))
                break;
            swapAt(p, i);
            i = p;
        }
    }
    void down(int i)
    {
        const int n = (int)h.size();
        while (true)
        {
            int c = 2 * i + 1;
            if (c >= n)
                break;
            if (c + 1 < n && less(h[(size_t)c + 1], h[(size_t)c]))
                c++;
            if (!less(h[(size_t)c], h[(size_t)i]))
                break;
            swapAt(c, i);
            i = c;
        }
    }
    /** @brief Insert the cell, or move its key. */
    void set(int cell, double key)
    {
        const int i = pos[(size_t)cell];
        if (i < 0)
        {
            h.push_back({key, cell});
            pos[(size_t)cell] = (int)h.size() - 1;
            up((int)h.size() - 1);
        }
        else
        {
            const bool lower = key < h[(size_t)i].first;
            h[(size_t)i].first = key;
            if (lower)
                up(i);
            else
                down(i);
        }
    }
    void remove(int cell)
    {
        const int i = pos[(size_t)cell];
        if (i < 0)
            return;
        pos[(size_t)cell] = -1;
        const int last = (int)h.size() - 1;
        if (i != last)
        {
            h[(size_t)i] = h[(size_t)last];
            pos[(size_t)h[(size_t)i].second] = i;
            h.pop_back();
            down(i);
            up(i);
        }
        else
        {
            h.pop_back();
        }
    }
    /** @brief Remove and return the minimum (key, cell). */
    std::pair<double, int> pop()
    {
        const std::pair<double, int> top = h.front();
        remove(top.second);
        return top;
    }
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
    /** Current device mask view (read and written by propagation). */
    DevWordView currentDev;
    // One control container per sweep: [0] changed cells, [1] contradiction
    // flag, [2] next frontier count. A single memset and a single host copy per
    // sweep instead of two of each (they were 16 to 20 % of the run on CPU).
    /** Device control words. */
    DevIntView ctrlDev;
    /** Host mirror of ctrlDev. */
    Kokkos::View<int *, Kokkos::HostSpace> ctrlHost;
    /** Frontier size below which a host loop beats a parallel_for. */
    static constexpr int kSerialSweep = 8;
    // Frontier list: only the cells whose domain changed in the previous sweep
    // are processed. A full-grid sweep costs cells*dirs*words*domain even when a
    // single cell moved, which dominated large grids.
    /** Frontier ring: cumulative slot indices, masked with frontierMask. */
    DevIntView frontier;
    /** Power-of-two ring size minus one. */
    int frontierMask = 0;
    // Per-cell stamp of the sweep it was queued for: a cell can be queued again
    // for the next sweep without any clearing pass, and duplicates are rejected
    // without a race.
    /** Queue stamp per cell. */
    DevIntView queueEpoch;
    /** Monotonic sweep stamp (never reset: queueEpoch is global). */
    int epochCounter = 0;
    // Cellules dont le domaine a change depuis le debut de la propagation
    // courante. Ecrit par le noyau (un octet par cellule, sans atomique), lu
    // une seule fois a la fin : sert a ne recalculer l'entropie que de ces
    // cellules, au lieu d'un balayage O(cells*T) apres chaque propagation.
    /** Per-cell dirty flag (0/1), int so it can be swapped atomically. */
    DevIntView dirtyDev;
    /** Compacted list of the dirty cells of the current propagation. */
    DevIntView dirtyListDev;
    /** Host mirror of dirtyListDev (only [0, nDirty) is meaningful). */
    Kokkos::View<int *, Kokkos::HostSpace> dirtyListHost;
    // The propagation only changes the dirty cells, so only their masks are
    // copied back: a full-grid copy was 655 KB per propagation (5.4 GB per run
    // at 128x128 forest, ~24 % of the time).
    /** Compacted masks of the dirty cells, gathered on the device. */
    DevWordView dirtyMaskDev;
    /** Host mirror of dirtyMaskDev. */
    HostWordView dirtyMaskHost;
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
     * @brief Backtracking checkpoint: two undo-journal positions.
     *
     * The masks themselves are not copied any more (they were O(cells) per
     * checkpoint: 670 MB at 128x128). Every change since the checkpoint is
     * journalled, and a restore ORs those words back: propagation only clears
     * bits, so OR is the exact inverse.
     */
    struct Snapshot
    {
        /** Device journal length at checkpoint time. */
        int devPos = 0;
        /** Host journal length at checkpoint time. */
        int hostPos = 0;
        /** Cell index where collapse happened. */
        int cell = -1;
        /** Chosen tile at collapse time. */
        int chosenTile = -1;
    };

    /** Device undo journal: (flat word index, previous value) pairs. */
    DevWordView journalDev;
    /** Host mirror of journalDev (only the needed slice is copied). */
    HostWordView journalHost;
    /** Host undo journal for collapse and exclusion writes. */
    std::vector<Word> hostJournal;
    /** Number of valid words in hostJournal. */
    int hostJournalCount = 0;
    /** Device journal capacity, in (index, value) slots. */
    int journalCapacity = 0;


    std::stack<Snapshot> history;
    int maxBacktracks;
    int lastBacktracks = 0; // backtracks used by the last run() call
    // L9 : tas de priorite pour pickLowestEntropy. Contient (entropie, cellIdx)
    // pour les cellules actives (1 < count < T). Evite un scan O(cells) a
    // chaque iteration de la boucle principale.
    CellHeap entropyHeap;
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
          frontier("frontier", (std::size_t)nextPow2(cells)),
          ctrlDev("ctrlDev", 6),
          ctrlHost("ctrlHost", 6),
          queueEpoch("queueEpoch", (std::size_t)cells),
          journalDev("journalDev", (std::size_t)cells * (std::size_t)wordsPerMask * 2),
          journalHost("journalHost", (std::size_t)cells * (std::size_t)wordsPerMask * 2),
          journalCapacity(cells * wordsPerMask),
          dirtyDev("dirtyDev", (std::size_t)cells),
          dirtyListDev("dirtyListDev", (std::size_t)cells),
          dirtyListHost("dirtyListHost", (std::size_t)cells),
          dirtyMaskDev("dirtyMaskDev", (std::size_t)cells * (std::size_t)wordsPerMask),
          dirtyMaskHost("dirtyMaskHost", (std::size_t)cells * (std::size_t)wordsPerMask),

          rng(seed), maxBacktracks(maxBacktracks)
    {
        if (T <= 0 || T > 8192)
        {
            std::cerr << "[ERROR] This implementation expects 1..8192 tiles, got " << T << "\n";
            std::exit(1);
        }

        hostJournal.resize((std::size_t)cells * (std::size_t)wordsPerMask * 2);
        entropyHeap.init(cells);
        frontierMask = nextPow2(cells) - 1;
        // The control counters are cumulative and read on the host before the
        // first sweep: zero them once, device and mirror.
        Kokkos::deep_copy(ctrlDev, 0);
        Kokkos::deep_copy(ctrlHost, ctrlDev);
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
                currentHost(cellWordOffset(cell, w, wordsPerMask)) = value;
            }
            state.entropy[cell] = entropyForCell(cell);
            entropyHeap.set(cell, state.entropy[cell]);
        }

        Kokkos::deep_copy(currentDev, currentHost);
        Kokkos::deep_copy(queueEpoch, 0);
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
        const std::size_t base = (std::size_t)cellIdx * (std::size_t)wordsPerMask;
        for (int w = 0; w < wordsPerMask; w++)
        {
            Word word = masks(base + (std::size_t)w);
            while (word != 0)
            {
                const int bit = ctzWord(word);
                word &= (word - 1);
                const int t = w * 64 + bit;
                if (t < T)
                {
                    sumW += freq[t];
                    sumWlogW += freq[t] * logFreq[t];
                }
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
        // currentHost is the single host copy of the domains (the old mirror
        // vector duplicated it and cost a cells*words host loop per collapse).
        Kokkos::deep_copy(currentHost, currentDev);
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
        auto t_pick0 = Clock::now();
        // L9 : extraire jusqu'a 5 candidats du tas, choisir le meilleur avec
        // bruit, remettre les autres. Le bruit etant minuscule (1e-6), le tas
        // trie par entropie de base donne presque toujours le bon ordre ; cette
        // marche de securite garantit la correctitude dans les cas rares.
        std::vector<std::pair<double, int>> candidates;
        while (!entropyHeap.empty() && (int)candidates.size() < 5)
        {
            auto top = entropyHeap.pop();
            int cell = top.second;

            int count = cellPopcount(cell);
            if (count == 0)
            {
                // Contradiction detectee via le tas
                if (wfcstats::on)
                    wfcstats::pick_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_pick0).count();
                return -2;
            }
            if (count == 1)
            {
                // Cellule collepassee entre-temps, ignorer definitivement
                continue;
            }
            candidates.push_back(top);
        }

        if (candidates.empty())
        {
            // Tas vide : soit tout est collepasse (-1), soit il reste une
            // cellule vide (-2) qui n'etait pas encore dans le tas. Scan
            // lineaire de securite (rare, seulement apres un backtrack).
            for (int cell = 0; cell < cells; cell++)
            {
                int count = cellPopcount(cell);
                if (count == 0)
                {
                    if (wfcstats::on)
                        wfcstats::pick_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_pick0).count();
                    return -2;
                }
                if (count > 1 && !entropyHeap.has(cell))
                {
                    // Cellule active oubliee : la remettre dans le tas
                    entropyHeap.set(cell, state.entropy[cell]);
                }
            }
            if (entropyHeap.empty())
            {
                if (wfcstats::on)
                    wfcstats::pick_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_pick0).count();
                return -1;
            }
            // Relancer avec le candidat tombe du scan
            candidates.push_back(entropyHeap.pop());
        }

        // Choisir le meilleur candidat avec bruit
        int bestCell = -1;
        double bestEntropy = std::numeric_limits<double>::infinity();
        for (const auto &cand : candidates)
        {
            double e = cand.first + noiseDist(rng);
            if (e < bestEntropy)
            {
                bestEntropy = e;
                bestCell = cand.second;
            }
        }

        // Remettre les non-choisis dans le tas
        for (const auto &cand : candidates)
        {
            if (cand.second != bestCell && !entropyHeap.has(cand.second))
            {
                entropyHeap.set(cand.second, cand.first);
            }
        }

        if (wfcstats::on)
            wfcstats::pick_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_pick0).count();
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
    /**
     * @brief Grow the undo journal so one full-grid propagation still fits.
     * @param used Slots already written.
     */
    void ensureJournalCapacity(int used)
    {
        const int headroom = cells * wordsPerMask;
        if (used + headroom <= journalCapacity)
        {
            return;
        }
        const int newCapacity = std::max(journalCapacity * 2, used + headroom);
        DevWordView biggerDev("journalDev", (std::size_t)newCapacity * 2);
        Kokkos::deep_copy(Kokkos::subview(biggerDev, std::make_pair(0, used * 2)),
                          Kokkos::subview(journalDev, std::make_pair(0, used * 2)));
        journalDev = biggerDev;
        journalHost = HostWordView("journalHost", (std::size_t)newCapacity * 2);
        journalCapacity = newCapacity;
    }

    Snapshot makeSnapshot(int cellIdx, int chosenTile)
    {
        Snapshot snap;
        snap.cell = cellIdx;
        snap.chosenTile = chosenTile;
        snap.hostPos = hostJournalCount;
        Kokkos::deep_copy(Kokkos::subview(ctrlHost, std::make_pair(4, 5)),
                          Kokkos::subview(ctrlDev, std::make_pair(4, 5)));
        snap.devPos = ctrlHost(4);
        ensureJournalCapacity(snap.devPos);
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
    bool restoreSnapshot(const Snapshot &snap)
    {
        // Device journal: OR back every word the propagation cleared since the
        // checkpoint (only the slice is copied, not the whole grid).
        Kokkos::deep_copy(Kokkos::subview(ctrlHost, std::make_pair(4, 5)),
                          Kokkos::subview(ctrlDev, std::make_pair(4, 5)));
        const int devCount = ctrlHost(4);
        if (ctrlHost(5) > 0)
        {
            // Journal overflow: the undo would be partial, refuse it.
            std::cerr << "[ERROR] undo journal overflow, cannot restore\n";
            return false;
        }
        if (devCount > snap.devPos)
        {
            // The slice starts at the checkpoint slot, not at 0.
            Kokkos::deep_copy(Kokkos::subview(journalHost, std::make_pair(0, 2 * (devCount - snap.devPos))),
                              Kokkos::subview(journalDev, std::make_pair(2 * snap.devPos, 2 * devCount)));
            for (int k = 0; k < devCount - snap.devPos; k++)
            {
                const std::size_t e = (std::size_t)k * 2;
                currentHost(journalHost(e)) |= journalHost(e + 1);
            }
        }
        Kokkos::deep_copy(Kokkos::subview(ctrlDev, std::make_pair(4, 5)), snap.devPos);
        // Host journal: collapse and exclusion writes.
        for (int k = snap.hostPos; k < hostJournalCount; k += 2)
        {
            currentHost(hostJournal[(std::size_t)k]) |= hostJournal[(std::size_t)k + 1];
        }
        hostJournalCount = snap.hostPos;
        hostDirty = true;
        // L9 : reconstruire le tas depuis l'etat restaure
        entropyHeap.init(cells);
        for (int cell = 0; cell < cells; cell++)
        {
            state.entropy[cell] = entropyFromMaskCell(cell, currentHost);
            if (cellPopcount(cell) > 1)
            {
                entropyHeap.set(cell, state.entropy[cell]);
            }
        }
        return true;
    }

    /**
     * @brief Run propagation on device until convergence or contradiction.
     *
     * Only the cells whose domain changed in the previous sweep are processed:
     * the frontier is built on the device and updated in place. The update is a
     * monotone intersection, so reading a neighbour that was already shrunk in
     * this sweep is sound (it can only under-propagate, and that neighbour's
     * change already queued this cell). Same fixed point as a full-grid sweep,
     * for a fraction of the work and one host round-trip per sweep.
     *
     * @param startCell Cell whose domain was just collapsed or restricted.
     * @return True if propagation converges without contradiction.
     */
    bool propagateUntilStable(int startCell)
    {
        wfcstats::bump(wfcstats::propagate_calls);
        auto stats_ms = [](std::chrono::high_resolution_clock::time_point a,
                           std::chrono::high_resolution_clock::time_point b)
        { return std::chrono::duration<double, std::milli>(b - a).count(); };

        if (hostDirty)
        {
            syncDeviceFromHost();
            hostDirty = false;
        }
        Kokkos::deep_copy(dirtyDev, 0);

        // [3] dirty-list and [4] journal length accumulate over the whole
        // propagation: only [3] is reset here, never [4].
        Kokkos::deep_copy(Kokkos::subview(ctrlDev, std::make_pair(3, 4)), 0);

        // Seed the frontier with the *neighbours* of the cell that just changed:
        // its own domain is already final (a collapse writes a singleton, and
        // re-processing it would report no change), but its neighbours must see
        // it. Invalid neighbours are written as -1 and skipped by the first
        // sweep, so the count does not have to be read back (one kernel instead
        // of a memset + a kernel + a copy).
        auto t_seed0 = std::chrono::high_resolution_clock::now();
        {
            auto seedList = frontier;
            auto seedEpoch = queueEpoch;
            auto seedCtrl = ctrlDev;
            const int seedStamp = ++epochCounter;
            const int seedCell = startCell;
            const int seedCols = cols;
            const int seedRows = rows;
            const int seedBase = ctrlHost(2);
            const int seedMask = frontierMask;
            Kokkos::parallel_for(
                "WFCSeed", 4, KOKKOS_LAMBDA(const int d) {
                    const int sy = seedCell / seedCols;
                    const int sx = seedCell % seedCols;
                    const int ny = sy + ((d == 0) ? 1 : (d == 1) ? -1 : 0);
                    const int nx = sx + ((d == 2) ? 1 : (d == 3) ? -1 : 0);
                    // [2] is the cumulative frontier counter: the first sweep
                    // starts right after the four seed slots.
                    if (d == 0)
                    {
                        seedCtrl(2) = seedBase + 4;
                    }
                    if (nx < 0 || ny < 0 || nx >= seedCols || ny >= seedRows)
                    {
                        seedList((seedBase + d) & seedMask) = -1;
                        return;
                    }
                    const int ncell = ny * seedCols + nx;
                    // Stamp for the first sweep so a duplicate neighbour is not
                    // enqueued twice.
                    seedList((seedBase + d) & seedMask) =
                        (Kokkos::atomic_exchange(&seedEpoch(ncell), seedStamp) != seedStamp)
                            ? ncell
                            : -1;
                });
        }
        if (wfcstats::on)
            wfcstats::seed_ms += std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() - t_seed0).count();
        int count = 4; // the first sweep skips the -1 entries
        // Ring base of the current level. The control counters are cumulative:
        // the host derives each level's base and size from their deltas, so no
        // per-sweep memset is needed (it was 20 % of the forest profile).
        int listBase = ctrlHost(2);
        const int contraBefore = ctrlHost(1);

        while (true)
        {
            wfcstats::bump(wfcstats::sweeps);
            const int nextEpoch = ++epochCounter;
            const int n = count;
            const int nextBase = listBase + n;
            const int ringMask = frontierMask;

            auto current = currentDev;
            auto curList = frontier;
            const int curBase = listBase;
            auto epochView = queueEpoch;
            auto ctrl = ctrlDev;
            auto compatView = compat.devView;
            auto uniView = compat.uniDev;
            auto fullView = compat.fullDev;
            const bool useUniK = compat.useUni;
            auto dirty = dirtyDev;
            auto dirtyList = dirtyListDev;
            auto journal = journalDev;
            const int localWords = wordsPerMask;
            const int localTiles = T;
            const int localCols = cols;
            const int localRows = rows;
            const int journalCap = journalCapacity;

            if (wfcstats::on)
            {
                wfcstats::frontier_cells += (unsigned long long)n;
                if ((unsigned long long)n > wfcstats::frontier_max)
                    wfcstats::frontier_max = (unsigned long long)n;
            }

            [[maybe_unused]] auto body = KOKKOS_LAMBDA(const int i) {
                const int cell = curList((curBase + i) & ringMask);
                if (cell < 0)
                {
                    return;
                }
                const int x = cell % localCols;
                const int y = cell / localCols;
                const std::size_t base = (std::size_t)cell * (std::size_t)localWords;
                Word before[16];
                for (int w = 0; w < localWords; w++)
                {
                    before[w] = current(base + (std::size_t)w);
                }
                bool changed = false;
                bool anyNonZero = false;

                for (int w = 0; w < localWords; w++)
                {
                    Word newWord = before[w];
                    // Cardinal offsets: the diagonal constraints are implied by
                    // the chain of unit steps (same fixed point).
                    for (int d = 0; d < 4; d++)
                    {
                        newWord &= allowedMaskFor(d, w, x, y, localCols, localRows, localWords,
                                                  localTiles, current, compatView, uniView, fullView,
                                                  useUniK);
                    }
                    current(base + (std::size_t)w) = newWord;
                    if (newWord != before[w])
                    {
                        changed = true;
                    }
                    if (newWord != 0)
                    {
                        anyNonZero = true;
                    }
                }

                if (changed)
                {
                    // Undo journal: one atomic for the whole cell, then plain
                    // stores. Journalled even on contradiction (words already
                    // cleared must be restored too).
                    const int jslot = Kokkos::atomic_fetch_add(&ctrl(4), localWords);
                    if (jslot + localWords <= journalCap)
                    {
                        for (int w = 0; w < localWords; w++)
                        {
                            journal((std::size_t)(jslot + w) * 2) = (Word)(base + (std::size_t)w);
                            journal((std::size_t)(jslot + w) * 2 + 1) = before[w];
                        }
                    }
                    else
                    {
                        Kokkos::atomic_add(&ctrl(5), 1);
                    }
                }

                if (!anyNonZero)
                {
                    Kokkos::atomic_add(&ctrl(1), 1);
                    return;
                }
                if (!changed)
                {
                    return;
                }
                if (Kokkos::atomic_exchange(&dirty(cell), 1) == 0)
                {
                    const int dslot = Kokkos::atomic_fetch_add(&ctrl(3), 1);
                    dirtyList(dslot) = cell;
                }
                for (int d = 0; d < 4; d++)
                {
                    const int ny = y + ((d == 0) ? 1 : (d == 1) ? -1 : 0);
                    const int nx = x + ((d == 2) ? 1 : (d == 3) ? -1 : 0);
                    if (nx < 0 || ny < 0 || nx >= localCols || ny >= localRows)
                    {
                        continue;
                    }
                    const int ncell = ny * localCols + nx;
                    if (Kokkos::atomic_exchange(&epochView(ncell), nextEpoch) != nextEpoch)
                    {
                        const int slot = Kokkos::atomic_fetch_add(&ctrl(2), 1);
                        curList(slot & ringMask) = ncell;
                    }
                }
            };

            auto t_launch0 = std::chrono::high_resolution_clock::now();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
            {
                // GPU: a frontier of ~20 cells fills a single warp with a range
                // policy. One team of 32 lanes per cell spreads the
                // (direction, word) lookups over the whole warp.
                using team_member =
                    typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
                const int scratchWords = 4 * localWords;
                Kokkos::parallel_for(
                    "WFCPropagate",
                    Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(n, 32)
                        .set_scratch_size(0, Kokkos::PerTeam((std::size_t)scratchWords * sizeof(Word))),
                    KOKKOS_LAMBDA(const team_member &team) {
                        const int i = team.league_rank();
                        const int cell = curList((curBase + i) & ringMask);
                        if (cell < 0)
                        {
                            return;
                        }
                        const int x = cell % localCols;
                        const int y = cell / localCols;
                        const std::size_t base = (std::size_t)cell * (std::size_t)localWords;
                        Kokkos::View<Word *, typename team_member::scratch_memory_space>
                            allowedView(team.team_scratch(0), (std::size_t)scratchWords);
                        Word *allowed = allowedView.data();

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, 4 * localWords),
                            [&](const int k) {
                                const int d = k / localWords;
                                const int w = k - d * localWords;
                                allowed[k] = allowedMaskFor(d, w, x, y, localCols, localRows,
                                                            localWords, localTiles, current,
                                                            compatView, uniView, fullView, useUniK);
                            });
                        team.team_barrier();

                        // Pack both flags in one int: high 16 bits changed words,
                        // low 16 bits non-empty words (words per mask is <= 16).
                        int acc = 0;
                        Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(team, localWords),
                            [&](const int w, int &partial) {
                                const Word before = current(base + (std::size_t)w);
                                Word after = before;
                                for (int d = 0; d < 4; d++)
                                {
                                    after &= allowed[d * localWords + w];
                                }
                                current(base + (std::size_t)w) = after;
                                allowed[w] = before; // reused as the undo copy
                                partial += (int)(after != before) * 65536 + (int)(after != 0);
                            },
                            acc);

                        if ((acc & 0xFFFF) == 0)
                        {
                            if (team.team_rank() == 0)
                            {
                                Kokkos::atomic_add(&ctrl(1), 1);
                            }
                            return;
                        }
                        const int rank = team.team_rank();
                        if ((acc >> 16) == 0)
                        {
                            return;
                        }
                        {
                            int jslot = 0;
                            if (rank == 0)
                            {
                                jslot = Kokkos::atomic_fetch_add(&ctrl(4), localWords);
                            }
                            team.team_broadcast(jslot, 0);
                            if (jslot + localWords <= journalCap)
                            {
                                Kokkos::parallel_for(
                                    Kokkos::TeamThreadRange(team, localWords),
                                    [&](const int w) {
                                        journal((std::size_t)(jslot + w) * 2) =
                                            (Word)(base + (std::size_t)w);
                                        journal((std::size_t)(jslot + w) * 2 + 1) = allowed[w];
                                    });
                                team.team_barrier();
                            }
                            else if (rank == 0)
                            {
                                Kokkos::atomic_add(&ctrl(5), 1);
                            }
                        }
                        if (rank == 0)
                        {
                            if (Kokkos::atomic_exchange(&dirty(cell), 1) == 0)
                            {
                                const int dslot = Kokkos::atomic_fetch_add(&ctrl(3), 1);
                                dirtyList(dslot) = cell;
                            }
                        }
                        if (rank < 4)
                        {
                            const int d = rank;
                            const int ny = y + ((d == 0) ? 1 : (d == 1) ? -1 : 0);
                            const int nx = x + ((d == 2) ? 1 : (d == 3) ? -1 : 0);
                            if (nx < 0 || ny < 0 || nx >= localCols || ny >= localRows)
                            {
                                return;
                            }
                            const int ncell = ny * localCols + nx;
                            if (Kokkos::atomic_exchange(&epochView(ncell), nextEpoch) != nextEpoch)
                            {
                                const int slot = Kokkos::atomic_fetch_add(&ctrl(2), 1);
                                curList(slot & ringMask) = ncell;
                            }
                        }
                    });
            }
#else
            // A tiny frontier does not pay for a parallel_for on a CPU
            // backend: run the sweep as a plain loop.
            if (n <= kSerialSweep)
            {
                for (int i = 0; i < n; i++)
                    body(i);
            }
            else
            {
                Kokkos::parallel_for(
                    "WFCPropagate",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n), body);
            }
#endif
            if (wfcstats::on)
                wfcstats::launch_ms += std::chrono::duration<double, std::milli>(
                                           std::chrono::high_resolution_clock::now() - t_launch0).count();
            auto t_kernel0 = std::chrono::high_resolution_clock::now();
            Kokkos::fence();
            auto t_kernel1 = std::chrono::high_resolution_clock::now();
            Kokkos::deep_copy(ctrlHost, ctrlDev);
            auto t_sync1 = std::chrono::high_resolution_clock::now();
            if (wfcstats::on)
            {
                wfcstats::work_ms += stats_ms(t_kernel0, t_kernel1);
                wfcstats::sync_ms += stats_ms(t_kernel1, t_sync1);
            }

            if (ctrlHost(1) > contraBefore)
            {
                break;
            }
            count = ctrlHost(2) - nextBase;
            if (count == 0)
            {
                break;
            }
            listBase = nextBase;
        }

        // [1] is cumulative: only a *new* contradiction fails the propagation.
        if (ctrlHost(1) > contraBefore)
        {
            return false;
        }

        ensureJournalCapacity(ctrlHost(4));
        const int nDirty = ctrlHost(3);
        auto t_hs0 = std::chrono::high_resolution_clock::now();
        copyDirtyList(nDirty);
        syncDirtyMasksFromDevice(nDirty);
        auto t_hs1 = std::chrono::high_resolution_clock::now();
        refreshEntropyFromDirty(nDirty);
        if (wfcstats::on)
        {
            wfcstats::hostsync_ms += std::chrono::duration<double, std::milli>(t_hs1 - t_hs0).count();
            wfcstats::refresh_ms += std::chrono::duration<double, std::milli>(
                                        std::chrono::high_resolution_clock::now() - t_hs1).count();
        }

        return true;
    }

    /**
     * @brief Refresh entropy and heap keys for the cells changed by propagation.
     *
     * One device->host copy of the per-cell dirty flags, then a host pass over
     * those cells only. The heap entry of a cell is erased with its previous
     * entropy and re-inserted with the new one, so the set stays ordered by the
     * current entropy (previously the keys were never refreshed, every cell kept
     * its initial entropy and the solver collapsed in raster order).
     */
    void copyDirtyList(int nDirty)
    {
        if (nDirty <= 0)
        {
            return;
        }
        const auto range = std::make_pair(0, nDirty);
        Kokkos::deep_copy(Kokkos::subview(dirtyListHost, range),
                          Kokkos::subview(dirtyListDev, range));
    }

    /**
     * @brief Copy back the masks of the dirty cells only.
     * @param nDirty Number of entries in the compacted dirty list.
     */
    void syncDirtyMasksFromDevice(int nDirty)
    {
        if (nDirty <= 0)
        {
            return;
        }
        auto cur = currentDev;
        auto list = dirtyListDev;
        auto out = dirtyMaskDev;
        const int W = wordsPerMask;
        Kokkos::parallel_for(
            "WFCGatherMasks", nDirty, KOKKOS_LAMBDA(const int i) {
                const std::size_t src = (std::size_t)list(i) * (std::size_t)W;
                const std::size_t dst = (std::size_t)i * (std::size_t)W;
                for (int w = 0; w < W; w++)
                {
                    out(dst + (std::size_t)w) = cur(src + (std::size_t)w);
                }
            });
        const auto range = std::make_pair(0, nDirty * W);
        Kokkos::deep_copy(Kokkos::subview(dirtyMaskHost, range),
                          Kokkos::subview(dirtyMaskDev, range));
        for (int i = 0; i < nDirty; i++)
        {
            const std::size_t dst = (std::size_t)dirtyListHost((std::size_t)i) * (std::size_t)W;
            const std::size_t src = (std::size_t)i * (std::size_t)W;
            for (int w = 0; w < W; w++)
            {
                currentHost(dst + (std::size_t)w) = dirtyMaskHost(src + (std::size_t)w);
            }
        }
    }

    /**
     * @brief Refresh entropy and heap keys for the cells changed by propagation.
     * @param nDirty Number of entries in the compacted dirty list.
     */
    void refreshEntropyFromDirty(int nDirty)
    {
        if (nDirty <= 0)
        {
            return;
        }
        for (int i = 0; i < nDirty; i++)
        {
            const int cell = dirtyListHost((std::size_t)i);
            const double oldH = state.entropy[cell];
            const double newH = entropyFromMaskCell(cell, currentHost);
            if (newH == oldH)
            {
                continue;
            }
            entropyHeap.remove(cell);
            state.entropy[cell] = newH;
            if (cellPopcount(cell) > 1)
            {
                entropyHeap.set(cell, newH);
            }
        }
    }

    /**
     * @brief Collapse one cell to a single tile on host and sync to device.
     * @param cellIdx Flattened cell index.
     * @param chosenTile Tile index to keep.
     */
    void applyCollapse(int cellIdx, int chosenTile)
    {
        auto t_collapse0 = Clock::now();
        std::vector<Word> mask(wordsPerMask, Word(0));
        int wordIdx = chosenTile / 64;
        int bitIdx = chosenTile % 64;
        mask[wordIdx] = (Word(1) << bitIdx);
        // L9 : retirer la cellule du tas avant de modifier son entropie
        entropyHeap.remove(cellIdx);
        for (int w = 0; w < wordsPerMask; w++)
        {
            const std::size_t idx = cellWordOffset(cellIdx, w, wordsPerMask);
            hostJournal[(std::size_t)hostJournalCount++] = (Word)idx;
            hostJournal[(std::size_t)hostJournalCount++] = currentHost(idx);
        }
        updateCellMaskHost(cellIdx, mask);
        syncCellToDevice(cellIdx);
        if (wfcstats::on)
            wfcstats::collapse_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_collapse0).count();
    }

    /**
     * @brief Backtrack by excluding previous choices until a valid state is found.
     * @param[in,out] backtracks Number of backtracks already performed.
     * @return True if a valid resumed branch is found.
     */
    bool backtrack(int &backtracks)
    {
        auto t_bt0 = Clock::now();
        while (!history.empty() && backtracks < maxBacktracks)
        {
            Snapshot snap = history.top();
            history.pop();
            backtracks++;

            if (!restoreSnapshot(snap))
            {
                return false;
            }

            int cellIdx = snap.cell;
            int chosenTile = snap.chosenTile;
            int wordIdx = chosenTile / 64;
            int bitIdx = chosenTile % 64;
            {
                const std::size_t idx = cellWordOffset(cellIdx, wordIdx, wordsPerMask);
                hostJournal[(std::size_t)hostJournalCount++] = (Word)idx;
                hostJournal[(std::size_t)hostJournalCount++] = currentHost(idx);
                currentHost(idx) &= ~(Word(1) << bitIdx);
            }
            state.entropy[cellIdx] = entropyFromMaskCell(cellIdx, currentHost);
            hostDirty = true; // single full sync at propagate() entry

            if (isZeroCell(currentHost, cellIdx, wordsPerMask))
            {
                continue;
            }

            std::cout << "[BACKTRACK] #" << backtracks
                      << " excluding tile " << chosenTile
                      << " at (" << (cellIdx / cols) << "," << (cellIdx % cols) << ")\n";

            if (propagateUntilStable(cellIdx))
            {
                if (wfcstats::on)
                    wfcstats::backtrack_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_bt0).count();
                return true;
            }
        }

        if (wfcstats::on)
            wfcstats::backtrack_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_bt0).count();
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

            auto t_ch0 = std::chrono::high_resolution_clock::now();
            int chosen = chooseTile(idx);
            if (wfcstats::on)
                wfcstats::choose_ms += std::chrono::duration<double, std::milli>(
                                           std::chrono::high_resolution_clock::now() - t_ch0).count();
            if (chosen < 0)
            {
                return false;
            }

            // Sparse checkpoint of the PRE-collapse masks (see member docs).
            bool checkpoint = (collapses == 0 || collapses % kCheckpointEvery == 0);
            if (checkpoint)
            {
                auto t_sn0 = std::chrono::high_resolution_clock::now();
                history.push(makeSnapshot(idx, chosen));
                if (wfcstats::on)
                    wfcstats::snap_ms += std::chrono::duration<double, std::milli>(
                                             std::chrono::high_resolution_clock::now() - t_sn0).count();
            }
            collapses++;
            wfcstats::bump(wfcstats::collapses);
            applyCollapse(idx, chosen);
            if (verbose)
                std::cout << "[COLLAPSE] (" << (idx / cols) << "," << (idx % cols) << ") -> tile " << chosen << "\n";

            if (!propagateUntilStable(idx))
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

        // --verbose may appear anywhere: drop it before positional parsing so it
        // cannot shift the arguments (it used to make the image path be parsed
        // as a number and abort the process).
        bool verbose = false;
        std::vector<const char *> args;
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--verbose")
                verbose = true;
            else
                args.push_back(argv[i]);
        }
        const int nargs = (int)args.size();

        if (nargs >= 1)
        {
            sample = loadImage(args[0], palette);
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
        int outCols = (nargs >= 2) ? parseInt(args[1], "outW") : inputCols;
        int outRows = (nargs >= 3) ? parseInt(args[2], "outH") : inputRows;
        int N = (nargs >= 4) ? parseInt(args[3], "N") : 3;
        int scale = (nargs >= 5) ? parseInt(args[4], "scale") : 32;
        bool useSymmetries = true;
        if (nargs >= 6)
        {
            std::string flag = args[5];
            useSymmetries = (flag != "0" && flag != "false" && flag != "False" &&
                             flag != "--no-symmetry");
        }
        unsigned int seed = (nargs >= 7) ? parseUInt(args[6], "seed") : 42u;
        int maxBacktracks = (nargs >= 8) ? parseInt(args[7], "maxBacktracks") : 200;

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
            auto t_build0 = MainClock::now();
            auto output = wfc.buildOutput();
            auto t_build1 = MainClock::now();
            if (wfcstats::on)
                wfcstats::build_ms += std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();
            auto t_verify0 = MainClock::now();
            if (!wfc.verify(output))
                continue;
            auto t_verify1 = MainClock::now();
            if (wfcstats::on)
                wfcstats::verify_ms += std::chrono::duration<double, std::milli>(t_verify1 - t_verify0).count();
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
        std::cout << "  Build output:          " << std::fixed << std::setprecision(2)
                  << wfcstats::build_ms << " ms\n";
        std::cout << "  Verify:                " << std::fixed << std::setprecision(2)
                  << wfcstats::verify_ms << " ms\n";
        std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
        wfcstats::print("kokkos");
        std::cout << std::string(50, '=') << "\n";
    }

    Kokkos::finalize();
    return exitCode;
}
