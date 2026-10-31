#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>
#include <string>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <deque>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <queue>
#include <functional>
#include <stack>

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
 * @file wfc.cpp
 * @brief Sequential Wave Function Collapse (WFC) implementation with overlap constraints.
 *
 * The program:
 * 1. Loads a sample image (or a built-in binary sample).
 * 2. Extracts all NxN overlapping tiles and their frequencies.
 * 3. Precomputes tile compatibility for every relative offset in the overlap window.
 * 4. Runs WFC using entropy-based selection, propagation, and bounded backtracking.
 * 5. Writes an output image and prints simple quality/timing diagnostics.
 */

// ---------------------------------------------------------------------------
// Image I/O
// ---------------------------------------------------------------------------

/**
 * @brief Load an RGB image and convert it to a palette-index grid.
 * @param path Input image path.
 * @param[out] palette Palette indexed by tile/pixel values (packed as 0xRRGGBB).
 * @return 2D grid of palette indices with shape [height][width].
 */
std::vector<std::vector<int>> loadImage(const std::string &path,
                                        std::vector<uint32_t> &palette)
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

// ---------------------------------------------------------------------------
// Tile
// ---------------------------------------------------------------------------

/**
 * @brief NxN tile represented as palette indices.
 */
struct Tile
{
    /** Tile edge size. */
    int N;
    /** Tile content as [row][col] indices. */
    std::vector<std::vector<int>> data; // N×N
    /** Lexicographic ordering for map/set keys. */
    bool operator<(const Tile &o) const { return data < o.data; }
    /** Equality on tile data. */
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
        for (int j = 0; j < N; j++)
            r.data[j][N - 1 - i] = t.data[i][j];
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
        std::reverse(r.data[i].begin(), r.data[i].end());
    return r;
}

// ---------------------------------------------------------------------------
// Tile extraction WITH symmetries
// ---------------------------------------------------------------------------

/**
 * @brief Extract unique overlapping NxN tiles from the sample and count frequencies.
 *
 * The sample is treated as periodic during extraction (wrap-around indexing).
 * Optionally augments tile counts with rotational and reflection symmetries.
 *
 * @param sample Input sample as palette-index grid.
 * @param N Tile size.
 * @param[out] tiles Unique extracted tiles.
 * @param[out] freq Frequency of each corresponding tile in @p tiles.
 * @param useSymmetries If true, include 4 rotations and their horizontal reflections.
 */
void extractTiles(const std::vector<std::vector<int>> &sample, int N,
                  std::vector<Tile> &tiles, std::vector<int> &freq,
                  bool useSymmetries = true)
{
    int rows = (int)sample.size();
    int cols = (int)sample[0].size();

    std::map<std::vector<std::vector<int>>, int> counts;

    auto addTile = [&](const Tile &t)
    { counts[t.data]++; };

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
                    base.data[i][j] = sample[(r + i) % rows][(c + j) % cols];
            }

            if (!useSymmetries)
            {
                addTile(base);
            }
            else
            {
                // 4 rotations × 2 reflections = up to 8 variants
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
    for (auto &kv : counts)
    {
        Tile t;
        t.N = N;
        t.data = kv.first;
        tiles.push_back(t);
        freq.push_back(kv.second);
    }
    std::cout << "[TILES] " << tiles.size() << " unique " << N << "x" << N
              << " tiles extracted" << (useSymmetries ? " (with symmetries)" : "") << "\n";
}

// ---------------------------------------------------------------------------
// Compatibility — precomputed as a flat array indexed by [a][dir_index]
// Directions are the (2N-1)² offsets, stored as a vector per (tile_a, offset).
// ---------------------------------------------------------------------------

/**
 * @brief Check whether two tiles are compatible for a relative offset.
 * @param a Reference tile.
 * @param b Neighbor tile.
 * @param dy Row offset from @p a to @p b.
 * @param dx Column offset from @p a to @p b.
 * @return True if all overlapping cells match.
 */
bool canOverlap(const Tile &a, const Tile &b, int dy, int dx)
{
    int N = a.N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int bi = i - dy, bj = j - dx;
            if (bi >= 0 && bi < N && bj >= 0 && bj < N)
                if (a.data[i][j] != b.data[bi][bj])
                    return false;
        }
    return true;
}

// compat[a][offset_index] = bitset of compatible tile indices
// offset_index encodes (dy+N-1)*(2N-1)+(dx+N-1)
/**
 * @brief Precomputed overlap-compatibility table for all tile pairs and offsets.
 *
 * Storage layout: table[a * D + offset][b] where:
 * - a is the source tile index,
 * - offset encodes (dy, dx),
 * - b is the neighbor tile index.
 */
/**
 * @brief Flat bit-mask helpers shared by the state and the compat table.
 *
 * Tile sets are stored as WORDS 64-bit words (bit t = tile t). Flat storage
 * keeps snapshots contiguous (fast backtrack copies, ~2x less memory than
 * vector<vector<bool>> whose per-cell overhead alone is 24 B).
 */
inline int maskWords(int T) { return (T + 63) / 64; }

struct CompatTable
{
    /** Cardinal directions, in the order used as table index. */
    static constexpr int DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    /** Number of unique tiles. */
    int T, N;
    /** Number of stored offsets: the four cardinal directions. */
    int D = 4;
    /** Words per mask. */
    int WORDS = 0;
    // compat[((a * D) + dir) * WORDS + w]: bit b = tile b compatible.
    std::vector<uint64_t> table; // size T*D*WORDS

    // Union lookup table: eight domain bits at a time.
    // uni[(((dir*WORDS + w) * 8 + byte) * 256 + v) * WORDS + k] is the OR of the
    // compatibility rows of the tiles selected by byte v in word w. It replaces
    // popcount(domain) row ORs by at most eight lookups per word, which is the
    // hot loop of the propagation. Only built for small masks (T <= 512) where
    // the table stays a few MB; the caller falls back to the bit loop otherwise.
    std::vector<uint64_t> uni;
    bool useUni = false;
    static const int kMaxUniWords = 8;

    /**
     * @brief Pointer to the union entry for direction, word, byte and value.
     * @param d Direction index.
     * @param w Word index inside the mask.
     * @param by Byte index inside the word (0..7).
     * @param v Byte value (1..255).
     * @return Pointer to WORDS words.
     */
    const uint64_t *uniEntry(int d, int w, int by, int v) const
    {
        return uni.data() + ((((size_t)d * WORDS + w) * 8 + by) * 256 + v) * WORDS;
    }

    /**
     * @brief Build the cardinal compatibility table from extracted tiles.
     *
     * Only the four cardinal offsets are stored. For the overlapping model the
     * constraint at (dy, dx) is implied by the chain of unit steps: if a is
     * compatible with b at (1,0) and b with c at (0,1), then a and c agree on
     * their (1,1) overlap because both agree with b on the same cells. The
     * propagation fixed point is therefore identical with 4 directions, for
     * 4/(2N-1)^2 of the work (verified: bit-identical outputs on all samples).
     *
     * @param tiles Unique tile set.
     * @param withUnion Build the 8-bit union table (worth it only when the grid
     *        is big enough: its fixed cost is paid by tiny runs).
     */
    void build(const std::vector<Tile> &tiles, bool withUnion)
    {
        T = (int)tiles.size();
        N = tiles[0].N;
        WORDS = maskWords(T);
        table.assign((size_t)T * (size_t)D * (size_t)WORDS, uint64_t(0));

        for (int a = 0; a < T; a++)
            for (int b = 0; b < T; b++)
                for (int d = 0; d < D; d++)
                {
                    if (canOverlap(tiles[a], tiles[b], DIRS[d][0], DIRS[d][1]))
                        table[((size_t)a * (size_t)D + (size_t)d) * (size_t)WORDS + (size_t)(b / 64)] |= uint64_t(1) << (b % 64);
                }
        useUni = withUnion && (WORDS <= kMaxUniWords);
        if (useUni)
        {
            uni.assign((size_t)D * WORDS * 8 * 256 * WORDS, uint64_t(0));
            for (int d = 0; d < D; d++)
                for (int w = 0; w < WORDS; w++)
                    for (int by = 0; by < 8; by++)
                        for (int v = 1; v < 256; v++)
                        {
                            uint64_t *dst = &uni[((((size_t)d * WORDS + w) * 8 + by) * 256 + v) * WORDS];
                            const int bit = __builtin_ctz((unsigned)v);
                            const int prev = v & (v - 1);
                            const uint64_t *src = prev ? &uni[((((size_t)d * WORDS + w) * 8 + by) * 256 + prev) * WORDS] : nullptr;
                            const int t = w * 64 + by * 8 + bit;
                            const uint64_t *row = (t < T) ? get(t, d) : nullptr;
                            for (int k = 0; k < WORDS; k++)
                                dst[k] = (src ? src[k] : 0) | (row ? row[k] : 0);
                        }
        }
        std::cout << "[COMPAT] Table built for " << T << " tiles ("
                  << D << " cardinal directions, union table "
                  << (useUni ? "on" : "off") << ")\n";
    }

    /**
     * @brief Get the compatibility mask for tile @p a in direction @p d.
     * @param a Source tile index.
     * @param d Direction index in [0, 4).
     * @return Pointer to WORDS words; bit b set means tile b compatible.
     */
    const uint64_t *get(int a, int d) const
    {
        return table.data() + ((size_t)a * (size_t)D + (size_t)d) * (size_t)WORDS;
    }
};

// ---------------------------------------------------------------------------
// WFC state
// ---------------------------------------------------------------------------

/**
 * @brief Mutable WFC domain state for all output cells.
 */
struct WFCState
{
    // Flat domain bits: bits[idx*WORDS + w], bit t = tile t still possible.
    // One contiguous buffer (fast snapshot copies for backtracking).
    std::vector<uint64_t> bits;

    // Cached entropy per cell (-1 = dirty / needs recompute)
    // We store a pair (entropy, noise) so the heap key is stable
    // entropy == -2.0 means "already collapsed" (skip)
    std::vector<double> entropy;

    // Count of possible tiles per cell (to detect contradictions quickly)
    std::vector<int> possCount;

    /** Grid height, width, number of tile states, and words per mask. */
    int rows, cols, T, WORDS;
};


/**
 * @brief Indexed min-heap of (entropy + noise, cell) with decrease-key.
 *
 * The lazy std::priority_queue kept one entry per domain change, so at 256x256
 * forest the heap held a million entries and each pick popped dozens of stale
 * ones (139 ms of the 560 ms run). Here a cell has at most one entry, and its
 * key is updated in place; a pick is exactly one extract-min.
 *
 * Ordering is lexicographic on (key, cell) so the sequence of picks does not
 * depend on the insertion order, which keeps the three CPU variants on the
 * same trajectory.
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
        std::swap(h[i], h[j]);
        pos[(size_t)h[i].second] = i;
        pos[(size_t)h[j].second] = j;
    }
    void up(int i)
    {
        while (i > 0)
        {
            const int p = (i - 1) / 2;
            if (!less(h[i], h[p]))
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
            if (c + 1 < n && less(h[c + 1], h[c]))
                c++;
            if (!less(h[c], h[i]))
                break;
            swapAt(c, i);
            i = c;
        }
    }
    /** @brief Insert the cell, or move its key (up or down). */
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
            up(i);
            down(i);
        }
        else
        {
            h.pop_back();
        }
    }
    /** @brief Remove and return the smallest cell, -1 when empty. */
    int pop()
    {
        if (h.empty())
            return -1;
        const int cell = h[0].second;
        remove(cell);
        return cell;
    }
};

// ---------------------------------------------------------------------------
// WFC engine
// ---------------------------------------------------------------------------

/**
 * @brief Entropy-driven Wave Function Collapse engine with propagation and backtracking.
 */
struct WFC
{
    /** Output grid dimensions and model cardinalities. */
    int rows, cols, N, T;
    /** Immutable model data. */
    const std::vector<Tile> &tiles;
    const std::vector<int> &freq;
    const CompatTable &compat;

    // Precomputed log(freq[t]) and sum helpers
    std::vector<double> logFreq;
    double totalLogFreqSum; // sum of freq[t]*log(freq[t]) over all t
    double totalFreqSum;    // sum of freq[t]

    std::mt19937 rng;
    std::uniform_real_distribution<double> noiseDist{0.0, 1e-6};
    // Bruit de departage tire une fois par cellule : le choix de la cellule a
    // collapse ne depend plus de l'ordre des tirages, donc ni du decoupage en
    // vagues ni du nombre de threads. Meme graine = meme chemin, et seq/omp/gif
    // suivent le meme chemin pour une meme configuration.
    std::vector<double> cellNoise;
    // Per-collapse logging is I/O-heavy on big grids; enable with --verbose.
    bool verbose = false;

    WFCState state;

    // Undo journal: one entry per domain shrink, [cell, possCountBefore,
    // entropyBits, removed words...]. Domains only shrink, so restoring a cell
    // is the OR of the bits it lost, in any order. This replaces the full-state
    // snapshot per collapse (850 kB copied 1024 times at 128x128).
    //
    // One segment per checkpoint: only the last maxBacktracks+1 checkpoints can
    // be restored, so older segments are dropped with the history entry. Memory
    // stays bounded to a few MB instead of growing with the whole run.
    std::deque<std::vector<uint64_t>> journalSeg;
    /** Absolute id of journalSeg.front(). */
    size_t journalSegBase = 0;
    /** Per-cell stamp, so a restore takes the earliest entry of each cell. */
    std::vector<uint32_t> journalGen;
    uint32_t journalGenId = 0;
    /** Scratch buffer for the bits removed by a collapse or an exclusion. */
    std::vector<uint64_t> undoScratch;
    /**
     * @brief Checkpoint: journal position plus the decision taken at it.
     */
    struct Snapshot
    {
        int cellIdx;
        int chosenTile;
        size_t journalPos;
    };
    std::deque<Snapshot> history;
    // Cells whose domain changed during the current propagation (deferred heap).
    std::vector<char> dirtyFlag;
    std::vector<int> dirtyList;
    // BFS frontier reused across propagations: a stamp per cell instead of a
    // cells-sized zeroing, and a plain vector instead of a deque (one allocation
    // per push).
    std::vector<uint32_t> inQueueGen;
    uint32_t inQueueId = 0;
    std::vector<int> bfsQueue;
    int maxBacktracks;
    int lastBacktracks = 0; // backtracks used by the last run() call
    int collapses = 0;

    // Min-heap: (entropy+noise, flat_cell_index)
    CellHeap heap;

    /**
     * @brief Construct a WFC solver for a target output size.
     * @param rows Output height.
     * @param cols Output width.
     * @param N Tile size.
     * @param tiles Unique model tiles.
     * @param freq Tile frequencies used for weighted collapse and entropy.
     * @param compat Precomputed compatibility table.
     * @param seed RNG seed.
     * @param maxBacktracks Maximum allowed backtrack operations.
     */
    WFC(int rows, int cols, int N,
        const std::vector<Tile> &tiles,
        const std::vector<int> &freq,
        const CompatTable &compat,
        unsigned int seed = 42,
        int maxBacktracks = 100)
        : rows(rows), cols(cols), N(N), T((int)tiles.size()),
          tiles(tiles), freq(freq), compat(compat),
          rng(seed), maxBacktracks(maxBacktracks)
    {
        logFreq.resize(T);
        totalFreqSum = totalLogFreqSum = 0;
        for (int t = 0; t < T; t++)
        {
            logFreq[t] = std::log((double)freq[t]);
            totalFreqSum += freq[t];
            totalLogFreqSum += freq[t] * logFreq[t];
        }

        dirtyFlag.assign((size_t)rows * cols, 0);
        journalGen.assign((size_t)rows * cols, 0);
        cellNoise.resize((size_t)rows * cols);
        for (int i = 0; i < rows * cols; i++)
            cellNoise[i] = noiseDist(rng);

        initState(state);
    }

    /**
     * @brief Initialize the state so every tile is possible in every cell.
     * @param[out] s State object to initialize.
     */
    void initState(WFCState &s)
    {
        int cells = rows * cols;
        s.rows = rows;
        s.cols = cols;
        s.T = T;
        s.WORDS = maskWords(T);
        s.bits.assign((size_t)cells * (size_t)s.WORDS, ~uint64_t(0));
        // Clear padding bits past T in the last word of every cell.
        if (T % 64)
        {
            uint64_t last = (uint64_t(1) << (T % 64)) - 1;
            for (int i = 0; i < cells; i++)
                s.bits[(size_t)i * (size_t)s.WORDS + (size_t)s.WORDS - 1] = last;
        }
        s.possCount.assign(cells, T);
        undoScratch.assign((size_t)s.WORDS, uint64_t(0));
        // Initial entropy is the same for every cell
        double h0 = shannonEntropy(totalFreqSum, totalLogFreqSum);
        s.entropy.assign(cells, h0);
    }

    /**
     * @brief Compute Shannon entropy from aggregate weighted sums.
     * @param sumW Sum of active weights.
     * @param sumWlogW Sum of active weight*log(weight).
     * @return Shannon entropy for the domain.
     */
    double shannonEntropy(double sumW, double sumWlogW) const
    {
        if (sumW <= 0)
            return 0.0;
        return std::log(sumW) - sumWlogW / sumW;
    }

    /**
     * @brief Recompute entropy of a single cell from its active domain.
     * @param idx Flat cell index.
     * @return Entropy value.
     */
    double computeEntropy(int idx) const
    {
        double sumW = 0, sumWlogW = 0;
        const uint64_t *bits = &state.bits[(size_t)idx * (size_t)state.WORDS];
        for (int w = 0; w < state.WORDS; w++)
        {
            uint64_t word = bits[w];
            while (word)
            {
                int bit = __builtin_ctzll(word);
                word &= word - 1;
                int t = w * 64 + bit;
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
     * @brief Rebuild the min-heap of candidate cells from scratch.
     */
    void rebuildHeap()
    {
        heap.init(rows * cols);
        for (int idx = 0; idx < rows * cols; idx++)
        {
            if (state.possCount[idx] <= 1)
                continue; // collapsed or contradiction
            heap.set(idx, state.entropy[idx] + cellNoise[idx]);
        }
    }

    /**
     * @brief Insert/update one cell in the min-heap after entropy changes.
     * @param idx Flat cell index.
     */
    void pushHeap(int idx)
    {
        if (state.possCount[idx] <= 1)
        {
            heap.remove(idx); // became a singleton or empty
            return;
        }
        heap.set(idx, state.entropy[idx] + cellNoise[idx]);
    }

    /**
     * @brief Pick the next cell with the smallest entropy.
     * @return Flat cell index, -1 if completed, or -2 on contradiction.
     */
    int pickLowestEntropy()
    {
        while (true)
        {
            const int idx = heap.pop();
            if (idx < 0)
                break;
            const int cnt = state.possCount[idx];
            if (cnt == 0)
                return -2; // contradiction
            if (cnt == 1)
                continue; // became a singleton through propagation
            return idx;
        }
        // Heap empty: check for contradictions or completion
        for (int idx = 0; idx < rows * cols; idx++)
            if (state.possCount[idx] == 0)
                return -2;
        return -1; // all collapsed
    }

    /**
     * @brief Collapse one cell to a single tile sampled by frequency.
     * @param idx Flat cell index.
     * @return Tile index chosen.
     */
    int collapse(int idx)
    {
        uint64_t *bits = &state.bits[(size_t)idx * (size_t)state.WORDS];
        std::vector<int> choices;
        std::vector<int> weights;
        for (int w = 0; w < state.WORDS; w++)
        {
            uint64_t word = bits[w];
            while (word)
            {
                int bit = __builtin_ctzll(word);
                word &= word - 1;
                int t = w * 64 + bit;
                if (t < T)
                {
                    choices.push_back(t);
                    weights.push_back(freq[t]);
                }
            }
        }
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int chosen = choices[dist(rng)];

        // Journal the tiles that this collapse removes, then keep the chosen one.
        for (int w = 0; w < state.WORDS; w++)
            undoScratch[w] = bits[w];
        undoScratch[chosen / 64] &= ~(uint64_t(1) << (chosen % 64));
        recordUndo(idx, undoScratch.data());
        for (int w = 0; w < state.WORDS; w++)
            bits[w] = uint64_t(0);
        bits[chosen / 64] = uint64_t(1) << (chosen % 64);
        state.possCount[idx] = 1;
        state.entropy[idx] = 0.0;

        int r = idx / cols, c = idx % cols;
        if (verbose)
            std::cout << "[COLLAPSE] (" << r << "," << c << ") -> tile " << chosen << "\n";
        return chosen;
    }

    /**
     * @brief Propagate domain reductions from a starting cell using BFS.
     * @param startIdx Flat index of the changed cell.
     * @return True on success, false if a contradiction is detected.
     * @note Worst-case complexity is O(cells * D * T^2), typically lower in practice.
     */
    bool propagate(int startIdx)
    {
        // Le nombre de mots est fixe pour tout le run : le specialiser a la
        // compilation deroule le OR des mots et garde le tampon de support en
        // registres (c'est 70 % du temps du noyau).
        switch (state.WORDS)
        {
        case 1: return propagateImpl<1>(startIdx);
        case 2: return propagateImpl<2>(startIdx);
        case 3: return propagateImpl<3>(startIdx);
        case 4: return propagateImpl<4>(startIdx);
        case 5: return propagateImpl<5>(startIdx);
        case 6: return propagateImpl<6>(startIdx);
        case 7: return propagateImpl<7>(startIdx);
        case 8: return propagateImpl<8>(startIdx);
        default: return propagateImpl<0>(startIdx);
        }
    }

    template <int WT>
    bool propagateImpl(int startIdx)
    {
        const int WORDS = WT ? WT : state.WORDS;
        wfcstats::bump(wfcstats::propagate_calls);
        if ((int)inQueueGen.size() != rows * cols)
            inQueueGen.assign((size_t)rows * cols, 0);
        const uint32_t qid = ++inQueueId;
        std::vector<int> &q = bfsQueue;
        q.clear();
        q.push_back(startIdx);
        inQueueGen[startIdx] = qid;
        wfcstats::bump(wfcstats::pushes);
        size_t qhead = 0;

        while (qhead < q.size())
        {
            int idx = q[qhead++];
            inQueueGen[idx] = 0;
            wfcstats::bump(wfcstats::pops);
            wfcstats::note_queue(q.size() - qhead + 1);
            int r = idx / cols, c = idx % cols;
            uint64_t supFixed[8];
            std::vector<uint64_t> supVec;
            uint64_t *supported = supFixed;
            if (WORDS > 8)
            {
                supVec.resize((size_t)WORDS);
                supported = supVec.data();
            }

            for (int d = 0; d < compat.D; d++)
            {
                {
                    const int nr = r + CompatTable::DIRS[d][0];
                    const int nc = c + CompatTable::DIRS[d][1];
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols)
                        continue;
                    int nidx = nr * cols + nc;

                    // --- Word-level support computation ---
                    // Union of tiles supported by any tile at (r,c) in
                    // direction d towards (nr,nc).
                    std::fill(supported, supported + WORDS, uint64_t(0));
                    const uint64_t *curBits = &state.bits[(size_t)idx * (size_t)WORDS];
                    if (compat.useUni)
                    {
                        for (int w = 0; w < WORDS; w++)
                        {
                            const uint64_t word = curBits[w];
                            for (int by = 0; by < 8; by++)
                            {
                                const int v = (int)((word >> (8 * by)) & 0xFF);
                                if (!v)
                                    continue;
                                const uint64_t *u = compat.uniEntry(d, w, by, v);
                                for (int k = 0; k < WORDS; k++)
                                    supported[k] |= u[k];
                            }
                        }
                    }
                    else
                    {
                        for (int w = 0; w < WORDS; w++)
                        {
                            uint64_t word = curBits[w];
                            while (word)
                            {
                                int bit = __builtin_ctzll(word);
                                word &= word - 1;
                                int ta = w * 64 + bit;
                                if (ta < T)
                                {
                                    const uint64_t *row = compat.get(ta, d);
                                    for (int k = 0; k < WORDS; k++)
                                        supported[k] |= row[k];
                                }
                            }
                        }
                    }

                    // Remove unsupported tiles from neighbor, journaling the
                    // bits it loses (restoring is an OR of those bits).
                    uint64_t *nbits = &state.bits[(size_t)nidx * (size_t)WORDS];
                    uint64_t remFixed[8];
                    std::vector<uint64_t> remVec;
                    uint64_t *removed = remFixed;
                    if (WORDS > 8)
                    {
                        remVec.resize((size_t)WORDS);
                        removed = remVec.data();
                    }
                    bool changed = false;
                    int removedCount = 0;
                    for (int w = 0; w < WORDS; w++)
                    {
                        uint64_t before = nbits[w];
                        nbits[w] &= supported[w];
                        removed[w] = before & ~nbits[w];
                        if (removed[w])
                        {
                            removedCount += __builtin_popcountll(removed[w]);
                            changed = true;
                        }
                    }
                    if (changed)
                    {
                        // The undo entry must carry the values *before* the
                        // shrink, so record it before updating them.
                        recordUndo(nidx, removed);
                        state.possCount[nidx] -= removedCount;
                    }

                    if (state.possCount[nidx] == 0)
                    {
                        std::cout << "[CONTRADICTION] at (" << nr << "," << nc << ")\n";
                        flushDirty();
                        return false;
                    }

                    if (changed)
                    {
                        // Update incremental entropy
                        state.entropy[nidx] = computeEntropy(nidx);
                        // Deferred: the heap gets one entry per cell at the end
                        // of the propagation instead of one per shrink.
                        if (!dirtyFlag[nidx])
                        {
                            dirtyFlag[nidx] = 1;
                            dirtyList.push_back(nidx);
                        }
                        if (inQueueGen[nidx] != qid)
                        {
                            inQueueGen[nidx] = qid;
                            q.push_back(nidx);
                            wfcstats::bump(wfcstats::pushes);
                        }
                    }
                }
            }
        }
        flushDirty();
        return true;
    }

    /**
     * @brief Push the cells changed by the last propagation into the heap.
     */
    void flushDirty()
    {
        for (int c : dirtyList)
        {
            dirtyFlag[c] = 0;
            pushHeap(c);
        }
        dirtyList.clear();
    }

    /**
     * @brief Execute WFC until completion or unrecoverable contradiction.
     * @return True if all cells collapse successfully, false otherwise.
     */
    bool run()
    {
        std::cout << "=== WFC " << rows << "x" << cols << " ===\n";
        journalSeg.clear();
        journalSegBase = 0;
        rebuildHeap();
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

            // Checkpoint before collapsing: the journal position is enough to
            // undo the collapse and everything it propagates.
            journalSeg.emplace_back();
            history.push_back({idx, -1, journalSegBase + journalSeg.size() - 1});
            while ((int)history.size() > maxBacktracks + 1)
            {
                history.pop_front();
                journalSeg.pop_front();
                journalSegBase++;
            }
            int chosen = collapse(idx);
            history.back().chosenTile = chosen;
            collapses++;
            wfcstats::bump(wfcstats::collapses);

            if (!propagate(idx))
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
     * @brief Record the bits a cell is about to lose, with its current state.
     * @param cell Flat cell index.
     * @param removed Bits removed by this shrink (WORDS words).
     */
    void recordUndo(int cell, const uint64_t *removed)
    {
        std::vector<uint64_t> &seg = journalSeg.back();
        seg.push_back((uint64_t)(uint32_t)cell);
        seg.push_back((uint64_t)(uint32_t)state.possCount[cell]);
        uint64_t ebits;
        double ent = state.entropy[cell];
        std::memcpy(&ebits, &ent, sizeof(ebits));
        seg.push_back(ebits);
        for (int w = 0; w < state.WORDS; w++)
            seg.push_back(removed[w]);
    }

    /**
     * @brief Undo every journalled change made since a checkpoint.
     * @param pos Journal position (in entries) of the checkpoint.
     */
    void restoreJournal(size_t segId)
    {
        const size_t st = (size_t)(3 + state.WORDS);
        const size_t first = segId - journalSegBase;
        for (size_t s = journalSeg.size(); s-- > first;)
        {
            const std::vector<uint64_t> &seg = journalSeg[s];
            for (size_t e = 0; e < seg.size(); e += st)
            {
                const uint64_t *u = &seg[e];
                uint64_t *m = &state.bits[(size_t)u[0] * (size_t)state.WORDS];
                for (int w = 0; w < state.WORDS; w++)
                    m[w] |= u[3 + w];
            }
        }
        // The earliest entry of a cell in the range carries its pre-change
        // entropy and tile count (segments are in chronological order).
        journalGenId++;
        for (size_t s = first; s < journalSeg.size(); s++)
        {
            const std::vector<uint64_t> &seg = journalSeg[s];
            for (size_t e = 0; e < seg.size(); e += st)
            {
                const uint64_t *u = &seg[e];
                const int cell = (int)u[0];
                if (journalGen[cell] == journalGenId)
                    continue;
                journalGen[cell] = journalGenId;
                state.possCount[cell] = (int)u[1];
                double ent;
                std::memcpy(&ent, &u[2], sizeof(ent));
                state.entropy[cell] = ent;
            }
        }
        journalSeg.resize(first);
        // The exclusion that follows belongs to the restored checkpoint.
        journalSeg.emplace_back();
    }

    /**
     * @brief Restore checkpoints until an alternate branch propagates.
     * @param[in,out] backtracks Number of backtracks performed.
     * @return True if a consistent resumed state was found.
     */
    bool backtrack(int &backtracks)
    {
        while (!history.empty() && backtracks < maxBacktracks)
        {
            Snapshot snap = history.back();
            history.pop_back();
            backtracks++;
            restoreJournal(snap.journalPos);

            const int idx = snap.cellIdx;
            uint64_t *bits = &state.bits[(size_t)idx * (size_t)state.WORDS];
            const int w = snap.chosenTile / 64, b = snap.chosenTile % 64;
            if (!((bits[w] >> b) & 1))
                continue; // tile no longer possible after the restore

            // Exclude the tile that failed and journal that removal too.
            for (int k = 0; k < state.WORDS; k++)
                undoScratch[k] = uint64_t(0);
            undoScratch[w] = uint64_t(1) << b;
            recordUndo(idx, undoScratch.data());
            bits[w] &= ~undoScratch[w];
            state.possCount[idx]--;
            if (state.possCount[idx] == 0)
                continue;
            state.entropy[idx] = computeEntropy(idx);

            int r = idx / cols, c = idx % cols;
            std::cout << "[BACKTRACK] #" << backtracks
                      << " excluding tile " << snap.chosenTile
                      << " at (" << r << "," << c << ")\n";
            rebuildHeap();
            if (propagate(idx))
                return true;
        }
        return false;
    }

    /**
     * @brief Build the final output index grid from the current collapsed state.
     * @return Output grid of palette indices with shape [rows][cols].
     */
    std::vector<std::vector<int>> buildOutput() const
    {
        std::vector<std::vector<int>> output(rows, std::vector<int>(cols, -1));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                int idx = r * cols + c;
                const uint64_t *bits = &state.bits[(size_t)idx * (size_t)state.WORDS];
                for (int w = 0; w < state.WORDS; w++)
                {
                    if (bits[w])
                    {
                        int t = w * 64 + __builtin_ctzll(bits[w]);
                        if (t < T)
                            output[r][c] = tiles[t].data[0][0];
                        break;
                    }
                }
            }
        return output;
    }

    /**
     * @brief Validate that each NxN output subgrid exists in the learned tile set.
     * @param output Generated palette-index grid.
     * @return True if all checked subgrids are valid.
     */
    bool verify(const std::vector<std::vector<int>> &output) const
    {
        // Hash tile set once so each subgrid check is O(N^2), not O(T*N^2).
        std::unordered_set<std::string> tileSet;
        for (const auto &t : tiles)
        {
            std::string key;
            key.reserve((size_t)N * (size_t)N * 4);
            for (const auto &row : t.data)
                for (int v : row)
                {
                    key += std::to_string(v);
                    key += ',';
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
                    for (int j = 0; j < N; j++)
                    {
                        sub += std::to_string(output[r + i][c + j]);
                        sub += ',';
                    }
                if (tileSet.find(sub) == tileSet.end())
                {
                    std::cout << "[VERIFY_FAIL] at (" << r << "," << c << ")\n";
                    errors++;
                }
            }
        }
        if (!errors)
            std::cout << "[VERIFY_OK] All subgrids valid.\n";
        else
            std::cout << "[VERIFY_FAIL] " << errors << " invalid subgrids.\n";
        return errors == 0;
    }

    /**
     * @brief Save the generated output as a PNG image.
     * @param filename Destination file path.
     * @param output Generated palette-index grid.
     * @param palette RGB palette encoded as 0xRRGGBB.
     * @param scale Pixel scale per logical output cell.
     * @return True if the file was written.
     */
    bool saveImage(const std::string &filename,
                   const std::vector<std::vector<int>> &output,
                   const std::vector<uint32_t> &palette,
                   int scale = 32)
    {
        int imgW = cols * scale, imgH = rows * scale;
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
                    for (int px = 0; px < scale; px++)
                    {
                        int i = ((r * scale + py) * imgW + (c * scale + px)) * 3;
                        pixels[i] = R;
                        pixels[i + 1] = G;
                        pixels[i + 2] = B;
                    }
            }
        }
        int ok = stbi_write_png(filename.c_str(), imgW, imgH, 3, pixels.data(), imgW * 3);
        if (ok)
            std::cout << "[IMAGE] Saved " << filename << " (" << imgW << "x" << imgH << ")\n";
        else
            std::cout << "[IMAGE] Failed to save " << filename << "\n";
        return ok != 0;
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/**
 * @brief Program entry point.
 *
 * Command line:
 * - argv[1]: input image path (optional)
 * - argv[2]: output width in cells (optional)
 * - argv[3]: output height in cells (optional)
 * - argv[4]: tile size N (optional, default 3)
 * - argv[5]: render scale in pixels per cell (optional, default 32)
 * - argv[6]: use symmetries (optional, default 1)
 *
 * If no input image is provided, a built-in 5x5 binary sample is used.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument vector.
 * @return 0 on normal termination.
 */
int main(int argc, char *argv[])
{
    using Clock = std::chrono::high_resolution_clock;

    // Usage: wfc [image] [outW] [outH] [N] [scale] [useSymmetries] [seed] [maxBacktracks] [--verbose]
    std::vector<std::vector<int>> sample;
    std::vector<uint32_t> palette;

    // --verbose may appear anywhere: drop it before positional parsing so it
    // cannot shift the arguments (it used to make the image path be parsed as
    // a number and abort the process).
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
        return 1;
    }

    std::cout << "[OUTPUT SIZE] " << outCols << "x" << outRows
              << "  N=" << N << "  scale=" << scale
              << "  symmetries=" << (useSymmetries ? "on" : "off")
              << "  seed=" << seed << "\n";

    auto t_start = Clock::now();

    // Step 1: Extract tiles
    std::vector<Tile> tiles;
    std::vector<int> freq;
    extractTiles(sample, N, tiles, freq, useSymmetries);

    if (tiles.empty())
    {
        std::cerr << "[ERROR] No tiles extracted\n";
        return 1;
    }

    auto t_extract = Clock::now();

    // Step 2: Build compatibility table
    CompatTable compat;
    compat.build(tiles, outRows * outCols >= 1024);

    auto t_compat = Clock::now();

    // Step 3: Run WFC (up to 3 trajectories; a failed collapse restarts
    // with the next seed instead of giving up on a single bad draw).
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

    auto t_end = Clock::now();
    std::cout << "[BACKTRACKS] " << lastBacktracks << " used, " << restarts << " restarts\n";

    int exitCode = 0;
    if (success)
        std::cout << "[RESULT] WFC succeeded.\n";
    else
    {
        std::cout << "[RESULT] WFC failed.\n";
        exitCode = 1;
    }

    // Display timing summary
    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    std::cout << "\n"
              << std::string(50, '=') << "\n";
    std::cout << "[TIMING] wfc.cpp\n";
    std::cout << "  Extract tiles:         " << std::fixed << std::setprecision(2)
              << elapsed_ms(t_start, t_extract) << " ms\n";
    std::cout << "  Build compatibility:   " << elapsed_ms(t_extract, t_compat) << " ms\n";
    std::cout << "  WFC algorithm:         " << elapsed_ms(t_compat, t_end) << " ms\n";
    std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
    wfcstats::print("seq");
    std::cout << std::string(50, '=') << "\n";

    return exitCode;
}