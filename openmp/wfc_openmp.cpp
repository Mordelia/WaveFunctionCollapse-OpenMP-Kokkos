#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

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
 * @file wfc_openmp.cpp
 * @brief OpenMP Wave Function Collapse (WFC) implementation using dynamic bit masks.
 *
 * This version keeps the same overlap model as the sequential implementation, but uses:
 * - OpenMP tasks for tile extraction, compatibility precomputation, and propagation work.
 * - A dynamic packed bit-mask representation for tile domains (unbounded tile count).
 * - Atomic mask updates for concurrent domain pruning during propagation.
 */

/** Number of 64-bit words required to represent T tile states. */
static int BM_K = 0;

/**
 * @brief Bit-mask helpers for domain operations over tile sets.
 *
 * Masks are arrays of BM_K words, each bit representing one tile index.
 */
struct BigMask
{
    /** Optional raw pointer field. */
    uint64_t *w;

    /**
     * @brief Initialize mask to all valid tile bits set.
     * @param dst Destination mask.
     * @param T Number of tiles.
     */
    static void setAll(uint64_t *dst, int T)
    {
        int fullWords = T / 64;
        int rem = T % 64;
        for (int i = 0; i < fullWords; i++)
            dst[i] = ~uint64_t(0);
        if (rem)
            dst[fullWords] = (uint64_t(1) << rem) - 1;
    }

    /**
     * @brief Set all mask words to zero.
     * @param dst Destination mask.
     */
    static void setZero(uint64_t *dst)
    {
        for (int i = 0; i < BM_K; i++)
            dst[i] = uint64_t(0);
    }

    /**
     * @brief Check whether a mask has no active bits.
     * @param a Input mask.
     * @return True if empty.
     */
    static bool isZero(const uint64_t *a)
    {
        for (int i = 0; i < BM_K; i++)
            if (a[i])
                return false;
        return true;
    }

    /**
     * @brief Compare two masks.
     * @param a Left mask.
     * @param b Right mask.
     * @return True if all words are identical.
     */
    static bool equal(const uint64_t *a, const uint64_t *b)
    {
        for (int i = 0; i < BM_K; i++)
            if (a[i] != b[i])
                return false;
        return true;
    }

    /**
     * @brief Compute bitwise AND of two masks.
     * @param dst Destination mask.
     * @param a Left operand.
     * @param b Right operand.
     */
    static void andInto(uint64_t *dst, const uint64_t *a, const uint64_t *b)
    {
        for (int i = 0; i < BM_K; i++)
            dst[i] = a[i] & b[i];
    }

    /**
     * @brief Apply bitwise OR assignment.
     * @param dst Destination mask.
     * @param src Source mask.
     */
    static void orInto(uint64_t *dst, const uint64_t *src)
    {
        for (int i = 0; i < BM_K; i++)
            dst[i] |= src[i];
    }

    /**
     * @brief Apply bitwise AND assignment.
     * @param dst Destination mask.
     * @param src Source mask.
     */
    static void andAssign(uint64_t *dst, const uint64_t *src)
    {
        for (int i = 0; i < BM_K; i++)
            dst[i] &= src[i];
    }

    /**
     * @brief Count active bits in a mask.
     * @param a Input mask.
     * @return Number of set bits.
     */
    static int popcount(const uint64_t *a)
    {
        int c = 0;
        for (int i = 0; i < BM_K; i++)
            c += __builtin_popcountll(a[i]);
        return c;
    }

    /**
     * @brief Find the first set bit index.
     * @param a Input mask.
     * @return Tile index, or -1 if mask is empty.
     */
    static int ctz(const uint64_t *a)
    {
        for (int i = 0; i < BM_K; i++)
            if (a[i])
                return i * 64 + __builtin_ctzll(a[i]);
        return -1;
    }

    /**
     * @brief Set a mask to a single tile bit.
     * @param dst Destination mask.
     * @param t Tile index.
     */
    static void setSingle(uint64_t *dst, int t)
    {
        setZero(dst);
        dst[t / 64] = uint64_t(1) << (t % 64);
    }

    /**
     * @brief Clear one tile bit.
     * @param dst Destination mask.
     * @param t Tile index.
     */
    static void clearBit(uint64_t *dst, int t)
    {
        dst[t / 64] &= ~(uint64_t(1) << (t % 64));
    }

    /**
     * @brief Test whether one tile bit is set.
     * @param a Input mask.
     * @param t Tile index.
     * @return True if set.
     */
    static bool testBit(const uint64_t *a, int t)
    {
        return (a[t / 64] >> (t % 64)) & 1;
    }

    /**
     * @brief Atomically intersect a domain mask with an allowed mask.
     * @param dst Atomic destination words.
     * @param keep Mask of bits to keep.
     * @param[out] anyChange Set to true if at least one word changed.
     * @return True if resulting mask is non-empty.
     */
    static bool atomicAndWords(std::atomic<uint64_t> *dst,
                               const uint64_t *keep,
                               bool &anyChange)
    {
        anyChange = false;
        for (int i = 0; i < BM_K; i++)
        {
            uint64_t old = dst[i].load(std::memory_order_relaxed);
            uint64_t next;
            do
            {
                next = old & keep[i];
                if (next == old)
                    goto next_word;
            } while (!dst[i].compare_exchange_weak(old, next,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_relaxed));
            anyChange = true;
        next_word:;
        }
        for (int i = 0; i < BM_K; i++)
            if (dst[i].load(std::memory_order_relaxed))
                return true;
        return false;
    }

    /**
     * @brief Load an atomic mask into a plain buffer.
     * @param src Atomic source words.
     * @param dst Plain destination words.
     */
    static void loadFrom(const std::atomic<uint64_t> *src, uint64_t *dst)
    {
        for (int i = 0; i < BM_K; i++)
            dst[i] = src[i].load(std::memory_order_relaxed);
    }

    /**
     * @brief Store a plain mask into atomic words.
     * @param dst Atomic destination words.
     * @param src Plain source words.
     */
    static void storeTo(std::atomic<uint64_t> *dst, const uint64_t *src)
    {
        for (int i = 0; i < BM_K; i++)
            dst[i].store(src[i], std::memory_order_relaxed);
    }
};

/**
 * @brief Load an RGB image and convert it to a palette-index grid.
 * @param path Input image path.
 * @param[out] palette Palette encoded as 0xRRGGBB values.
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
    /** Equality on tile content. */
    bool operator==(const Tile &o) const { return data == o.data; }
};

/**
 * @brief Rotate a tile by 90 degrees clockwise.
 * @param t Source tile.
 * @return Rotated tile.
 */
static Tile rotate90(const Tile &t)
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
static Tile reflectH(const Tile &t)
{
    Tile r;
    r.N = t.N;
    r.data = t.data;
    for (auto &row : r.data)
        std::reverse(row.begin(), row.end());
    return r;
}

/**
 * @brief Extract unique overlapping NxN tiles from a sample and count frequencies.
 *
 * The sample is treated as periodic (wrap-around indexing). Rows are processed using
 * OpenMP tasks; each task accumulates local counts and merges them into a global map.
 *
 * @param sample Input sample as palette-index grid.
 * @param N Tile size.
 * @param[out] tiles Unique extracted tiles.
 * @param[out] freq Frequency for each corresponding tile.
 * @param useSymmetries If true, include rotations and horizontal reflections.
 */
void extractTiles(const std::vector<std::vector<int>> &sample, int N,
                  std::vector<Tile> &tiles, std::vector<int> &freq,
                  bool useSymmetries = true)
{
    int rows = (int)sample.size(), cols = (int)sample[0].size();
    std::map<std::vector<std::vector<int>>, int> counts;

#pragma omp parallel
    {
#pragma omp single
        {
            for (int r = 0; r < rows; r++)
            {
#pragma omp task firstprivate(r) shared(counts, sample, N, rows, cols, useSymmetries)
                {
                    std::map<std::vector<std::vector<int>>, int> local;
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
                            local[base.data]++;
                        }
                        else
                        {
                            Tile cur = base;
                            for (int rot = 0; rot < 4; rot++)
                            {
                                local[cur.data]++;
                                local[reflectH(cur).data]++;
                                cur = rotate90(cur);
                            }
                        }
                    }
#pragma omp critical(wfc_tile_merge)
                    {
                        for (const auto &kv : local)
                            counts[kv.first] += kv.second;
                    }
                }
            }
#pragma omp taskwait
        }
    }

    tiles.clear();
    freq.clear();
    for (const auto &kv : counts)
    {
        Tile t;
        t.N = N;
        t.data = kv.first;
        tiles.push_back(t);
        freq.push_back(kv.second);
    }
    std::cout << "[TILES] " << tiles.size() << " unique " << N << "x" << N
              << " tiles" << (useSymmetries ? " (with symmetries)" : "") << "\n";
}

/**
 * @brief Check whether two tiles can overlap for an offset (dy, dx).
 * @param a Reference tile.
 * @param b Neighbor tile.
 * @param dy Row offset from @p a to @p b.
 * @param dx Column offset from @p a to @p b.
 * @return True if all overlapping cells match.
 */
static bool canOverlap(const Tile &a, const Tile &b, int dy, int dx)
{
    int N = a.N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int bi = i - dy, bj = j - dx;
            if (bi >= 0 && bi < N && bj >= 0 && bj < N && a.data[i][j] != b.data[bi][bj])
                return false;
        }
    return true;
}

/**
 * @brief Precomputed overlap-compatibility table stored as packed bit masks.
 *
 * Storage layout: table[((a * D) + offset) * BM_K + word].
 */
struct CompatTable
{
    /** Number of tiles, tile size, and number of relative offsets. */
    int T, N, D;
    /** Flat storage for compatibility masks. */
    std::vector<uint64_t> table;

    /**
     * @brief Encode offset (dy, dx) to a linear index.
     * @param dy Row offset.
     * @param dx Column offset.
     * @return Offset index in [0, D).
     */
    int offsetIdx(int dy, int dx) const
    {
        return (dy + N - 1) * (2 * N - 1) + (dx + N - 1);
    }

    /**
     * @brief Get mutable pointer to compatibility mask for tile a and offset.
     * @param a Source tile index.
     * @param dy Row offset.
     * @param dx Column offset.
     * @return Pointer to first word of mask.
     */
    uint64_t *entry(int a, int dy, int dx)
    {
        return table.data() + ((size_t)a * D + offsetIdx(dy, dx)) * BM_K;
    }
    /**
     * @brief Get const pointer to compatibility mask for tile a and offset.
     * @param a Source tile index.
     * @param dy Row offset.
     * @param dx Column offset.
     * @return Pointer to first word of mask.
     */
    const uint64_t *entry(int a, int dy, int dx) const
    {
        return table.data() + ((size_t)a * D + offsetIdx(dy, dx)) * BM_K;
    }

    /**
     * @brief Build compatibility masks for all tile pairs and relative offsets.
     * @param tiles Unique tile set.
     */
    void build(const std::vector<Tile> &tiles)
    {
        T = (int)tiles.size();
        N = tiles[0].N;
        D = (2 * N - 1) * (2 * N - 1);
        BM_K = (T + 63) / 64;

        table.assign((size_t)T * D * BM_K, uint64_t(0));

#pragma omp parallel
        {
#pragma omp single
            {
                for (int a = 0; a < T; a++)
                {
#pragma omp task firstprivate(a) shared(tiles)
                    {
                        for (int b = 0; b < T; b++)
                            for (int dy = -(N - 1); dy <= N - 1; dy++)
                                for (int dx = -(N - 1); dx <= N - 1; dx++)
                                {
                                    if (dy == 0 && dx == 0)
                                        continue;
                                    if (canOverlap(tiles[a], tiles[b], dy, dx))
                                    {
                                        uint64_t *e = entry(a, dy, dx);
                                        e[b / 64] |= uint64_t(1) << (b % 64);
                                    }
                                }
                    }
                }
#pragma omp taskwait
            }
        }
        std::cout << "[COMPAT] Table built for " << T << " tiles"
                  << "  (K=" << BM_K << " words/mask)\n";
    }
};

/**
 * @brief Mutable WFC state using atomic dynamic masks per cell.
 */
struct WFCState
{
    /** Grid dimensions and tile count. */
    int rows, cols, T;
    /** Number of cells in flattened layout. */
    size_t ncells = 0;
    /** Flat atomic masks, BM_K words per cell. */
    std::unique_ptr<std::atomic<uint64_t>[]> masks;
    /** Cached entropy per cell. */
    std::vector<double> entropy;

    /** Default constructor. */
    WFCState() = default;

    /**
     * @brief Deep-copy constructor for snapshot/backtracking support.
     * @param o Source state.
     */
    WFCState(const WFCState &o)
        : rows(o.rows), cols(o.cols), T(o.T), ncells(o.ncells),
          masks(o.ncells ? new std::atomic<uint64_t>[o.ncells * BM_K] : nullptr),
          entropy(o.entropy)
    {
        size_t total = o.ncells * BM_K;
        for (size_t i = 0; i < total; i++)
            masks[i].store(o.masks[i].load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
    }

    /**
     * @brief Deep-copy assignment for snapshot/backtracking support.
     * @param o Source state.
     * @return Reference to this state.
     */
    WFCState &operator=(const WFCState &o)
    {
        if (this == &o)
            return *this;
        rows = o.rows;
        cols = o.cols;
        T = o.T;
        ncells = o.ncells;
        size_t total = o.ncells * BM_K;
        masks.reset(total ? new std::atomic<uint64_t>[total] : nullptr);
        for (size_t i = 0; i < total; i++)
            masks[i].store(o.masks[i].load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
        entropy = o.entropy;
        return *this;
    }

    /**
     * @brief Get mutable pointer to cell mask words.
     * @param idx Flat cell index.
     * @return Pointer to BM_K atomic words.
     */
    std::atomic<uint64_t> *cell(int idx)
    {
        return masks.get() + (size_t)idx * BM_K;
    }
    /**
     * @brief Get const pointer to cell mask words.
     * @param idx Flat cell index.
     * @return Pointer to BM_K atomic words.
     */
    const std::atomic<uint64_t> *cell(int idx) const
    {
        return masks.get() + (size_t)idx * BM_K;
    }

    /**
     * @brief Load one cell mask to a plain buffer.
     * @param idx Flat cell index.
     * @param[out] dst Destination buffer with BM_K words.
     */
    void load(int idx, uint64_t *dst) const
    {
        BigMask::loadFrom(cell(idx), dst);
    }

    /**
     * @brief Store a plain mask into one cell.
     * @param idx Flat cell index.
     * @param src Source buffer with BM_K words.
     */
    void store(int idx, const uint64_t *src)
    {
        BigMask::storeTo(cell(idx), src);
    }

    /**
     * @brief Count possible tiles in one cell.
     * @param idx Flat cell index.
     * @return Number of active tile states.
     */
    int count(int idx) const
    {
        uint64_t tmp[64];
        load(idx, tmp);
        return BigMask::popcount(tmp);
    }

    /**
     * @brief Atomically intersect one cell with an allowed mask.
     * @param idx Flat cell index.
     * @param keep Bits to keep.
     * @param[out] anyChange True if the mask changed.
     * @return True if resulting mask is non-empty.
     */
    bool atomicAnd(int idx, const uint64_t *keep, bool &anyChange)
    {
        return BigMask::atomicAndWords(cell(idx), keep, anyChange);
    }
};

/**
 * @brief Entropy-driven OpenMP WFC solver using dynamic bit masks.
 */
struct WFC
{
    /** Output dimensions, tile size, and number of tile states. */
    int rows, cols, N, T;
    /** Immutable model data. */
    const std::vector<Tile> &tiles;
    const std::vector<int> &freq;
    const CompatTable &compat;

    /** Precomputed log-frequency per tile. */
    std::vector<double> logFreq;
    /** All-tiles-active mask used for initialization. */
    std::vector<uint64_t> fullMaskBuf;

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

    // L5 : version par cellule. `cellGen` est incrementee a chaque
    // retrecissement d'un domaine ; le thread qui traite une cellule retient
    // dans `cellGenSeen` la version lue *avant* de charger le domaine. Une
    // cellule modifiee pendant une vague n'a besoin d'etre re-enfilee que si
    // sa version a bouge depuis : sinon le retrecissement est anterieur au
    // chargement, le traitement de la vague vient de le propager, et un
    // second passage ne ferait que reparcourir le meme voisinage.
    std::unique_ptr<std::atomic<uint32_t>[]> cellGen;
    std::vector<uint32_t> cellGenSeen;

    /**
     * @brief Snapshot saved before collapse to support backtracking.
     */
    struct Snapshot
    {
        /** Full WFC state snapshot. */
        WFCState state;
        /** Collapsed cell index for this decision. */
        int cellIdx;
        /** Tile chosen at collapse time. */
        int chosenTile;
    };
    std::stack<Snapshot> history;
    int maxBacktracks;
    int lastBacktracks = 0; // backtracks used by the last run() call
    // Sparse checkpoints: snapshot every K collapses (+ always #0), see
    // sequential variant. Memory O(cells^2/K); undoes up to K per pop.
    int collapses = 0;
    static const int kCheckpointEvery = 16;

    using HeapEntry = std::pair<double, int>;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;

    /**
     * @brief Construct a WFC solver instance.
     * @param rows Output height.
     * @param cols Output width.
     * @param N Tile size.
     * @param tiles Unique model tiles.
     * @param freq Tile frequencies for weighted collapse.
     * @param compat Precomputed compatibility masks.
     * @param seed RNG seed.
     * @param maxBacktracks Maximum number of backtracks.
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

        fullMaskBuf.resize(BM_K);
        BigMask::setAll(fullMaskBuf.data(), T);

        logFreq.resize(T);
        for (int t = 0; t < T; t++)
            logFreq[t] = std::log((double)freq[t]);

        cellNoise.resize((size_t)rows * cols);
        for (int i = 0; i < rows * cols; i++)
            cellNoise[i] = noiseDist(rng);

        initState(state);
    }

    /**
     * @brief Initialize all cells with full domain and uniform initial entropy.
     * @param[out] s State to initialize.
     */
    void initState(WFCState &s)
    {
        s.rows = rows;
        s.cols = cols;
        s.T = T;
        s.ncells = (size_t)(rows * cols);
        size_t total = s.ncells * BM_K;
        s.masks.reset(new std::atomic<uint64_t>[total]);
        for (size_t i = 0; i < s.ncells; i++)
            s.store((int)i, fullMaskBuf.data());
        double h0 = shannonEntropy(fullMaskBuf.data());
        s.entropy.assign(s.ncells, h0);
    }

    /**
     * @brief Compute Shannon entropy from an active tile mask.
     * @param m Cell mask buffer.
     * @return Entropy value.
     */
    double shannonEntropy(const uint64_t *m) const
    {
        double sumW = 0.0, sumWlogW = 0.0;
        for (int w = 0; w < BM_K; w++)
        {
            uint64_t word = m[w];
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
        if (sumW <= 0.0)
            return 0.0;
        return std::log(sumW) - sumWlogW / sumW;
    }

    /**
     * @brief Rebuild the entropy heap from current state.
     */
    void rebuildHeap()
    {
        while (!heap.empty())
            heap.pop();
        for (int idx = 0; idx < rows * cols; idx++)
        {
            if (state.count(idx) <= 1)
                continue;
            heap.push({state.entropy[idx] + cellNoise[idx], idx});
        }
    }

    /**
     * @brief Push one updated cell into the entropy heap.
     * @param idx Flat cell index.
     */
    void pushHeap(int idx)
    {
        if (state.count(idx) <= 1)
            return;
        heap.push({state.entropy[idx] + cellNoise[idx], idx});
    }

    /**
     * @brief Pick next cell with minimum entropy.
     * @return Flat cell index, -1 if complete, -2 on contradiction.
     */
    int pickLowestEntropy()
    {
        uint64_t tmp[64];
        while (!heap.empty())
        {
            auto [h, idx] = heap.top();
            heap.pop();
            (void)h;
            int cnt = state.count(idx);
            if (cnt == 0)
                return -2;
            if (cnt == 1)
                continue;
            state.load(idx, tmp);
            double fresh = shannonEntropy(tmp);
            if (std::abs(fresh - state.entropy[idx]) > 1e-9)
            {
                state.entropy[idx] = fresh;
                heap.push({fresh + cellNoise[idx], idx});
                continue;
            }
            return idx;
        }
        for (int idx = 0; idx < rows * cols; idx++)
            if (state.count(idx) == 0)
                return -2;
        return -1;
    }

    /**
     * @brief Collapse one cell to a weighted random tile.
     * @param idx Flat cell index.
     * @return Chosen tile index.
     */
    int collapse(int idx)
    {
        uint64_t tmp[64];
        state.load(idx, tmp);
        std::vector<int> choices;
        std::vector<int> weights;
        for (int w = 0; w < BM_K; w++)
        {
            uint64_t word = tmp[w];
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

        uint64_t only[64];
        BigMask::setSingle(only, chosen);
        state.store(idx, only);
        state.entropy[idx] = 0.0;

        int r = idx / cols, c = idx % cols;
        if (verbose)
            std::cout << "[COLLAPSE] (" << r << "," << c << ") -> tile " << chosen << "\n";
        return chosen;
    }

    /**
     * @brief Propagate constraints from a starting cell, one BFS wave at a time.
     *
     * L1 (campagne leviers) : le voisinage n'est plus découpé en 24 tâches par
     * pop (une std::vector allouée et une barrière de taskgroup chacune) mais
     * traité par une boucle parallèle sur la vague BFS courante. Les cellules
     * modifiées sont accumulées dans un tampon par thread et fusionnées une
     * fois par vague : plus de section critique par enqueue, plus de barrière
     * par pop. Les domaines ne font que rétrécir, donc traiter une vague dans
     * le désordre atteint le même point fixe.
     *
     * @param startIdx Flat index of changed cell.
     * @return True if propagation succeeds, false if contradiction is found.
     */
    bool propagate(int startIdx)
    {
        // Le nombre de mots est fixe pour tout le run : le specialiser a la
        // compilation deroule les boucles de masque (c'est la part dominante du
        // noyau) au lieu de garder BM_K comme compteur runtime.
        switch (BM_K)
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
        const int K = WT ? WT : BM_K;
        wfcstats::bump(wfcstats::propagate_calls);
        const int cells = rows * cols;
        std::vector<char> inQ((size_t)cells, 0);
        std::vector<int> cur, next;
        cur.push_back(startIdx);
        inQ[startIdx] = 1;
        std::vector<std::vector<int>> bufs((size_t)omp_get_max_threads());
        // L5 : deux tableaux alloues une fois pour le run et jamais remis a
        // zero. Pas besoin : la comparaison ne porte que sur des cellules
        // presentes dans un tampon, donc retrecies pendant cet appel, dont la
        // version ne peut plus etre celle vue a leur dernier traitement.
        if (!cellGen)
        {
            cellGen.reset(new std::atomic<uint32_t>[(size_t)cells]);
            for (int i = 0; i < cells; i++)
                cellGen[i].store(0, std::memory_order_relaxed);
            cellGenSeen.assign((size_t)cells, 0);
        }
        std::vector<int> skipped;
        std::atomic<bool> ok(true);
        bool stop = false;

        wfcstats::bump(wfcstats::regions);
#pragma omp parallel if (cells > 4096) shared(bufs, cur, next, inQ, ok, stop)
        {
            std::vector<int> &local = bufs[(size_t)omp_get_thread_num()];
            uint64_t curBuf[64], supported[64];
            while (true)
            {
                const int n = (int)cur.size();
                if (wfcstats::on)
                    wfcstats::tasks.fetch_add(1, std::memory_order_relaxed);

#pragma omp for schedule(dynamic, 4)
                for (int i = 0; i < n; i++)
                {
                    const int idx = cur[i];
                    // La cellule est depilee : elle redevient enfilable, comme
                    // le `inQueue[idx] = false` du sequentiel. Sans cela elle
                    // ne serait traitee qu'une fois par propagation et les
                    // retrecissements subis pendant la vague ne seraient plus
                    // propages.
                    inQ[idx] = 0;
                    // L5 : version lue avant le chargement du domaine. Tout
                    // retrecissement posterieur a cette lecture incremente la
                    // version (release) et sera donc vu a la fusion (acquire
                    // / barriere), ce qui declenche le re-enfilage.
                    const uint32_t gen0 = cellGen[idx].load(std::memory_order_acquire);
                    const int r = idx / cols, c = idx % cols;
                    state.load(idx, curBuf);
                    cellGenSeen[idx] = gen0;

                    for (int dy = -(N - 1); dy <= N - 1; dy++)
                    {
                        for (int dx = -(N - 1); dx <= N - 1; dx++)
                        {
                            if (dy == 0 && dx == 0)
                                continue;
                            const int nr = r + dy, nc = c + dx;
                            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols)
                                continue;
                            const int nidx = nr * cols + nc;

                            for (int w = 0; w < K; w++)
                                supported[w] = uint64_t(0);
                            for (int w = 0; w < K; w++)
                            {
                                uint64_t word = curBuf[w];
                                while (word)
                                {
                                    const int bit = __builtin_ctzll(word);
                                    word &= word - 1;
                                    const int ta = w * 64 + bit;
                                    if (ta < T)
                                    {
                                        const uint64_t *row = compat.entry(ta, dy, dx);
                                        for (int k = 0; k < K; k++)
                                            supported[k] |= row[k];
                                    }
                                }
                            }

                            bool anyChange = false;
                            const bool nonZero = state.atomicAnd(nidx, supported, anyChange);
                            if (!nonZero)
                            {
#pragma omp critical(wfc_log)
                                std::cout << "[CONTRADICTION] at (" << nr << "," << nc << ")\n";
                                ok.store(false, std::memory_order_relaxed);
                            }
                            else if (anyChange)
                            {
                                cellGen[nidx].fetch_add(1, std::memory_order_release);
                                uint64_t newBuf[64];
                                state.load(nidx, newBuf);
                                state.entropy[nidx] = shannonEntropy(newBuf);
                                local.push_back(nidx);
                            }
                        }
                    }
                }

// Une seule fusion par vague (le thread maître) : plus de section critique.
#pragma omp single
                {
                    if (wfcstats::on)
                        wfcstats::pops += cur.size();
                    next.clear();
                    skipped.clear();
                    for (auto &b : bufs)
                    {
                        for (int cell : b)
                        {
                            if (!inQ[cell])
                            {
                                inQ[cell] = 1;
                                pushHeap(cell);
                                // L5 : n'enfiler que si la version a bouge
                                // depuis le traitement de la cellule. Le
                                // pushHeap reste inconditionnel : le tas doit
                                // voir exactement les memes entrees (et les
                                // memes tirages de bruit) qu'avant, sinon
                                // l'ordre de collapse changerait.
                                if (cellGenSeen[cell] != cellGen[cell].load(std::memory_order_relaxed))
                                {
                                    next.push_back(cell);
                                    wfcstats::bump(wfcstats::pushes);
                                }
                                else
                                    skipped.push_back(cell);
                            }
                        }
                        b.clear();
                    }
                    // Cellules deja propagees : ni dans cur ni dans next, donc
                    // de nouveau enfilables pour les vagues suivantes.
                    for (int cell : skipped)
                        inQ[cell] = 0;
                    cur.swap(next);
                    wfcstats::note_queue(cur.size());
                    wfcstats::bump(wfcstats::critical_entries);
                    stop = cur.empty() || !ok.load(std::memory_order_relaxed);
                }
                if (stop)
                    break;
            }
        }
        return ok.load(std::memory_order_relaxed);
    }

    /**
     * @brief Restore snapshots until a valid alternate branch is found.
     * @param[in,out] backtracks Current number of performed backtracks.
     * @return True if a consistent resumed state is found.
     */
    bool backtrack(int &backtracks)
    {
        uint64_t mBuf[64], bitBuf[64], newBuf[64];
        while (!history.empty() && backtracks < maxBacktracks)
        {
            Snapshot snap = history.top();
            history.pop();
            backtracks++;
            state = snap.state;
            int idx = snap.cellIdx;

            state.load(idx, mBuf);
            if (!BigMask::testBit(mBuf, snap.chosenTile))
                continue;

            BigMask::setSingle(bitBuf, snap.chosenTile);
            for (int i = 0; i < BM_K; i++)
                bitBuf[i] = ~bitBuf[i];
            bool anyChange = false;
            bool nonZero = state.atomicAnd(idx, bitBuf, anyChange);
            if (!nonZero)
                continue;

            state.load(idx, newBuf);
            state.entropy[idx] = shannonEntropy(newBuf);

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
     * @brief Execute the full WFC loop with propagation and bounded backtracking.
     * @return True on success, false on unrecoverable contradiction.
     */
    bool run()
    {
        std::cout << "=== WFC " << rows << "x" << cols << " (T=" << T << ") ===\n";
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
            // Sparse checkpoint of the PRE-collapse state (see member docs).
            bool checkpoint = (collapses == 0 || collapses % kCheckpointEvery == 0);
            if (checkpoint)
                history.push({state, idx, -1});
            int chosen = collapse(idx);
            if (checkpoint)
                history.top().chosenTile = chosen;
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
     * @brief Build an output palette-index grid from the final state.
     * @return Output grid of shape [rows][cols].
     */
    std::vector<std::vector<int>> buildOutput() const
    {
        std::vector<std::vector<int>> output(rows, std::vector<int>(cols, -1));
        uint64_t tmp[64];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                int idx = r * cols + c;
                state.load(idx, tmp);
                int t = BigMask::ctz(tmp);
                if (t >= 0 && t < T)
                    output[r][c] = tiles[t].data[0][0];
            }
        return output;
    }

    /**
     * @brief Validate that every NxN output subgrid belongs to learned tiles.
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
        if (!errors)
            std::cout << "[VERIFY_OK] All subgrids valid.\n";
        else
            std::cout << "[VERIFY_FAIL] " << errors << " invalid subgrids.\n";
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
                   const std::vector<uint32_t> &palette, int scale = 32)
    {
        int imgW = cols * scale, imgH = rows * scale;
        std::vector<unsigned char> pixels(imgW * imgH * 3);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                uint32_t color = (output[r][c] < (int)palette.size())
                                     ? palette[output[r][c]]
                                     : 0;
                unsigned char R = (color >> 16) & 0xFF, G = (color >> 8) & 0xFF, B = color & 0xFF;
                for (int py = 0; py < scale; py++)
                    for (int px = 0; px < scale; px++)
                    {
                        int i = ((r * scale + py) * imgW + (c * scale + px)) * 3;
                        pixels[i] = R;
                        pixels[i + 1] = G;
                        pixels[i + 2] = B;
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
 * - --verbose anywhere: log every collapse (off by default; heavy on big grids)
 *
 * If no input image is provided, a built-in 5x5 binary sample is used.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument vector.
 * @return Process exit code.
 */
int main(int argc, char *argv[])
{
    using Clock = std::chrono::high_resolution_clock;

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
            {1, 0, 1, 1, 1}, {1, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 0, 0, 0, 0}};
        palette = {0x000000, 0xFFFFFF};
        std::cout << "[INPUT] Using built-in 5x5 sample.\n";
    }

    int inputRows = (int)sample.size(), inputCols = (int)sample[0].size();
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
        return 1;
    }

    std::cout << "[OUTPUT SIZE] " << outCols << "x" << outRows
              << "  N=" << N << "  scale=" << scale
              << "  symmetries=" << (useSymmetries ? "on" : "off")
              << "  seed=" << seed << "\n";
#ifdef _OPENMP
    std::cout << "[THREADS] max=" << omp_get_max_threads() << "\n";
#endif

    auto t_start = Clock::now();
    std::vector<Tile> tiles;
    std::vector<int> freq;
    extractTiles(sample, N, tiles, freq, useSymmetries);

    if (tiles.empty())
    {
        std::cerr << "[ERROR] No tiles extracted\n";
        return 1;
    }

    auto t_extract = Clock::now();
    CompatTable compat;
    compat.build(tiles);

    auto t_compat = Clock::now();
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

    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cout << "\n"
              << std::string(50, '=') << "\n";
    std::cout << "[TIMING] wfc_openmp.cpp\n";
    std::cout << "  Extract tiles:         " << std::fixed << std::setprecision(2)
              << elapsed_ms(t_start, t_extract) << " ms\n";
    std::cout << "  Build compatibility:   " << elapsed_ms(t_extract, t_compat) << " ms\n";
    std::cout << "  WFC algorithm:         " << elapsed_ms(t_compat, t_end) << " ms\n";
    std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
    wfcstats::print("omp");
    std::cout << std::string(50, '=') << "\n";

    return exitCode;
}