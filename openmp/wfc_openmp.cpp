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
#include <utility>
#include <vector>

#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

    WFCState state;

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
            heap.push({state.entropy[idx] + noiseDist(rng), idx});
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
        heap.push({state.entropy[idx] + noiseDist(rng), idx});
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
                heap.push({fresh + noiseDist(rng), idx});
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
        std::cout << "[COLLAPSE] (" << r << "," << c << ") -> tile " << chosen << "\n";
        return chosen;
    }

    /**
     * @brief Propagate constraints from a starting cell.
     *
     * Neighbor updates are parallelized with OpenMP tasks. Domain intersections are
     * applied atomically, and changed cells are queued for further propagation.
     *
     * @param startIdx Flat index of changed cell.
     * @return True if propagation succeeds, false if contradiction is found.
     */
    bool propagate(int startIdx)
    {
        int cells = rows * cols;
        std::queue<int> globalQ;
        std::vector<bool> inGlobal(cells, false);
        globalQ.push(startIdx);
        inGlobal[startIdx] = true;

        std::atomic<bool> ok(true);

#pragma omp parallel
        {
#pragma omp single nowait
            {
                while (!globalQ.empty() && ok.load(std::memory_order_relaxed))
                {
                    int idx = globalQ.front();
                    globalQ.pop();
                    inGlobal[idx] = false;

                    int r = idx / cols, c = idx % cols;

                    uint64_t curBuf[64];
                    state.load(idx, curBuf);

                    // Gather newly constrained cells from parallel tasks and enqueue after taskgroup.
                    std::vector<int> toEnqueue;

#pragma omp taskgroup
                    {
                        for (int dy = -(N - 1); dy <= N - 1; dy++)
                        {
                            for (int dx = -(N - 1); dx <= N - 1; dx++)
                            {
                                if (dy == 0 && dx == 0)
                                    continue;
                                int nr = r + dy, nc = c + dx;
                                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols)
                                    continue;
                                int nidx = nr * cols + nc;
                                std::vector<uint64_t> curVec(curBuf, curBuf + BM_K);

#pragma omp task firstprivate(nidx, dy, dx, nr, nc, curVec) \
    shared(toEnqueue, ok)
                                {
                                    if (ok.load(std::memory_order_relaxed))
                                    {
                                        uint64_t supported[64] = {};
                                        for (int w = 0; w < BM_K; w++)
                                        {
                                            uint64_t word = curVec[w];
                                            while (word)
                                            {
                                                int bit = __builtin_ctzll(word);
                                                word &= word - 1;
                                                int ta = w * 64 + bit;
                                                if (ta < T)
                                                    BigMask::orInto(supported,
                                                                    compat.entry(ta, dy, dx));
                                            }
                                        }

                                        bool anyChange = false;
                                        bool nonZero = state.atomicAnd(nidx, supported, anyChange);

                                        if (!nonZero)
                                        {
#pragma omp critical(wfc_log)
                                            std::cout << "[CONTRADICTION] at ("
                                                      << nr << "," << nc << ")\n";
                                            ok.store(false, std::memory_order_relaxed);
                                        }
                                        else if (anyChange)
                                        {
                                            uint64_t newBuf[64];
                                            state.load(nidx, newBuf);
                                            state.entropy[nidx] = shannonEntropy(newBuf);
#pragma omp critical(wfc_enqueue)
                                            {
                                                toEnqueue.push_back(nidx);
                                            }
                                        }
                                    }
                                } // task
                            }
                        }
                    } // taskgroup

                    if (!ok.load(std::memory_order_relaxed))
                        break;

                    for (int cell : toEnqueue)
                    {
                        if (!inGlobal[cell])
                        {
                            globalQ.push(cell);
                            inGlobal[cell] = true;
                            pushHeap(cell);
                        }
                    }
                }
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
                    std::cout << "[FAIL] Contradiction, aucune option de backtrack ("
                              << backtracks << " effectués)\n";
                    return false;
                }
                continue;
            }
            if (idx == -1)
            {
                std::cout << "=== Toutes les cellules effondrées ! ===\n";
                return true;
            }
            int chosen = collapse(idx);
            history.push({state, idx, chosen});
            if (!propagate(idx))
            {
                if (!backtrack(backtracks))
                {
                    std::cout << "[FAIL] Contradiction après collapse, pas de backtrack\n";
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
        int errors = 0;
        for (int r = 0; r + N <= rows; r++)
            for (int c = 0; c + N <= cols; c++)
            {
                std::vector<std::vector<int>> sub(N, std::vector<int>(N));
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        sub[i][j] = output[r + i][c + j];
                bool found = false;
                for (const auto &t : tiles)
                    if (t.data == sub)
                    {
                        found = true;
                        break;
                    }
                if (!found)
                {
                    std::cout << "[VERIFY ERROR] at (" << r << "," << c << ")\n";
                    errors++;
                }
            }
        if (!errors)
            std::cout << "[VERIFY] Tous les sous-grilles valides.\n";
        else
            std::cout << "[VERIFY] " << errors << " sous-grilles invalides.\n";
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
 * If no input image is provided, a built-in 5x5 binary sample is used.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument vector.
 * @return 0 on normal termination.
 */
int main(int argc, char *argv[])
{
    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();

    std::vector<std::vector<int>> sample;
    std::vector<uint32_t> palette;

    if (argc >= 2)
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

    std::cout << "[OUTPUT SIZE] " << outCols << "x" << outRows
              << "  N=" << N << "  scale=" << scale << "\n";

    auto t_extract = Clock::now();
    std::vector<Tile> tiles;
    std::vector<int> freq;
    extractTiles(sample, N, tiles, freq, /*useSymmetries=*/false);

    auto t_compat = Clock::now();
    CompatTable compat;
    compat.build(tiles);

    auto t_wfc = Clock::now();
    unsigned int seed = (unsigned int)
                            std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count();
    WFC wfc(outRows, outCols, N, tiles, freq, compat, seed, /*maxBacktracks=*/200);
    bool success = wfc.run();
    auto t_end = Clock::now();

    if (success)
    {
        auto output = wfc.buildOutput();
        wfc.verify(output);
        wfc.saveImage("output.png", output, palette, scale);

        int inNonBg = 0;
        for (const auto &row : sample)
            for (int v : row)
                inNonBg += (v != 0);
        double inDensity = (double)inNonBg / (inputRows * inputCols);

        int outNonBg = 0;
        for (const auto &row : output)
            for (int v : row)
                outNonBg += (v != 0);
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
    std::cout << "  WFC algorithm:         " << elapsed_ms(t_compat, t_wfc) << " ms\n";
    std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
    std::cout << std::string(50, '=') << "\n";

    return 0;
}