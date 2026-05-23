#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>
#include <string>
#include <iomanip>
#include <cstdint>
#include <queue>
#include <functional>
#include <stack>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
struct CompatTable
{
    /** Number of unique tiles. */
    int T, N;
    /** Number of offsets, equals (2N-1)^2. */
    int D; // (2N-1)²
    // compat[a * D + offset] = vector<bool> of length T
    std::vector<std::vector<bool>> table; // size T*D, each of length T

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
     * @brief Build the full compatibility table from extracted tiles.
     * @param tiles Unique tile set.
     */
    void build(const std::vector<Tile> &tiles)
    {
        T = (int)tiles.size();
        N = tiles[0].N;
        D = (2 * N - 1) * (2 * N - 1);
        table.assign(T * D, std::vector<bool>(T, false));

        for (int a = 0; a < T; a++)
            for (int b = 0; b < T; b++)
                for (int dy = -(N - 1); dy <= N - 1; dy++)
                    for (int dx = -(N - 1); dx <= N - 1; dx++)
                    {
                        if (dy == 0 && dx == 0)
                            continue;
                        if (canOverlap(tiles[a], tiles[b], dy, dx))
                            table[a * D + offsetIdx(dy, dx)][b] = true;
                    }
        std::cout << "[COMPAT] Table built for " << T << " tiles\n";
    }

    /**
     * @brief Get all tile indices compatible with tile @p a at offset (dy, dx).
     * @param a Source tile index.
     * @param dy Row offset.
     * @param dx Column offset.
     * @return Boolean vector of size T; true means compatible.
     */
    const std::vector<bool> &get(int a, int dy, int dx) const
    {
        return table[a * D + offsetIdx(dy, dx)];
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
    // possible[r*cols+c][t] = true if tile t is still possible at (r,c)
    std::vector<std::vector<bool>> possible;

    // Cached entropy per cell (-1 = dirty / needs recompute)
    // We store a pair (entropy, noise) so the heap key is stable
    // entropy == -2.0 means "already collapsed" (skip)
    std::vector<double> entropy;

    // Count of possible tiles per cell (to detect contradictions quickly)
    std::vector<int> possCount;

    /** Grid height, width, and number of tile states. */
    int rows, cols, T;
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

    WFCState state;

    // Backtracking stack: each entry is a full snapshot of the state
    // plus the cell that was collapsed (for diagnostics)
    /**
     * @brief Snapshot saved before each collapse step for backtracking.
     */
    struct Snapshot
    {
        WFCState state;
        std::pair<int, int> cell;
    };
    std::stack<Snapshot> history;
    int maxBacktracks;

    // Min-heap: (entropy+noise, flat_cell_index)
    using HeapEntry = std::pair<double, int>;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;

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
        s.possible.assign(cells, std::vector<bool>(T, true));
        s.possCount.assign(cells, T);
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
        const auto &poss = state.possible[idx];
        for (int t = 0; t < T; t++)
        {
            if (poss[t])
            {
                sumW += freq[t];
                sumWlogW += freq[t] * logFreq[t];
            }
        }
        return shannonEntropy(sumW, sumWlogW);
    }

    /**
     * @brief Rebuild the min-heap of candidate cells from scratch.
     */
    void rebuildHeap()
    {
        while (!heap.empty())
            heap.pop();
        for (int idx = 0; idx < rows * cols; idx++)
        {
            if (state.possCount[idx] <= 1)
                continue; // collapsed or contradiction
            double h = state.entropy[idx] + noiseDist(rng);
            heap.push({h, idx});
        }
    }

    /**
     * @brief Insert/update one cell in the min-heap after entropy changes.
     * @param idx Flat cell index.
     */
    void pushHeap(int idx)
    {
        if (state.possCount[idx] <= 1)
            return;
        double h = state.entropy[idx] + noiseDist(rng);
        heap.push({h, idx});
        // Stale entries remain in the heap but are skipped in pickLowestEntropy
    }

    /**
     * @brief Pick the next cell with the smallest entropy.
     * @return Flat cell index, -1 if completed, or -2 on contradiction.
     */
    int pickLowestEntropy()
    {
        while (!heap.empty())
        {
            auto [h, idx] = heap.top();
            heap.pop();
            int cnt = state.possCount[idx];
            if (cnt == 0)
                return -2; // contradiction
            if (cnt == 1)
                continue; // already collapsed, stale entry
            // Check staleness: recompute and re-push if entropy changed significantly
            double fresh = computeEntropy(idx);
            if (std::abs(fresh - state.entropy[idx]) > 1e-9)
            {
                state.entropy[idx] = fresh;
                heap.push({fresh + noiseDist(rng), idx});
                continue;
            }
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
     */
    void collapse(int idx)
    {
        auto &poss = state.possible[idx];
        std::vector<int> choices;
        std::vector<int> weights;
        for (int t = 0; t < T; t++)
        {
            if (poss[t])
            {
                choices.push_back(t);
                weights.push_back(freq[t]);
            }
        }
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int chosen = choices[dist(rng)];

        // Clear all but chosen
        for (int t = 0; t < T; t++)
            poss[t] = false;
        poss[chosen] = true;
        state.possCount[idx] = 1;
        state.entropy[idx] = 0.0;

        int r = idx / cols, c = idx % cols;
        std::cout << "[COLLAPSE] (" << r << "," << c << ") -> tile " << chosen << "\n";
    }

    /**
     * @brief Propagate domain reductions from a starting cell using BFS.
     * @param startIdx Flat index of the changed cell.
     * @return True on success, false if a contradiction is detected.
     * @note Worst-case complexity is O(cells * D * T^2), typically lower in practice.
     */
    bool propagate(int startIdx)
    {
        std::vector<bool> inQueue(rows * cols, false);
        std::queue<int> q;
        q.push(startIdx);
        inQueue[startIdx] = true;

        while (!q.empty())
        {
            int idx = q.front();
            q.pop();
            inQueue[idx] = false;
            int r = idx / cols, c = idx % cols;

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

                    // --- O(T²) support computation ---
                    // Build union of tiles supported by any tile at (r,c)
                    // for the offset (dy,dx) towards (nr,nc)
                    std::vector<bool> supported(T, false);
                    const auto &poss = state.possible[idx];
                    for (int ta = 0; ta < T; ta++)
                    {
                        if (!poss[ta])
                            continue;
                        const auto &compat_row = compat.get(ta, dy, dx);
                        for (int tb = 0; tb < T; tb++)
                            if (compat_row[tb])
                                supported[tb] = true;
                    }

                    // Remove unsupported tiles from neighbor
                    auto &nposs = state.possible[nidx];
                    bool changed = false;
                    for (int tb = 0; tb < T; tb++)
                    {
                        if (nposs[tb] && !supported[tb])
                        {
                            nposs[tb] = false;
                            state.possCount[nidx]--;
                            changed = true;
                        }
                    }

                    if (state.possCount[nidx] == 0)
                    {
                        std::cout << "[CONTRADICTION] at (" << nr << "," << nc << ")\n";
                        return false;
                    }

                    if (changed)
                    {
                        // Update incremental entropy
                        state.entropy[nidx] = computeEntropy(nidx);
                        pushHeap(nidx);
                        if (!inQueue[nidx])
                        {
                            q.push(nidx);
                            inQueue[nidx] = true;
                        }
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Execute WFC until completion or unrecoverable contradiction.
     * @return True if all cells collapse successfully, false otherwise.
     */
    bool run()
    {
        std::cout << "=== WFC " << rows << "x" << cols << " ===\n";
        rebuildHeap();
        int backtracks = 0;

        while (true)
        {
            int idx = pickLowestEntropy();

            if (idx == -2)
            {
                // Contradiction — try to backtrack
                if (history.empty() || backtracks >= maxBacktracks)
                {
                    std::cout << "[FAIL] Contradiction, no more backtrack options ("
                              << backtracks << " done)\n";
                    return false;
                }
                std::cout << "[BACKTRACK] #" << ++backtracks << "\n";
                auto &snap = history.top();
                state = snap.state;
                // Force-remove the tile that was chosen at this cell
                // by collapsing to something else on retry
                // Simple approach: restore and let pickLowestEntropy retry with noise
                history.pop();
                rebuildHeap();
                continue;
            }

            if (idx == -1)
            {
                std::cout << "=== All cells collapsed! ===\n";
                return true;
            }

            // Save snapshot before collapsing
            history.push({state, {idx / cols, idx % cols}});

            collapse(idx);

            if (!propagate(idx))
            {
                if (history.empty() || backtracks >= maxBacktracks)
                {
                    std::cout << "[FAIL] Contradiction after collapse, no backtrack\n";
                    return false;
                }
                std::cout << "[BACKTRACK] #" << ++backtracks << " after propagation\n";
                auto &snap = history.top();
                state = snap.state;
                history.pop();
                rebuildHeap();
            }
        }
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
                for (int t = 0; t < T; t++)
                    if (state.possible[idx][t])
                    {
                        output[r][c] = tiles[t].data[0][0];
                        break;
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
        int errors = 0;
        for (int r = 0; r + N <= rows; r++)
        {
            for (int c = 0; c + N <= cols; c++)
            {
                std::vector<std::vector<int>> sub(N, std::vector<int>(N));
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        sub[i][j] = output[r + i][c + j];
                bool found = false;
                for (auto &t : tiles)
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
        }
        if (!errors)
            std::cout << "[VERIFY] All subgrids valid.\n";
        else
            std::cout << "[VERIFY] " << errors << " invalid subgrids.\n";
        return errors == 0;
    }

    /**
     * @brief Save the generated output as a PNG image.
     * @param filename Destination file path.
     * @param output Generated palette-index grid.
     * @param palette RGB palette encoded as 0xRRGGBB.
     * @param scale Pixel scale per logical output cell.
     */
    void saveImage(const std::string &filename,
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

    // Usage: wfc [image] [outW] [outH] [N] [scale]
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

    auto t_extract = Clock::now();

    // Step 1: Extract tiles
    std::vector<Tile> tiles;
    std::vector<int> freq;
    extractTiles(sample, N, tiles, freq, /*useSymmetries=*/false);

    auto t_compat = Clock::now();

    // Step 2: Build compatibility table
    CompatTable compat;
    compat.build(tiles);

    auto t_wfc = Clock::now();

    // Step 3: Run WFC
    unsigned int seed = (unsigned int)
                            std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count();

    WFC wfc(outRows, outCols, N, tiles, freq, compat, seed, /*maxBacktracks=*/200);
    bool success = wfc.run();

    auto t_end = Clock::now();

    if (success)
    {
        std::cout << "[RESULT] WFC succeeded.\n";
        auto output = wfc.buildOutput();
        wfc.verify(output);
        wfc.saveImage("output.png", output, palette, scale);

        // Density similarity metric
        int inNonBg = 0;
        for (auto &row : sample)
            for (int v : row)
                inNonBg += (v != 0);
        double inDensity = (double)inNonBg / (inputRows * inputCols);

        int outNonBg = 0;
        for (auto &row : output)
            for (int v : row)
                outNonBg += (v != 0);
        double outDensity = (double)outNonBg / (outRows * outCols);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[SIMILARITY] Input density:  " << inDensity << "\n";
        std::cout << "[SIMILARITY] Output density: " << outDensity << "\n";
        std::cout << "[SIMILARITY] Difference:     " << std::abs(outDensity - inDensity) << "\n";
    }
    else
    {
        std::cout << "[RESULT] WFC failed.\n";
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
    std::cout << "  WFC algorithm:         " << elapsed_ms(t_compat, t_wfc) << " ms\n";
    std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
    std::cout << std::string(50, '=') << "\n";

    return 0;
}