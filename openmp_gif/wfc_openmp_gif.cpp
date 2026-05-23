#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
 * @file wfc_openmp_gif.cpp
 * @brief OpenMP Wave Function Collapse (WFC) implementation with GIF animation recording.
 *
 * This version extends the dynamic bit-mask WFC algorithm with GIF encoding capabilities:
 * - OpenMP tasks for tile extraction, compatibility precomputation, and propagation.
 * - Dynamic packed bit-mask representation for tile domains (unbounded tile count).
 * - Atomic mask updates for concurrent domain pruning during propagation.
 * - GIF frame recording with LZW compression for step-by-step visualization.
 */

/**
 * @namespace gif
 * @brief GIF encoding routines: bit writer, LZW compression, and file writer.
 */
namespace gif
{

    /**
     * @brief Bit-level output writer for variable-width encoding.
     *
     * Buffers output codes with variable bit widths, accumulating them into 8-bit bytes
     * before appending to an output vector.
     */
    struct BitWriter
    {
        /** Output byte buffer. */
        std::vector<uint8_t> &out;
        /** Bit accumulator. */
        uint32_t buf = 0;
        /** Number of valid bits in accumulator. */
        int bits = 0;

        /**
         * @brief Write a code with variable bit width.
         * @param code Bit pattern to write.
         * @param nbits Number of bits to write (1-32).
         */
        void write(uint32_t code, int nbits)
        {
            buf |= code << bits;
            bits += nbits;
            while (bits >= 8)
            {
                out.push_back(buf & 0xFF);
                buf >>= 8;
                bits -= 8;
            }
        }
        /**
         * @brief Flush any remaining bits in the accumulator.
         */
        void flush()
        {
            if (bits)
            {
                out.push_back(buf & 0xFF);
                buf = 0;
                bits = 0;
            }
        }
    };

    /**
     * @brief LZW compression for palette-based image data.
     *
     * Implements the LZW algorithm used by GIF files, building a dynamic code table
     * and resetting periodically to prevent overflow.
     *
     * @param pixels Input palette indices.
     * @param n Number of pixels.
     * @param minCodeSize Initial code size (typically 2-8 for GIF).
     * @param[out] out LZW-compressed data with block boundaries.
     */
    void lzwCompress(const uint8_t *pixels, int n,
                     int minCodeSize,
                     std::vector<uint8_t> &out)
    {
        const int CLEAR = 1 << minCodeSize;
        const int EOI = CLEAR + 1;
        constexpr int HASH_SZ = 8192;
        constexpr int HASH_MASK = HASH_SZ - 1;

        struct Entry
        {
            int prefix;
            uint8_t suffix;
        };
        Entry table[4096];
        int tableSize = 0;

        int hash[HASH_SZ];

        auto hashKey = [](int prefix, uint8_t suffix) -> int
        {
            return ((prefix * 251) ^ (suffix * 7919)) & HASH_MASK;
        };
        auto hashFind = [&](int prefix, uint8_t suffix) -> int
        {
            int k = hashKey(prefix, suffix);
            for (;;)
            {
                if (hash[k] == -1)
                    return -1;
                int idx = hash[k];
                if (table[idx].prefix == prefix && table[idx].suffix == suffix)
                    return idx;
                k = (k + 1) & HASH_MASK;
            }
        };
        auto hashInsert = [&](int idx)
        {
            int k = hashKey(table[idx].prefix, table[idx].suffix);
            while (hash[k] != -1)
                k = (k + 1) & HASH_MASK;
            hash[k] = idx;
        };

        std::vector<uint8_t> lzwBuf;
        lzwBuf.reserve(n + 256);
        BitWriter bw{lzwBuf};
        int codeSize = minCodeSize + 1;

        out.reserve(out.size() + n + 512);

        auto fullReset = [&]()
        {
            std::fill(hash, hash + HASH_SZ, -1);
            tableSize = CLEAR + 2;
            codeSize = minCodeSize + 1;
            bw.write(CLEAR, codeSize);
        };
        fullReset();

        int prefix = pixels[0];
        for (int i = 1; i < n; i++)
        {
            uint8_t suf = pixels[i];
            int found = hashFind(prefix, suf);
            if (found != -1)
            {
                prefix = found;
            }
            else
            {
                bw.write(prefix, codeSize);
                int nextCode = tableSize;
                if (nextCode < 4096)
                {
                    table[nextCode] = {prefix, suf};
                    hashInsert(nextCode);
                    tableSize++;
                    if (nextCode == (1 << codeSize))
                        codeSize++;
                }
                else
                {
                    fullReset();
                }
                prefix = suf;
            }
        }
        bw.write(prefix, codeSize);
        bw.write(EOI, codeSize);
        bw.flush();

        size_t pos = 0;
        while (pos < lzwBuf.size())
        {
            uint8_t chunk = (uint8_t)std::min((size_t)255, lzwBuf.size() - pos);
            out.push_back(chunk);
            out.insert(out.end(), lzwBuf.begin() + pos, lzwBuf.begin() + pos + chunk);
            pos += chunk;
        }
        out.push_back(0);
    }

    /**
     * @brief Color palette for GIF encoding.
     *
     * Stores up to 256 RGB colors and encodes the power-of-2 table size required by GIF format.
     */
    struct Palette
    {
        /** RGB color entries: [index][0=R, 1=G, 2=B]. */
        uint8_t rgb[256][3] = {};
        /** Number of colors in use. */
        int size = 0;
        /** Power such that (1 << (pow2+1)) = actual palette size. */
        int pow2 = 0;
    };

    /**
     * @brief GIF file writer with per-frame compression.
     *
     * Constructs a valid GIF89a file with animated frames. Each frame is independently
     * LZW-compressed and can have a per-frame display delay.
     */
    class Writer
    {
    public:
        /** Image dimensions. */
        int w, h;
        /** Output palette. */
        Palette pal;
        /** File handle (null when not open). */
        FILE *fp = nullptr;

        /**
         * @brief Open a GIF file and write header.
         * @param path Output file path.
         * @param width Frame width in pixels.
         * @param height Frame height in pixels.
         * @param palette Color palette.
         * @return True on success.
         */
        bool open(const std::string &path, int width, int height,
                  const Palette &palette)
        {
            fp = fopen(path.c_str(), "wb");
            if (!fp)
                return false;
            w = width;
            h = height;
            pal = palette;
            writeHeader();
            return true;
        }

        /**
         * @brief Add a raw frame to the GIF, compressing on-the-fly.
         * @param pixels Palette-indexed pixel data.
         * @param delay Frame delay in centiseconds (default 10 = 0.1 sec).
         */
        void addFrame(const std::vector<uint8_t> &pixels, int delay = 10)
        {
            writeGCE(delay);
            writeImageDesc();
            int minCode = std::max(2, pal.pow2 + 1);
            std::vector<uint8_t> lzw;
            lzwCompress(pixels.data(), (int)pixels.size(), minCode, lzw);
            uint8_t mc = (uint8_t)minCode;
            fwrite(&mc, 1, 1, fp);
            fwrite(lzw.data(), 1, lzw.size(), fp);
        }

        /**
         * @brief Add a pre-compressed frame to the GIF.
         * @param lzwPayload Pre-compressed LZW data block.
         * @param minCode Minimum code size used during compression.
         * @param delay Frame delay in centiseconds (default 10).
         */
        void addPrecompressed(const std::vector<uint8_t> &lzwPayload,
                              int minCode, int delay = 10)
        {
            writeGCE(delay);
            writeImageDesc();
            uint8_t mc = (uint8_t)minCode;
            fwrite(&mc, 1, 1, fp);
            fwrite(lzwPayload.data(), 1, lzwPayload.size(), fp);
        }

        /**
         * @brief Finalize and close the GIF file.
         */
        void close()
        {
            if (fp)
            {
                fputc(0x3B, fp);
                fclose(fp);
                fp = nullptr;
            }
        }

        /** Destructor ensures file is closed. */
        ~Writer() { close(); }

    private:
        /** Write GIF89a file header with palette. */
        /** Write GIF89a file header with palette. */
        void writeHeader()
        {
            fwrite("GIF89a", 1, 6, fp);
            uint16_t lw = (uint16_t)w, lh = (uint16_t)h;
            fwrite(&lw, 2, 1, fp);
            fwrite(&lh, 2, 1, fp);
            uint8_t packed = 0x80 | (pal.pow2 << 4) | pal.pow2;
            fputc(packed, fp);
            fputc(0, fp);
            fputc(0, fp);
            int tableSize = 1 << (pal.pow2 + 1);
            fwrite(pal.rgb, 3, tableSize, fp);
        }

        /**
         * @brief Write Graphics Control Extension (frame timing, transparency).
         * @param delay Display time in centiseconds.
         */
        void writeGCE(int delay)
        {
            fputc(0x21, fp);
            fputc(0xF9, fp);
            fputc(4, fp);
            fputc(0x00, fp);
            uint16_t d = (uint16_t)delay;
            fwrite(&d, 2, 1, fp);
            fputc(0, fp);
            fputc(0, fp);
        }

        /** Write Image Descriptor block. */
        void writeImageDesc()
        {
            fputc(0x2C, fp);
            uint16_t zero = 0, lw = (uint16_t)w, lh = (uint16_t)h;
            fwrite(&zero, 2, 1, fp);
            fwrite(&zero, 2, 1, fp);
            fwrite(&lw, 2, 1, fp);
            fwrite(&lh, 2, 1, fp);
            fputc(0x00, fp);
        }
    };

}

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
        int fullWords = T / 64, rem = T % 64;
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
            dst[i] = 0;
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
                               const uint64_t *keep, bool &anyChange)
    {
        anyChange = false;
        for (int i = 0; i < BM_K; i++)
        {
            uint64_t old = dst[i].load(std::memory_order_relaxed), next;
            do
            {
                next = old & keep[i];
                if (next == old)
                    goto next_word;
            } while (!dst[i].compare_exchange_weak(old, next,
                                                   std::memory_order_acq_rel, std::memory_order_relaxed));
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
    std::atomic<uint64_t> *cell(int idx) { return masks.get() + (size_t)idx * BM_K; }
    /**
     * @brief Get const pointer to cell mask words.
     * @param idx Flat cell index.
     * @return Pointer to BM_K atomic words.
     */
    const std::atomic<uint64_t> *cell(int idx) const { return masks.get() + (size_t)idx * BM_K; }

    /**
     * @brief Load one cell mask to a plain buffer.
     * @param idx Flat cell index.
     * @param[out] dst Destination buffer with BM_K words.
     */
    void load(int idx, uint64_t *dst) const { BigMask::loadFrom(cell(idx), dst); }

    /**
     * @brief Store a plain mask into one cell.
     * @param idx Flat cell index.
     * @param src Source buffer with BM_K words.
     */
    void store(int idx, const uint64_t *src) { BigMask::storeTo(cell(idx), src); }

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
 * @brief GIF frame recording and rendering for WFC step visualization.
 *
 * Captures WFC solver state at key points (initial state, after each collapse)
 * and stores rendered snapshots for later GIF encoding. Handles palette management,
 * state visualization (superposition, contradiction, collapsed cells), and
 * parallel LZW compression.
 */
struct GifRecorder
{
    /** Grid dimensions, tile count, and pixel scale for rendering. */
    int rows, cols, T, scale;
    /** Maximum number of frames to capture. */
    int maxFrames;
    /** Reference to global palette. */
    const std::vector<uint32_t> &palette;
    /** Reference to extracted tiles. */
    const std::vector<Tile> &tiles;

    /** GIF-format palette with power-of-2 sizing. */
    gif::Palette gifPal;
    /** Special palette indices for rendering. */
    int idxSuperposition;
    int idxContradiction;
    int idxBackground;

    /**
     * @brief Stored frame with pre-rendered pixels.
     */
    struct FrameData
    {
        /** Scaled pixel data in palette indices. */
        std::vector<uint8_t> pixels;
    };
    /** Captured frames in order. */
    std::vector<FrameData> frames;

    /** Expected total collapses (for adaptive capture rate). */
    int totalCollapsesExpected = 0;
    /** Capture every N-th collapse. */
    int captureEvery = 1;
    /** Collapses since last capture. */
    int collapsesSinceCapture = 0;

    /**
     * @brief Initialize GIF recorder.
     * @param rows Output grid height.
     * @param cols Output grid width.
     * @param T Number of tiles.
     * @param scale Pixel scale per cell.
     * @param maxFrames Maximum frames to capture.
     * @param pal Global palette (0xRRGGBB format).
     * @param tiles Tile definitions for rendering.
     * @param totalCells Total cells to adapt capture frequency.
     */
    GifRecorder(int rows, int cols, int T, int scale, int maxFrames,
                const std::vector<uint32_t> &pal,
                const std::vector<Tile> &tiles,
                int totalCells)
        : rows(rows), cols(cols), T(T), scale(scale), maxFrames(maxFrames),
          palette(pal), tiles(tiles)
    {

        totalCollapsesExpected = totalCells;
        captureEvery = std::max(1, totalCells / maxFrames);

        buildGifPalette();
    }

    /**
     * @brief Build GIF-compatible palette from the global palette.
     *
     * Allocates power-of-2 colors and reserves special indices for:
     * - Superposition (gray) - cells with multiple possible tiles
     * - Contradiction (red) - cells with no valid tiles
     * - Background (dark gray) - for out-of-range palette indices
     */
    void buildGifPalette()
    {
        int needed = (int)palette.size() + 3;

        int pow2 = 0;
        int size = 2;
        while (size < needed && size < 256)
        {
            size <<= 1;
            pow2++;
        }
        if (size < needed)
        {
            std::cerr << "[GIF] Attention: " << palette.size()
                      << " couleurs, troncature à 253.\n";
        }
        gifPal.pow2 = pow2;
        gifPal.size = size;

        int n = (int)std::min((size_t)253, palette.size());
        for (int i = 0; i < n; i++)
        {
            gifPal.rgb[i][0] = (palette[i] >> 16) & 0xFF;
            gifPal.rgb[i][1] = (palette[i] >> 8) & 0xFF;
            gifPal.rgb[i][2] = palette[i] & 0xFF;
        }
        idxSuperposition = n;
        idxContradiction = n + 1;
        idxBackground = n + 2;

        gifPal.rgb[idxSuperposition][0] = 200;
        gifPal.rgb[idxSuperposition][1] = 200;
        gifPal.rgb[idxSuperposition][2] = 200;

        gifPal.rgb[idxContradiction][0] = 255;
        gifPal.rgb[idxContradiction][1] = 0;
        gifPal.rgb[idxContradiction][2] = 0;

        gifPal.rgb[idxBackground][0] = 30;
        gifPal.rgb[idxBackground][1] = 30;
        gifPal.rgb[idxBackground][2] = 30;
    }

    /**
     * @brief Render current WFC state to palette indices.
     *
     * For each cell:
     * - Contradiction (no valid tiles) → idxContradiction
     * - Collapsed (one tile) → first color of that tile
     * - Superposition (multiple tiles) → idxSuperposition
     *
     * @param st Current WFC state.
     * @return Unscaled pixel buffer [rows * cols].
     */
    std::vector<uint8_t> renderState(const WFCState &st) const
    {
        std::vector<uint8_t> px(rows * cols);
        uint64_t tmp[64];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int idx = r * cols + c;
                st.load(idx, tmp);
                int cnt = BigMask::popcount(tmp);
                if (cnt == 0)
                {
                    px[idx] = (uint8_t)idxContradiction;
                }
                else if (cnt == 1)
                {
                    int t = BigMask::ctz(tmp);
                    if (t >= 0 && t < T)
                    {
                        int colorIdx = tiles[t].data[0][0];
                        if (colorIdx >= 0 && colorIdx < idxSuperposition)
                        {
                            px[idx] = (uint8_t)colorIdx;
                        }
                        else
                        {
                            px[idx] = (uint8_t)idxBackground;
                        }
                    }
                    else
                    {
                        px[idx] = (uint8_t)idxBackground;
                    }
                }
                else
                {
                    px[idx] = (uint8_t)idxSuperposition;
                }
            }
        }
        return px;
    }

    /**
     * @brief Scale pixel buffer by replicating each cell.
     * @param src Unscaled source buffer.
     * @return Scaled buffer with dimensions [rows*scale][cols*scale].
     */
    std::vector<uint8_t> scaleUp(const std::vector<uint8_t> &src) const
    {
        int W = cols * scale, H = rows * scale;
        std::vector<uint8_t> dst(W * H);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                uint8_t v = src[r * cols + c];
                for (int py = 0; py < scale; py++)
                    for (int px = 0; px < scale; px++)
                        dst[(r * scale + py) * W + (c * scale + px)] = v;
            }
        return dst;
    }

    /**
     * @brief Capture current WFC state if capture period has elapsed.
     * @param st Current WFC state.
     */
    void capture(const WFCState &st)
    {
        collapsesSinceCapture++;
        if (collapsesSinceCapture < captureEvery)
            return;
        collapsesSinceCapture = 0;

        FrameData f;
        f.pixels = scaleUp(renderState(st));
        frames.push_back(std::move(f));
    }

    /**
     * @brief Encode and save all captured frames as GIF.
     *
     * Uses OpenMP to parallelize LZW compression across frames. The final frame
     * is displayed for 2 seconds while others display for 0.1 seconds each.
     *
     * @param path Output GIF file path.
     * @return True on success.
     */
    bool save(const std::string &path)
    {
        if (frames.empty())
        {
            std::cerr << "[GIF] Aucune frame à écrire.\n";
            return false;
        }
        int W = cols * scale, H = rows * scale;
        int nf = (int)frames.size();
        int minCode = std::max(2, gifPal.pow2 + 1);

        std::vector<std::vector<uint8_t>> compressed(nf);

        auto t_lzw0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < nf; i++)
        {
            gif::lzwCompress(frames[i].pixels.data(),
                             (int)frames[i].pixels.size(),
                             minCode,
                             compressed[i]);
        }

        auto t_lzw1 = std::chrono::high_resolution_clock::now();
        double lzwMs = std::chrono::duration<double, std::milli>(t_lzw1 - t_lzw0).count();
        std::cout << "[GIF] LZW compression (" << nf << " frames, "
                  << omp_get_max_threads() << " threads): "
                  << std::fixed << std::setprecision(1) << lzwMs << " ms\n";

        gif::Writer writer;
        if (!writer.open(path, W, H, gifPal))
        {
            std::cerr << "[GIF] Impossible d'ouvrir " << path << "\n";
            return false;
        }
        for (int i = 0; i < nf; i++)
        {
            int delay = (i == nf - 1) ? 200 : 10; // dernière frame : 2 s
            writer.addPrecompressed(compressed[i], minCode, delay);
        }
        writer.close();

        std::cout << "[GIF] Saved " << path
                  << " (" << nf << " frames, "
                  << W << "x" << H << ")\n";
        return true;
    }
};

/**
 * @brief Entropy-driven OpenMP WFC solver with GIF recording support.
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

    /** Random number generator and noise distribution. */
    std::mt19937 rng;
    std::uniform_real_distribution<double> noiseDist{0.0, 1e-6};

    /** Current WFC state. */
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
    /** History stack for backtracking. */
    std::stack<Snapshot> history;
    /** Maximum allowed backtracking attempts. */
    int maxBacktracks;

    /** Optional GIF recorder for visualization. */
    GifRecorder *gifRec = nullptr;

    /** Entropy-based cell selection priority queue. */
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
     * @brief Attach a GIF recorder for step-by-step visualization.
     * @param rec GIF recorder instance (can be nullptr).
     */
    void setGifRecorder(GifRecorder *rec) { gifRec = rec; }

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
     * @brief Collapse one cell to a weighted random tile and capture frame if enabled.
     * @param idx Flat cell index.
     * @return Chosen tile index.
     */
    int collapse(int idx)
    {
        uint64_t tmp[64];
        state.load(idx, tmp);
        std::vector<int> choices, weights;
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

        if (gifRec)
            gifRec->capture(state);

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
                                }
                            }
                        }
                    }

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
     * @brief Execute the full WFC loop with propagation, backtracking, and GIF capture.
     * @return True on success, false on unrecoverable contradiction.
     */
    bool run()
    {
        std::cout << "=== WFC " << rows << "x" << cols << " (T=" << T << ") ===\n";
        rebuildHeap();

        if (gifRec)
            gifRec->capture(state);

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
                if (gifRec)
                {
                    gifRec->collapsesSinceCapture = gifRec->captureEvery; // force capture
                    gifRec->capture(state);
                }
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
 * - argv[6]: maximum GIF frames (optional, default 120)
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
    bool useSymmetries = true;
    int maxGifFrames = 120;
    if (argc >= 7)
    {
        std::string arg6 = argv[6];
        bool looksLikeSymmetryFlag = (arg6 == "0" || arg6 == "1" || arg6 == "false" ||
                                      arg6 == "False" || arg6 == "true" || arg6 == "True" ||
                                      arg6 == "--no-symmetry" || arg6 == "--symmetry");
        if (looksLikeSymmetryFlag)
        {
            useSymmetries = (arg6 != "0" && arg6 != "false" && arg6 != "False" &&
                             arg6 != "--no-symmetry");
            if (argc >= 8)
            {
                maxGifFrames = std::stoi(argv[7]);
            }
        }
        else
        {
            maxGifFrames = std::stoi(argv[6]);
            if (argc >= 8)
            {
                std::string flag = argv[7];
                useSymmetries = (flag != "0" && flag != "false" && flag != "False" &&
                                 flag != "--no-symmetry");
            }
        }
    }

    std::cout << "[OUTPUT SIZE] " << outCols << "x" << outRows
              << "  N=" << N << "  scale=" << scale
              << "  symmetries=" << (useSymmetries ? "on" : "off")
              << "  maxGifFrames=" << maxGifFrames << "\n";

    auto t_extract = Clock::now();
    std::vector<Tile> tiles;
    std::vector<int> freq;
    extractTiles(sample, N, tiles, freq, useSymmetries);

    auto t_compat = Clock::now();
    CompatTable compat;
    compat.build(tiles);

    // Le scale GIF est limité à 16 pour garder un fichier raisonnable.
    int gifScale = std::min(scale, 16);
    GifRecorder gifRec(outRows, outCols, (int)tiles.size(), gifScale,
                       maxGifFrames, palette, tiles, outRows * outCols);

    auto t_wfc = Clock::now();
    unsigned int seed = (unsigned int)
                            std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count();
    WFC wfc(outRows, outCols, N, tiles, freq, compat, seed, /*maxBacktracks=*/200);
    wfc.setGifRecorder(&gifRec);
    bool success = wfc.run();
    auto t_end = Clock::now();

    if (success)
    {
        auto output = wfc.buildOutput();
        wfc.verify(output);
        wfc.saveImage("output.png", output, palette, scale);
        gifRec.save("output.gif");

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
    std::cout << "[TIMING] (wfc_openmp_gif.cpp)\n";
    std::cout << "  Extract tiles:         " << std::fixed << std::setprecision(2)
              << elapsed_ms(t_start, t_extract) << " ms\n";
    std::cout << "  Build compatibility:   " << elapsed_ms(t_extract, t_compat) << " ms\n";
    std::cout << "  WFC + GIF capture:     " << elapsed_ms(t_compat, t_wfc) << " ms\n";
    std::cout << "  TOTAL:                 " << elapsed_ms(t_start, t_end) << " ms\n";
    std::cout << std::string(50, '=') << "\n";

    return 0;
}