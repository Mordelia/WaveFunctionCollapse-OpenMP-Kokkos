# WaveFunctionCollapse in C++

This repository contains several C++ implementations of the Wave Function Collapse (WFC) algorithm, developed as a university project by Mordelia and Sanstitr.

The project is based on the original WFC repository by mxgmn:

- https://github.com/mxgmn/WaveFunctionCollapse

The goal was to implement the algorithm in C++, then parallelize it using OpenMP tasks or Kokkos.

The animation below was recorded with the OpenMP GIF version on one of the bundled samples, with enough frames to make the collapse process easy to follow.

<p align="center">
	<img src="docs/examples/intro.gif" alt="Wave Function Collapse animation" width="300" />
</p>

## What is included

- `sequential/`: a standalone sequential C++ implementation.
- `openmp/`: an OpenMP task-based parallel version.
- `openmp_gif/`: an OpenMP version that also records a GIF of the collapse process.
- `kokkos/`: a Kokkos-based implementation with multiple backends.
- `samples/`: a few small input images you can use right away.

Each implementation uses the vendored `stb_image.h` and `stb_image_write.h` headers, so there is no external image library dependency.

## Visual examples

These renders were generated with the sequential implementation using the samples shipped with the repository. They show how the result changes when tile symmetries are enabled or disabled.

| Sample | Symmetries on | Symmetries off |
| --- | --- | --- |
| Forest | ![Forest with symmetries on](docs/examples/forest-sym-on.png) | ![Forest with symmetries off](docs/examples/forest-sym-off.png) |
| Water | ![Water with symmetries on](docs/examples/water-sym-on.png) | ![Water with symmetries off](docs/examples/water-sym-off.png) |

## Requirements

For the sequential and OpenMP versions, you need:

- A C++17 compiler such as `g++` or `clang++`
- `make`
- OpenMP support for the OpenMP variants

For the Kokkos version, you also need:

- A working Kokkos installation
- Either `pkg-config` support for Kokkos or a `KOKKOS_PREFIX` pointing to the install directory
- A supported backend, selected with `BACKEND`

On Debian or Ubuntu, the basic tools are usually available with:

```bash
sudo apt install build-essential make pkg-config
```

Then install Kokkos through your package manager or your own build if you want to use the `kokkos/` directory.

## Build and run

### Sequential version

```bash
cd sequential
make
./wfc
```

The default run uses a built-in 5x5 sample. To use one of the provided images:

```bash
./wfc ../samples/forest.png 128 128 3 32 1
```

Arguments:

```text
wfc [image] [outW] [outH] [N] [scale] [useSymmetries] [seed] [maxBacktracks]
```

- `image`: input image path, optional
- `outW`, `outH`: output size in cells
- `N`: tile size
- `scale`: pixel size used when writing `output.png`
- `useSymmetries`: `1` to enable tile symmetries, `0` or `--no-symmetry` to disable them
- `seed`: RNG seed (default `42`). Same seed + same inputs = same output for a given variant, whatever the thread count. `sequential`, `openmp` and `openmp_gif` share the same propagation fixed point, the same selection heuristic and the same checkpoint policy (undo journal, one checkpoint per collapse, the failing tile is excluded), so they follow the exact same collapse trajectory and produce the same image, backtracks included. This is verified up to 128x128; at 256x256 with hundreds of backtracks the OpenMP wave schedule can still change the trajectory (known limitation, under investigation). Kokkos uses a different selection heuristic (it draws a small pool of candidates), so its images differ.
- `maxBacktracks`: backtrack budget per attempt (default `200`). Each of the 3 attempts restarts with `seed`, `seed+1`, `seed+2`.
- `--verbose` anywhere: log every collapse (off by default; heavy on big grids). The flag is removed from the positional arguments before parsing, so it can be placed anywhere without shifting them.

Generated files:

- `output.png`

### OpenMP version

```bash
cd openmp
make
./wfc_openmp
```

Example:

```bash
./wfc_openmp ../samples/water.png 128 128 3 32 1
```

Arguments:

```text
wfc_openmp [image] [outW] [outH] [N] [scale] [useSymmetries] [seed] [maxBacktracks]
```

This variant uses OpenMP tasks to parallelize tile extraction and constraint propagation.

Limits and tuning:

- Up to 4 096 unique tiles (the hot path keeps masks in fixed 64-word stack buffers). Beyond that the run is refused with an error; use the sequential variant, which has no such limit.
- Propagation runs serially below 64x64 cells, where the team fork/join would dominate.
- **Set `OMP_PROC_BIND=close OMP_PLACES=cores`.** Thread migration costs a lot when the propagation hammers shared cache lines: on ROMEO (32 cores, wall 128x128) the same run takes 8.9 s at 32 threads without binding and 0.71 s with it, and 287 -> 205 ms at 8 threads.
- The propagation is a breadth-first wave of 4 to 10 cells on average, so it barely scales: one thread is often the fastest, and the parallel variants pay off only on grids big enough for the wave to be wider than the team. For a many-core node, run several independent generations instead of one.

Generated files:

- `output.png`

### OpenMP GIF version

```bash
cd openmp_gif
make
./wfc_openmp_gif
```

Example:

```bash
./wfc_openmp_gif ../samples/wall.png 96 96 3 16 120 1
```

Arguments:

```text
wfc_openmp_gif [image] [outW] [outH] [N] [scale] [maxGifFrames] [useSymmetries] [seed] [maxBacktracks]
```

Notes:

- `maxGifFrames` controls how many frames are captured for the GIF (must be > 0).
- `useSymmetries` works the same way as in the other variants.
- Argument order is fixed: unlike older builds, `0`/`1` in 6th position is always `maxGifFrames`, never a symmetry flag.
- The program writes both `output.png` and `output.gif`.

Generated files:

- `output.png`
- `output.gif`

### Kokkos version

The Kokkos project supports several backends through the `BACKEND` variable:

- `openmp` default
- `cpu`
- `cuda`
- `hip`

Build with the default backend:

```bash
cd kokkos
make
./build/openmp/wfc_kokkos
```

Build another backend:

```bash
make BACKEND=cpu
make BACKEND=cuda
make BACKEND=hip
make BACKEND=openmp
```

Example run:

```bash
./build/openmp/wfc_kokkos ../samples/forest.png 128 128 3 32 1
```

Arguments:

```text
wfc_kokkos [image] [outW] [outH] [N] [scale] [useSymmetries] [seed] [maxBacktracks]
```

If Kokkos is not found automatically, you can point the build to your installation with:

```bash
make KOKKOS_PREFIX=/path/to/kokkos
```

or rely on `pkg-config` if your installation exposes `kokkos.pc`.

Generated files:

- `build/<backend>/wfc_kokkos`
- `output.png`

## Cleaning up

Each subproject has its own clean target:

```bash
make clean
make clean-output
```

For Kokkos:

```bash
make clean
```

The repository-level `.gitignore` already excludes the generated images, build directories, object files, dependency files, and local binaries.

## Project notes

- The sequential implementation is the reference version.
- The OpenMP implementations parallelize the algorithm with tasks.
- The Kokkos version is structured so it can be built for several execution backends.
- All variants preserve the same core idea: extract overlapping tiles from a sample image, compute compatibility constraints, and collapse a target grid until it forms a new image.
- All variants share the same overlap model and the same fixed point. Propagation only needs the **four cardinal directions**: for overlapping tiles the constraint at a diagonal offset is implied by the chain of unit steps (if `a` matches `b` on the right and `b` matches `c` below, `a` and `c` agree on their diagonal overlap). That is `4/(2N-1)^2` of the work of the old full-window `(2N-1)^2` version; outputs were verified bit-identical on every sample.
- Cell selection uses a lazy min-heap of `(entropy + per-cell tie-break noise)`; a popped entry whose key is stale is re-pushed with its current entropy, so the chosen cell really is the minimum.
- Each run tries up to 3 trajectories (seeds `seed`, `seed+1`, `seed+2`); a failed collapse restarts instead of giving up on one bad draw.
- Backtracking snapshots the pre-collapse domain and excludes the failed tile, so every backtrack resumes a live alternate branch (this was broken in the parallel variants: post-collapse snapshots made backtrack a no-op).
- `[TIMING]` phases are `extractTiles`, `compat.build`, and `run()` (pure solver); `TOTAL` covers everything. Only `TOTAL` is comparable with runs from before this fix.
- OpenMP `propagate()` runs serially below 64x64 cells (team fork/join would dominate) and in parallel above; small grids no longer slow down with more threads.
- Memory: `sequential`, `openmp` and `openmp_gif` keep a bounded undo journal (the last `maxBacktracks+1` checkpoints, a few MB); Kokkos still snapshots the whole state every 16 collapses (about 1 GB at 128x128). Give 256x256 Kokkos runs a few GB.
- Exit codes: `0` = output verified and written; `1` = invalid arguments, contradiction (after 3 attempts), failed verify, or failed write.

## Tests

```bash
./tests/test_wfc.sh
```

Builds the three local variants, checks a valid run per variant (`VERIFY_OK`), invalid inputs (`N=0`, `scale<=0`, `0x0`, `maxFrames=0` all exit 1), bad command-line numbers (exit 1 instead of a crash), `--verbose` placement, seed reproducibility, and that sequential and OpenMP produce the same image when no backtrack occurred. Kokkos is skipped (needs a Kokkos install).

## Credits

- Original Wave Function Collapse concept and repository: mxgmn
- Coursework implementation and parallelization: Mordelia and Sanstitr
