#!/usr/bin/env bash
# prebuild.sh [x64|arm] — build every variant ONCE per arch into prebuilt/.
# Task runners (run_case3.sh) reuse these read-only binaries instead of
# rebuilding per task (~30 s x 400+ tasks saved).
set -u
MODE="${1:-x64}"
ROOT="/scratch_p/corentinrik/wfc-bench/rebuild-12fixes"
SRC="$ROOT/src"
PB="$ROOT/prebuilt/$MODE" # per-arch: x64 and ARM binaries must not mix
KI="/scratch_p/corentinrik/wfc-bench/wfc-bench-staging/kokkos-install"
case "$MODE" in
  x64) KPFX="x64cpu" ;;
  arm) KPFX="armgpu" ;;
  *) echo "usage: $0 [x64|arm]"; exit 2 ;;
esac
export OMP_NUM_THREADS=4
mkdir -p "$PB"
PASS=0
FAIL=0

# Un binaire est installe par renommage atomique : les taches d'une campagne
# en cours gardent l'inode de l'ancien binaire au lieu de lire un fichier
# tronque par cp (SIGBUS observe sur le lot X).
# Note : x64 et arm construisent dans le meme $SRC, et `make clean` y efface
# build/ : ne jamais lancer deux prebuild en parallele.
install_bin() { # <binaire source> <repertoire de destination> <nom>
  mkdir -p "$2" && cp "$1" "$2/.tmp-$$" && mv -f "$2/.tmp-$$" "$2/$3"
}

build_plain() { # <srcdir> <binary> <outdir>
  (cd "$SRC/$1" && make clean >/dev/null 2>&1; make) >"$PB/build-$1.log" 2>&1 \
    && install_bin "$SRC/$1/$2" "$PB/$3" "$2" \
    && echo "PASS: prebuilt $1" && PASS=$((PASS + 1)) \
    || { echo "FAIL: prebuilt $1"; FAIL=$((FAIL + 1)); }
}

build_plain sequential wfc sequential
build_plain openmp wfc_openmp openmp
build_plain openmp_gif wfc_openmp_gif openmp_gif

KOKKOS_BACKENDS="openmp cpu"
[ "$MODE" = "arm" ] && KOKKOS_BACKENDS="openmp cuda"
for BE in $KOKKOS_BACKENDS; do
  KP="$KI/${KPFX}-${BE}"
  KPABS="$(cd "$KP" && pwd)"
  [ -d "$KPABS/bin" ] && export PATH="$KPABS/bin:$PATH"
  DEPSTRING=""
  if [ "$BE" = "cuda" ]; then
    DEPSTRING="DEPFLAGS="
    export PATH="/apps/2025/manual_install/cuda_eviden/12.6/bin:$PATH"
    export LD_LIBRARY_PATH="/apps/2025/manual_install/cuda_eviden/12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  # shellcheck disable=SC2086
  (cd "$SRC/kokkos" && make clean >/dev/null 2>&1; make BACKEND="$BE" KOKKOS_PREFIX="$KPABS" $DEPSTRING) >"$PB/build-kokkos-$BE.log" 2>&1 \
    && install_bin "$SRC/kokkos/build/$BE/wfc_kokkos" "$PB/kokkos-$BE" wfc_kokkos \
    && echo "PASS: prebuilt kokkos-$BE" && PASS=$((PASS + 1)) \
    || { echo "FAIL: prebuilt kokkos-$BE"; FAIL=$((FAIL + 1)); }
done

echo "==== prebuild $MODE: $PASS passed, $FAIL failed ===="
[ "$FAIL" = "0" ]
