#!/usr/bin/env bash
# rebuild-12fixes build + smoke test. Runs on a ROMEO compute node.
# Usage: build_test.sh [x64|arm]   (arch selects Kokkos installs)
set -u
MODE="${1:-x64}"
ROOT="/scratch_p/corentinrik/wfc-bench/rebuild-12fixes"
KI="/scratch_p/corentinrik/wfc-bench/wfc-bench-staging/kokkos-install"
case "$MODE" in
  x64) KPFX="x64cpu" ;;
  arm) KPFX="armgpu" ;;
  *) echo "usage: $0 [x64|arm]"; exit 2 ;;
esac
export OMP_NUM_THREADS=4
cd "$ROOT" || exit 2
mkdir -p logs
SRC="$ROOT/src"

# Always start from clean artifacts: x64 and ARM (and CUDA) object files
# share these directories and are mutually unusable.
for D in sequential openmp openmp_gif kokkos; do
  (cd "$SRC/$D" && make clean >/dev/null 2>&1)
done

PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); echo "PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }

build_ok=1
(cd "$SRC/sequential" && make) >logs/build-seq.log 2>&1 && ok "build sequential" || { fail "build sequential"; build_ok=0; }
(cd "$SRC/openmp" && make) >logs/build-omp.log 2>&1 && ok "build openmp" || { fail "build openmp"; build_ok=0; }
(cd "$SRC/openmp_gif" && make) >logs/build-gif.log 2>&1 && ok "build openmp_gif" || { fail "build openmp_gif"; build_ok=0; }

KOKKOS_BACKENDS="openmp cpu"
[ "$MODE" = "arm" ] && KOKKOS_BACKENDS="openmp cuda"
for BE in $KOKKOS_BACKENDS; do
  KP="$KI/${KPFX}-${BE}"
  if [ ! -d "$KP" ]; then fail "kokkos install missing: $KP"; continue; fi
  KPABS="$(cd "$KP" && pwd)"
  [ -d "$KPABS/bin" ] && export PATH="$KPABS/bin:$PATH"
  DEPSTRING=""
  if [ "$BE" = "cuda" ]; then
    DEPSTRING="DEPFLAGS="
    # nvcc_wrapper needs a real nvcc in PATH; the cuda module load is a
    # silent no-op on ROMEO, so use the direct toolkit path (bench G-cuda).
    export PATH="/apps/2025/manual_install/cuda_eviden/12.6/bin:$PATH"
    # Same for the runtime loader (libcudart.so.12).
    export LD_LIBRARY_PATH="/apps/2025/manual_install/cuda_eviden/12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  # shellcheck disable=SC2086
  (cd "$SRC/kokkos" && make BACKEND="$BE" KOKKOS_PREFIX="$KPABS" $DEPSTRING) >"logs/build-kokkos-$BE.log" 2>&1 \
    && ok "build kokkos-$BE" || { fail "build kokkos-$BE"; }
done

run() { # <workdir> <binary> <logname> <args...> -> echoes exit code
  local wd="$1" bin="$2" name="$3"; shift 3
  (cd "$SRC/$wd" && rm -f output.png output.gif && ./"$bin" "$@" >"$ROOT/logs/run-$name.log" 2>&1)
  echo $?
}

check() { # <pattern> <logname> <label>
  if grep -q "$1" "logs/$2.log"; then ok "$3"; else fail "$3"; fi
}

if [ "$build_ok" = "1" ]; then
  S=../samples/wall.png
  ec=$(run sequential wfc seq "$S" 24 24 3 4 1 42); [ "$ec" = "0" ] && ok "seq valid exit=0" || fail "seq valid exit=$ec"
  check "VERIFY_OK" run-seq "seq VERIFY_OK"
  check "WFC algorithm:" run-seq "seq timing labels"
  ec=$(run openmp wfc_openmp omp "$S" 24 24 3 4 1 42); [ "$ec" = "0" ] && ok "omp valid exit=0" || fail "omp valid exit=$ec"
  check "VERIFY_OK" run-omp "omp VERIFY_OK"
  ec=$(run openmp_gif wfc_openmp_gif gif "$S" 16 16 3 4 5 1 42); [ "$ec" = "0" ] && ok "gif valid exit=0" || fail "gif valid exit=$ec"
  check "VERIFY_OK" run-gif "gif VERIFY_OK"
  [ -s "$SRC/openmp_gif/output.gif" ] && ok "gif file non-empty" || fail "gif file non-empty"

  for BE in $KOKKOS_BACKENDS; do
    BIN="build/$BE/wfc_kokkos"
    if [ -x "$SRC/kokkos/$BIN" ]; then
      ec=$(run kokkos "$BIN" kkos-"$BE" "$S" 24 24 3 4 1 42); [ "$ec" = "0" ] && ok "kokkos-$BE valid exit=0" || fail "kokkos-$BE valid exit=$ec"
      check "VERIFY_OK" run-kkos-"$BE" "kokkos-$BE VERIFY_OK"
    fi
  done

  # Tile-count alignment across backends (same sample+sym must give same count).
  TS=$(grep -oE "[0-9]+ unique" logs/run-seq.log | head -1)
  TO=$(grep -oE "[0-9]+ unique" logs/run-omp.log | head -1)
  if [ -n "$TS" ] && [ "$TS" = "$TO" ]; then ok "tiles seq==omp ($TS)"; else fail "tiles seq($TS)==omp($TO)"; fi
  if [ -x "$SRC/kokkos/build/openmp/wfc_kokkos" ]; then
    TK=$(grep -oE "[0-9]+ unique" logs/run-kkos-openmp.log | head -1)
    if [ -n "$TK" ] && [ "$TK" = "$TS" ]; then ok "tiles kokkos==seq ($TK)"; else fail "tiles kokkos($TK)==seq($TS)"; fi
  fi

  # Invalid inputs must exit 1 (no crash: 134/139).
  ec=$(run sequential wfc neg1 "$S" 12 12 0 4 1); [ "$ec" = "1" ] && ok "seq N=0 exit=1" || fail "seq N=0 exit=$ec"
  ec=$(run openmp wfc_openmp neg2 "$S" 12 12 3 0 1); [ "$ec" = "1" ] && ok "omp scale=0 exit=1" || fail "omp scale=0 exit=$ec"
  ec=$(run openmp_gif wfc_openmp_gif neg3 "$S" 12 12 3 4 0 1); [ "$ec" = "1" ] && ok "gif maxFrames=0 exit=1" || fail "gif maxFrames=0 exit=$ec"

  # Determinism: same seed twice -> identical PNG.
  (cd "$SRC/sequential" && ./wfc ../samples/checkerboard.png 12 12 2 4 1 7 >/dev/null 2>&1 && md5sum output.png | cut -d' ' -f1) > logs/md5a
  (cd "$SRC/sequential" && ./wfc ../samples/checkerboard.png 12 12 2 4 1 7 >/dev/null 2>&1 && md5sum output.png | cut -d' ' -f1) > logs/md5b
  [ "$(cat logs/md5a)" = "$(cat logs/md5b)" ] && ok "seed determinism" || fail "seed determinism"
fi

echo "==== $MODE: $PASS passed, $FAIL failed ===="
[ "$FAIL" = "0" ]
