#!/usr/bin/env bash
# run_case2.sh SPEC TID AID [TIMEOUT_S]
# Rebench runner for the 12-fix codebase: fixed CLI order, explicit SEED,
# new log tokens (VERIFY_OK/VERIFY_FAIL), true timing phases.
# Writes result.txt + DONE. Never fails the SLURM task itself (exit 0).
set -u
SPEC="${1:?spec required}"
TID="${2:-0}"
AID="${3:-0}"
TIMEOUT_S="${4:-3300}"

PHASE="$(cut -d'|' -f1 <<<"$SPEC")"
CODE="$(cut -d'|' -f2 <<<"$SPEC")"
SAMPLE="$(cut -d'|' -f3 <<<"$SPEC")"
W="$(cut -d'|' -f4 <<<"$SPEC")"
H="$(cut -d'|' -f5 <<<"$SPEC")"
N="$(cut -d'|' -f6 <<<"$SPEC")"
SCALE="$(cut -d'|' -f7 <<<"$SPEC")"
MAXFR="$(cut -d'|' -f8 <<<"$SPEC")"
SYM="$(cut -d'|' -f9 <<<"$SPEC")"
THREADS="$(cut -d'|' -f10 <<<"$SPEC")"
BIND="$(cut -d'|' -f11 <<<"$SPEC")"
REP="$(cut -d'|' -f12 <<<"$SPEC")"
KEEP="$(cut -d'|' -f13 <<<"$SPEC")"
SEED="$(cut -d'|' -f14 <<<"$SPEC")"
[ -z "$SEED" ] && SEED=42

SAN="$(tr '|/' '__' <<<"$SPEC")"
D="runs/$PHASE/${AID}_${TID}_${SAN}"
mkdir -p "$D"
cd "$D" || exit 0
# Drop any objects copied by an earlier task reusing this sandbox
# (e.g. another arch): x64 and ARM binaries share these directories.
rm -rf sequential openmp openmp_gif kokkos samples output.png output.gif

ARCH="$(uname -m)"
STATUS="CRASH"
EXITC=99
VERIFY=0
BT=0
SUCC=0
TEXTRACT=""
TCOMPAT=""
TWFC=""
TTOTAL=""
GIF_OK=""
GIF_BYTES=""
WALL_S=0
T0=$SECONDS

export OMP_NUM_THREADS="$THREADS"
if [ "$BIND" != "none" ]; then
  export OMP_PROC_BIND="$BIND"
  export OMP_PLACES=cores
fi

BIN=""
ARGS=""
# Prebuilt binaries (one build per arch, shared read-only). Fall back to a
# local build when prebuilt/ is missing (e.g. first run before prebuild.sh).
case "$(uname -m)" in
  x86_64) KPFXARCH="x64" ;;
  *)      KPFXARCH="arm" ;;
esac
PB="../../../prebuilt/$KPFXARCH" # per-arch prebuilt (x64 vs ARM must not mix)
use_prebuilt() { # <name> <relbin> -> sets BIN on success
  if [ -x "$PB/$1/$2" ]; then BIN="$PB/$1/$2"; return 0; fi
  return 1
}
case "$CODE" in
  seq) use_prebuilt sequential wfc || { cp -r ../../../src/sequential . && (cd sequential && make >../build-seq.log 2>&1) && BIN="sequential/wfc"; } && ARGS="samples/$SAMPLE.png $W $H $N $SCALE $SYM $SEED" ;;
  omp) use_prebuilt openmp wfc_openmp || { cp -r ../../../src/openmp . && (cd openmp && make >../build-omp.log 2>&1) && BIN="openmp/wfc_openmp"; } && ARGS="samples/$SAMPLE.png $W $H $N $SCALE $SYM $SEED" ;;
  gif) use_prebuilt openmp_gif wfc_openmp_gif || { cp -r ../../../src/openmp_gif . && (cd openmp_gif && make >../build-gif.log 2>&1) && BIN="openmp_gif/wfc_openmp_gif"; } && ARGS="samples/$SAMPLE.png $W $H $N $SCALE $MAXFR $SYM $SEED" ;;
  kokkos-*)
    BE="${CODE#kokkos-}"
    case "$ARCH" in
      x86_64) KP="/scratch_p/corentinrik/wfc-bench/wfc-bench-staging/kokkos-install/x64cpu-$BE" ;;
      *)      KP="/scratch_p/corentinrik/wfc-bench/wfc-bench-staging/kokkos-install/armgpu-$BE" ;;
    esac
    if [ -d "$KP" ]; then
      KPABS="$(cd "$KP" && pwd)" # resolve BEFORE cd kokkos (relative!)
      [ -d "$KPABS/bin" ] && export PATH="$KPABS/bin:$PATH" # nvcc_wrapper
      if [ "$BE" = "cuda" ]; then
        # Needed at BUILD time (nvcc) and at RUN time (libcudart), even
        # when using the prebuilt binary.
        export PATH="/apps/2025/manual_install/cuda_eviden/12.6/bin:$PATH"
        export LD_LIBRARY_PATH="/apps/2025/manual_install/cuda_eviden/12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        DEPSTRING="DEPFLAGS="
      else
        DEPSTRING=""
      fi
      if ! use_prebuilt "kokkos-$BE" wfc_kokkos; then
        cp -r ../../../src/kokkos . && cp -r ../../../src/samples .
        # shellcheck disable=SC2086
        (cd kokkos && make BACKEND="$BE" KOKKOS_PREFIX="$KPABS" $DEPSTRING >../build-kokkos.log 2>&1) \
          && BIN="kokkos/build/$BE/wfc_kokkos"
      fi
      [ -n "$BIN" ] && ARGS="samples/$SAMPLE.png $W $H $N $SCALE $SYM $SEED"
    else
      STATUS="SKIP"
    fi
    ;;
  *) STATUS="SKIP" ;;
esac
cp -r ../../../src/samples . 2>/dev/null
case "$BIN" in
  /*) ;; # prebuilt absolute path runs as-is
  *) BIN="./$BIN" ;; # sandbox-relative fallback build
esac

if [ "$STATUS" != "SKIP" ]; then
  if [ -n "$BIN" ] && [ -x "$BIN" ]; then
    # shellcheck disable=SC2086
    timeout "$TIMEOUT_S" $BIN $ARGS >run.log 2>&1
    EXITC=$?
    if [ "$EXITC" -eq 124 ] || [ "$EXITC" -eq 137 ]; then
      STATUS="TIMEOUT"
    elif grep -q "WFC failed" run.log; then
      STATUS="WFC_FAIL"
    elif [ "$EXITC" -ne 0 ]; then
      STATUS="CRASH"
    else
      SUCC="$(grep -c "WFC succeeded" run.log || true)"
      if [ "$SUCC" -ge 1 ]; then STATUS="OK"; else STATUS="WFC_FAIL"; fi
    fi
    VERIFY=0
    if grep -q "VERIFY_OK" run.log && [ "$(grep -c "VERIFY_FAIL" run.log || true)" -eq 0 ]; then VERIFY=1; fi
    BT="$(grep -c "^\[BACKTRACKS\]" run.log || true)"
    BTN="$(grep "^\[BACKTRACKS\]" run.log | head -1 | grep -oE "[0-9]+" | head -1)"
    [ -z "$BTN" ] && BTN=0
    # Timing labels are TRUE phases since the 12-fix rework.
    TEXTRACT="$(awk '/Extract tiles:/{v=$(NF-1)} END{print v}' run.log)"
    TCOMPAT="$(awk '/Build compatibility:/{v=$(NF-1)} END{print v}' run.log)"
    TWFC="$(awk '/WFC algorithm:|WFC \+ GIF capture:/{v=$(NF-1)} END{print v}' run.log)"
    TTOTAL="$(awk '/TOTAL:/{v=$(NF-1)} END{print v}' run.log)"
    SEEDLOG="$(grep -oE "seed=[0-9]+" run.log | head -1 | cut -d= -f2)"
    if [ "$CODE" = "gif" ] && [ -f output.gif ]; then
      GIF_BYTES="$(wc -c <output.gif | tr -d ' ')"
      [ "$GIF_BYTES" -gt 0 ] && GIF_OK=1 || GIF_OK=0
    fi
    [ "$KEEP" != "1" ] && rm -f output.gif output.png
  else
    STATUS="BUILD_FAIL"
    EXITC=98
    echo "build failed for $CODE" >run.log
  fi
fi

WALL_S=$((SECONDS - T0))
tail -n 50 run.log >run.log.tail 2>/dev/null && mv run.log.tail run.log 2>/dev/null
{
  echo "phase|code|sample|W|H|N|scale|maxfr|sym|threads|bind|rep|seed|arch|status|exit|verify|backtracks_n|t_extract_ms|t_compat_ms|t_wfc_ms|t_total_ms|wall_s|gif_ok|gif_bytes"
  echo "$PHASE|$CODE|$SAMPLE|$W|$H|$N|$SCALE|$MAXFR|$SYM|$THREADS|$BIND|$REP|$SEED|$ARCH|$STATUS|$EXITC|$VERIFY|$BTN|$TEXTRACT|$TCOMPAT|$TWFC|$TTOTAL|$WALL_S|$GIF_OK|$GIF_BYTES"
} >result.txt
touch DONE
exit 0
