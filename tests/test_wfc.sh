#!/bin/bash
# Regression tests for the WFC variants (points 1,2,5,6,7,9,10,11).
# Usage: ./test_wfc.sh  (run from repo root; skips kokkos without a Kokkos install)
# Fails (exit 1) on the first broken expectation.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
T0=$SECONDS

ok()   { PASS=$((PASS + 1)); echo "PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }

expect_exit() { # <want> <label> <cmd...>
  local want="$1"; shift
  local label="$1"; shift
  "$@" >/tmp/wfc_test.log 2>&1
  local got=$?
  if [ "$got" -eq "$want" ]; then ok "$label (exit=$got)"; else fail "$label (want $want, got $got)"; fi
}

expect_grep() { # <pattern> <label>
  if grep -q "$1" /tmp/wfc_test.log; then ok "$2"; else fail "$2 (missing '$1')"; fi
}

build() { # <dir> <target>
  (cd "$ROOT/$1" && make "$2" >/tmp/wfc_build.log 2>&1) || { fail "build $1/$2"; return 1; }
  ok "build $1/$2"
}

run_case() { # <dir> <binary> <args...>
  local dir="$1"; shift
  (cd "$ROOT/$dir" && rm -f output.png output.gif && "./$1" "${@:2}" >/tmp/wfc_test.log 2>&1)
}

for v in "sequential wfc" "openmp wfc_openmp" "openmp_gif wfc_openmp_gif"; do
  set -- $v
  build "$1" "$2" || exit 1
done

# 1. Valid small run succeeds, verifies, reports solver timing (seq).
run_case sequential wfc ../samples/checkerboard.png 12 12 2 4 1 42
[ $? -eq 0 ] && ok "seq valid run exit=0" || fail "seq valid run exit=0"
expect_grep "VERIFY_OK" "seq verify token"
expect_grep "WFC algorithm:" "seq timing labels"
expect_grep "BACKTRACKS" "seq backtrack report"

# 1b. Same for openmp + gif (gif uses fixed order: img W H N scale maxFrames sym seed).
run_case openmp wfc_openmp ../samples/checkerboard.png 12 12 2 4 1 42
[ $? -eq 0 ] && ok "omp valid run exit=0" || fail "omp valid run exit=0"
expect_grep "VERIFY_OK" "omp verify token"
run_case openmp_gif wfc_openmp_gif ../samples/checkerboard.png 12 12 2 4 3 1 42
[ $? -eq 0 ] && ok "gif valid run exit=0" || fail "gif valid run exit=0"
expect_grep "VERIFY_OK" "gif verify token"
[ -s "$ROOT/openmp_gif/output.gif" ] && ok "gif file written" || fail "gif file written"

# 2. Invalid inputs exit 1, no crash (128+SIG = crash).
expect_exit 1 "seq N=0" "$ROOT/sequential/wfc" "$ROOT/samples/wall.png" 12 12 0 4 1
expect_exit 1 "seq scale=0" "$ROOT/sequential/wfc" "$ROOT/samples/wall.png" 12 12 3 0 1
expect_exit 1 "seq 0x0" "$ROOT/sequential/wfc" "$ROOT/samples/wall.png" 0 0 3 4 1
expect_exit 1 "omp scale<0" "$ROOT/openmp/wfc_openmp" "$ROOT/samples/wall.png" 12 12 3 -2 1
expect_exit 1 "gif maxFrames=0" "$ROOT/openmp_gif/wfc_openmp_gif" "$ROOT/samples/wall.png" 12 12 3 4 0 1
expect_exit 1 "seq maxbt=0" "$ROOT/sequential/wfc" "$ROOT/samples/wall.png" 12 12 3 4 1 42 0
expect_exit 1 "omp maxbt<0" "$ROOT/openmp/wfc_openmp" "$ROOT/samples/wall.png" 12 12 3 4 1 42 -5
expect_exit 1 "gif maxbt=0" "$ROOT/openmp_gif/wfc_openmp_gif" "$ROOT/samples/wall.png" 12 12 3 4 3 1 42 0

# 2b. Command line: --verbose must not shift the positional arguments, and a
# non-numeric argument must exit 1 (it used to abort with SIGABRT).
expect_exit 0 "seq --verbose first" "$ROOT/sequential/wfc" --verbose "$ROOT/samples/checkerboard.png" 12 12 2 4 1 42
expect_exit 0 "omp --verbose middle" "$ROOT/openmp/wfc_openmp" "$ROOT/samples/checkerboard.png" 12 12 2 4 1 42 --verbose
expect_exit 0 "gif --verbose first" "$ROOT/openmp_gif/wfc_openmp_gif" --verbose "$ROOT/samples/checkerboard.png" 12 12 2 4 3 1 42
expect_exit 1 "seq non-numeric arg" "$ROOT/sequential/wfc" "$ROOT/samples/wall.png" abc 12
expect_exit 1 "omp non-numeric arg" "$ROOT/openmp/wfc_openmp" "$ROOT/samples/wall.png" 12 12 3 4 1 42 xyz

# 3. Seed reproducibility: same seed -> identical PNG.
(cd "$ROOT/sequential" && ./wfc ../samples/checkerboard.png 12 12 2 4 1 7 >/dev/null 2>&1 && md5sum output.png | cut -d' ' -f1) > /tmp/wfc_s1
(cd "$ROOT/sequential" && ./wfc ../samples/checkerboard.png 12 12 2 4 1 7 >/dev/null 2>&1 && md5sum output.png | cut -d' ' -f1) > /tmp/wfc_s2
if [ "$(cat /tmp/wfc_s1)" = "$(cat /tmp/wfc_s2)" ]; then ok "seed reproducibility"; else fail "seed reproducibility"; fi

# 4. Restart/backtrack report line + --verbose gate.
run_case sequential wfc ../samples/wall.png 16 16 3 4 1 42 200
expect_grep "BACKTRACKS" "seq backtrack report line"
(cd "$ROOT/sequential" && ./wfc ../samples/wall.png 16 16 3 2 1 42 --verbose 2>&1 | grep -q COLLAPSE) \
  && ok "verbose prints collapses" || fail "verbose prints collapses"
(cd "$ROOT/sequential" && ./wfc ../samples/wall.png 16 16 3 2 1 42 2>&1 | grep -q COLLAPSE) \
  && fail "quiet hides collapses" || ok "quiet hides collapses"

# 5. Profiling counters are opt-in: [STATS] only with WFC_STATS=1, coherent.
(cd "$ROOT/sequential" && WFC_STATS=1 ./wfc ../samples/checkerboard.png 12 12 2 4 1 42 >/tmp/wfc_stats.log 2>&1)
prop=$(sed -n 's/^\[STATS\] code=seq .*propagate_calls=\([0-9]*\) .*/\1/p' /tmp/wfc_stats.log)
if [ -n "$prop" ] && [ "$prop" -ge 1 ]; then ok "seq WFC_STATS counters (propagate_calls=$prop)"; else fail "seq WFC_STATS counters"; fi
(cd "$ROOT/openmp" && WFC_STATS=1 ./wfc_openmp ../samples/checkerboard.png 12 12 2 4 1 42 >/tmp/wfc_stats_omp.log 2>&1)
grep -q '^\[STATS\] code=omp ' /tmp/wfc_stats_omp.log && ok "omp WFC_STATS line" || fail "omp WFC_STATS line"
run_case sequential wfc ../samples/checkerboard.png 12 12 2 4 1 42
grep -q '\[STATS\]' /tmp/wfc_test.log && fail "no STATS by default" || ok "no STATS by default"

# 6. Cross-variant: with the same seed and no backtrack, sequential and OpenMP
#    must follow the same trajectory and produce the same image.
CV_ARGS="checkerboard.png 16 16 2 4 1 42"
(cd "$ROOT/sequential" && ./wfc ../samples/$CV_ARGS >/tmp/wfc_cv_seq.log 2>&1 && md5sum output.png | cut -d' ' -f1) >/tmp/wfc_cv_s
(cd "$ROOT/openmp" && OMP_NUM_THREADS=1 ./wfc_openmp ../samples/$CV_ARGS >/tmp/wfc_cv_omp.log 2>&1 && md5sum output.png | cut -d' ' -f1) >/tmp/wfc_cv_o
if grep -q "\[BACKTRACKS\] 0 used" /tmp/wfc_cv_seq.log; then
  if [ "$(cat /tmp/wfc_cv_s)" = "$(cat /tmp/wfc_cv_o)" ]; then
    ok "seq and omp same image (no backtrack)"
  else
    fail "seq and omp same image (no backtrack)"
  fi
else
  fail "seq and omp same image (checkerboard 16x16 needed a backtrack)"
fi

# 7. Same seed, different thread count: OpenMP must be thread-count independent.
(cd "$ROOT/openmp" && OMP_NUM_THREADS=2 ./wfc_openmp ../samples/$CV_ARGS >/dev/null 2>&1 && md5sum output.png | cut -d' ' -f1) >/tmp/wfc_cv_o2
if [ "$(cat /tmp/wfc_cv_o)" = "$(cat /tmp/wfc_cv_o2)" ]; then
  ok "omp thread-count independence"
else
  fail "omp thread-count independence"
fi

echo "---- $PASS passed, $FAIL failed ----"

# Append one JSON line to the dashboard history (~/.wfc-bench/dashboard reads it).
HIST="${WFC_TESTS_HISTORY:-$HOME/.wfc-bench/tests-history.txt}"
if [ "$FAIL" -eq 0 ] || [ -n "${WFC_TESTS_RECORD:-}" ]; then
  mkdir -p "$(dirname "$HIST")"
  printf '{"date":"%s","pass":%d,"fail":%d,"dur_s":%d,"git":"%s","dirty":%d}\n' \
    "$(date -Iseconds)" "$PASS" "$FAIL" "$((SECONDS - T0))" \
    "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')" \
    "$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')" >> "$HIST"
fi

[ "$FAIL" -eq 0 ]
