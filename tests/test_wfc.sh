#!/bin/bash
# Regression tests for the WFC variants (points 1,2,5,6,7,9,10,11).
# Usage: ./test_wfc.sh  (run from repo root; skips kokkos without a Kokkos install)
# Fails (exit 1) on the first broken expectation.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0

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
expect_exit 1 "seq N=0" "$ROOT/sequential/wfc" ../samples/wall.png 12 12 0 4 1
expect_exit 1 "seq scale=0" "$ROOT/sequential/wfc" ../samples/wall.png 12 12 3 0 1
expect_exit 1 "seq 0x0" "$ROOT/sequential/wfc" ../samples/wall.png 0 0 3 4 1
expect_exit 1 "omp scale<0" "$ROOT/openmp/wfc_openmp" ../samples/wall.png 12 12 3 -2 1
expect_exit 1 "gif maxFrames=0" "$ROOT/openmp_gif/wfc_openmp_gif" ../samples/wall.png 12 12 3 4 0 1
expect_exit 1 "seq maxbt=0" "$ROOT/sequential/wfc" ../samples/wall.png 12 12 3 4 1 42 0
expect_exit 1 "omp maxbt<0" "$ROOT/openmp/wfc_openmp" ../samples/wall.png 12 12 3 4 1 42 -5
expect_exit 1 "gif maxbt=0" "$ROOT/openmp_gif/wfc_openmp_gif" ../samples/wall.png 12 12 3 4 3 1 42 0

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

echo "---- $PASS passed, $FAIL failed ----"
[ "$FAIL" -eq 0 ]
