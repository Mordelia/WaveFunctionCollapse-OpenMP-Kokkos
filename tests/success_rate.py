#!/usr/bin/env python3
"""Success rates with Wilson 95% CI over result.txt data (point 4: seeds as a dimension).

Usage:
  success_rate.py all_results.txt            # main phases (25-col format)
  success_rate.py --f F_results.txt          # negative battery (5-col format)
Reads files given as args, or runs/*/*/result.txt under cwd when no args.
"""
import glob
import math
import sys
from collections import defaultdict

MAIN_COLS = "phase|code|sample|W|H|N|scale|maxfr|sym|threads|bind|rep|seed|arch|status|exit|verify|backtracks_n|t_extract_ms|t_compat_ms|t_wfc_ms|t_total_ms|wall_s|gif_ok|gif_bytes".split("|")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def load_main(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("phase|"):
                    continue
                rows.append(dict(zip(MAIN_COLS, line.split("|"))))
    return rows


def main(argv):
    args = list(argv)
    only_f = False
    if args and args[0] == "--f":
        only_f = True
        args = args[1:]
    paths = args
    if not paths:
        paths = glob.glob("runs/*/*/result.txt") + ["all_results.txt", "F_results.txt"]
        import os
        paths = [p for p in paths if os.path.exists(p)]
    f_paths = [p for p in paths if "F_results" in p or "runs/F" in p]
    m_paths = [p for p in paths if p not in f_paths]

    if only_f:
        m_paths = []
    if m_paths:
        rows = load_main(m_paths)
        groups = defaultdict(list)
        for r in rows:
            groups[(r["phase"], r["code"], r["sample"],
                    f"{r['W']}x{r['H']}", f"N{r['N']}", "sym" + r["sym"],
                    r["arch"])].append(r["status"])
        print(f"{'config':60s} {'n':>3} {'ok':>3} {'rate':>6} 95% CI")
        for k in sorted(groups):
            st = groups[k]
            n, ok = len(st), sum(1 for s in st if s == "OK")
            lo, hi = wilson(ok, n)
            print(f"{'/'.join(k):60s} {n:>3} {ok:>3} {ok/n:6.2f} [{lo:.2f},{hi:.2f}]")
    for p in f_paths:
        tot = neg = 0
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("phase|"):
                    continue
                tot += 1
                if "PASS_NEG" in line:
                    neg += 1
        lo, hi = wilson(neg, tot)
        print(f"{p}: PASS_NEG {neg}/{tot} = {neg/tot:.2f} [{lo:.2f},{hi:.2f}]" if tot else f"{p}: empty")


if __name__ == "__main__":
    main(sys.argv[1:])
