"""Statistics for comparing checkpoints evaluated on the same problems.

Every checkpoint in an evaluate.py --json file answered the same problems in
the same order, so comparisons are paired: two checkpoints that differ by a
few points of accuracy can be told apart far more reliably by looking at the
problems where exactly one of them is right than by comparing two independent
proportions.

    wilson_ci(k, n)              confidence interval for an accuracy k / n
    pass_at_k(n, c, k)           unbiased pass@k from c correct out of n samples
    paired_bootstrap(a, b, ...)  mean of b - a with a bootstrap CI over problems
    mcnemar_exact(a, b)          exact two-sided McNemar test
    paired_z_test(a, b, ...)     large-sample test of mean(b - a) = 0 for
                                 non-binary pairs (seed-averaged correctness)

    python stats.py results/eval.json --baseline sft [--group regex]

prints each checkpoint's accuracy with a Wilson interval and, against the
baseline, the paired difference with a bootstrap CI and McNemar's p, per domain
and overall. Checkpoints whose labels differ only in a seed suffix (rlhf_s1,
rlhf_s2, ...) are also aggregated, provided their stage and run configs agree
apart from the seed and output path: mean and standard deviation of accuracy
over seeds, and one pooled paired comparison in which each problem's
correctness is averaged over the seeds. Correctness is Lean's verdict when
every checkpoint was compiled with Lean (evaluate.py --lean true), the
checker's otherwise.
"""

import argparse
import json
import math
import re

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96):
    """(lo, hi) Wilson score interval for k successes in n trials. Unlike the
    normal approximation it stays inside [0, 1] and is not degenerate at 0 or
    n successes, which small domains and weak checkpoints hit often."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At k = 0 or k = n one end is exactly 0 or 1; rounding must not move it.
    return (0.0 if k == 0 else max(0.0, center - half)), (1.0 if k == n else min(1.0, center + half))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k from n samples of which c are correct
    (Chen et al. 2021): 1 - C(n - c, k) / C(n, k), computed as a product so it
    stays exact enough for large n."""
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def _strata_indices(n: int, strata):
    if strata is None:
        return [np.arange(n)]
    strata = np.asarray(strata)
    if len(strata) != n:
        raise ValueError(f"{len(strata)} strata labels for {n} problems")
    return [np.flatnonzero(strata == s) for s in dict.fromkeys(strata.tolist())]


def paired_bootstrap(a, b, n_boot: int = 10_000, seed: int = 0, strata=None, level: float = 0.95):
    """(mean of b - a, lo, hi): the mean paired difference with a percentile
    bootstrap interval. Problems are resampled with replacement, keeping each
    problem's pair together; with `strata` (one label per problem, e.g. its
    domain) each stratum is resampled to its own size, as the evaluation set was
    drawn with a fixed number of problems per domain."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be non-empty 1-d sequences of the same length")
    d = b - a
    n = len(d)
    rng = np.random.default_rng(seed)
    totals = np.zeros(n_boot)
    for idx in _strata_indices(n, strata):
        values = d[idx]
        step = max(1, 2_000_000 // len(idx))  # bounds the (replicates, problems) index array
        for start in range(0, n_boot, step):
            m = min(step, n_boot - start)
            totals[start:start + m] += values[rng.integers(0, len(idx), size=(m, len(idx)))].sum(axis=1)
    lo, hi = np.quantile(totals / n, [(1 - level) / 2, (1 + level) / 2])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_exact(a, b) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes: under the
    null, each problem that exactly one of the two gets right is equally
    likely to favour either, so the count favouring b is Binomial(discordant,
    1/2). Concordant problems carry no information about the difference."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2 ** n)  # exact integers; int / int rounds correctly


def paired_z_test(a, b, strata=None) -> float:
    """Two-sided p-value for mean(b - a) = 0 from the normal approximation, with
    the variance of a stratified mean when `strata` is given. Used for pooled
    seeds, where b is a per-problem fraction (correct seeds / seeds) and
    McNemar does not apply; treating every (seed, problem) pair as independent
    would overstate the evidence, as a problem's seeds are correlated."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(d)
    var = sum(len(idx) * d[idx].var(ddof=1) for idx in _strata_indices(n, strata) if len(idx) > 1) / n ** 2
    mean = float(d.mean())
    if var == 0:
        return 1.0 if mean == 0 else 0.0
    return math.erfc(abs(mean) / math.sqrt(var) / math.sqrt(2))


# ----------------------------------------------------------------------- CLI

SEED_SUFFIX = re.compile(r"^(.*)_s(\d+)$")


def correctness(entries):
    """("lean" or "checker", {label: bool array}): Lean verdicts only if every
    checkpoint has them for every problem, so all comparisons use one judge."""
    use_lean = all(r.get("lean_ok") is not None for e in entries for r in e["records"])
    key = "lean_ok" if use_lean else "checker_ok"
    return ("lean" if use_lean else "checker"), {e["label"]: np.array([bool(r[key]) for r in e["records"]])
                                                 for e in entries}


def seed_groups(labels):
    """{base: [labels]} for bases with at least two seeds, in first-seen order."""
    groups = {}
    for label in labels:
        m = SEED_SUFFIX.match(label)
        if m:
            groups.setdefault(m.group(1), []).append(label)
    return {base: members for base, members in groups.items() if len(members) > 1}


# Run-config fields in which the seeds of one arm may differ.
PER_SEED = ("seed", "out", "device")


def check_seed_group(name: str, members, by_label) -> None:
    """Raises unless the members of a seed group have the same stage and run
    configs that differ only in PER_SEED. Labels alone decide what is pooled,
    so a mislabelled run (another reward source, fewer iterations) would
    otherwise be averaged into the arm without a word."""
    first = by_label[members[0]]
    for label in members[1:]:
        e = by_label[label]
        a, b = first.get("run_config", {}), e.get("run_config", {})
        diff = ["stage"] if e.get("stage") != first.get("stage") else []
        diff += sorted(k for k in a.keys() | b.keys() if k not in PER_SEED and a.get(k) != b.get(k))
        if diff:
            raise SystemExit(f"seed group {name!r}: {label} differs from {members[0]} in {', '.join(diff)}, "
                             "so they are not seeds of one arm; rename one of them")


def _signed(x: float) -> str:
    return f"{x:+.3f}"


def _p(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.4f}"


def report(data: dict, baseline: str, group: str = None, n_boot: int = 10_000, seed: int = 0) -> str:
    entries = data["checkpoints"]
    by_label = {e["label"]: e for e in entries}
    if baseline not in by_label:
        raise SystemExit(f"baseline {baseline!r} is not in the file; labels: {', '.join(by_label)}")
    first = [(r["domain"], r["prompt"]) for r in entries[0]["records"]]
    for e in entries[1:]:
        if [(r["domain"], r["prompt"]) for r in e["records"]] != first:
            raise SystemExit(f"{e['label']} was evaluated on different problems from {entries[0]['label']}")
    judge, ok = correctness(entries)
    doms = np.array([d for d, _ in first])
    domains = list(dict.fromkeys(doms.tolist()))
    rows = [(d, doms == d) for d in domains] + [("all", np.ones(len(doms), dtype=bool))]
    shown = [e["label"] for e in entries if group is None or re.search(group, e["label"])]
    base = ok[baseline].astype(float)

    sizes = ", ".join(f"{d} {int(m.sum())}" for d, m in rows[:-1])
    width = max(12, max(map(len, shown), default=0) + 2)
    lines = [f"correctness: {'Lean 4 (every answer compiled)' if judge == 'lean' else 'checker'}; "
             f"baseline: {baseline}; {len(doms)} problems ({sizes})",
             "", "accuracy", f"{'label':<{width}}" + "".join(f"{d:>9}" for d, _ in rows)]
    for label in shown:
        lines.append(f"{label:<{width}}" + "".join(f"{ok[label][m].mean():>9.3f}" for _, m in rows))

    vs = len(baseline) + 5      # width of the difference column
    header = f"  {'domain':<8}{'n':>6}{'acc':>8}  {'95% Wilson CI':<16}{'vs ' + baseline:>{vs}}  " \
             f"{'95% bootstrap CI':<18}{'McNemar p':>10}"
    for label in shown:
        e = by_label[label]
        lines += ["", f"== {label} (stage {e.get('stage')}) ==", header]
        for d, m in rows:
            k, n = int(ok[label][m].sum()), int(m.sum())
            lo, hi = wilson_ci(k, n)
            line = f"  {d:<8}{n:>6}{k / n:>8.3f}  [{lo:.3f}, {hi:.3f}]  "
            if label != baseline:
                strata = doms[m] if d == "all" else None
                diff, blo, bhi = paired_bootstrap(base[m], ok[label][m], n_boot, seed, strata)
                p = mcnemar_exact(ok[baseline][m], ok[label][m])
                line += f"{_signed(diff):>{vs}}  [{_signed(blo)}, {_signed(bhi)}]{_p(p):>12}"
            lines.append(line.rstrip())

    groups = seed_groups([label for label in shown if label != baseline])
    for name, members in groups.items():
        check_seed_group(name, members, by_label)
    if groups:
        lines += ["", "seed groups: accuracy mean and std over seeds; pooled paired difference vs "
                      f"{baseline} with each problem's correctness averaged over seeds (the CI covers "
                      "problem sampling only; the std column is the seed-to-seed spread)"]
    for name, members in groups.items():
        pooled = np.mean([ok[label] for label in members], axis=0)
        lines += ["", f"== {name}: {len(members)} seeds ({', '.join(members)}) ==",
                  f"  {'domain':<8}{'mean acc':>9}{'std':>8}{'vs ' + baseline:>{vs}}  {'95% bootstrap CI':<18}"
                  f"{'paired p':>10}"]
        for d, m in rows:
            accs = [ok[label][m].mean() for label in members]
            strata = doms[m] if d == "all" else None
            diff, blo, bhi = paired_bootstrap(base[m], pooled[m], n_boot, seed, strata)
            p = paired_z_test(base[m], pooled[m], strata)
            lines.append(f"  {d:<8}{np.mean(accs):>9.3f}{np.std(accs, ddof=1):>8.3f}{_signed(diff):>{vs}}  "
                         f"[{_signed(blo)}, {_signed(bhi)}]{_p(p):>12}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="JSON written by evaluate.py --json")
    parser.add_argument("--baseline", default="sft", help="label (checkpoint file stem) to compare against")
    parser.add_argument("--group", default=None, help="only show checkpoints whose label matches this regex")
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    print(report(data, args.baseline, args.group, args.n_boot, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
