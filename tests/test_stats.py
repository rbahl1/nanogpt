"""stats.py against scipy and against synthetic data with a known answer."""

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats as sps
from scipy.special import comb

import stats


def test_wilson_endpoints_solve_the_score_equation():
    # The Wilson interval is {p : |k/n - p| <= z sqrt(p (1 - p) / n)}; its
    # endpoints are where equality holds.
    z = 1.96
    for k, n in [(0, 10), (1, 10), (5, 10), (81, 263), (499, 500), (500, 500)]:
        lo, hi = stats.wilson_ci(k, n, z)
        assert 0 <= lo <= k / n <= hi <= 1
        for p in (lo, hi):
            if p not in (0.0, 1.0):
                assert (k / n - p) ** 2 == pytest.approx(z * z * p * (1 - p) / n, abs=1e-12)
    assert stats.wilson_ci(81, 263) == pytest.approx((0.2553, 0.3662), abs=1e-4)
    assert stats.wilson_ci(0, 0) == (0.0, 1.0)


def test_pass_at_k_known_values():
    for n in (1, 4, 16, 50):
        for c in range(n + 1):
            assert stats.pass_at_k(n, c, 1) == pytest.approx(c / n)
            for k in range(1, n + 1):
                exact = 1 - Fraction(math.comb(n - c, k), math.comb(n, k))
                assert stats.pass_at_k(n, c, k) == pytest.approx(float(exact), abs=1e-12)
                assert stats.pass_at_k(n, c, k) == pytest.approx(1 - comb(n - c, k) / comb(n, k), abs=1e-9)
    assert stats.pass_at_k(16, 0, 16) == 0.0
    assert stats.pass_at_k(16, 1, 16) == 1.0
    assert stats.pass_at_k(200, 3, 100) == pytest.approx(1 - (100 * 99 * 98) / (200 * 199 * 198))
    with pytest.raises(ValueError):
        stats.pass_at_k(4, 1, 5)


def test_mcnemar_matches_scipy_binomtest():
    rng = np.random.default_rng(0)
    for trial in range(300):
        n = int(rng.integers(1, 120))
        a = rng.random(n) < rng.random()
        b = rng.random(n) < rng.random()
        only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
        expected = 1.0 if only_a + only_b == 0 else sps.binomtest(only_b, only_a + only_b, 0.5).pvalue
        assert stats.mcnemar_exact(a, b) == pytest.approx(expected, rel=1e-9, abs=1e-15)
    # symmetric in the two arms, and concordant pairs do not matter
    a = [1, 1, 1, 0, 0, 1, 0]
    b = [0, 0, 1, 1, 0, 1, 0]
    assert stats.mcnemar_exact(a, b) == stats.mcnemar_exact(b, a) == stats.mcnemar_exact(a + [1] * 50, b + [1] * 50)
    # large discordant counts stay exact (no float overflow in the binomial)
    a = np.array([1] * 900 + [0] * 1100, dtype=bool)
    b = np.array([0] * 800 + [1] * 1200, dtype=bool)
    assert stats.mcnemar_exact(a, b) == pytest.approx(sps.binomtest(1100, 1900, 0.5).pvalue, rel=1e-9)


def paired_sample(rng, n, p_a_only, p_b_only, p_both):
    """Paired binary outcomes with P(only a right) = p_a_only etc.; the true
    difference in accuracy (b - a) is p_b_only - p_a_only."""
    u = rng.random(n)
    a = (u < p_a_only) | ((u >= p_a_only + p_b_only) & (u < p_a_only + p_b_only + p_both))
    b = (u >= p_a_only) & (u < p_a_only + p_b_only + p_both)
    return a.astype(float), b.astype(float)


def test_bootstrap_ci_covers_the_true_difference():
    rng = np.random.default_rng(1)
    a, b = paired_sample(rng, 2000, 0.10, 0.15, 0.40)
    mean, lo, hi = stats.paired_bootstrap(a, b, seed=0)
    assert mean == pytest.approx(b.mean() - a.mean())
    assert lo < 0.05 < hi
    # its width is what the normal approximation predicts for the paired differences
    se = np.std(b - a, ddof=1) / math.sqrt(len(a))
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.1)
    # same seed, same interval
    assert stats.paired_bootstrap(a, b, seed=0) == (mean, lo, hi)


def test_bootstrap_coverage_is_near_nominal():
    rng = np.random.default_rng(2)
    covered = 0
    trials = 200
    for t in range(trials):
        a, b = paired_sample(rng, 300, 0.12, 0.18, 0.30)
        _, lo, hi = stats.paired_bootstrap(a, b, n_boot=1000, seed=t)
        covered += lo <= 0.06 <= hi
    assert 0.88 <= covered / trials <= 0.99


def test_stratified_bootstrap_keeps_stratum_sizes():
    # b beats a on every problem of stratum "x" and loses on every one of "y":
    # resampling within strata fixes the mean exactly, resampling problems
    # freely does not.
    a = np.array([0.0] * 30 + [1.0] * 10)
    b = np.array([1.0] * 30 + [0.0] * 10)
    strata = ["x"] * 30 + ["y"] * 10
    mean, lo, hi = stats.paired_bootstrap(a, b, n_boot=2000, strata=strata)
    assert mean == lo == hi == pytest.approx(0.5)
    _, lo, hi = stats.paired_bootstrap(a, b, n_boot=2000)
    assert hi - lo > 0.3


def test_paired_z_test():
    rng = np.random.default_rng(3)
    a, b = paired_sample(rng, 5000, 0.1, 0.1, 0.4)
    d = b - a
    expected = 2 * sps.norm.sf(abs(d.mean()) / (d.std(ddof=1) / math.sqrt(len(d))))
    assert stats.paired_z_test(a, b) == pytest.approx(expected)
    assert stats.paired_z_test(a, a) == 1.0
    a, b = paired_sample(rng, 2000, 0.05, 0.15, 0.4)
    assert stats.paired_z_test(a, b) < 1e-6
    # seed-averaged (fractional) outcomes are allowed
    assert 0 <= stats.paired_z_test(a, (a + b) / 2) <= 1


def fake_eval(labels, outcomes, domains):
    prompts = [f"problem {i}" for i in range(len(domains))]
    return {"checkpoints": [
        {"label": label, "stage": "x", "records": [
            {"domain": d, "prompt": p, "answer": "", "status": "finished", "checker_ok": bool(o), "lean_ok": None}
            for d, p, o in zip(domains, prompts, outs)]}
        for label, outs in zip(labels, outcomes)]}


def test_cli_groups_seeds(tmp_path, capsys):
    rng = np.random.default_rng(4)
    domains = ["nat"] * 60 + ["linear"] * 60
    base = rng.random(120) < 0.4
    arms = [base | (rng.random(120) < 0.3) for _ in range(3)]
    other = rng.random(120) < 0.4
    data = fake_eval(["sft", "rlhf_s1", "rlhf_s2", "rlhf_s3", "rft_s1"], [base, *arms, other], domains)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data))
    assert stats.main([str(path), "--baseline", "sft", "--n_boot", "500"]) == 0
    out = capsys.readouterr().out
    assert "correctness: checker" in out
    assert "== rlhf: 3 seeds (rlhf_s1, rlhf_s2, rlhf_s3) ==" in out
    assert "== rft:" not in out                        # one seed is not a group
    group = out.split("== rlhf: 3 seeds")[1].splitlines()
    row = next(line for line in group if line.strip().startswith("all"))
    accs = [arm.mean() for arm in arms]
    assert f"{np.mean(accs):.3f}" in row and f"{np.std(accs, ddof=1):.3f}" in row
    assert f"{np.mean(accs) - base.mean():+.3f}" in row
    # every arm only adds correct answers, so McNemar finds rlhf_s1 better
    block = out.split("== rlhf_s1 (stage x) ==")[1].split("==")[0]
    p = float(next(line for line in block.splitlines() if line.strip().startswith("all")).split()[-1])
    assert p == pytest.approx(stats.mcnemar_exact(base, arms[0]), rel=0.02)   # printed to 2 digits
    assert p < 0.001

    # --group filters, and Lean verdicts are used when every record has one
    for e in data["checkpoints"]:
        for r in e["records"]:
            r["lean_ok"] = r["checker_ok"]
    path.write_text(json.dumps(data))
    stats.main([str(path), "--baseline", "sft", "--group", "rft", "--n_boot", "200"])
    out = capsys.readouterr().out
    assert "Lean 4" in out and "== rft_s1" in out and "rlhf" not in out

    with pytest.raises(SystemExit):
        stats.main([str(path), "--baseline", "nope"])


def test_seed_groups_pool_only_runs_that_differ_in_seed():
    """Labels decide which checkpoints are pooled as seeds of one arm. Runs
    that differ in more than their seed (and where they were written) are
    different arms, and averaging them would report a mixture as one arm."""
    domains = ["nat"] * 20 + ["linear"] * 20
    rng = np.random.default_rng(5)
    data = fake_eval(["sft", "rlhf_s1", "rlhf_s2"], [rng.random(40) < 0.5 for _ in range(3)], domains)
    configs = [{"seed": 2024, "out": "out/sft.pt"},
               {"reward_source": "rm", "iters": 150, "seed": 1, "out": "out/rlhf_s1.pt", "device": "cpu"},
               {"reward_source": "rm", "iters": 150, "seed": 2, "out": "out/rlhf_s2.pt", "device": "auto"}]
    for e, c in zip(data["checkpoints"], configs):
        e["run_config"] = c
    assert "== rlhf: 2 seeds (rlhf_s1, rlhf_s2) ==" in stats.report(data, "sft", n_boot=200)
    data["checkpoints"][2]["run_config"]["reward_source"] = "oracle"
    with pytest.raises(SystemExit, match="rlhf_s2.*reward_source"):
        stats.report(data, "sft", n_boot=200)
    data["checkpoints"][2]["run_config"]["reward_source"] = "rm"
    data["checkpoints"][2]["stage"] = "ppo-checker"
    with pytest.raises(SystemExit, match="stage"):
        stats.report(data, "sft", n_boot=200)
