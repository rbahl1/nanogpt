"""Independent recomputation of the headline statistics from results/eval.json (no stats.py/analyze.py)."""
import json, math
import numpy as np
from math import comb, erfc, sqrt

D = json.load(open("/home/user/nanoGPT/results/eval.json"))
cks = D["checkpoints"]
labels = [c["label"] for c in cks]
recs0 = [(r["domain"], r["prompt"]) for r in cks[0]["records"]]
for c in cks:
    assert [(r["domain"], r["prompt"]) for r in c["records"]] == recs0, c["label"]
    assert all(r["lean_ok"] is not None for r in c["records"])
dom = np.array([d for d, _ in recs0])
DOMS = ["prop", "fol", "nat", "linear"]
print("n problems", len(dom), {d: int((dom == d).sum()) for d in DOMS})
# distinct canonical forms?
import sys; sys.path.insert(0, "/home/user/nanoGPT")
import data as nd
keys = [nd.split_key(p) for _, p in recs0]
print("distinct split keys", len(set(keys)), "splits", {s: sum(nd.split_of_key(k) == s for k in keys) for s in ("train","val","test")})

L = {c["label"]: np.array([bool(r["lean_ok"]) for r in c["records"]]) for c in cks}
C = {c["label"]: np.array([bool(r["checker_ok"]) for r in c["records"]]) for c in cks}
ST = {c["label"]: np.array([r["status"] for r in c["records"]]) for c in cks}
masks = {d: dom == d for d in DOMS}; masks["all"] = np.ones(len(dom), bool)
ARMS = ["rlhf", "ppochecker", "rft", "sftplus"]

def acc(x, m): return 100 * x[m].mean()

print("\n== per-checkpoint Lean accuracy ==")
for lab in labels:
    print(f"{lab:14s}", " ".join(f"{d}:{acc(L[lab], masks[d]):5.1f}" for d in DOMS + ["all"]),
          " checker all:", f"{acc(C[lab], masks['all']):5.2f}")

# checker vs lean
fp = sum(int((C[l] & ~L[l]).sum()) for l in labels); fn = sum(int((~C[l] & L[l]).sum()) for l in labels)
print("\nchecker+ lean-:", fp, " checker- lean+:", fn, " total answers", len(labels) * len(dom))
for l in labels:
    if (~C[l] & L[l]).any():
        idx = np.flatnonzero(~C[l] & L[l])
        for i in idx: print("   FN", l, cks[labels.index(l)]["records"][i]["domain"], repr(cks[labels.index(l)]["records"][i]["prompt"]), "::", repr(cks[labels.index(l)]["records"][i]["answer"]))
unfinished_ok = sum(int(((ST[l] != "finished") & L[l]).sum()) for l in labels)
print("unfinished but lean_ok:", unfinished_ok)

def boot(d, strata=None, B=10000, seed=12345):
    """percentile CI for mean of d, resampling problems (within strata if given)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    groups = [np.arange(n)] if strata is None else [np.flatnonzero(strata == s) for s in dict.fromkeys(strata)]
    tot = np.zeros(B)
    for g in groups:
        v = d[g]
        for s in range(0, B, 1000):
            tot[s:s+1000] += v[rng.integers(0, len(g), size=(1000, len(g)))].sum(1)
    return np.quantile(tot / n, [0.025, 0.975])

def ztest(d, strata=None):
    n = len(d)
    if strata is None:
        var = d.var(ddof=1) / n
    else:
        var = sum(len(g) * d[g].var(ddof=1) for g in [np.flatnonzero(strata == s) for s in dict.fromkeys(strata)]) / n**2
    if var == 0: return 1.0 if d.mean() == 0 else 0.0
    return erfc(abs(d.mean()) / sqrt(var) / sqrt(2))

def mcnemar(a, b):
    oa, ob = int((a & ~b).sum()), int((~a & b).sum()); n = oa + ob
    if n == 0: return oa, ob, 1.0
    tail = sum(comb(n, i) for i in range(min(oa, ob) + 1))
    return oa, ob, min(1.0, 2 * tail / 2**n)

def pooled(arm): return np.mean([L[f"{arm}_s{s}"] for s in (1, 2, 3)], axis=0)

def compare(a, b, name, out):
    """a, b: per-problem (possibly fractional) correctness arrays; b - a."""
    for d in DOMS + ["all"]:
        m = masks[d]
        diff = (b - a)[m]
        strata = dom[m] if d == "all" else None
        lo, hi = boot(diff, strata)
        p = ztest(diff, strata)
        out.append((name, d, 100 * diff.mean(), 100 * lo, 100 * hi, p))

print("\n== seed groups ==")
res = []
for arm in ARMS:
    for d in DOMS + ["all"]:
        a = [acc(L[f"{arm}_s{s}"], masks[d]) for s in (1, 2, 3)]
        print(f"{arm:11s} {d:7s} seeds {a[0]:5.1f} {a[1]:5.1f} {a[2]:5.1f}  mean {np.mean(a):5.2f} std {np.std(a, ddof=1):4.2f}")
    compare(L["sft"].astype(float), pooled(arm), f"{arm} - sft", res)
for a, b in [("sftplus", "rlhf"), ("sftplus", "ppochecker"), ("sftplus", "rft"), ("ppochecker", "rft"), ("ppochecker", "rlhf"), ("rft", "rlhf")]:
    compare(pooled(a), pooled(b), f"{b} - {a}", res)
print("\n== pooled paired differences (points), bootstrap 95% CI (stratified for all), paired z p ==")
for name, d, m, lo, hi, p in res:
    print(f"{name:24s} {d:7s} {m:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] p={p:.2g}")

# Alternative inference that respects seed variance: t-test on 3 seed-level differences, and cluster
# (two-level) bootstrap resampling both problems and seeds.
print("\n== seed-level: per-seed diff vs sft, t on 3 seeds ==")
from scipy import stats as ss
for arm in ARMS:
    ds = [acc(L[f"{arm}_s{s}"], masks["all"]) - acc(L["sft"], masks["all"]) for s in (1, 2, 3)]
    t = np.mean(ds) / (np.std(ds, ddof=1) / sqrt(3))
    print(arm, [f"{x:+.2f}" for x in ds], f"t={t:.2f} p={2*ss.t.sf(abs(t), 2):.3g}")

print("\n== per-seed McNemar vs sft (Lean) ==")
mcn = []
for arm in ARMS:
    for s in (1, 2, 3):
        lab = f"{arm}_s{s}"
        for d in DOMS + ["all"]:
            m = masks[d]
            oa, ob, p = mcnemar(L["sft"][m], L[lab][m])
            mcn.append((lab, d, acc(L[lab], m) - acc(L["sft"], m), oa, ob, p))
for row in mcn:
    if row[1] == "all" or row[5] < 0.05:
        print(f"{row[0]:14s} {row[1]:7s} {row[2]:+5.1f}  sft-only {row[3]:4d} arm-only {row[4]:4d}  p={row[5]:.3g}")
json.dump({"pooled": res, "mcnemar": mcn}, open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/mystats_out.json", "w"))
