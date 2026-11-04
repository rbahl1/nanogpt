import json
import numpy as np
from math import comb, erfc, sqrt
from scipy import stats as ss
D = json.load(open("/home/user/nanoGPT/results/eval.json"))["checkpoints"]
L = {c["label"]: np.array([bool(r["lean_ok"]) for r in c["records"]]) for c in D}
dom = np.array([r["domain"] for r in D[0]["records"]]); DOMS = ["prop", "fol", "nat", "linear"]
ARMS = ["rlhf", "ppochecker", "rft", "sftplus"]
S = {a: np.stack([L[f"{a}_s{s}"] for s in (1, 2, 3)]).astype(float) for a in ARMS}
S["sft"] = L["sft"][None].astype(float)
gidx = [np.flatnonzero(dom == d) for d in DOMS]

def hier_boot(a, b, B=10000, seed=3):
    """two-level bootstrap of mean(b) - mean(a): resample problems within domain AND seeds within arm."""
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for i in range(B):
        pi = np.concatenate([rng.choice(g, size=len(g)) for g in gidx])
        sa = rng.integers(0, S[a].shape[0], S[a].shape[0]); sb = rng.integers(0, S[b].shape[0], S[b].shape[0])
        out[i] = S[b][sb][:, pi].mean() - S[a][sa][:, pi].mean()
    obs = S[b].mean() - S[a].mean()
    lo, hi = np.quantile(out, [.025, .975])
    p = min(1.0, 2 * min(np.mean(out <= 0), np.mean(out >= 0)))
    return 100 * obs, 100 * lo, 100 * hi, p

def mcn(a, b):
    oa, ob = int((a & ~b).sum()), int((~a & b).sum()); n = oa + ob
    return oa, ob, (1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(oa, ob) + 1)) / 2**n))

pairs = [("sft", "rlhf"), ("sft", "ppochecker"), ("sft", "rft"), ("sft", "sftplus"),
         ("sftplus", "rlhf"), ("sftplus", "ppochecker"), ("sftplus", "rft"), ("ppochecker", "rft"), ("ppochecker", "rlhf"), ("rft", "rlhf")]
print("comparison (b - a)        diff  hier-boot CI      p_boot | seed-level Welch t p | per-seed-pair McNemar (b>a / b<a sig at .05, of 9)")
for a, b in pairs:
    d, lo, hi, p = hier_boot(a, b)
    accA = 100 * S[a].mean(1); accB = 100 * S[b].mean(1)
    if a == "sft":
        t = ss.ttest_1samp(accB - accA[0], 0); tp = t.pvalue
    else:
        tp = ss.ttest_ind(accB, accA, equal_var=False).pvalue
    pm = []
    for i in range(S[a].shape[0]):
        for j in range(S[b].shape[0]):
            oa, ob, p_ = mcn(S[a][i].astype(bool), S[b][j].astype(bool)); pm.append((ob - oa, p_))
    nb = sum(1 for x, q in pm if x > 0 and q < .05); na = sum(1 for x, q in pm if x < 0 and q < .05)
    pmax = max(q for _, q in pm); signs = sum(1 for x, _ in pm if x < 0)
    print(f"{b:>10s} - {a:<10s} {d:+6.2f} [{lo:+5.2f}, {hi:+5.2f}] p={p:.4f} | Welch/1-samp t p={tp:.3g} | sig b>a {nb}, sig b<a {na}, b<a sign {signs}/{len(pm)}, max p {pmax:.3g}")

# Holm across 12 per-seed McNemar (all) vs sft
ps = []
for a in ARMS:
    for s in (1, 2, 3):
        oa, ob, p = mcn(L["sft"], L[f"{a}_s{s}"]); ps.append((f"{a}_s{s}", p))
ps_sorted = sorted(ps, key=lambda x: x[1]); m = len(ps)
adj, run = {}, 0
for k, (l, p) in enumerate(ps_sorted):
    run = max(run, min(1, (m - k) * p)); adj[l] = run
print("\nHolm-adjusted per-seed McNemar (all, vs sft):", {l: round(adj[l], 4) for l, _ in ps})

# PPO / RFT val trajectories from logs
print("\nval trajectories (correct/300); iteration-0 = SFT 96")
for a in ("rlhf", "ppochecker"):
    for s in (1, 2, 3):
        lg = json.load(open(f"/home/user/nanoGPT/results/logs/{a}_s{s}.json"))
        ev = [(e["iteration"], e["results"]["all"]["correct"]) for e in lg["evals"]]
        it = [c for i, c in ev if i > 0]
        best = max(ev[1:], key=lambda x: x[1])
        print(f"{a}_s{s}: {[c for _, c in ev]}  iterates>96: {sum(c > 96 for c in it)}/{len(it)}  ==96: {sum(c == 96 for c in it)}  <96: {sum(c < 96 for c in it)}  best {best} (+{100*(best[1]-96)/300:.1f})  mean of iterates {100*np.mean(it)/300:.2f}")
    # per-iteration rollout accuracy trend first vs last
for s in (1, 2, 3):
    lg = json.load(open(f"/home/user/nanoGPT/results/logs/rft_s{s}.json"))
    print(f"rft_s{s}: rounds val", [r["val"]["all"]["correct"] for r in lg["rounds"]], "evals key:", [ (e.get('round'), e['results']['all']['correct']) for e in lg['evals']] if lg.get('evals') else None)
