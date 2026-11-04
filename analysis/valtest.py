import json, sys
import numpy as np
from math import comb, erfc, sqrt
from scipy import stats as ss
A = "/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/"
V = json.load(open(A + "val_records.json"))["checkpoints"]
T = {c["label"]: c for c in json.load(open("/home/user/nanoGPT/results/eval.json"))["checkpoints"]}
DOMS = ["prop", "fol", "nat", "linear"]
labs = ["sft"] + [f"{a}_s{s}" for a in ("rlhf", "ppochecker", "rft", "sftplus") for s in (1, 2, 3)]
arms = labs[1:]
vp = [(r["domain"], r["prompt"]) for r in V["sft"]["records"]]
for l in labs: assert [(r["domain"], r["prompt"]) for r in V[l]["records"]] == vp
vdom = np.array([d for d, _ in vp]); tdom = np.array([r["domain"] for r in T["sft"]["records"]])
# overlap val/test prompts?
print("val/test prompt overlap:", len(set(p for _, p in vp) & set(r["prompt"] for r in T["sft"]["records"])))
VL = {l: np.array([r["lean_ok"] for r in V[l]["records"]]) for l in labs}
VC = {l: np.array([r["checker_ok"] for r in V[l]["records"]]) for l in labs}
TL = {l: np.array([bool(r["lean_ok"]) for r in T[l]["records"]]) for l in labs}
print("val checker vs lean disagreements:", sum(int((VL[l] != VC[l]).sum()) for l in labs))
# compare with logged final val
logs = {}
for l in arms:
    a = l.split("_")[0]
    if a in ("rlhf", "ppochecker"):
        lg = json.load(open(f"/home/user/nanoGPT/results/logs/{l}.json")); logs[l] = lg["evals"][-1]["results"]["all"]["correct"]
    elif a == "rft":
        lg = json.load(open(f"/home/user/nanoGPT/results/logs/{l}.json")); logs[l] = lg["rounds"][-1]["val"]["all"]["correct"]
    sv = V[l]["saved_val"]
    if sv: logs[l] = (logs.get(l), sv["all"]["correct"])
print("logged/saved final val correct:", logs)
print("my val correct:", {l: int(VC[l].sum()) for l in labs})

def mcn(a, b):
    oa, ob = int((a & ~b).sum()), int((~a & b).sum()); n = oa + ob
    return oa, ob, (1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(oa, ob) + 1)) / 2**n))

def strat_boot_idx(dom, rng):
    return np.concatenate([rng.choice(np.flatnonzero(dom == d), size=(dom == d).sum()) for d in DOMS])

rng = np.random.default_rng(7)
B = 10000
vidx = [strat_boot_idx(vdom, rng) for _ in range(B)]
rows = []
print("\nlabel          val%   vdiff  [95% CI]       McN p | test%  tdiff  McN p | val-test gap")
for l in labs:
    va, ta = 100 * VL[l].mean(), 100 * TL[l].mean()
    if l == "sft":
        print(f"{l:14s} {va:5.1f}                                | {ta:5.1f}               | {va - ta:+5.1f}"); continue
    d = (VL[l].astype(float) - VL["sft"])
    bs = np.array([d[i].mean() for i in vidx]) * 100
    lo, hi = np.quantile(bs, [.025, .975])
    _, _, pv = mcn(VL["sft"], VL[l]); oa, ob, pt = mcn(TL["sft"], TL[l])
    vd, td = 100 * d.mean(), 100 * (TL[l].mean() - TL["sft"].mean())
    rows.append((l, va, vd, lo, hi, pv, ta, td, pt, va - ta))
    print(f"{l:14s} {va:5.1f}  {vd:+5.2f} [{lo:+5.1f},{hi:+5.1f}]  {pv:.3f} | {ta:5.1f} {td:+5.2f}  {pt:.3g} | {va - ta:+5.1f}")
print("\nby domain val acc:")
for l in labs:
    print(f"{l:14s}", " ".join(f"{d}:{100*VL[l][vdom==d].mean():5.1f}({int(VL[l][vdom==d].sum())})" for d in DOMS),
          "| test", " ".join(f"{100*TL[l][tdom==d].mean():5.1f}" for d in DOMS))
nv_up = sum(r[2] > 0 for r in rows); nt_dn = sum(r[7] < 0 for r in rows)
rev = [r[0] for r in rows if r[2] > 0 and r[7] < 0]
print(f"\nval>sft: {nv_up}/12, val==sft: {sum(r[2]==0 for r in rows)}, test<sft: {nt_dn}/12, val>sft & test<sft: {len(rev)} {rev}")
print("val>=sft & test<sft:", [r[0] for r in rows if r[2] >= 0 and r[7] < 0])
print("sign agreement val vs test:", sum(np.sign(r[2]) == np.sign(r[7]) for r in rows))
# arm-level pooled
for a in ("rlhf", "ppochecker", "rft", "sftplus"):
    pv_ = np.mean([VL[f"{a}_s{s}"] for s in (1, 2, 3)], 0) - VL["sft"]
    pt_ = np.mean([TL[f"{a}_s{s}"] for s in (1, 2, 3)], 0) - TL["sft"]
    bs = np.array([pv_[i].mean() for i in vidx]) * 100
    print(f"arm {a:10s} pooled val diff {100*pv_.mean():+5.2f} [{np.quantile(bs,.025):+5.2f},{np.quantile(bs,.975):+5.2f}]  test diff {100*pt_.mean():+5.2f}")
allv = np.mean([VL[l] for l in arms], 0) - VL["sft"]; allt = np.mean([TL[l] for l in arms], 0) - TL["sft"]
rlv = np.mean([VL[l] for l in arms[:9]], 0) - VL["sft"]; rlt = np.mean([TL[l] for l in arms[:9]], 0) - TL["sft"]
print(f"mean over 12 arms: val {100*allv.mean():+.2f} test {100*allt.mean():+.2f}; over 9 RL/RFT arms: val {100*rlv.mean():+.2f} test {100*rlt.mean():+.2f}")
# rank of SFT
vr = sorted(labs, key=lambda l: -VL[l].sum()); tr = sorted(labs, key=lambda l: -TL[l].sum())
print("SFT val rank (1=best):", 1 + sum(VL[l].sum() > VL["sft"].sum() for l in labs), "ties", sum(VL[l].sum() == VL["sft"].sum() for l in arms),
      "| test rank:", 1 + sum(TL[l].sum() > TL["sft"].sum() for l in labs))

# ---- SFT val vs test: two-sample (independent problem sets) stratified
def strat_var(x, dom):
    n = len(x); return sum((dom == d).sum()**2 * x[dom == d].var(ddof=1) / (dom == d).sum() for d in DOMS) / n**2
for l in ["sft"]:
    dv = VL[l].mean() - TL[l].mean()
    se = sqrt(strat_var(VL[l].astype(float), vdom) + strat_var(TL[l].astype(float), tdom))
    se_u = sqrt(VL[l].var(ddof=1) / 300 + TL[l].var(ddof=1) / 2000)
    print(f"\nSFT val-test {100*dv:+.2f}: stratified SE {100*se:.2f} z={dv/se:.2f} p={erfc(abs(dv/se)/sqrt(2)):.3f}; unstratified SE {100*se_u:.2f} p={erfc(abs(dv/se_u)/sqrt(2)):.3f}")
    # Fisher exact per domain and overall
    for d in DOMS + ["all"]:
        mv = vdom == d if d != "all" else np.ones(300, bool); mt = tdom == d if d != "all" else np.ones(2000, bool)
        kv, nv, kt, nt = int(VL[l][mv].sum()), int(mv.sum()), int(TL[l][mt].sum()), int(mt.sum())
        _, pf = ss.fisher_exact([[kv, nv - kv], [kt, nt - kt]])
        # binomial: prob of val count <= observed given test rate
        pb = ss.binom.cdf(kv, nv, kt / nt) if kv / nv < kt / nt else ss.binom.sf(kv - 1, nv, kt / nt)
        print(f"  {d:7s} val {kv}/{nv}={100*kv/nv:5.1f} test {kt}/{nt}={100*kt/nt:5.1f}  Fisher p={pf:.3g}  one-sided binom tail (test rate) {pb:.3g}")

# ---- Gap comparison: SFT's val-test gap vs the arms'
gaps = np.array([r[9] for r in rows])
print(f"\narms' val-test gaps: mean {gaps.mean():+.2f} sd {gaps.std(ddof=1):.2f} min {gaps.min():+.2f} max {gaps.max():+.2f}; SFT gap {100*(VL['sft'].mean()-TL['sft'].mean()):+.2f}")
print("how many arms have a gap <= SFT's:", int((gaps <= 100*(VL['sft'].mean()-TL['sft'].mean())).sum()))

# ---- permutation test: re-split val+test problems at random (within domain) 75/500
P = 20000
prng = np.random.default_rng(11)
Mall = {l: np.concatenate([VL[l], TL[l]]) for l in labs}
mdom = np.concatenate([vdom, tdom])
groups = [np.flatnonzero(mdom == d) for d in DOMS]
X = np.stack([Mall[l] for l in labs]).astype(float)   # 13 x 2300
def stats_for(vi, ti):
    va = X[:, vi].mean(1); ta = X[:, ti].mean(1)
    vd = va[1:] - va[0]; td = ta[1:] - ta[0]
    return dict(sft_gap=va[0] - ta[0],
                mean_vd_minus_td=(vd - td).mean(),
                rl_vd_minus_td=(vd[:9] - td[:9]).mean(),
                n_rev=int(((vd > 0) & (td < 0)).sum()),
                n_rev_ge=int(((vd >= 0) & (td < 0)).sum()),
                n_vup=int((vd > 0).sum()),
                sft_val_rank=int((va[1:] > va[0]).sum()),
                sft_gap_minus_arm_gap=(va[0] - ta[0]) - ((va[1:] - ta[1:]).mean()))
vi0 = np.arange(300); ti0 = np.arange(300, 2300)
obs = stats_for(vi0, ti0)
print("\nobserved:", {k: round(float(v), 4) for k, v in obs.items()})
sims = {k: [] for k in obs}
for _ in range(P):
    vi, ti = [], []
    for g in groups:
        perm = prng.permutation(g); vi.append(perm[:75]); ti.append(perm[75:])
    s = stats_for(np.concatenate(vi), np.concatenate(ti))
    for k in s: sims[k].append(s[k])
sims = {k: np.array(v) for k, v in sims.items()}
for k in obs:
    o = obs[k]; s = sims[k]
    print(f"  {k:24s} obs {float(o):+8.4f}  perm mean {s.mean():+8.4f} sd {s.std():.4f}  P(sim<=obs) {np.mean(s <= o + 1e-12):.4f}  P(sim>=obs) {np.mean(s >= o - 1e-12):.4f}")
print("  distribution of n_rev (val>sft & test<sft) under random re-splits:", np.bincount(sims["n_rev"], minlength=13))
json.dump({"rows": rows}, open(A + "valtest_out.json", "w"))
