import json
import numpy as np
A = "/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/"
V = json.load(open(A + "val_records.json"))["checkpoints"]
T = {c["label"]: c for c in json.load(open("/home/user/nanoGPT/results/eval.json"))["checkpoints"]}
DOMS = ["prop", "fol", "nat", "linear"]
labs = ["sft"] + [f"{a}_s{s}" for a in ("rlhf", "ppochecker", "rft", "sftplus") for s in (1, 2, 3)]
vdom = np.array([r["domain"] for r in V["sft"]["records"]]); tdom = np.array([r["domain"] for r in T["sft"]["records"]])
VL = np.stack([[r["lean_ok"] for r in V[l]["records"]] for l in labs]).astype(float)
TL = np.stack([[bool(r["lean_ok"]) for r in T[l]["records"]] for l in labs]).astype(float)
print("per-domain contribution to (mean over 12 arms) val diff and test diff vs SFT (points of overall accuracy):")
tot_v = tot_t = 0
for d in DOMS:
    mv, mt = vdom == d, tdom == d
    cv = (VL[1:, mv].sum(1) - VL[0, mv].sum()).mean() / 300 * 100
    ct = (TL[1:, mt].sum(1) - TL[0, mt].sum()).mean() / 2000 * 100
    nv_up = int((VL[1:, mv].sum(1) > VL[0, mv].sum()).sum()); nt_up = int((TL[1:, mt].sum(1) > TL[0, mt].sum()).sum())
    print(f"  {d:7s} val {cv:+5.2f}  test {ct:+5.2f}  val-test {cv-ct:+5.2f}   arms above SFT: val {nv_up}/12, test {nt_up}/12; within-domain mean arm diff val {100*(VL[1:,mv].mean(1)-VL[0,mv].mean()).mean():+5.1f} test {100*(TL[1:,mt].mean(1)-TL[0,mt].mean()).mean():+5.1f}")
    tot_v += cv; tot_t += ct
print(f"  total   val {tot_v:+5.2f} test {tot_t:+5.2f}")
# permutation test per domain and excluding linear
X = np.concatenate([VL, TL], 1); mdom = np.concatenate([vdom, tdom])
rng = np.random.default_rng(5); P = 20000
def stat(vi, ti):
    va, ta = X[:, vi].mean(1), X[:, ti].mean(1)
    return ((va[1:] - va[0]) - (ta[1:] - ta[0])).mean(), int((va[1:] > va[0]).sum()), va[0] - ta[0]
for name, ds in [("linear only", ["linear"]), ("prop only", ["prop"]), ("fol only", ["fol"]), ("nat only", ["nat"]), ("excluding linear", ["prop", "fol", "nat"])]:
    groups = [np.flatnonzero(mdom == d) for d in ds]
    vi0 = np.concatenate([g[g < 300] for g in groups]); ti0 = np.concatenate([g[g >= 300] for g in groups])
    o = stat(vi0, ti0)
    sims = []
    for _ in range(P):
        vi, ti = [], []
        for g in groups:
            p = rng.permutation(g); vi.append(p[:75]); ti.append(p[75:])
        sims.append(stat(np.concatenate(vi), np.concatenate(ti)))
    sims = np.array(sims)
    print(f"{name:17s} obs mean(vd-td) {100*o[0]:+5.2f} perm p(>=) {np.mean(sims[:,0] >= o[0]-1e-12):.4f}; arms above SFT on val {o[1]}/12 p(>=) {np.mean(sims[:,1] >= o[1]):.4f}; SFT val-test {100*o[2]:+5.1f} p(<=) {np.mean(sims[:,2] <= o[2]+1e-12):.4f}")
