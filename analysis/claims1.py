import json, sys
import numpy as np
sys.path.insert(0, "/home/user/nanoGPT")
from stats import paired_bootstrap, paired_z_test, mcnemar_exact
d = json.load(open("results/eval.json"))
cks = {c["label"]: c for c in d["checkpoints"]}
ok = {l: np.array([bool(r["lean_ok"]) for r in c["records"]]) for l, c in cks.items()}
chk = {l: np.array([bool(r["checker_ok"]) for r in c["records"]]) for l, c in cks.items()}
doms = np.array([r["domain"] for r in cks["sft"]["records"]])
def pool(a): return np.mean([ok[f"{a}_s{s}"] for s in (1,2,3)], axis=0)
arms = ["rlhf","ppochecker","rft","sftplus"]
print("pairwise pooled arm comparisons (all domains)")
for i,a in enumerate(arms):
    for b in arms[i+1:]:
        m, lo, hi = paired_bootstrap(pool(a), pool(b), strata=doms)
        p = paired_z_test(pool(a), pool(b), strata=doms)
        print(f"  {b} - {a}: {m*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] p={p:.3g}")
print("discordant problems vs sft (per seed): only_sft / only_arm / total changed answers")
for l in cks:
    if l in ("sft",): continue
    a, b = ok["sft"], ok[l]
    ans_changed = sum(r1["answer"] != r2["answer"] for r1, r2 in zip(cks["sft"]["records"], cks[l]["records"]))
    print(f"  {l}: {int((a&~b).sum())} / {int((~a&b).sum())}; answers differing from SFT: {ans_changed}")
# compiled counts
fin = sum(r["status"]=="finished" for c in d["checkpoints"] for r in c["records"])
tot = sum(len(c["records"]) for c in d["checkpoints"])
decls = set(r["prompt"]+" "+r["answer"] for c in d["checkpoints"] for r in c["records"] if r["status"]=="finished")
print("total records", tot, "finished", fin, "distinct finished decls", len(decls))
fp = sum(r["checker_ok"] and not r["lean_ok"] for c in d["checkpoints"] for r in c["records"])
fn = sum(r["lean_ok"] and not r["checker_ok"] for c in d["checkpoints"] for r in c["records"])
print("checker FP", fp, "FN", fn)
# per-domain significance for sftplus & rft
for a in arms:
    for dm in ["prop","fol","nat","linear"]:
        m = doms==dm
        mean, lo, hi = paired_bootstrap(ok["sft"][m].astype(float), pool(a)[m])
        p = paired_z_test(ok["sft"][m].astype(float), pool(a)[m])
        print(f"  {a} {dm}: {mean*100:+.1f} [{lo*100:+.1f},{hi*100:+.1f}] p={p:.3g}")
# checker-vs-lean accuracy of sft
print("sft checker acc", chk["sft"].mean(), "lean", ok["sft"].mean())
# answer lengths
for l in ["sft","rlhf_s1","ppochecker_s1","rft_s1","sftplus_s1"]:
    L=[len(r["answer"]) for r in cks[l]["records"]]
    print(l, "mean answer chars", np.mean(L))
