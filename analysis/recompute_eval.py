import json, math, sys
import numpy as np
sys.path.insert(0, "/home/user/nanoGPT")
import stats

d = json.load(open("/home/user/nanoGPT/results/eval.json"))
cks = d["checkpoints"]
by = {c["label"]: c for c in cks}
doms = np.array([r["domain"] for r in cks[0]["records"]])
D = ["prop", "fol", "nat", "linear"]
ok = {c["label"]: np.array([bool(r["lean_ok"]) for r in c["records"]]) for c in cks}
chk = {c["label"]: np.array([bool(r["checker_ok"]) for r in c["records"]]) for c in cks}

# same problems order
first = [(r["domain"], r["prompt"]) for r in cks[0]["records"]]
for c in cks:
    assert [(r["domain"], r["prompt"]) for r in c["records"]] == first, c["label"]
print("n problems", len(first), {x: int((doms == x).sum()) for x in D})
print("distinct prompts", len(set(p for _, p in first)))

print("\nPer-checkpoint accuracy (lean), percent, 3 decimals")
for l in ok:
    row = [100 * ok[l][doms == x].mean() for x in D] + [100 * ok[l].mean()]
    print(f"{l:15s}", " ".join(f"{v:7.3f}" for v in row))

print("\nArm means and std (ddof=1), percent")
arms = ["rlhf", "ppochecker", "rft", "sftplus"]
for a in arms:
    ms = [f"{a}_s{i}" for i in (1, 2, 3)]
    out = []
    for x in D + ["all"]:
        m = np.ones(len(doms), bool) if x == "all" else doms == x
        accs = [100 * ok[l][m].mean() for l in ms]
        out.append(f"{np.mean(accs):6.3f}+-{np.std(accs, ddof=1):5.3f}")
    print(f"{a:12s}", " | ".join(out))

print("\nPooled diffs vs sft")
base = ok["sft"].astype(float)
for a in arms:
    ms = [f"{a}_s{i}" for i in (1, 2, 3)]
    pooled = np.mean([ok[l] for l in ms], axis=0)
    for x in D + ["all"]:
        m = np.ones(len(doms), bool) if x == "all" else doms == x
        strata = doms[m] if x == "all" else None
        diff, lo, hi = stats.paired_bootstrap(base[m], pooled[m], 10000, 0, strata)
        p = stats.paired_z_test(base[m], pooled[m], strata)
        # also a different bootstrap seed for robustness
        diff2, lo2, hi2 = stats.paired_bootstrap(base[m], pooled[m], 10000, 123, strata)
        print(f"{a:11s} {x:7s} diff {100*diff:+.3f} CI [{100*lo:+.3f}, {100*hi:+.3f}] p={p:.5g}  (seed123 CI [{100*lo2:+.3f}, {100*hi2:+.3f}])")

print("\nChecker vs Lean disagreements")
fp = fn = tot = fin = 0
for l in ok:
    recs = by[l]["records"]
    for r in recs:
        tot += 1
        if r["status"] == "finished":
            fin += 1
        if r["checker_ok"] and not r["lean_ok"]:
            fp += 1
        if (not r["checker_ok"]) and r["lean_ok"]:
            fn += 1
print("total answers", tot, "finished", fin, "checker true lean false", fp, "checker false lean true", fn)
# lean_ok None counts
print("lean_ok None count", sum(1 for c in cks for r in c["records"] if r["lean_ok"] is None))
decls = set()
decls_fin = set()
for c in cks:
    for r in c["records"]:
        decls.add(r["prompt"] + " " + r["answer"])
        if r["status"] == "finished":
            decls_fin.add(r["prompt"] + " " + r["answer"])
print("distinct (prompt, answer)", len(decls), "distinct finished", len(decls_fin))
decls2 = set((r["prompt"], r["answer"].strip()) for c in cks for r in c["records"] if r["status"] == "finished")
print("distinct finished stripped", len(decls2))
# status counts
for c in cks:
    print(c["label"], c["status_counts"]["all"])

print("\nSFT gain over pretrain", 100 * (ok["sft"].mean() - ok["pretrain"].mean()))
