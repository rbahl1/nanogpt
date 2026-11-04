import json, sys, math
import numpy as np
sys.path.insert(0, "/home/user/nanoGPT")
from stats import pass_at_k

P = json.load(open("/home/user/nanoGPT/results/passk.json"))
E = json.load(open("/home/user/nanoGPT/results/eval.json"))
N = P["config"]["passk_samples"]
doms = P["domains"]
cks = {c["label"]: c for c in P["checkpoints"]}
labels = list(cks)
print("config", {k: P["config"][k] for k in ("n","split","passk_n","passk_samples","temperature","lean","seed")})

# problems identical across checkpoints?
probs0 = [(p["domain"], p["prompt"]) for p in cks["sft"]["passk"]["problems"]]
for l in labels:
    assert [(p["domain"], p["prompt"]) for p in cks[l]["passk"]["problems"]] == probs0, l
# same as first 100 per domain of eval.json?
ev = {c["label"]: c for c in E["checkpoints"]}
erec = ev["sft"]["records"]
first = {d: [r["prompt"] for r in erec if r["domain"] == d][:100] for d in doms}
print("passk problems == first 100/domain of eval.json:",
      all([p for dd, p in probs0 if dd == d] == first[d] for d in doms))

counts = {l: np.array([p["correct"] for p in cks[l]["passk"]["problems"]]) for l in labels}
dom = np.array([d for d, _ in probs0])

def pk(c, k):
    return np.array([pass_at_k(N, int(x), k) for x in c])

# reproduce stored values
for l in labels:
    for key, r in cks[l]["passk"]["results"].items():
        m = dom == key if key != "all" else np.ones(len(dom), bool)
        for k in (1, 2, 4, 8, 16):
            v = pk(counts[l][m], k).mean()
            assert abs(v - r[f"pass@{k}"]) < 1e-9, (l, key, k, v, r[f"pass@{k}"])
print("stored pass@{1,2,4,8,16} reproduced from per-problem counts: OK")

ks = list(range(1, 17))
out = {}
for key in doms + ["all"]:
    m = dom == key if key != "all" else np.ones(len(dom), bool)
    print(f"\n== {key} (n={m.sum()}) pass@k, %")
    print("ckpt".ljust(15) + "".join(f"{k:>6}" for k in ks))
    for l in labels:
        row = [100 * pk(counts[l][m], k).mean() for k in ks]
        out[(l, key)] = row
        print(l.ljust(15) + "".join(f"{v:6.1f}" for v in row))

# greedy accuracy on these 400 problems (from eval.json, Lean & checker) and from passk.json records (checker)
print("\n== greedy on the same 400 problems")
def greedy(label):
    recs = ev[label]["records"]
    idx = []
    cnt = {d: 0 for d in doms}
    for i, r in enumerate(recs):
        if cnt[r["domain"]] < 100:
            idx.append(i); cnt[r["domain"]] += 1
    sel = [recs[i] for i in idx]
    by = {r["prompt"]: r for r in sel}
    return [by[p] for _, p in probs0]   # aligned with passk problem order
for l in labels:
    g = greedy(l)
    byp = {r["prompt"]: r for r in cks[l]["records"]}
    gp = [byp[p] for _, p in probs0]
    same_ans = sum(a["answer"] == b["answer"] for a, b in zip(g, gp))
    lean = np.array([r["lean_ok"] for r in g], float)
    chk = np.array([r["checker_ok"] for r in g], float)
    chkp = np.array([r["checker_ok"] for r in gp], float)
    per = {d: 100 * lean[dom == d].mean() for d in doms}
    print(l.ljust(15), f"eval.json greedy Lean {100*lean.mean():.1f} checker {100*chk.mean():.1f}; passk.json greedy checker {100*chkp.mean():.1f}; identical greedy answers {same_ans}/400;",
          " ".join(f"{d} {v:.0f}" for d, v in per.items()))

# distribution of per-problem correct counts
print("\n== distribution of c (correct of 16) per problem, all domains")
bins = [(0,0),(1,1),(2,3),(4,7),(8,11),(12,15),(16,16)]
print("ckpt".ljust(15) + "".join(f"{f'{a}-{b}' if a!=b else str(a):>7}" for a,b in bins) + "  mean c")
for l in labels:
    c = counts[l]
    print(l.ljust(15) + "".join(f"{((c>=a)&(c<=b)).sum():7d}" for a,b in bins) + f"  {c.mean():.2f}")

# paired comparisons vs SFT: pass@1 diff, pass@16 diff (coverage), bootstrap CI stratified by domain
rng = np.random.default_rng(0)
B = 10000
strata = [np.flatnonzero(dom == d) for d in doms]
def boot(a, b, m=None):
    idx_all = np.arange(len(a)) if m is None else np.flatnonzero(m)
    d = b - a
    if m is None:
        bs = np.empty(B)
        for i in range(B):
            s = np.concatenate([rng.choice(st, len(st)) for st in strata])
            bs[i] = d[s].mean()
    else:
        bs = np.array([d[rng.choice(idx_all, len(idx_all))].mean() for _ in range(B)])
    return d[idx_all].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

from math import comb
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)

print("\n== vs SFT: pass@1 and pass@16 (coverage) differences, 400 problems, points [95% bootstrap CI stratified by domain]")
s1 = pk(counts["sft"], 1); s16 = (counts["sft"] > 0).astype(float)
for l in labels[1:]:
    a1 = pk(counts[l], 1); a16 = (counts[l] > 0).astype(float)
    d1 = boot(s1, a1); d16 = boot(s16, a16)
    gained = int(((counts[l] > 0) & (counts["sft"] == 0)).sum()); lost = int(((counts[l] == 0) & (counts["sft"] > 0)).sum())
    # sign test on per-problem c
    up = int((counts[l] > counts["sft"]).sum()); dn = int((counts[l] < counts["sft"]).sum())
    # paired t on c
    dc = counts[l] - counts["sft"]
    t = dc.mean() / (dc.std(ddof=1) / math.sqrt(len(dc)))
    print(f"{l:15} pass@1 {100*d1[0]:+5.1f} [{100*d1[1]:+5.1f},{100*d1[2]:+5.1f}]  pass@16 {100*d16[0]:+5.1f} [{100*d16[1]:+5.1f},{100*d16[2]:+5.1f}]"
          f"  coverage gained {gained} lost {lost} (McNemar p={mcnemar(gained, lost):.3f})  c up/down {up}/{dn}, paired t on c {t:+.2f}")
    for d in doms:
        m = dom == d
        g = int(((counts[l] > 0) & (counts["sft"] == 0) & m).sum()); lo = int(((counts[l] == 0) & (counts["sft"] > 0) & m).sum())
        print(f"     {d:7} pass@1 {100*(a1[m].mean()-s1[m].mean()):+5.1f}  pass@16 {100*(a16[m].mean()-s16[m].mean()):+5.1f}  gained {g} lost {lo} (p={mcnemar(g, lo):.3f})")

# Where do arms lose coverage: SFT's c on lost problems
print("\n== SFT's correct count on problems an arm stops covering, and arm's c on problems SFT doesn't cover")
for l in labels[1:]:
    lostc = counts["sft"][(counts[l] == 0) & (counts["sft"] > 0)]
    gainc = counts[l][(counts[l] > 0) & (counts["sft"] == 0)]
    print(f"{l:15} lost: SFT c = {sorted(lostc.tolist())}   gained: arm c = {sorted(gainc.tolist())}")

# Mass on rare problems: for problems with SFT c in 1..3, arm mean c
print("\n== arm mean c, by SFT's c bucket (sharpening = up on easy, down on hard)")
buckets = [(0,0),(1,2),(3,5),(6,10),(11,15),(16,16)]
print("ckpt".ljust(15) + "".join(f"{f'sft {a}-{b}':>11}" for a,b in buckets))
print("n".ljust(15) + "".join(f"{((counts['sft']>=a)&(counts['sft']<=b)).sum():>11d}" for a,b in buckets))
for l in labels:
    print(l.ljust(15) + "".join(f"{counts[l][(counts['sft']>=a)&(counts['sft']<=b)].mean():>11.2f}" for a,b in buckets))

# Greedy vs pass@16 headroom for SFT
g = greedy("sft"); lean = np.array([r["lean_ok"] for r in g], float)
print("\n== SFT headroom on the 400: greedy Lean %.1f, sampled pass@1 %.1f, pass@16 %.1f; headroom pass@16 - greedy = %.1f pts" %
      (100*lean.mean(), 100*s1.mean(), 100*s16.mean(), 100*(s16.mean()-lean.mean())))
for d in doms:
    m = dom == d
    print(f"   {d:7} greedy {100*lean[m].mean():5.1f}  pass@1 {100*s1[m].mean():5.1f}  pass@16 {100*s16[m].mean():5.1f}  headroom {100*(s16[m].mean()-lean[m].mean()):+5.1f}")
# greedy-correct but never sampled correct, and vice versa
gc = lean.astype(bool); cov = counts["sft"] > 0
print("   greedy right & 0/16 sampled:", int((gc & ~cov).sum()), " greedy wrong & >=1/16 sampled:", int((~gc & cov).sum()))
for l in labels[1:]:
    gl = np.array([r["lean_ok"] for r in greedy(l)], bool)
    print(f"   {l:15} greedy {100*gl.mean():.1f} pass@1 {100*pk(counts[l],1).mean():.1f} pass@16 {100*(counts[l]>0).mean():.1f} headroom {100*((counts[l]>0).mean()-gl.mean()):.1f}")

# full 2000 SFT greedy, for reference
allg = np.array([r["lean_ok"] for r in ev["sft"]["records"]], float)
print("   SFT greedy Lean on all 2000:", 100 * allg.mean())
json.dump({f"{l}|{k}": v for (l, k), v in out.items()}, open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/passk_table.json", "w"))
