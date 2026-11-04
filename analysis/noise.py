import json, sys, math
import numpy as np
sys.path.insert(0, "/home/user/nanoGPT")
from stats import pass_at_k
A = "/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/"
C = {k: np.array(v) for k, v in json.load(open(A + "samples_lean_counts.json")).items()}
P = json.load(open("/home/user/nanoGPT/results/passk.json"))
dom = np.array([p["domain"] for p in P["checkpoints"][0]["passk"]["problems"]])
doms = ["prop", "fol", "nat", "linear"]
def pk(c, n, k): return np.array([pass_at_k(n, int(x), k) for x in c])
print("== SFT vs SFT (same model, different sampling seeds): the noise floor")
base = C["sft@31337"]
for other in ("sft@1", "sft@2"):
    o = C[other]
    g = int(((o > 0) & (base == 0)).sum()); l = int(((o == 0) & (base > 0)).sum())
    print(f"  {other} vs sft@31337: pass@1 {100*(pk(o,16,1).mean()-pk(base,16,1).mean()):+.1f}  pass@16 {100*((o>0).mean()-(base>0).mean()):+.1f}  coverage gained {g} lost {l}; "
          + " ".join(f"{d} {100*((o[dom==d]>0).mean()-(base[dom==d]>0).mean()):+.0f}" for d in doms))
pool = C["sft@31337"] + C["sft@1"] + C["sft@2"]   # 48 samples per problem
print("\n== arms vs SFT pooled over 3 draws (48 samples/problem; unbiased pass@k from 48)")
rng = np.random.default_rng(0)
strata = [np.flatnonzero(dom == d) for d in doms]
def boot(d):
    bs = [d[np.concatenate([rng.choice(s, len(s)) for s in strata])].mean() for _ in range(5000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
s1, s16 = pk(pool, 48, 1), pk(pool, 48, 16)
print(f"  SFT pooled: pass@1 {100*s1.mean():.1f} pass@16 {100*s16.mean():.1f}; by domain pass@16 " + " ".join(f"{d} {100*s16[dom==d].mean():.1f}" for d in doms))
print(f"  SFT single-draw pass@16 range over 3 seeds: " + ", ".join(f"{100*(C[k]>0).mean():.1f}" for k in ("sft@31337","sft@1","sft@2")))
for l in ("rlhf_s1", "ppochecker_s1", "rft_s1", "sftplus_s1"):
    a = C[l + "@31337"]
    a1, a16 = pk(a, 16, 1), (a > 0).astype(float)
    d1, d16 = a1 - s1, a16 - s16
    lo1, hi1 = boot(d1); lo16, hi16 = boot(d16)
    print(f"  {l:14} pass@1 {100*d1.mean():+5.1f} [{100*lo1:+.1f},{100*hi1:+.1f}]   pass@16 {100*d16.mean():+5.1f} [{100*lo16:+.1f},{100*hi16:+.1f}]   by domain pass@16: "
          + " ".join(f"{d} {100*d16[dom==d].mean():+.1f}" for d in doms))
