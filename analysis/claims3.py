import json, sys
import numpy as np
sys.path.insert(0, "/home/user/nanoGPT")
from stats import paired_bootstrap, pass_at_k
P = json.load(open("results/passk.json"))
print("passk config", {k: P["config"][k] for k in ("n","passk_n","passk_samples","temperature","lean")})
ent = {c["label"]: c for c in P["checkpoints"]}
print("labels", list(ent))
sft = ent["sft"]["passk"]
doms = np.array([p["domain"] for p in sft["problems"]])
for k in (1, 16):
    base = np.array([pass_at_k(16, p["correct"], k) for p in sft["problems"]])
    for l, e in ent.items():
        v = np.array([pass_at_k(16, p["correct"], k) for p in e["passk"]["problems"]])
        m, lo, hi = paired_bootstrap(base, v, strata=doms)
        print(f"pass@{k} {l}: {v.mean()*100:.1f}  diff vs sft {m*100:+.1f} [{lo*100:+.1f},{hi*100:+.1f}]")
for l, e in ent.items():
    print(l, {d: round(r["pass@16"]*100,1) for d, r in e["passk"]["results"].items()})
# val gains excluding linear
import glob
sft_val = None
for name in ["rlhf_s1","rlhf_s2","rlhf_s3","ppochecker_s1","ppochecker_s2","ppochecker_s3"]:
    L = json.load(open(f"results/logs/{name}.json"))
    f, l = L["evals"][0]["results"], L["evals"][-1]["results"]
    gains = {d: round((l[d]["correct"] - f[d]["correct"]),1) for d in ("prop","fol","nat","linear","all")}
    print(name, "val gain in problems (of 75/domain, 300 all):", gains)
for name in ["rft_s1","rft_s2","rft_s3"]:
    L = json.load(open(f"results/logs/{name}.json"))
    f, l = L["evals"][0]["results"], L["rounds"][-1]["val"]
    print(name, "val gain in problems:", {d: (l[d]["correct"] - f[d]["correct"]) for d in ("prop","fol","nat","linear","all")})
