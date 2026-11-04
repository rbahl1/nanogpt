import json, sys, time, math, collections
import numpy as np
sys.path.insert(0, "/home/user/nanoGPT")
import lean_oracle
from stats import pass_at_k
A = "/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/"
S = json.load(open(A + "samples.json"))
P = json.load(open("/home/user/nanoGPT/results/passk.json"))
probs = [(p["domain"], p["prompt"]) for p in P["checkpoints"][0]["passk"]["problems"]]
prompts = [p for _, p in probs for _ in range(16)]
dom = np.array([d for d, _ in probs])
decls = list(dict.fromkeys(pr + " " + a for run in S.values() for pr, a, s in zip(prompts, run["answers"], run["status"]) if s == "finished"))
print("runs", list(S), "distinct finished sampled declarations", len(decls), flush=True)
t = time.time()
v = dict(zip(decls, lean_oracle.check_in_chunks(decls, "/opt/lean/lean-4.34.0-linux/bin/lean", chunk=1500)))
print(f"Lean {time.time()-t:.0f}s")
out = {}
for name, run in S.items():
    lean = [s == "finished" and v[pr + " " + a] for pr, a, s in zip(prompts, run["answers"], run["status"])]
    chk = run["checker"]
    fp = [(pr, a) for pr, a, c, l in zip(prompts, run["answers"], chk, lean) if c and not l]
    fn = [(pr, a) for pr, a, c, l in zip(prompts, run["answers"], chk, lean) if l and not c]
    counts = np.array([sum(lean[i * 16:(i + 1) * 16]) for i in range(len(probs))])
    p1 = np.mean([pass_at_k(16, int(c), 1) for c in counts]); p16 = np.mean(counts > 0)
    cc = np.array(run["counts"])
    st = collections.Counter(run["status"])
    print(f"{name:20} samples {len(lean)} status {dict(st)}; checker+Lean- {len(fp)}, checker-Lean+ {len(fn)}; "
          f"Lean pass@1 {100*p1:.2f} pass@16 {100*p16:.1f} (checker pass@1 {100*np.mean([pass_at_k(16,int(c),1) for c in cc]):.2f} pass@16 {100*np.mean(cc>0):.1f})")
    for pr, a in fp[:5]: print("    FP:", pr, a)
    for pr, a in list(dict.fromkeys(fn))[:8]: print("    FN:", pr, repr(a))
    out[name] = counts.tolist()
json.dump(out, open(A + "samples_lean_counts.json", "w"))
