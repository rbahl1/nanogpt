import json, sys, re, itertools, collections
import numpy as np
sys.path.insert(0, "/home/user/nanoGPT")
import data, formal
from formal import prop as P, linear as L
src = open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/integrity.py").read()
ns = {}
exec(src[src.index("def prop_loose"):src.index("def loose")], {"P": P, "L": L, "N": formal.nat, "re": re, "itertools": itertools, "data": data}, ns)
prop_loose, shape, lin_norm = ns["prop_loose"], ns["shape"], ns["lin_norm"]
E = json.load(open("/home/user/nanoGPT/results/eval.json"))
ck = {c["label"]: c for c in E["checkpoints"]}
recs = ck["sft"]["records"]
sft = data.sample_problems(80000, "train", 2024, "all")
tr_prop = set(prop_loose(p.prompt) for p in sft if p.domain == "prop")
tr_fol = set(shape(p.prompt) for p in sft if p.domain == "fol")
tr_lin = collections.Counter(lin_norm(p.prompt) for p in sft if p.domain == "linear")
print("distinct linear normal forms (A,B) in SFT train:", len(tr_lin), " in test:", len(set(lin_norm(r["prompt"]) for r in recs if r["domain"] == "linear")))
nd = {}
for i, r in enumerate(recs):
    if r["domain"] == "prop": nd[i] = prop_loose(r["prompt"]) in tr_prop
    elif r["domain"] == "fol": nd[i] = shape(r["prompt"]) in tr_fol
for d in ("prop", "fol"):
    idx = [i for i in nd if recs[i]["domain"] == d]
    yes = [i for i in idx if nd[i]]; no = [i for i in idx if not nd[i]]
    print(f"\n{d}: {len(yes)} test problems with a near-duplicate in SFT train ({'renaming + AC of ∧∨↔ + premise order' if d=='prop' else 'same shape: identifiers and numerals erased'}), {len(no)} without")
    for l in ck:
        a = np.mean([ck[l]["records"][i]["lean_ok"] for i in yes]); b = np.mean([ck[l]["records"][i]["lean_ok"] for i in no])
        print(f"   {l:15} acc near-dup {100*a:5.1f}   others {100*b:5.1f}")
    # examples
    for i in yes[:3]: print("     e.g.", recs[i]["prompt"])
