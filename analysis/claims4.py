import json, sys, collections
sys.path.insert(0, "/home/user/nanoGPT")
import data
probs = data.sample_problems(6000, "train", 7, "all", unique=True)
ref = {p.prompt: p.reference for p in probs}
pairs = [json.loads(l) for l in open("out/preferences.jsonl")]
print("pairs", len(pairs), "prompts in prefs", len({p['prompt'] for p in pairs}), "matched", sum(p["prompt"] in ref for p in pairs))
c = collections.Counter()
for p in pairs:
    r = ref.get(p["prompt"])
    kind = "ref chosen" if p["chosen"] == r else ("ref rejected" if p["rejected"] == r else "both samples")
    c[(p["domain"], kind)] += 1
    c[("all", kind)] += 1
for k in sorted(c): print(k, c[k])
