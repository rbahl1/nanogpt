# Estimate: in how many preference pairs was the chosen answer (the reference) among the SFT model's own samples?
import json, sys, random, collections
sys.path.insert(0, "/home/user/nanoGPT")
import torch, data, common
probs = data.sample_problems(6000, "train", 7, "all", unique=True)
ref = {p.prompt: p for p in probs}
pairs = [json.loads(l) for l in open("out/preferences.jsonl")]
byprompt = collections.defaultdict(list)
for q in pairs: byprompt[q["prompt"]].append(q)
rng = random.Random(0)
sel = []
for d in ("prop","fol","nat","linear"):
    ps = [p for p in byprompt if byprompt[p][0]["domain"] == d]
    sel += rng.sample(ps, 150)
model, tok, _ = common.load_checkpoint("out/sft.pt", "cpu", kind="policy")
flat = [ref[p] for p in sel for _ in range(8)]
g = torch.Generator().manual_seed(123)
ans, st = common.generate_answers(model, tok, flat, max_new_tokens=200, greedy=False, temperature=1.0, generator=g, batch_size=128, statuses=True)
w = collections.Counter(); tot = collections.Counter()
for i, p in enumerate(sel):
    samples = set(a for a, s in zip(ans[i*8:(i+1)*8], st[i*8:(i+1)*8]) if s == "finished")
    d = byprompt[p][0]["domain"]
    n_refpairs = sum(q["chosen"] == ref[p].reference for q in byprompt[p])
    tot[d] += n_refpairs
    if ref[p].reference in samples: w[d] += n_refpairs
for d in tot: print(d, f"pairs with reference chosen whose reference also appears among 8 fresh SFT samples: {w[d]/tot[d]*100:.0f}%")
print("all", f"{sum(w.values())/sum(tot.values())*100:.0f}%")
