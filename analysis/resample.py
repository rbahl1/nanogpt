"""Regenerate the pass@k samples (same seed as results/passk.json) keeping the answers,
plus two extra SFT draws with other seeds as a sampling-noise floor."""
import json, sys, time
sys.path.insert(0, "/home/user/nanoGPT")
import torch
import data, formal
from common import generate_answers, load_checkpoint
torch.set_num_threads(1)
P = json.load(open("/home/user/nanoGPT/results/passk.json"))
probs = [(p["domain"], p["prompt"]) for p in P["checkpoints"][0]["passk"]["problems"]]
problems = data.sample_problems(400, "test", 31337, "all", unique=True)
by = {d: [p for p in problems if p.domain == d] for d in P["domains"]}
pk_problems = [p for d in P["domains"] for p in by[d][:100]]
assert [(p.domain, p.prompt) for p in pk_problems] == probs
OUT = "/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/samples.json"
runs = [("sft", 31337), ("sft", 1), ("sft", 2), ("rlhf_s1", 31337), ("ppochecker_s1", 31337), ("rft_s1", 31337), ("sftplus_s1", 31337)]
res = {}
for label, seed in runs:
    t = time.time()
    model, tok, ck = load_checkpoint(f"/home/user/nanoGPT/out/{label}.pt", "cpu", kind="policy")
    prompts = [p for p in pk_problems for _ in range(16)]
    g = torch.Generator(device="cpu").manual_seed(seed)
    ans, st = generate_answers(model, tok, prompts, fmt="task", max_new_tokens=200, greedy=False, temperature=1.0,
                               generator=g, batch_size=64, statuses=True)
    ok = [s == "finished" and formal.verify(p.prompt, a) for p, a, s in zip(prompts, ans, st)]
    counts = [sum(ok[i * 16:(i + 1) * 16]) for i in range(len(pk_problems))]
    res[f"{label}@{seed}"] = {"answers": ans, "status": st, "checker": ok, "counts": counts}
    stored = None
    if seed == 31337:
        stored = [p["correct"] for c in P["checkpoints"] if c["label"] == label for p in c["passk"]["problems"]]
        same = sum(a == b for a, b in zip(stored, counts))
    print(f"{label}@{seed}: {time.time()-t:.0f}s, checker-correct samples {sum(ok)}/6400"
          + (f", per-problem counts identical to passk.json: {same}/400" if stored else ""), flush=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False)
