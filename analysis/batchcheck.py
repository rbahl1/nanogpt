import json, sys, os
sys.path.insert(0, "/home/user/nanoGPT"); os.chdir("/home/user/nanoGPT")
import torch; torch.set_num_threads(1)
import data
from common import evaluate, load_checkpoint
A = "/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/"
V = json.load(open(A + "val_records.json"))["checkpoints"]
problems = data.sample_problems(300, "val", 0, unique=True)
model, tok, _ = load_checkpoint("out/sft.pt", "cpu", kind="policy")
for bs in (1, 300):
    r = evaluate(model, tok, problems, max_new_tokens=200, batch_size=bs, records=True)
    same = sum(a["answer"] == b["answer"] for a, b in zip(r["records"], V["sft"]["records"]))
    print("batch", bs, "correct", r["all"]["correct"], {k: v["correct"] for k, v in r.items() if k not in ("records", "all")}, "identical answers to batch-64 run:", same, "/300", flush=True)
# a test subset with batch 1: first 100 linear + 100 prop test problems
T = json.load(open("results/eval.json"))["checkpoints"][1]
tp = data.sample_problems(2000, "test", 31337, unique=True)
assert [p.prompt for p in tp] == [x["prompt"] for x in T["records"]]
sub = [i for i, p in enumerate(tp) if p.domain == "linear"][:200]
r = evaluate(model, tok, [tp[i] for i in sub], max_new_tokens=200, batch_size=1, records=True)
same = sum(a["answer"] == T["records"][i]["answer"] for a, i in zip(r["records"], sub))
print("test linear first 200, batch 1: correct", r["all"]["correct"], "vs eval.json checker", sum(T["records"][i]["checker_ok"] for i in sub), "identical answers", same, "/200")
