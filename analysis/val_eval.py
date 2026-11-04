"""Greedy val evaluation (the 300 val problems the training runs used) of sft and the 12 arm checkpoints,
per-problem records, checker + Lean verdicts. Writes val_records.json in this directory."""
import json, os, sys, time
sys.path.insert(0, "/home/user/nanoGPT")
os.chdir("/home/user/nanoGPT")
import torch
torch.set_num_threads(1)
import data, formal, lean_oracle
from common import evaluate, load_checkpoint

OUT = "/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/val_records.json"
labels = ["sft"] + [f"{a}_s{s}" for a in ("rlhf", "ppochecker", "rft", "sftplus") for s in (1, 2, 3)]
problems = data.sample_problems(75 * 4, "val", 0, unique=True)
print(len(problems), [p.domain for p in problems[:8]], flush=True)
res = {}
if os.path.exists(OUT):
    res = json.load(open(OUT))["checkpoints"]
for lab in labels:
    if lab in res:
        continue
    t = time.time()
    model, tok, ckpt = load_checkpoint(f"out/{lab}.pt", "cpu", kind="policy")
    r = evaluate(model, tok, problems, fmt="task", max_new_tokens=200, batch_size=64, records=True)
    res[lab] = {"stage": ckpt["stage"], "iteration": ckpt.get("iteration"), "saved_val": ckpt.get("val"),
                "records": [{"domain": x["domain"], "prompt": x["prompt"], "answer": x["answer"],
                             "status": x["status"], "checker_ok": bool(x["ok"])} for x in r["records"]]}
    print(lab, f"{r['all']['accuracy']:.4f}", {k: v["correct"] for k, v in r.items() if k != "records"},
          f"{time.time()-t:.0f}s", flush=True)
    json.dump({"checkpoints": res}, open(OUT, "w"), ensure_ascii=False)
lean = "/opt/lean/lean-4.34.0-linux/bin/lean"
decls = list(dict.fromkeys(x["prompt"] + " " + x["answer"] for v in res.values() for x in v["records"]
                           if x["status"] == "finished"))
print("compiling", len(decls), flush=True)
verdict = dict(zip(decls, lean_oracle.check_in_chunks(decls, lean)))
for v in res.values():
    for x in v["records"]:
        x["lean_ok"] = bool(x["status"] == "finished" and verdict[x["prompt"] + " " + x["answer"]])
json.dump({"checkpoints": res}, open(OUT, "w"), ensure_ascii=False)
print("done", flush=True)
