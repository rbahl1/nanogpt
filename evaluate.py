"""Evaluates checkpoints on one fixed set of held-out problems.

    python evaluate.py --preset small out/pretrain.pt out/sft.pt out/rlhf.pt \
        [--n 500] [--lean true] [--passk_n 200] [--json results/eval.json]

Every checkpoint answers the same problems (n per domain from the test split,
no two with the same split key) greedily; an answer is correct if it finished
(ended in <eos>) and the domain's checker accepts it. The pretrained model has
never seen the task format, so it is given `<bos> header ` as the start of a
document and its continuation is the answer; every fine-tuned model gets
`<bos> header <ans>`.

--lean true also compiles every finished answer with Lean 4 and reports Lean's
accuracy beside the checker's, with the two kinds of disagreement: answers the
checker accepts and Lean rejects (checker false positives: a soundness bug)
and answers the checker rejects and Lean accepts (false negatives: the
checker's fragment of Lean is narrower than Lean). An unfinished answer counts
as wrong for both and is not compiled.

--passk_n k draws --passk_samples answers at --temperature for the first k
problems of each domain and reports the unbiased pass@k estimate (judged by
the checker) for k = 1, 2, 4, ... up to the number of samples.

--json writes everything, including one record per problem in the same order
for every checkpoint, which stats.py uses for paired comparisons.
"""

import json
import os
import time
from dataclasses import asdict

import torch

import data
import formal
import lean_oracle
from common import evaluate, generate_answers, get_device, load_checkpoint
from config import EvalConfig, parse_config
from stats import pass_at_k

STATUSES = ("finished", "cut off", "malformed")


def label_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def format_of(stage: str) -> str:
    return "document" if stage == "pretrain" else "task"


def sampled_pass_at_k(model, tokenizer, problems, fmt: str, config: EvalConfig, device: str) -> dict:
    """passk_samples answers per problem at config.temperature, judged by the
    checker. The generator is reseeded for every checkpoint, so all of them
    draw from the same random stream."""
    n = config.passk_samples
    prompts = [p for p in problems for _ in range(n)]
    generator = torch.Generator(device=device).manual_seed(config.seed)
    answers, statuses = generate_answers(model, tokenizer, prompts, fmt=fmt, max_new_tokens=config.max_new_tokens,
                                         greedy=False, temperature=config.temperature, generator=generator,
                                         batch_size=config.batch_size, statuses=True)
    correct = [0] * len(problems)
    for i, (p, a, s) in enumerate(zip(prompts, answers, statuses)):
        correct[i // n] += int(s == "finished" and formal.verify(p.prompt, a))
    ks = [k for k in (1, 2, 4, 8, 16) if k <= n]
    out = {}
    for key in [d for d in formal.DOMAINS if any(p.domain == d for p in problems)] + ["all"]:
        counts = [c for p, c in zip(problems, correct) if key in (p.domain, "all")]
        out[key] = {f"pass@{k}": sum(pass_at_k(n, c, k) for c in counts) / len(counts) for k in ks}
        out[key]["problems"] = len(counts)
    return {"samples": n, "temperature": config.temperature, "results": out,
            "problems": [{"domain": p.domain, "prompt": p.prompt, "correct": c} for p, c in zip(problems, correct)]}


def add_lean_verdicts(entries, lean: str) -> float:
    """Sets lean_ok on every record. Declarations are pooled across checkpoints
    and deduplicated (checkpoints often give the same answer), and compiled one
    Lean process per chunk. Returns the seconds spent."""
    started = time.time()
    decls = list(dict.fromkeys(r["prompt"] + " " + r["answer"] for e in entries for r in e["records"]
                               if r["status"] == "finished"))
    print(f"compiling {len(decls):,} distinct declarations with Lean...", flush=True)
    verdict = dict(zip(decls, lean_oracle.check_in_chunks(decls, lean)))
    for e in entries:
        for r in e["records"]:
            r["lean_ok"] = r["status"] == "finished" and verdict[r["prompt"] + " " + r["answer"]]
    return time.time() - started


def summarise(records, with_lean: bool):
    """(results, status_counts): per domain and "all", accuracy (and Lean's,
    with the disagreement counts) and how answers ended."""
    results, statuses = {}, {}
    for r in records:
        for key in (r["domain"], "all"):
            s = results.setdefault(key, {"n": 0, "correct": 0})
            s["n"] += 1
            s["correct"] += int(r["checker_ok"])
            if with_lean:
                s["lean_correct"] = s.get("lean_correct", 0) + int(r["lean_ok"])
                s["checker_false_pos"] = s.get("checker_false_pos", 0) + int(r["checker_ok"] and not r["lean_ok"])
                s["checker_false_neg"] = s.get("checker_false_neg", 0) + int(r["lean_ok"] and not r["checker_ok"])
            c = statuses.setdefault(key, dict.fromkeys(STATUSES, 0))
            c[r["status"]] += 1
    for s in results.values():
        s["accuracy"] = s["correct"] / s["n"]
        if with_lean:
            s["lean_accuracy"] = s["lean_correct"] / s["n"]
    order = [d for d in formal.DOMAINS if d in results] + ["all"]
    return {k: results[k] for k in order}, {k: statuses[k] for k in order}


def table(entries, with_lean: bool) -> str:
    keys = list(entries[0]["results"])
    width = max(12, max(len(e["label"]) for e in entries) + 2)
    lines = ["checker accuracy (greedy)", f"{'checkpoint':<{width}}{'stage':<15}" + "".join(f"{k:>9}" for k in keys)
             + f"{'cut off':>9}{'malformed':>10}"]
    for e in entries:
        st = e["status_counts"]["all"]
        lines.append(f"{e['label']:<{width}}{e['stage']:<15}"
                     + "".join(f"{e['results'][k]['accuracy']:>9.3f}" for k in keys)
                     + f"{st['cut off']:>9}{st['malformed']:>10}")
    if with_lean:
        lines += ["", "Lean accuracy (every finished answer compiled); checker vs Lean disagreements over all domains",
                  f"{'checkpoint':<{width}}{'stage':<15}" + "".join(f"{k:>9}" for k in keys)
                  + f"{'chk+ Lean-':>12}{'chk- Lean+':>12}"]
        for e in entries:
            a = e["results"]["all"]
            lines.append(f"{e['label']:<{width}}{e['stage']:<15}"
                         + "".join(f"{e['results'][k]['lean_accuracy']:>9.3f}" for k in keys)
                         + f"{a['checker_false_pos']:>12}{a['checker_false_neg']:>12}")
    sampled = [e for e in entries if e["passk"]]
    if sampled:
        ks = [k for k in sampled[0]["passk"]["results"]["all"] if k.startswith("pass@")]
        p = sampled[0]["passk"]
        lines += ["", f"pass@k ({p['results']['all']['problems']} problems, {p['samples']} samples each at "
                      f"temperature {p['temperature']}, checker-judged)",
                  f"{'checkpoint':<{width}}{'domain':<9}" + "".join(f"{k:>9}" for k in ks)]
        for e in sampled:
            for d, r in e["passk"]["results"].items():
                lines.append(f"{e['label']:<{width}}{d:<9}" + "".join(f"{r[k]:>9.3f}" for k in ks))
    return "\n".join(lines)


def main(argv=None):
    config, preset, paths = parse_config(EvalConfig, argv, positional="checkpoints")
    device = get_device(config.device)
    domains = formal.resolve(config.domains)
    labels = [label_of(p) for p in paths]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"checkpoints are labelled by file name, which must differ: {labels}")
    if config.passk_n > config.n:
        raise SystemExit(f"--passk_n ({config.passk_n}) is drawn from the evaluated problems and cannot exceed "
                         f"--n ({config.n})")
    lean = None
    if config.lean:
        lean = lean_oracle.find_lean()
        if lean is None:
            raise SystemExit("--lean true needs Lean 4: set LEAN to the lean binary or put it on PATH")

    problems = data.sample_problems(config.n * len(domains), config.split, config.seed, domains, unique=True)
    by_domain = {d: [p for p in problems if p.domain == d] for d in domains}
    passk_problems = [p for d in domains for p in by_domain[d][:config.passk_n]]
    print(f"{len(problems)} {config.split} problems ({config.n} per domain: {', '.join(domains)}), "
          f"{len(paths)} checkpoints, device {device}", flush=True)

    entries = []
    for path, label in zip(paths, labels):
        started = time.time()
        model, tokenizer, ckpt = load_checkpoint(path, device, kind="policy")
        fmt = format_of(ckpt["stage"])
        res = evaluate(model, tokenizer, problems, fmt=fmt, max_new_tokens=config.max_new_tokens,
                       batch_size=config.batch_size, records=True)
        records = [{"domain": r["domain"], "prompt": r["prompt"], "answer": r["answer"], "status": r["status"],
                    "checker_ok": r["ok"], "lean_ok": None} for r in res["records"]]
        passk = sampled_pass_at_k(model, tokenizer, passk_problems, fmt, config, device) if passk_problems else None
        entries.append({"label": label, "path": path, "stage": ckpt["stage"], "format": fmt,
                        "run_config": ckpt.get("run_config", {}), "passk": passk, "records": records})
        print(f"{label}: checker accuracy {res['all']['accuracy']:.3f} ({time.time() - started:.0f}s)", flush=True)

    if lean:
        print(f"Lean: {add_lean_verdicts(entries, lean):.0f}s", flush=True)
    for e in entries:
        e["results"], e["status_counts"] = summarise(e["records"], with_lean=lean is not None)

    print()
    print(table(entries, with_lean=lean is not None))

    if config.json:
        out = {"config": asdict(config), "preset": preset, "domains": domains,
               "checkpoints": [{k: e[k] for k in ("label", "path", "stage", "format", "run_config", "results",
                                                  "status_counts", "passk", "records")} for e in entries]}
        os.makedirs(os.path.dirname(os.path.abspath(config.json)), exist_ok=True)
        with open(config.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {config.json}")
    return entries


if __name__ == "__main__":
    main()
