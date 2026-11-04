"""The supervised pipeline end to end on the tiny preset, restricted to the nat
and linear domains: pretrain -> SFT -> continued SFT (two data seeds) ->
evaluate (every answer compiled with Lean when a toolchain is available) ->
stats.py -> export_web.py, with web/gpt.js under Node reproducing the
exported model. Everything is written under a temporary directory."""

import json
import os
import subprocess
import sys

import pytest
import torch

import data
import evaluate
import export_web
import lean_oracle
import pretrain
import sample
import sft
from common import load_checkpoint, lr_at
from stats import pass_at_k

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["--domains", "nat,linear", "--device", "cpu"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    # One thread: the machine is shared, and with busy cores PyTorch's spinning
    # worker threads slow a model this small down by orders of magnitude.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    out = tmp_path_factory.mktemp("pipeline")
    paths = {name: str(out / f"{name}.pt") for name in ("pretrain", "sft", "sftplus_s1", "sftplus_s2")}
    try:
        pretrain.main(["--preset", "tiny", *DOMAINS, "--out", paths["pretrain"]])
        sft.main(["--preset", "tiny", *DOMAINS, "--init", paths["pretrain"], "--out", paths["sft"]])
        for seed in (1, 2):
            sft.main(["--preset", "tiny", *DOMAINS, "--init", paths["sft"], "--out", paths[f"sftplus_s{seed}"],
                      "--max_iters", "30", "--warmup_iters", "3", "--lr", "1e-4", "--eval_interval", "15",
                      "--seed", str(1000 + seed)])
        lean = lean_oracle.find_lean()
        eval_json = str(out / "eval.json")
        evaluate.main(["--preset", "tiny", *DOMAINS, *paths.values(), "--lean", str(lean is not None),
                       "--json", eval_json])
        yield {"out": out, "paths": paths, "eval": eval_json, "lean": lean}
    finally:
        torch.set_num_threads(threads)


def test_pretrain_checkpoint(run):
    model, _, ckpt = load_checkpoint(run["paths"]["pretrain"])
    assert ckpt["stage"] == "pretrain" and ckpt["kind"] == "policy"
    assert ckpt["run_config"]["domains"] == "nat,linear" and ckpt["run_config"]["n_layer"] == 2
    history = ckpt["history"]
    assert [h["iter"] for h in history] == [50, 100, 150]
    # from ln(121) ~ 4.8 at initialisation to well below it, on held-out documents too
    assert history[-1]["val_loss"] < 2.0 and history[-1]["val_loss"] < history[0]["val_loss"]
    assert all(h["tokens_per_s"] > 0 for h in history)


def test_lr_schedule():
    """Linear warmup to the peak, then a cosine that reaches min_lr at the end."""
    lr, min_lr, warmup, total = 1e-3, 1e-4, 10, 110
    lrs = [lr_at(it, lr, min_lr, warmup, total) for it in range(total + 5)]
    assert lrs[0] == pytest.approx(lr / warmup) and lrs[warmup - 1] == pytest.approx(lr)
    assert all(a < b for a, b in zip(lrs[:warmup - 1], lrs[1:warmup]))
    assert lrs[warmup] == pytest.approx(lr)
    assert all(a >= b for a, b in zip(lrs[warmup:total], lrs[warmup + 1:total + 1]))
    assert lrs[warmup + (total - warmup) // 2] == pytest.approx((lr + min_lr) / 2)
    assert lrs[total - 1] == pytest.approx(min_lr, rel=1e-2) and lrs[total:] == [min_lr] * 5


def test_pretrain_refuses_documents_longer_than_the_context(tmp_path):
    with pytest.raises(ValueError, match="longer than the context"):
        pretrain.main(["--preset", "tiny", *DOMAINS, "--block_size", "32", "--n_docs", "200",
                       "--out", str(tmp_path / "never.pt")])
    assert not (tmp_path / "never.pt").exists()


def test_sft_checkpoints(run):
    _, _, base = load_checkpoint(run["paths"]["sft"])
    assert base["stage"] == "sft" and base["init"]["stage"] == "pretrain"
    assert [h["iter"] for h in base["history"]] == [50, 100]
    for h in base["history"]:
        assert list(h["val"]) == ["nat", "linear", "all"] and h["val"]["all"]["n"] == 10
    assert base["history"][-1]["train_loss"] < base["history"][0]["train_loss"]

    continued = [load_checkpoint(run["paths"][f"sftplus_s{s}"]) for s in (1, 2)]
    for model, _, ckpt in continued:
        assert ckpt["stage"] == "sft-continued"
        assert ckpt["init"] == {"path": run["paths"]["sft"], "stage": "sft", "run_config": base["run_config"]}
        assert [h["iter"] for h in ckpt["history"]] == [15, 30]
    # the data seed decides which fresh examples each run sees
    w1, w2 = (m.transformer.wte.weight for m, _, _ in continued)
    assert not torch.equal(w1, w2)
    with pytest.raises(ValueError):
        sft.stage_after("rlhf")


def test_evaluation_json(run):
    with open(run["eval"], encoding="utf-8") as f:
        entries = json.load(f)["checkpoints"]
    assert [e["label"] for e in entries] == ["pretrain", "sft", "sftplus_s1", "sftplus_s2"]
    assert [e["stage"] for e in entries] == ["pretrain", "sft", "sft-continued", "sft-continued"]
    assert [e["format"] for e in entries] == ["document", "task", "task", "task"]
    problems = [(r["domain"], r["prompt"]) for r in entries[0]["records"]]
    assert len(problems) == 20 and len(set(problems)) == 20
    assert {data.split_of(p) for _, p in problems} == {"test"}
    assert len({data.split_key(p) for _, p in problems}) == 20
    assert {d for d, _ in problems} == {"nat", "linear"}
    for e in entries:
        assert [(r["domain"], r["prompt"]) for r in e["records"]] == problems
        assert list(e["results"]) == ["nat", "linear", "all"]
        assert e["results"]["all"]["correct"] == sum(r["checker_ok"] for r in e["records"])
        assert sum(e["status_counts"]["all"].values()) == 20
        assert all(r["checker_ok"] <= (r["status"] == "finished") for r in e["records"])
        # pass@k: 2 problems per domain, 4 samples each
        passk = e["passk"]
        assert passk["samples"] == 4 and len(passk["problems"]) == 4
        assert list(passk["results"]["all"]) == ["pass@1", "pass@2", "pass@4", "problems"]
        counts = [p["correct"] for p in passk["problems"]]
        assert passk["results"]["all"]["pass@2"] == pytest.approx(sum(pass_at_k(4, c, 2) for c in counts) / 4)


def test_lean_agrees_with_the_checker(run):
    if run["lean"] is None:
        pytest.skip("no Lean 4 toolchain (set LEAN or put lean on PATH)")
    with open(run["eval"], encoding="utf-8") as f:
        entries = json.load(f)["checkpoints"]
    for e in entries:
        assert all(isinstance(r["lean_ok"], bool) for r in e["records"])
        assert all(r["lean_ok"] <= (r["status"] == "finished") for r in e["records"])
        all_ = e["results"]["all"]
        assert all_["lean_correct"] == sum(r["lean_ok"] for r in e["records"])
        # the checkers are sound: nothing they accept is rejected by Lean
        assert all_["checker_false_pos"] == 0, [r for r in e["records"] if r["checker_ok"] and not r["lean_ok"]]


def test_stats_cli(run):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "stats.py"), run["eval"], "--baseline", "sft",
                           "--n_boot", "1000"], capture_output=True, text=True, check=True, cwd=ROOT)
    out = proc.stdout
    assert ("Lean 4" if run["lean"] else "checker") in out.splitlines()[0]
    for label in ("pretrain", "sft", "sftplus_s1", "sftplus_s2"):
        assert f"== {label} (stage" in out
    assert "== sftplus: 2 seeds (sftplus_s1, sftplus_s2) ==" in out


def test_export_web(run, node):
    dist, art = run["out"] / "dist", run["out"] / "artifact"
    export_web.main([run["paths"]["sft"], "--out", str(dist)])
    export_web.main([run["paths"]["sft"], "--out", str(art), "--artifact"])
    model, tokenizer, ckpt = load_checkpoint(run["paths"]["sft"])
    with open(dist / "model.json", encoding="utf-8") as f:
        bundle = json.load(f)
    assert bundle["stage"] == "sft" and bundle["params"] == model.num_params()
    assert bundle["trained_domains"] == ["nat", "linear"] and len(bundle["examples"]) == 12
    assert "lm_head.weight" not in bundle["tensors"]
    reloaded = export_web.model_from_bundle(bundle)
    for k, v in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[k], v)

    # web/gpt.js reproduces the exported model
    ids, _ = tokenizer.example_ids(bundle["examples"][0]["prompt"], "12 := by decide")
    proc = subprocess.run([node, os.path.join(ROOT, "tests", "js_logits.js"), str(dist / "model.json"),
                           json.dumps(ids)], capture_output=True, text=True, check=True)
    with torch.no_grad():
        expected = model(torch.tensor([ids]))[0][0]
    assert (torch.tensor(json.loads(proc.stdout)) - expected).abs().max().item() < 1e-4

    page = (dist / "index.html").read_text(encoding="utf-8")
    artifact = (art / "index.html").read_text(encoding="utf-8")
    assert page.startswith('<!doctype html>\n<meta charset="utf-8">\n')
    assert artifact == page[len(export_web.STANDALONE_HEAD):]
    assert page.lower().count("<!doctype") == 1 and "</body></html>" not in page
    # what the publisher's skeleton gave the page, the standalone head must give it too
    head = page[:len(export_web.STANDALONE_HEAD)]
    assert 'name="viewport"' in head and "body{margin:0}" in head and "viewport" not in artifact[:200]
    assert "<script src=" not in page
    for name in ("gpt.js", "checker.js"):
        with open(os.path.join(ROOT, "web", name), encoding="utf-8") as f:
            assert f.read().rstrip("\n") in page
    assert 'fetch("model.json")' in page


def test_sample_cli(run, capsys):
    sample.main([run["paths"]["sft"], "--n", "1", "--domains", "nat,linear", "--device", "cpu"])
    out = capsys.readouterr().out
    assert "stage sft, task format, greedy" in out
    assert out.count("[nat] ") == 1 and out.count("[linear] ") == 1 and out.count("reference: ") == 2
    sample.main([run["paths"]["pretrain"], "--prompt", "example : 2 + 3 =", "--samples", "3", "--temperature", "0.8",
                 "--device", "cpu"])
    out = capsys.readouterr().out
    assert "stage pretrain, document format, 3 samples at temperature 0.8" in out
    assert out.count("  -> ") == 3 and "reference" not in out
