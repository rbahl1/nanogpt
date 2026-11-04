"""Turns the experiment's outputs into the tables and figures of REPORT.md.

    python analyze.py --out out --results results

Reads results/eval.json and results/passk.json (evaluate.py), the PPO and RFT
logs (out/*_s*.json), and the SFT and reward-model checkpoints; writes
results/summary.md, results/summary.json, results/fig_*.png, and copies the
training logs to results/logs/ (the checkpoints themselves stay in out/).
Missing inputs are skipped, so it also works on a partial run.
"""

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import torch

import stats

ARMS = {  # file stem prefix -> name in the report
    "rlhf": "RLHF (PPO vs reward model)",
    "ppochecker": "PPO vs checker",
    "rft": "rejection-sampling FT",
    "sftplus": "more SFT (control)",
}
DOMAINS = ["prop", "fol", "nat", "linear"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(x):
    return f"{100 * x:.1f}"


def test_accuracy(evals: dict, n_boot: int):
    """Per checkpoint and per arm (pooled over seeds): accuracy by domain, and
    the paired difference from SFT with a bootstrap CI and p-value."""
    entries = evals["checkpoints"]
    judge, ok = stats.correctness(entries)
    first = entries[0]["records"]
    doms = np.array([r["domain"] for r in first])
    masks = {d: doms == d for d in DOMAINS if (doms == d).any()}
    masks["all"] = np.ones(len(doms), dtype=bool)
    out = {"judge": judge, "n": {d: int(m.sum()) for d, m in masks.items()}, "checkpoints": {}, "arms": {}}
    for e in entries:
        label = e["label"]
        out["checkpoints"][label] = {
            "stage": e["stage"],
            "accuracy": {d: float(ok[label][m].mean()) for d, m in masks.items()},
            "checker_false_pos": sum(r.get("checker_ok", False) and r.get("lean_ok") is False for r in e["records"]),
            "checker_false_neg": sum(bool(r.get("lean_ok")) and not r.get("checker_ok") for r in e["records"]),
            "finished": sum(r["status"] == "finished" for r in e["records"]) / len(e["records"]),
            "answer_chars": float(np.mean([len(r["answer"]) for r in e["records"] if r["status"] == "finished"]
                                          or [0])),
        }
    base = ok["sft"].astype(float)
    for prefix in ARMS:
        members = sorted(label for label in ok if re.fullmatch(prefix + r"_s\d+", label))
        if not members:
            continue
        pooled = np.mean([ok[m] for m in members], axis=0)
        arm = {"seeds": members, "by_domain": {}}
        for d, m in masks.items():
            accs = [float(ok[label][m].mean()) for label in members]
            strata = doms[m] if d == "all" else None
            diff, lo, hi = stats.paired_bootstrap(base[m], pooled[m], n_boot, 0, strata)
            arm["by_domain"][d] = {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)) if len(accs) > 1
                                   else 0.0, "seeds": accs, "diff": diff, "lo": lo, "hi": hi,
                                   "p": stats.paired_z_test(base[m], pooled[m], strata)}
        out["arms"][prefix] = arm
    return out


def passk_table(passk: dict):
    out = {}
    for e in passk["checkpoints"]:
        if e.get("passk"):
            out[e["label"]] = e["passk"]
    return out


def ppo_dynamics(out_dir: str):
    """Per PPO run: the rollout score, rollout accuracy (the checker's verdict on
    the samples) and KL over iterations, and the val evaluations."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*_s*.json"))):
        log = _load(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        if not log or "iterations" not in log:
            continue
        its = log["iterations"]
        runs[stem] = {
            "iteration": [i["iteration"] for i in its],
            "score": [i["score"] for i in its],
            "rm_score": [i.get("rm_score") for i in its],
            "accuracy": [i["accuracy"] for i in its],
            "kl": [i["kl"] for i in its],
            "answer_tokens": [i.get("answer_tokens") for i in its],
            "val": [(e["iteration"], e["results"]["all"]["accuracy"]) for e in log["evals"]],
        }
    return runs


def rft_rounds(out_dir: str):
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "rft_s*.json"))):
        log = _load(path)
        if log and "rounds" in log:
            runs[os.path.splitext(os.path.basename(path))[0]] = log
    return runs


def reward_model(out_dir: str):
    path = os.path.join(out_dir, "reward.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    keys = ("reward_mean", "reward_std", "val_accuracy", "auc", "auc_by_domain", "n_comparisons",
            "n_val_comparisons", "n_finished_samples", "history")
    return {k: ckpt.get(k) for k in keys}


def sft_history(out_dir: str):
    path = os.path.join(out_dir, "sft.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True).get("history")


def _smooth(x, k=10):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def figures(res_dir: str, acc, passk, ppo, rm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    made = []

    if acc:
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        cols = DOMAINS + ["all"]
        rows = [("pretrain", "pretrained"), ("sft", "SFT")] + [(p, ARMS[p]) for p in ARMS if p in acc["arms"]]
        width = 0.8 / len(rows)
        for i, (key, name) in enumerate(rows):
            xs = np.arange(len(cols)) + (i - (len(rows) - 1) / 2) * width
            if key in acc["arms"]:
                arm = acc["arms"][key]["by_domain"]
                ax.bar(xs, [100 * arm[c]["mean"] for c in cols], width, label=name)
                for x, c in zip(xs, cols):
                    ax.scatter([x] * len(arm[c]["seeds"]), [100 * s for s in arm[c]["seeds"]], s=4, color="k",
                               zorder=3)
            elif key in acc["checkpoints"]:
                ax.bar(xs, [100 * acc["checkpoints"][key]["accuracy"][c] for c in cols], width, label=name)
        ax.set_xticks(np.arange(len(cols)), cols)
        ax.set_ylabel(f"test accuracy (%, {acc['judge']})")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, ncols=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, "fig_test_accuracy.png"), dpi=200)
        plt.close(fig)
        made.append("fig_test_accuracy.png")

    if ppo:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
        for stem, run in ppo.items():
            color = "C0" if stem.startswith("rlhf") else "C1"
            name = "RLHF" if stem.startswith("rlhf") else "PPO vs checker"
            it = run["iteration"]
            sm = _smooth(run["score"])
            axes[0].plot(it[len(it) - len(sm):], sm, color=color, alpha=0.8, lw=1,
                         label=name if stem.endswith("_s1") else None)
            sm = _smooth(run["accuracy"])
            axes[1].plot(it[len(it) - len(sm):], [100 * a for a in sm], color=color, alpha=0.8, lw=1)
            if run["val"]:
                axes[1].plot([v[0] for v in run["val"]], [100 * v[1] for v in run["val"]], ls=":", color=color,
                             lw=1)
        axes[0].set_title("reward being optimised (normalised)", fontsize=9)
        axes[1].set_title("checked accuracy: rollouts (solid), val (dotted)", fontsize=9)
        axes[1].set_ylabel("%")
        for ax in axes:
            ax.set_xlabel("PPO iteration")
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, "fig_ppo.png"), dpi=200)
        plt.close(fig)
        made.append("fig_ppo.png")

    if passk:
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for label, pk in passk.items():
            ks = sorted(int(k.split("@")[1]) for k in pk["all"])
            ax.plot(ks, [100 * pk["all"][f"pass@{k}"] for k in ks], marker="o", ms=3, lw=1, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (samples at T = 1)")
        ax.set_ylabel("pass@k (%)")
        ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, "fig_passk.png"), dpi=200)
        plt.close(fig)
        made.append("fig_passk.png")

    if rm and rm.get("history"):
        h = rm["history"]
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        it = [e["iter"] for e in h]
        ax.plot(it, [e["train_loss"] for e in h], marker="o", ms=3, label="train loss")
        ax.plot(it, [e["val_loss"] for e in h], marker="o", ms=3, label="held-out loss")
        ax2 = ax.twinx()
        ax2.plot(it, [100 * e["val_accuracy"] for e in h], color="C2", marker="s", ms=3, label="held-out acc")
        ax2.set_ylim(40, 70)
        ax2.set_ylabel("held-out pairwise accuracy (%)")
        ax.set_xlabel("reward-model step")
        ax.set_ylabel("Bradley-Terry loss")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax2.legend(fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, "fig_reward_model.png"), dpi=200)
        plt.close(fig)
        made.append("fig_reward_model.png")
    return made


def markdown(acc, passk, ppo, rft, rm, sft_hist) -> str:
    lines = []
    if acc:
        lines += [f"## Test accuracy ({acc['judge']} verdicts; "
                  + ", ".join(f"{d} {n}" for d, n in acc["n"].items() if d != "all") + " problems)", "",
                  "| checkpoint | " + " | ".join(DOMAINS + ["all"]) + " | Δ all vs SFT [95% CI] | p |",
                  "|---|" + "--:|" * (len(DOMAINS) + 3)]
        for key, name in (("pretrain", "pretrained"), ("sft", "SFT")):
            if key in acc["checkpoints"]:
                a = acc["checkpoints"][key]["accuracy"]
                lines.append(f"| {name} | " + " | ".join(_pct(a[d]) for d in DOMAINS + ["all"]) + " | | |")
        for prefix, arm in acc["arms"].items():
            b = arm["by_domain"]
            cells = [f"{_pct(b[d]['mean'])} ± {_pct(b[d]['std'])}" for d in DOMAINS + ["all"]]
            a = b["all"]
            lines.append(f"| {ARMS[prefix]} ({len(arm['seeds'])} seeds) | " + " | ".join(cells)
                         + f" | {100 * a['diff']:+.1f} [{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}] | {a['p']:.2g} |")
        lines += ["", "Per checkpoint (finished answers, mean answer length, checker vs Lean disagreements):", "",
                  "| checkpoint | all | finished | answer chars | checker true, Lean false | checker false, Lean true |",
                  "|---|--:|--:|--:|--:|--:|"]
        for label, c in acc["checkpoints"].items():
            lines.append(f"| {label} | {_pct(c['accuracy']['all'])} | {_pct(c['finished'])} | "
                         f"{c['answer_chars']:.1f} | {c['checker_false_pos']} | {c['checker_false_neg']} |")
    if passk:
        ks = sorted({int(k.split("@")[1]) for pk in passk.values() for k in pk["all"]})
        lines += ["", "## Sampled pass@k (T = 1, all domains)", "",
                  "| checkpoint | " + " | ".join(f"pass@{k}" for k in ks) + " |", "|---|" + "--:|" * len(ks)]
        for label, pk in passk.items():
            lines.append(f"| {label} | " + " | ".join(_pct(pk["all"][f"pass@{k}"]) for k in ks) + " |")
    if rm:
        lines += ["", "## Reward model", "",
                  f"- comparisons: {rm['n_comparisons']} ({rm['n_val_comparisons']} held out)",
                  f"- held-out pairwise accuracy: {_pct(rm['val_accuracy'])}%",
                  f"- AUC of the score for checker correctness (held-out prompts): {rm['auc']:.3f} ("
                  + ", ".join(f"{d} {v:.3f}" for d, v in (rm["auc_by_domain"] or {}).items() if v is not None) + ")"]
        for e in rm.get("history") or []:
            lines.append(f"- step {e['iter']}: train loss {e['train_loss']:.3f}, held-out loss {e['val_loss']:.3f}, "
                         f"held-out accuracy {_pct(e['val_accuracy'])}%")
    if ppo:
        lines += ["", "## PPO runs", "", "| run | rollout acc, first 10 iters | last 10 iters | score first 10 | "
                  "last 10 | KL last 10 | val acc start → end |", "|---|--:|--:|--:|--:|--:|--:|"]
        for stem, r in ppo.items():
            f10, l10 = slice(0, 10), slice(-10, None)
            val = f"{_pct(r['val'][0][1])} → {_pct(r['val'][-1][1])}" if r["val"] else ""
            lines.append(f"| {stem} | {_pct(np.mean(r['accuracy'][f10]))} | {_pct(np.mean(r['accuracy'][l10]))} | "
                         f"{np.mean(r['score'][f10]):+.3f} | {np.mean(r['score'][l10]):+.3f} | "
                         f"{np.mean(r['kl'][l10]):.3f} | {val} |")
    if rft:
        lines += ["", "## Rejection-sampling fine-tuning rounds", ""]
        for stem, log in rft.items():
            for r in log["rounds"]:
                y = ", ".join(f"{d} {v:.2f}" for d, v in r["yield"].items())
                val = _pct(r["val"]["all"]["accuracy"]) if r.get("val") else "-"
                lines.append(f"- {stem} round {r['round']}: {r['examples']} verified examples (yield {y}), "
                             f"{r.get('steps', '?')} steps, val accuracy {val}%")
    if sft_hist:
        lines += ["", "## SFT validation accuracy", ""]
        for e in sft_hist:
            if "val" in e:
                lines.append(f"- step {e['iter']}: " + ", ".join(f"{d} {_pct(v['accuracy'])}"
                                                                  for d, v in e["val"].items()))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out")
    parser.add_argument("--results", default="results")
    parser.add_argument("--n_boot", type=int, default=10_000)
    args = parser.parse_args(argv)
    os.makedirs(os.path.join(args.results, "logs"), exist_ok=True)
    for path in glob.glob(os.path.join(args.out, "*.json")) + glob.glob(os.path.join(args.out, "logs", "*.log")):
        shutil.copy(path, os.path.join(args.results, "logs", os.path.basename(path)))

    evals = _load(os.path.join(args.results, "eval.json"))
    acc = test_accuracy(evals, args.n_boot) if evals else None
    pk = _load(os.path.join(args.results, "passk.json"))
    passk = passk_table(pk) if pk else None
    ppo, rft, rm, sft_hist = ppo_dynamics(args.out), rft_rounds(args.out), reward_model(args.out), \
        sft_history(args.out)
    summary = {"test": acc, "passk": passk, "reward_model": {k: v for k, v in (rm or {}).items()},
               "ppo": ppo, "rft": {k: v["rounds"] for k, v in rft.items()}, "sft_history": sft_hist}
    with open(os.path.join(args.results, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    with open(os.path.join(args.results, "summary.md"), "w", encoding="utf-8") as f:
        f.write(markdown(acc, passk, ppo, rft, rm, sft_hist))
    made = figures(args.results, acc, passk, ppo, rm)
    print(f"wrote {args.results}/summary.md, summary.json" + (", " + ", ".join(made) if made else ""))


if __name__ == "__main__":
    main()
