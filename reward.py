"""Stage 3: a reward model trained on comparisons (the "RM" of RLHF).

    python reward.py --preset small --sft out/sft.pt --out out/reward.pt

RLHF does not optimise the checker directly: it optimises a model of a
rater's preferences, learned from comparisons. Here the rater is the checker
(formal.prefer: an answer that checks beats one that does not; between two
proofs that check, the shorter wins), so the comparisons are noise-free and
any gap between the reward model and the checker is the reward model's own.

1. Comparisons. For n_prompts train-split problems, samples_per_prompt answers
   from the SFT policy at `temperature`; the finished ones (plus the reference
   answer if include_reference) are deduplicated, and every pair the rater
   ranks is a comparison, at most max_pairs_per_prompt per prompt. They are
   written to --preferences as JSONL {"prompt", "chosen", "rejected",
   "domain"}. With --human_preferences file.jsonl the comparisons are read
   from that file instead (people's judgements; ties, prompts no domain
   claims and prompts outside the train split are skipped); the policy is
   still sampled, for step 4.
2. Hold-out. val_frac of the PROMPTS (by a hash of their split key, so all
   pairs of a prompt, and every respelling of it, fall on one side) are kept
   for validation: pairs of a seen prompt would flatter the accuracy.
3. Training. The SFT trunk with a fresh scalar head, Bradley-Terry loss
   -log sigmoid(r(chosen) - r(rejected)) on `<bos> prompt <ans> answer <eos>`
   (the score is read at the final <eos>).
4. Normalisation. A reward model's scale is arbitrary, so PPO uses
   (score - reward_mean) / reward_std, with the mean and std of the scores of
   the finished SFT samples from step 1 on the held-out prompts: the
   distribution PPO starts from, on prompts the reward model was not trained on.
   Also recorded: the AUC with which the score separates answers the checker
   accepts from ones it rejects, on the held-out prompts' samples.
"""

import hashlib
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

import data
import formal
from common import finish_progress, generate_answers, get_device, load_checkpoint, load_progress, lr_at, \
    make_optimizer, pad_batch, save_checkpoint, save_progress, seed_everything
from config import RewardConfig, parse_config


# ------------------------------------------------------------ comparisons

def fits(tokenizer, prompt: str, answer: str, block_size: int) -> bool:
    return len(tokenizer.example_ids(prompt, answer)[0]) <= block_size


def sample_answers(policy, tokenizer, problems, samples_per_prompt: int, temperature: float, max_new_tokens: int,
                   generator) -> list:
    """For each problem, its samples_per_prompt sampled (answer, status) pairs."""
    k = samples_per_prompt
    flat = [p for p in problems for _ in range(k)]
    answers, statuses = generate_answers(policy, tokenizer, flat, max_new_tokens=max_new_tokens, greedy=False,
                                         temperature=temperature, generator=generator, batch_size=128,
                                         statuses=True)
    return [list(zip(answers[i * k:(i + 1) * k], statuses[i * k:(i + 1) * k])) for i in range(len(problems))]


def candidates(problem, samples, include_reference: bool, tokenizer, block_size: int) -> list:
    """The distinct answers to compare for one problem: finished samples that
    fit the context, then the reference if asked for (and it fits)."""
    out = [a for a, status in samples if status == "finished" and fits(tokenizer, problem.prompt, a, block_size)]
    if include_reference and fits(tokenizer, problem.prompt, problem.reference, block_size):
        out.append(problem.reference)
    return list(dict.fromkeys(out))


def comparisons(problem, answers, max_pairs: int, rng) -> list:
    """Every pair of `answers` the rater ranks, as {"prompt", "chosen",
    "rejected", "domain"}; a random subset of max_pairs if there are more."""
    pairs = []
    for i, a in enumerate(answers):
        for b in answers[i + 1:]:
            pref = formal.prefer(problem.prompt, a, b)
            if pref:
                pairs.append((a, b) if pref > 0 else (b, a))
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
    return [{"prompt": problem.prompt, "chosen": c, "rejected": r, "domain": problem.domain} for c, r in pairs]


def load_human_preferences(path: str, tokenizer, block_size: int):
    """(pairs, skipped) from a JSONL file of {"prompt", "chosen", "rejected"}.

    Skipped, and counted in `skipped`: ties (chosen == rejected), prompts no
    domain claims (the checker could not score the policy's answers to them),
    prompts outside the train split (training on them would leak val/test
    problems into the policy through the reward), and pairs too long for the
    context. A line that is not such an object, or has text outside the
    model's alphabet, raises ValueError naming path:line."""
    pairs = []
    skipped = {"tie": 0, "unclaimed": 0, "not train": 0, "too long": 0}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: not valid JSON ({e.msg})") from None
            fields = ("prompt", "chosen", "rejected")
            if not isinstance(rec, dict) or not all(isinstance(rec.get(k), str) for k in fields):
                raise ValueError(f"{where}: expected an object with string fields prompt, chosen and rejected")
            prompt, chosen, rejected = rec["prompt"], rec["chosen"], rec["rejected"]
            if chosen == rejected:
                skipped["tie"] += 1
                continue
            try:
                domain = formal.domain_for(prompt)
            except ValueError:
                skipped["unclaimed"] += 1
                continue
            if data.split_of(prompt) != "train":
                skipped["not train"] += 1
                continue
            try:
                too_long = not all(fits(tokenizer, prompt, a, block_size) for a in (chosen, rejected))
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            if too_long:
                skipped["too long"] += 1
                continue
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "domain": domain.name})
    return pairs, skipped


def held_out(prompt: str, val_frac: float) -> bool:
    """Whether a prompt's comparisons are kept for validation. Salted, so the
    choice is independent of the train/val/test split (also a hash of the key)."""
    key = data.split_key(prompt)
    h = int.from_bytes(hashlib.sha256(("reward-val/" + key).encode("utf-8")).digest()[:8], "big")
    return h / 2 ** 64 < val_frac


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def score(rm, tokenizer, items, device: str, batch_size: int = 256) -> torch.Tensor:
    """Reward-model scores (float, on the CPU) of (prompt, answer) pairs, each
    read as `<bos> prompt <ans> answer <eos>`. Sorted by length internally so
    batches carry little padding."""
    was_training = rm.training
    rm.eval()
    seqs = [tokenizer.example_ids(p, a)[0] for p, a in items]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = torch.zeros(len(seqs))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        ids, mask = pad_batch([seqs[i] for i in chunk])
        out[chunk] = rm.score(ids.to(device), mask.to(device)).float().cpu()
    rm.train(was_training)
    return out


def pairwise_accuracy(rm, tokenizer, pairs, device: str) -> dict:
    """{"loss", "accuracy", "by_domain"} of the reward model on comparisons
    (accuracy: the chosen answer scores higher; an exact tie counts half)."""
    chosen = score(rm, tokenizer, [(p["prompt"], p["chosen"]) for p in pairs], device)
    rejected = score(rm, tokenizer, [(p["prompt"], p["rejected"]) for p in pairs], device)
    diff = chosen - rejected
    right = (diff > 0).float() + 0.5 * (diff == 0).float()
    by_domain = {}
    for d in formal.DOMAINS:
        sel = torch.tensor([p["domain"] == d for p in pairs], dtype=torch.bool)
        if sel.any():
            by_domain[d] = right[sel].mean().item()
    return {"loss": -F.logsigmoid(diff).mean().item(), "accuracy": right.mean().item(), "by_domain": by_domain}


def auc(scores, labels):
    """Probability that a random positive outscores a random negative (ties
    count half), i.e. the Mann-Whitney U statistic over n_pos * n_neg; None if
    either class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------- training

def pair_batch(pairs, tokenizer, device: str):
    """(ids, mask) for the chosen answers of `pairs`, then the rejected ones."""
    seqs = [tokenizer.example_ids(p["prompt"], p["chosen"])[0] for p in pairs]
    seqs += [tokenizer.example_ids(p["prompt"], p["rejected"])[0] for p in pairs]
    ids, mask = pad_batch(seqs)
    return ids.to(device), mask.to(device)


def train_reward_model(rm, tokenizer, train_pairs, val_pairs, config: RewardConfig, device: str) -> list:
    """Bradley-Terry training on train_pairs (batches drawn with replacement);
    every eval_interval iterations and at the end, the loss and pairwise
    accuracy on val_pairs. Returns the history."""
    if not train_pairs:
        raise ValueError("no training comparisons")
    optimizer = make_optimizer(rm, config.lr, config.weight_decay)
    gen = torch.Generator().manual_seed(config.seed)
    history, losses, start = [], [], 0
    run_config = asdict(config)
    state = load_progress(config.out, run_config)
    if state is not None:
        rm.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        gen.set_state(state["gen"])
        history, losses, start = state["history"], state["losses"], state["iter"]
        print(f"resuming reward-model training from iteration {start}", flush=True)
    started = time.time()
    rm.train()
    for it in range(start, config.max_iters):
        lr = lr_at(it, config.lr, config.min_lr, config.warmup_iters, config.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = [train_pairs[i] for i in torch.randint(len(train_pairs), (config.batch_size,), generator=gen).tolist()]
        ids, mask = pair_batch(batch, tokenizer, device)
        r = rm.score(ids, mask)
        loss = -F.logsigmoid(r[:len(batch)] - r[len(batch):]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(rm.parameters(), config.grad_clip)
        optimizer.step()
        losses.append(loss.item())

        if (it + 1) % config.eval_interval == 0 or it + 1 == config.max_iters:
            entry = {"iter": it + 1, "lr": lr, "train_loss": sum(losses) / len(losses),
                     "elapsed_s": time.time() - started}
            line = f"iter {it + 1:>6}  lr {lr:.2e}  train loss {entry['train_loss']:.4f}"
            if val_pairs:
                ev = pairwise_accuracy(rm, tokenizer, val_pairs, device)
                entry.update(val_loss=ev["loss"], val_accuracy=ev["accuracy"], val_accuracy_by_domain=ev["by_domain"])
                by_domain = " ".join(f"{d} {a:.3f}" for d, a in ev["by_domain"].items())
                line += f"  val loss {ev['loss']:.4f}  val acc {ev['accuracy']:.3f} ({by_domain})"
            history.append(entry)
            print(line + f"  {entry['elapsed_s']:.0f}s", flush=True)
            losses = []
        if config.save_interval > 0 and (it + 1) % config.save_interval == 0 and it + 1 < config.max_iters:
            save_progress(config.out, {"run_config": run_config, "iter": it + 1, "model": rm.state_dict(),
                                       "optimizer": optimizer.state_dict(), "gen": gen.get_state(),
                                       "history": history, "losses": losses})
    return history


# ------------------------------------------------------------------- main

def main(argv=None):
    config, preset = parse_config(RewardConfig, argv)
    if config.temperature <= 0:
        raise ValueError("preferences need sampled answers: temperature must be > 0")
    seed_everything(config.seed)
    device = get_device(config.device)
    domains = formal.resolve(config.domains)
    policy, tokenizer, sft = load_checkpoint(config.sft, device, kind="policy")
    block_size = policy.config.block_size
    started = time.time()

    # 1. The SFT policy's answers: comparisons (unless people supplied them),
    # and the score distribution PPO will start from.
    problems = data.sample_problems(config.n_prompts, "train", config.seed, domains, unique=True)
    # Sampling is the slow part; an interrupted run reuses the samples it saved.
    cache = config.out + ".samples.json"
    samples = None
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            saved = json.load(f)
        if saved["run_config"] == asdict(config):
            samples = [[tuple(x) for x in s] for s in saved["samples"]]
            print(f"reusing the samples in {cache}", flush=True)
    if samples is None:
        gen = torch.Generator(device=device).manual_seed(config.seed)
        samples = sample_answers(policy, tokenizer, problems, config.samples_per_prompt, config.temperature,
                                 config.max_new_tokens, gen)
        with open(cache + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"run_config": asdict(config), "samples": samples}, f, ensure_ascii=False)
        os.replace(cache + ".tmp", cache)
    del policy
    finished = [(p, a) for p, s in zip(problems, samples) for a, status in s
                if status == "finished" and fits(tokenizer, p.prompt, a, block_size)]
    correct = [formal.verify(p.prompt, a) for p, a in finished]
    n_samples = len(problems) * config.samples_per_prompt
    print(f"reward: {len(problems)} prompts x {config.samples_per_prompt} samples from {config.sft} "
          f"({sft['stage']}): {len(finished)} finished, {sum(correct)} correct, {time.time() - started:.0f}s",
          flush=True)
    if len(finished) < 2:
        raise ValueError(f"only {len(finished)} of {n_samples} samples finished: too few to normalise the reward")

    if config.human_preferences:
        pairs, skipped = load_human_preferences(config.human_preferences, tokenizer, block_size)
        print(f"{len(pairs)} comparisons from {config.human_preferences}; skipped {skipped}")
    else:
        pick = random.Random(config.seed)
        pairs = []
        for p, s in zip(problems, samples):
            pairs += comparisons(p, candidates(p, s, config.include_reference, tokenizer, block_size),
                                 config.max_pairs_per_prompt, pick)
        if config.preferences:
            os.makedirs(os.path.dirname(os.path.abspath(config.preferences)), exist_ok=True)
            with open(config.preferences, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"{len(pairs)} checker-labelled comparisons" + (f", written to {config.preferences}"
                                                             if config.preferences else ""))
    for d in domains:
        n = sum(p["domain"] == d for p in pairs)
        done = [ok for (p, _), ok in zip(finished, correct) if p.domain == d]
        print(f"  {d}: {n} comparisons, {len(done)} finished samples, {sum(done)} correct")

    # 2. Hold out whole prompts.
    val_pairs = [p for p in pairs if held_out(p["prompt"], config.val_frac)]
    train_pairs = [p for p in pairs if not held_out(p["prompt"], config.val_frac)]
    print(f"{len(train_pairs)} train comparisons, {len(val_pairs)} held out "
          f"({len({p['prompt'] for p in val_pairs})} prompts)", flush=True)

    # 3. Bradley-Terry training from the SFT trunk.
    rm, _, _ = load_checkpoint(config.sft, device, kind="reward")
    history = train_reward_model(rm, tokenizer, train_pairs, val_pairs, config, device)
    val_accuracy = history[-1].get("val_accuracy")

    # 4. Normalisation over the policy's own finished answers, and the AUC for
    # checker correctness on the held-out prompts' answers.
    scores = score(rm, tokenizer, [(p.prompt, a) for p, a in finished], device)
    held = [i for i, (p, _) in enumerate(finished) if held_out(p.prompt, config.val_frac)]
    # The statistics come from the held-out prompts' samples: PPO scores answers
    # to prompts the reward model has not been trained on, where its scores are
    # less spread out than on its own training samples. (All samples if too few.)
    basis = held if len(held) >= 100 else list(range(len(finished)))
    reward_mean, reward_std = scores[basis].mean().item(), scores[basis].std(unbiased=False).item()
    train_mean, train_std = scores.mean().item(), scores.std(unbiased=False).item()
    if reward_std == 0:
        raise ValueError("every finished sample has the same reward-model score; cannot normalise")
    rm_auc = auc(scores[held].numpy(), [correct[i] for i in held])
    auc_by_domain = {}
    for d in domains:
        sel = [i for i in held if finished[i][0].domain == d]
        auc_by_domain[d] = auc(scores[sel].numpy(), [correct[i] for i in sel])
    print(f"reward mean {reward_mean:.4f} std {reward_std:.4f} over {len(basis)} finished SFT samples of held-out "
          f"prompts (all {len(finished)}: mean {train_mean:.4f} std {train_std:.4f}); "
          f"held-out AUC for correctness {rm_auc if rm_auc is None else round(rm_auc, 4)} "
          f"({' '.join(f'{d} {a:.3f}' for d, a in auc_by_domain.items() if a is not None)})")

    save_checkpoint(config.out, rm, tokenizer, stage="reward", run_config=asdict(config), preset=preset,
                    reward_mean=reward_mean, reward_std=reward_std, all_samples_mean=train_mean,
                    all_samples_std=train_std, val_accuracy=val_accuracy, auc=rm_auc,
                    auc_by_domain=auc_by_domain, history=history, n_comparisons=len(pairs),
                    n_val_comparisons=len(val_pairs), n_finished_samples=len(finished),
                    sft={"path": config.sft, "stage": sft["stage"]})
    finish_progress(config.out)
    if os.path.exists(cache):
        os.remove(cache)
    print(f"saved {config.out}  ({time.time() - started:.0f}s)")
    return rm, history


if __name__ == "__main__":
    main()
