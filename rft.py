"""Stage 4, third arm: rejection-sampling fine-tuning (expert iteration).

    python rft.py --preset small --sft out/sft.pt --out out/rft.pt --seed 1

The simplest way to use a checker: each round, sample answers from the
current policy, keep the ones the checker accepts, and fine-tune on them.

  * Prompts: prompts_per_round train-split problems per round, interleaved
    domain by domain (each round's domains differ in count by at most one),
    never reused across rounds.
  * Samples: samples_per_prompt answers per prompt at `temperature` from the
    CURRENT policy; finished answers the checker accepts are deduplicated
    and the shortest keep_per_prompt of them kept.
  * Fine-tuning: epochs_per_round passes over those examples (loss on the
    answer and <eos>, as in SFT); the learning-rate schedule and the optimiser
    restart each round. With epochs_per_round 0, a fixed iters_per_round steps
    instead, optionally with replay_frac of every batch drawn from fresh
    reference examples (train-split problems never seen in this run).
    A few epochs, not a fixed step count: the verified set is small (about one
    example per solved prompt) and its size depends on the policy, so a fixed
    number of steps would mean dozens of epochs over it in one run and a few
    in another.

The policy-sample budget, rounds * prompts_per_round * samples_per_prompt,
equals PPO's iters * rollout_batch in the small preset, so the arms are
compared at the same number of samples. A JSON log <out stem>.json records,
per round, each domain's yield (the fraction of prompts with at least one
verified answer), the number of training examples, the train loss and the
greedy val accuracy (also before the first round: the SFT policy).
"""

import json
import math
import os
import time
from dataclasses import asdict

import torch

import data
import formal
from common import evaluate, format_results, generate_answers, get_device, load_checkpoint, lr_at, \
    make_optimizer, save_checkpoint, seed_everything, sft_batch
from config import RFTConfig, parse_config

# The val problems every run is scored on (the same as sft.py's).
VAL_SEED = 0


def select_verified(prompt: str, answers, statuses, keep: int) -> list:
    """The answers to train on for one prompt: finished, accepted by the
    checker, distinct, shortest first (ties in sampling order), at most keep."""
    good = [a for a, s in zip(answers, statuses) if s == "finished" and formal.verify(prompt, a)]
    return sorted(dict.fromkeys(good), key=len)[:keep]


def steps_for(config: RFTConfig, n_examples: int) -> int:
    """Fine-tuning steps for a round with n_examples verified examples."""
    if config.epochs_per_round > 0:
        return max(1, math.ceil(config.epochs_per_round * n_examples / config.batch_size))
    return config.iters_per_round


def round_problems(config: RFTConfig, domains) -> tuple:
    """(one list of problems per round, the fresh reference problems for
    replay). All distinct (by split key); the interleaved stream is cut into
    rounds in order, so each round's domains differ in count by at most one,
    and the replay problems come after every round's."""
    n_rounds = config.rounds * config.prompts_per_round
    n_replay = config.rounds * config.iters_per_round * replay_per_batch(config) if config.epochs_per_round <= 0 else 0
    stream = data.sample_problems(n_rounds + n_replay, "train", config.seed, domains, unique=True)
    rounds = [stream[r * config.prompts_per_round:(r + 1) * config.prompts_per_round] for r in range(config.rounds)]
    return rounds, stream[n_rounds:]


def replay_per_batch(config: RFTConfig) -> int:
    return round(config.replay_frac * config.batch_size)


def fine_tune(policy, examples, replay, config: RFTConfig, device: str, gen) -> list:
    """steps_for(config, len(examples)) steps on `examples` (drawn with
    replacement) plus, in each batch, the next replay_per_batch examples of
    `replay`. Returns the per-step losses."""
    n_replay = replay_per_batch(config)
    n_steps = steps_for(config, len(examples))
    optimizer = make_optimizer(policy, config.lr, config.weight_decay)
    losses = []
    policy.train()
    for it in range(n_steps):
        lr = lr_at(it, config.lr, config.min_lr, min(config.warmup_iters, n_steps // 4), n_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        n_own = config.batch_size - n_replay
        own = torch.randint(len(examples), (n_own,), generator=gen).tolist() if n_own else []
        batch = [examples[i] for i in own] + replay[it * n_replay:(it + 1) * n_replay]
        x, y, mask = sft_batch(batch, device)
        _, loss = policy(x, targets=y, attention_mask=mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
        optimizer.step()
        losses.append(loss.item())
    policy.eval()
    return losses


def main(argv=None):
    config, preset = parse_config(RFTConfig, argv)
    if config.temperature <= 0:
        raise ValueError("rejection sampling needs sampled answers: temperature must be > 0")
    if config.keep_per_prompt < 1 or not 0 <= config.replay_frac <= 1:
        raise ValueError("keep_per_prompt must be >= 1 and replay_frac in [0, 1]")
    if config.epochs_per_round > 0 and config.replay_frac > 0:
        raise ValueError("replay needs a fixed step count: set epochs_per_round 0 and iters_per_round")
    seed_everything(config.seed)
    device = get_device(config.device)
    domains = formal.resolve(config.domains)
    policy, tokenizer, sft = load_checkpoint(config.sft, device, kind="policy")
    policy.eval()

    rounds, replay_problems = round_problems(config, domains)
    replay = [tokenizer.example_ids(p.prompt, p.reference) for p in replay_problems]
    n_replay = replay_per_batch(config)
    val_problems = data.sample_problems(config.eval_problems * len(domains), "val", VAL_SEED, domains, unique=True) \
        if config.eval_problems > 0 else []
    sample_gen = torch.Generator(device=device).manual_seed(config.seed)
    batch_gen = torch.Generator().manual_seed(config.seed)
    k = config.samples_per_prompt
    log = {"stage": "rft", "preset": preset, "config": asdict(config),
           "sft": {"path": config.sft, "stage": sft["stage"]},
           "policy_samples": config.rounds * config.prompts_per_round * k, "rounds": [], "evals": []}
    stem = os.path.splitext(config.out)[0]
    print(f"rft: {config.rounds} rounds x {config.prompts_per_round} prompts x {k} samples = "
          f"{log['policy_samples']:,} policy samples, from {config.sft} ({sft['stage']}), device {device}", flush=True)

    def run_eval(r):
        results = evaluate(policy, tokenizer, val_problems, max_new_tokens=config.max_new_tokens)
        log["evals"].append({"round": r, "results": results})
        print(f"  val after round {r}: {format_results(results)}", flush=True)
        return results

    def write_log():
        os.makedirs(os.path.dirname(os.path.abspath(stem + ".json")), exist_ok=True)
        with open(stem + ".json.tmp", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1, ensure_ascii=False)
        os.replace(stem + ".json.tmp", stem + ".json")

    results = run_eval(0) if val_problems else None
    n_samples = 0
    for r, problems in enumerate(rounds, 1):
        started = time.time()
        flat = [p for p in problems for _ in range(k)]
        answers, statuses = generate_answers(policy, tokenizer, flat, max_new_tokens=config.max_new_tokens,
                                             greedy=False, temperature=config.temperature, generator=sample_gen,
                                             batch_size=128, statuses=True)
        n_samples += len(flat)
        examples, stats = [], {d: {"prompts": 0, "solved": 0, "samples": 0, "verified": 0} for d in domains}
        for i, p in enumerate(problems):
            a, s = answers[i * k:(i + 1) * k], statuses[i * k:(i + 1) * k]
            kept = select_verified(p.prompt, a, s, config.keep_per_prompt)
            examples += [tokenizer.example_ids(p.prompt, ans) for ans in kept]
            d = stats[p.domain]
            d["prompts"] += 1
            d["solved"] += bool(kept)
            d["samples"] += k
            d["verified"] += sum(st == "finished" and formal.verify(p.prompt, x) for x, st in zip(a, s))
        yields = {d: v["solved"] / v["prompts"] for d, v in stats.items() if v["prompts"]}
        sample_accuracy = {d: v["verified"] / v["samples"] for d, v in stats.items() if v["samples"]}
        entry = {"round": r, "yield": yields, "sample_accuracy": sample_accuracy, "examples": len(examples),
                 "finished": statuses.count("finished") / len(statuses)}
        print(f"round {r}: {len(flat)} samples, {len(examples)} verified examples; yield "
              + " ".join(f"{d} {y:.2f}" for d, y in yields.items()), flush=True)

        if examples or n_replay == config.batch_size:
            round_replay = replay[(r - 1) * config.iters_per_round * n_replay:r * config.iters_per_round * n_replay]
            losses = fine_tune(policy, examples, round_replay, config, device, batch_gen)
            entry["train_loss"] = sum(losses) / len(losses)
            entry["final_loss"] = losses[-1]
            entry["steps"] = len(losses)
            print(f"  fine-tuned {len(losses)} steps: loss {entry['train_loss']:.4f} "
                  f"(last {losses[-1]:.4f})", flush=True)
        else:
            entry["train_loss"] = None
            print("  no verified answers: nothing to fine-tune on this round", flush=True)
        if val_problems:
            results = run_eval(r)
            entry["val"] = results
        entry["seconds"] = time.time() - started
        log["rounds"].append(entry)
        write_log()

    log["policy_samples_drawn"] = n_samples
    save_checkpoint(config.out, policy, tokenizer, stage="rft", run_config=asdict(config), preset=preset,
                    rounds=config.rounds, policy_samples=n_samples, val=results,
                    sft={"path": config.sft, "stage": sft["stage"]})
    write_log()
    print(f"saved {config.out} ({n_samples:,} policy samples)")
    return policy, log


if __name__ == "__main__":
    main()
