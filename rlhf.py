"""Stage 4: PPO from the SFT policy, against a learned reward model (RLHF) or
against the checker itself.

    python rlhf.py --preset small --sft out/sft.pt --reward out/reward.pt --out out/rlhf.pt --reward_source rm
    python rlhf.py ... --reward_source oracle --out out/ppochecker.pt      # PPO against the checker
    python rlhf.py ... --advantage group                                  # GRPO-style, no critic

The InstructGPT recipe: sample answers from the policy, score each finished
one, and take a few epochs of clipped-surrogate steps on the batch, with a
KL penalty that keeps the policy near the SFT model (the reward model is only
trustworthy near the distribution it was trained on). Each rule below fixes
a bug an earlier version of this script had:

  * Rollout prompts rotate through the domains evenly, every iteration, so no
    domain is starved of updates.
  * Scores. "rm": the reward model's score normalised by the mean and std of
    the SFT policy's scores (saved by reward.py) and clipped to
    [-score_clip, score_clip], so the KL coefficient means the same thing
    whatever the reward model's scale. "oracle": +1 if the checker accepts,
    -1 if not. An unfinished answer (cut off, or a special token before
    <eos>) scores the lowest finished score minus no_eos_penalty, in both
    modes, so running on past <eos> can never pay.
  * advantage "gae": the KL penalty is folded into a per-token reward
    (r_t = -kl_coef * (log pi - log pi_ref), plus the score on the last
    token) and a critic estimates values; GAE over the answer tokens, then
    the advantages are whitened over all answer tokens of the batch. In "rm"
    mode the critic starts as the reward model with the normalisation folded
    into its head, so its first values are in the units of the scores; in
    "oracle" mode it starts from the SFT trunk with a zero head.
  * advantage "group" (GRPO): group_size answers per prompt; an answer's
    advantage is its score's z-score within its group, the same on all its
    tokens. No critic; the KL to the reference is a separate loss term
    (kl_coef * the k3 estimator), not part of the advantage, where it would
    be whitened away.
  * Log-probabilities are those of the distribution actions were sampled from
    (logits / temperature), so the importance ratio starts at exactly 1.
  * Dropout is off during PPO for the same reason.

Every eval_interval iterations, at iteration 0 (the SFT policy) and at the
end: greedy accuracy on the val problems. A JSON log <out stem>.json holds
the config, every iteration's metrics (score, raw reward-model score, KL,
the rollouts' true accuracy...) and every evaluation.
"""

import copy
import json
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

import data
import formal
from common import evaluate, format_results, generate_answers, get_device, load_checkpoint, make_optimizer, \
    pad_batch, response_status, save_checkpoint, seed_everything
from config import RLHFConfig, parse_config

# The val problems every run is scored on (the same as sft.py's).
VAL_SEED = 0


# ------------------------------------------------------------ the maths

def masked_mean(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(x, mask, eps: float = 1e-8):
    """x shifted and scaled to mean 0 and (population) std 1 over the
    positions where mask is set; 0 elsewhere."""
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps) * mask.to(x.dtype)


def gae(rewards, values, mask, gamma: float, lam: float):
    """Generalised advantage estimation along dim 1 (time). rewards, values
    and mask are (B, T); each row's masked positions are one contiguous
    episode, which ends at its last masked position (the value after it is
    0). Returns (advantages, returns = advantages + values), 0 off the mask."""
    mask = mask.to(rewards.dtype)
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        next_mask = mask[:, t + 1] if t + 1 < T else torch.zeros_like(running)
        next_value = values[:, t + 1] * next_mask if t + 1 < T else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = (delta + gamma * lam * next_mask * running) * mask[:, t]
        advantages[:, t] = running
    return advantages, (advantages + values) * mask


def clipped_surrogate(logp, old_logp, advantages, mask, clip_range: float):
    """PPO's policy loss, per token, averaged over the masked tokens: the
    larger (more pessimistic) of -A * ratio and -A * clip(ratio, 1 - eps,
    1 + eps). Returns (loss, fraction of tokens where the clipped term won)."""
    ratio = torch.exp(logp - old_logp)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = masked_mean(torch.max(unclipped, clipped), mask)
    return loss, masked_mean((clipped > unclipped).float(), mask)


def value_loss(values, old_values, returns, mask, vf_clip: float):
    """Squared error to the returns, with the value's move from old_values
    clipped to +-vf_clip (the larger of the two errors, as in PPO)."""
    clipped = old_values + (values - old_values).clamp(-vf_clip, vf_clip)
    return 0.5 * masked_mean(torch.max((values - returns) ** 2, (clipped - returns) ** 2), mask)


def action_logprobs(model, ids, mask, temperature: float):
    """log p(ids[:, t + 1] | ids[:, :t + 1]) under logits / temperature, (B, T-1)."""
    logits, _ = model(ids, attention_mask=mask)
    logp = F.log_softmax(logits[:, :-1].float() / temperature, dim=-1)
    return logp.gather(-1, ids[:, 1:, None]).squeeze(-1)


# --------------------------------------------------------------- scores

def score_floor(config: RLHFConfig) -> float:
    """The lowest score a finished answer can get."""
    return -config.score_clip if config.reward_source == "rm" else -1.0


def answer_scores(config: RLHFConfig, finished, correct, raw=None, reward_mean: float = 0.0,
                  reward_std: float = 1.0) -> torch.Tensor:
    """The score of each answer. finished, correct: one bool per answer; raw:
    the reward model's score of each finished answer (any value for the
    others), needed in "rm" mode. Unfinished answers score
    score_floor - no_eos_penalty."""
    finished = torch.as_tensor(finished, dtype=torch.bool)
    if config.reward_source == "rm":
        raw = torch.as_tensor(raw, dtype=torch.float)
        scores = ((raw - reward_mean) / reward_std).clamp(-config.score_clip, config.score_clip)
    elif config.reward_source == "oracle":
        scores = torch.as_tensor(correct, dtype=torch.float) * 2.0 - 1.0
    else:
        raise ValueError(f"reward_source must be rm or oracle, got {config.reward_source!r}")
    return torch.where(finished, scores, torch.tensor(score_floor(config) - config.no_eos_penalty))


def critic_from_reward_model(rm, reward_mean: float, reward_std: float):
    """A copy of the reward model whose values are (score - mean) / std: the
    normalisation folded into the head's weight and bias."""
    critic = copy.deepcopy(rm)
    with torch.no_grad():
        critic.head.weight.div_(reward_std)
        critic.head.bias.sub_(reward_mean).div_(reward_std)
    return critic


# -------------------------------------------------------------- rollouts

def rollout_problems(config: RLHFConfig, domains) -> list:
    """The problems of every iteration: distinct train-split problems,
    interleaved domain by domain and cut into iterations in order, so each
    iteration's domains differ in count by at most one and the extra one
    rotates. In group mode each problem appears group_size times in a row."""
    per_iter = config.rollout_batch // config.group_size if config.advantage == "group" else config.rollout_batch
    repeat = config.group_size if config.advantage == "group" else 1
    stream = data.sample_problems(config.iters * per_iter, "train", config.seed, domains, unique=True)
    return [[p for p in stream[i * per_iter:(i + 1) * per_iter] for _ in range(repeat)]
            for i in range(config.iters)]


@torch.no_grad()
def rollout(policy, ref_policy, tokenizer, problems, config: RLHFConfig, generator=None, critic=None,
            reward_model=None, reward_mean: float = 0.0, reward_std: float = 1.0):
    """Samples one answer per entry of `problems` (in group mode, consecutive
    runs of group_size entries are a group) and prepares the PPO batch.

    Returns (batch, stats). batch holds right-padded `ids`/`mask` (B, T) of
    `<bos> prompt <ans> answer`, and over the (B, T-1) next-token positions:
    `action_mask` (the answer's tokens: up to and including <eos> or the first
    other special token; a cut-off answer has neither), `old_logp`,
    `ref_logp`, `advantages`, and in gae mode `old_values` and `returns`;
    plus `scores` (B,)."""
    device = next(policy.parameters()).device
    answers, statuses, raw_ids = generate_answers(
        policy, tokenizer, problems, max_new_tokens=config.max_new_tokens, greedy=False,
        temperature=config.temperature, generator=generator, batch_size=len(problems), statuses=True,
        return_ids=True)
    heads = [tokenizer.prompt_ids(p.prompt) for p in problems]
    actions = []
    for out in raw_ids:
        body, status = response_status(out)
        actions.append(out[:len(body) + (status != "cut off")])
    ids, mask = pad_batch([h + a for h, a in zip(heads, actions)])
    ids, mask = ids.to(device), mask.to(device)
    B, T = ids.shape
    action_mask = torch.zeros((B, T - 1), dtype=torch.bool, device=device)
    for i, (h, a) in enumerate(zip(heads, actions)):
        action_mask[i, len(h) - 1:len(h) - 1 + len(a)] = True

    finished = [s == "finished" for s in statuses]
    correct = [f and formal.verify(p.prompt, a) for p, a, f in zip(problems, answers, finished)]
    raw = torch.zeros(B)
    if reward_model is not None and any(finished):
        rows = torch.tensor(finished, device=device)
        raw[torch.tensor(finished)] = reward_model.score(ids[rows], mask[rows]).float().cpu()
    scores = answer_scores(config, finished, correct, raw, reward_mean, reward_std)

    old_logp = action_logprobs(policy, ids, mask, config.temperature) * action_mask
    ref_logp = action_logprobs(ref_policy, ids, mask, config.temperature) * action_mask
    batch = {"ids": ids, "mask": mask, "action_mask": action_mask, "old_logp": old_logp, "ref_logp": ref_logp,
             "scores": scores.to(device)}
    if config.advantage == "gae":
        values = critic.values(ids, mask)[:, :-1].float() * action_mask
        rewards = -config.kl_coef * (old_logp - ref_logp)
        last = action_mask.long().cumsum(dim=1).argmax(dim=1)          # each row's last answer position
        has_actions = action_mask.any(dim=1)
        rewards[has_actions, last[has_actions]] += batch["scores"][has_actions]
        advantages, returns = gae(rewards * action_mask, values, action_mask, config.gamma, config.lam)
        batch.update(advantages=masked_whiten(advantages, action_mask), returns=returns, old_values=values)
    elif config.advantage == "group":
        groups = batch["scores"].view(-1, config.group_size)
        z = (groups - groups.mean(dim=1, keepdim=True)) / (groups.std(dim=1, keepdim=True) + 1e-6)
        batch["advantages"] = z.reshape(-1, 1) * action_mask
    else:
        raise ValueError(f"advantage must be gae or group, got {config.advantage!r}")

    kl = (old_logp - ref_logp).sum(dim=1)
    by_domain = {}
    for d in dict.fromkeys(p.domain for p in problems):
        sel = [i for i, p in enumerate(problems) if p.domain == d]
        by_domain[d] = sum(correct[i] for i in sel) / len(sel)
    stats = {
        "score": scores.mean().item(),
        "rm_score": raw[torch.tensor(finished)].mean().item() if reward_model is not None and any(finished) else None,
        "kl": kl.mean().item(),
        "accuracy": sum(correct) / B,
        "finished": sum(finished) / B,
        "accuracy_by_domain": by_domain,
        "answer_tokens": action_mask.sum(dim=1).float().mean().item(),
        "examples": [{"prompt": p.prompt, "answer": a, "status": s, "score": round(float(sc), 4), "ok": ok}
                     for p, a, s, sc, ok in list(zip(problems, answers, statuses, scores.tolist(), correct))[:3]],
    }
    return batch, stats


# ----------------------------------------------------------- the update

def ppo_update(policy, batch, config: RLHFConfig, policy_opt, generator=None, critic=None, critic_opt=None) -> dict:
    """ppo_epochs passes over the batch in shuffled minibatches of
    minibatch_size sequences: a clipped-surrogate step on the policy (plus
    kl_coef * k3 to the reference in group mode) and, in gae mode, a clipped
    value-loss step on the critic. Returns the mean losses and diagnostics."""
    B = batch["ids"].shape[0]
    sums, steps = {}, 0
    for _ in range(config.ppo_epochs):
        perm = torch.randperm(B, generator=generator).tolist()
        for start in range(0, B, config.minibatch_size):
            idx = torch.tensor(perm[start:start + config.minibatch_size], device=batch["ids"].device)
            T = int(batch["mask"][idx].sum(dim=1).max())       # right-padded: trim the shared padding
            ids, mask = batch["ids"][idx, :T], batch["mask"][idx, :T]
            act = batch["action_mask"][idx, :T - 1]
            old_logp, adv = batch["old_logp"][idx, :T - 1], batch["advantages"][idx, :T - 1]

            logp = action_logprobs(policy, ids, mask, config.temperature)
            pg_loss, clip_frac = clipped_surrogate(logp, old_logp, adv, act, config.clip_range)
            loss = pg_loss
            # Masked before the exp: off the answer ref_logp is 0 while logp (of
            # padding, say) can be very negative, and exp would overflow to inf,
            # which the mask cannot zero (inf * 0 = nan).
            log_ratio_ref = (batch["ref_logp"][idx, :T - 1] - logp) * act
            k3 = masked_mean(torch.exp(log_ratio_ref) - log_ratio_ref - 1.0, act)
            if config.advantage == "group":
                loss = loss + config.kl_coef * k3
            policy_opt.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
            policy_opt.step()
            stats = {"policy_loss": pg_loss.item(), "clip_frac": clip_frac.item(), "k3": k3.item(),
                     "approx_kl": masked_mean(0.5 * (logp.detach() - old_logp) ** 2, act).item()}

            if critic is not None:
                values = critic.values(ids, mask)[:, :-1].float()
                vf_loss = value_loss(values, batch["old_values"][idx, :T - 1], batch["returns"][idx, :T - 1], act,
                                     config.vf_clip)
                critic_opt.zero_grad(set_to_none=True)
                vf_loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
                critic_opt.step()
                stats["value_loss"] = vf_loss.item()
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
    return {k: v / steps for k, v in sums.items()}


# ------------------------------------------------------------------- main

def check_config(config: RLHFConfig) -> None:
    if config.reward_source not in ("rm", "oracle"):
        raise ValueError(f"reward_source must be rm or oracle, got {config.reward_source!r}")
    if config.advantage not in ("gae", "group"):
        raise ValueError(f"advantage must be gae or group, got {config.advantage!r}")
    if config.temperature <= 0:
        raise ValueError("PPO needs sampled answers: temperature must be > 0")
    if config.score_clip <= 0 or config.no_eos_penalty <= 0:
        # Otherwise an unfinished answer could tie or beat a finished one, or
        # every finished answer would score the same.
        raise ValueError("score_clip and no_eos_penalty must be > 0")
    if config.advantage == "group" and (config.group_size < 2 or config.rollout_batch % config.group_size):
        raise ValueError("group mode needs group_size >= 2 dividing rollout_batch")
    if config.select_best and config.eval_problems <= 0:
        raise ValueError("select_best needs val evaluations: eval_problems must be > 0")


def main(argv=None):
    config, preset = parse_config(RLHFConfig, argv)
    check_config(config)
    seed_everything(config.seed)
    device = get_device(config.device)
    domains = formal.resolve(config.domains)
    stage = "rlhf" if config.reward_source == "rm" else "ppo-checker"
    stem = os.path.splitext(config.out)[0]
    last_path, log_path = stem + "_last.pt", stem + ".json"
    if os.path.exists(last_path):
        os.remove(last_path)   # a stale last iterate from an earlier run must not pass for this one's

    policy, tokenizer, sft = load_checkpoint(config.sft, device, kind="policy")
    ref_policy, _, _ = load_checkpoint(config.sft, device, kind="policy")
    ref_policy.requires_grad_(False)
    # No dropout anywhere: the ratio and the KL must compare like with like.
    policy.eval()
    ref_policy.eval()

    # The reward model: the reward in rm mode, a diagnostic (its score of the
    # rollouts is logged) in oracle mode when there is one.
    reward_model, reward_mean, reward_std = None, 0.0, 1.0
    if config.reward_source == "rm" or (config.reward and os.path.exists(config.reward)):
        reward_model, rm_tokenizer, rm_ckpt = load_checkpoint(config.reward, device)
        if rm_ckpt["kind"] != "reward" or "reward_mean" not in rm_ckpt:
            raise ValueError(f"{config.reward} is not a reward.py checkpoint")
        if rm_tokenizer.alphabet != tokenizer.alphabet:
            raise ValueError("the reward model and the policy use different alphabets")
        reward_model.requires_grad_(False)
        reward_model.eval()
        reward_mean, reward_std = rm_ckpt["reward_mean"], rm_ckpt["reward_std"]

    critic = critic_opt = None
    if config.advantage == "gae":
        if config.reward_source == "rm":
            critic = critic_from_reward_model(reward_model, reward_mean, reward_std)
            critic.requires_grad_(True)
        else:
            critic, _, _ = load_checkpoint(config.sft, device, kind="reward")
            torch.nn.init.zeros_(critic.head.weight)
            torch.nn.init.zeros_(critic.head.bias)
        critic.eval()
        critic_opt = make_optimizer(critic, config.critic_lr, 0.0)
    policy_opt = make_optimizer(policy, config.lr, 0.0)

    schedule = rollout_problems(config, domains)
    val_problems = data.sample_problems(config.eval_problems * len(domains), "val", VAL_SEED, domains, unique=True) \
        if config.eval_problems > 0 else []
    sample_gen = torch.Generator(device=device).manual_seed(config.seed)
    shuffle_gen = torch.Generator().manual_seed(config.seed)
    log = {"stage": stage, "preset": preset, "config": asdict(config),
           "sft": {"path": config.sft, "stage": sft["stage"]}, "reward_mean": reward_mean, "reward_std": reward_std,
           "policy_samples": config.iters * config.rollout_batch, "iterations": [], "evals": []}
    target = f"the reward model {config.reward}" if config.reward_source == "rm" else "the checker"
    print(f"{stage}: PPO ({config.advantage}) against {target}, "
          f"from {config.sft} ({sft['stage']}), {config.iters} iterations x {config.rollout_batch} samples, "
          f"device {device}", flush=True)

    best = {"accuracy": -1.0, "iteration": None, "state": None, "results": None}

    def run_eval(it):
        results = evaluate(policy, tokenizer, val_problems, max_new_tokens=config.max_new_tokens)
        log["evals"].append({"iteration": it, "results": results})
        print(f"  val @ {it}: {format_results(results)}", flush=True)
        if results["all"]["accuracy"] > best["accuracy"]:
            best.update(accuracy=results["all"]["accuracy"], iteration=it, results=results,
                        state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                        if config.select_best else None)
        return results

    def write_log():
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1, ensure_ascii=False)
        os.replace(log_path + ".tmp", log_path)

    if val_problems:
        run_eval(0)
    last_results = log["evals"][-1]["results"] if log["evals"] else None
    for it in range(1, config.iters + 1):
        started = time.time()
        batch, stats = rollout(policy, ref_policy, tokenizer, schedule[it - 1], config, sample_gen, critic,
                               reward_model, reward_mean, reward_std)
        update = ppo_update(policy, batch, config, policy_opt, shuffle_gen, critic, critic_opt)
        entry = {"iteration": it, **stats, **update, "seconds": time.time() - started}
        log["iterations"].append(entry)
        rm_text = f"  rm {stats['rm_score']:+.3f}" if stats["rm_score"] is not None else ""
        by_domain = " ".join(f"{d} {a:.2f}" for d, a in stats["accuracy_by_domain"].items())
        vf_text = f"  vf {update['value_loss']:.4f}" if "value_loss" in update else ""
        print(f"iter {it:>4}  score {stats['score']:+.3f}{rm_text}  kl {stats['kl']:.3f}  acc {stats['accuracy']:.3f} "
              f"({by_domain})  fin {stats['finished']:.2f}  pg {update['policy_loss']:+.4f}{vf_text}  "
              f"clip {update['clip_frac']:.3f}  {entry['seconds']:.1f}s", flush=True)
        if val_problems and ((config.eval_interval > 0 and it % config.eval_interval == 0) or it == config.iters):
            last_results = run_eval(it)
        write_log()

    extra = {"run_config": asdict(config), "preset": preset, "sft": {"path": config.sft, "stage": sft["stage"]},
             "reward": config.reward if reward_model is not None else None}
    if config.select_best:
        save_checkpoint(last_path, policy, tokenizer, stage=stage, iteration=config.iters, val=last_results, **extra)
        policy.load_state_dict(best["state"])
        save_checkpoint(config.out, policy, tokenizer, stage=stage, iteration=best["iteration"], val=best["results"],
                        **extra)
        log["selected_iteration"] = best["iteration"]
        print(f"saved {config.out} (best val iterate, iteration {best['iteration']}) and {last_path} (last)")
    else:
        save_checkpoint(config.out, policy, tokenizer, stage=stage, iteration=config.iters, val=last_results, **extra)
        log["selected_iteration"] = config.iters
        print(f"saved {config.out}")
    write_log()
    return policy, log


if __name__ == "__main__":
    main()
