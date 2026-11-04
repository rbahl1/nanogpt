"""Shared machinery: devices, checkpoints, batching, the learning-rate schedule,
sampling answers, and checker-verified evaluation."""

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import formal
from model import GPT, GPTConfig, RewardModel
from tokenizer import ANS, BOS, EOS, PAD, SPECIAL_TOKENS, Tokenizer


# --------------------------------------------------------------------- setup

def get_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


# --------------------------------------------------------------- checkpoints

def save_checkpoint(path: str, model, tokenizer: Tokenizer, stage: str, run_config=None, **extra) -> None:
    """Saves weights and metadata only (no optimiser state): enough to evaluate
    or to start the next stage. kind is "policy" for a GPT, "reward" for a
    RewardModel."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kind = "reward" if isinstance(model, RewardModel) else "policy"
    cfg = model.config
    ckpt = {
        "kind": kind,
        "stage": stage,
        "model_config": {k: getattr(cfg, k) for k in GPTConfig.__dataclass_fields__},
        "alphabet": tokenizer.alphabet,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "run_config": dict(run_config or {}),
    }
    ckpt.update(extra)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, device: str = "cpu", kind: str = None):
    """Returns (model, tokenizer, ckpt). kind=None loads whatever was saved;
    kind="policy" requires a GPT checkpoint; kind="reward" returns a
    RewardModel, built from a reward checkpoint or, from a policy checkpoint,
    with the policy as its trunk and a fresh head."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = GPTConfig(**ckpt["model_config"])
    tokenizer = Tokenizer(ckpt["alphabet"])
    saved = ckpt["kind"]
    want = kind or saved
    if want == "policy":
        if saved != "policy":
            raise ValueError(f"{path} is a {saved} checkpoint, not a policy")
        model = GPT(config)
        model.load_state_dict(ckpt["state_dict"])
    elif want == "reward":
        model = RewardModel(config)
        if saved == "reward":
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.gpt.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError(f"unknown checkpoint kind {want!r}")
    return model.to(device), tokenizer, ckpt


# ------------------------------------------------------------------ resuming
# Long stages save their full training state (weights, optimiser, random
# states, history) every few hundred steps to <out>.progress, and pick it up
# again when rerun with the same configuration, so an interrupted run loses
# minutes, not the run. The file is removed once the final checkpoint exists.

def progress_path(out: str) -> str:
    return out + ".progress"


def save_progress(out: str, state: dict) -> None:
    state = dict(state, rng={"torch": torch.get_rng_state(), "python": random.getstate(),
                             "numpy": np.random.get_state()})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = progress_path(out) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, progress_path(out))


def load_progress(out: str, run_config: dict):
    """The saved state of an interrupted run with the same configuration, or
    None. (Our own file, holding random-number states, so not weights_only.)"""
    path = progress_path(out)
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("run_config") != run_config:
        print(f"ignoring {path}: it was saved by a run with a different configuration", flush=True)
        return None
    torch.set_rng_state(state["rng"]["torch"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    return state


def finish_progress(out: str) -> None:
    if os.path.exists(progress_path(out)):
        os.remove(progress_path(out))


# ------------------------------------------------------------------ batching

def pad_batch(seqs, pad: int = PAD, left: bool = False):
    """(ids, mask) LongTensor/BoolTensor (B, T) for a list of id lists."""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if left:
            ids[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, T - len(s):] = True
        else:
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = True
    return ids, mask


def sft_batch(examples, device: str):
    """examples: list of (ids, n_prompt) from Tokenizer.example_ids. Returns
    (x, y, mask) for next-token training with the loss on answer tokens only
    (targets of prompt and padding positions are -1)."""
    x, mask = pad_batch([ids[:-1] for ids, _ in examples])
    y = torch.full_like(x, -1)
    for i, (ids, n_prompt) in enumerate(examples):
        targets = torch.tensor(ids[1:], dtype=torch.long)
        targets[: n_prompt - 1] = -1
        y[i, : len(targets)] = targets
    return x.to(device), y.to(device), mask.to(device)


def lr_at(it: int, lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to min_lr at `total`."""
    if it < warmup:
        return lr * (it + 1) / warmup
    if it >= total:
        return min_lr
    ratio = (it - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


def make_optimizer(model, lr: float, weight_decay: float, beta2: float = 0.99):
    """AdamW with weight decay on matrices only (as in nanoGPT)."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, beta2))


def token_logprobs(model, ids, mask):
    """log p(ids[:, t] | ids[:, :t]) for t >= 1, shape (B, T-1)."""
    logits, _ = model(ids, attention_mask=mask)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, ids[:, 1:, None]).squeeze(-1)


def check_fits(tokenizer: Tokenizer, problems, block_size: int) -> None:
    """Raises if a reference example (prompt, answer and the special tokens)
    does not fit the context: it could be neither trained on nor produced."""
    for p in problems:
        ids, _ = tokenizer.example_ids(p.prompt, p.reference)
        if len(ids) > block_size:
            raise ValueError(f"example longer than the context ({len(ids)} > {block_size}): {p.prompt}")


# --------------------------------------------------------------- generation

def response_status(new_ids):
    """(answer, status) for generated ids: "finished" if they end in <eos> with
    no other special token before it, "malformed" if a special token other than
    <eos> comes first, "cut off" if no <eos> was produced."""
    n = len(SPECIAL_TOKENS)
    body = []
    for t in new_ids:
        if t == EOS:
            return body, "finished"
        if t < n:
            return body, "malformed"
        body.append(t)
    return body, "cut off"


def prompt_ids_for(tokenizer: Tokenizer, prompt: str, fmt: str) -> list:
    """The ids a model is given for `prompt`: task format `<bos> prompt <ans>`
    (fine-tuned models), document format `<bos> prompt + " "` (the pretrained
    model completes the declaration)."""
    if fmt == "task":
        return tokenizer.prompt_ids(prompt)
    if fmt == "document":
        return [BOS] + tokenizer.encode(prompt + " ")
    raise ValueError(f"unknown format {fmt!r}")


@torch.no_grad()
def generate_answers(model, tokenizer: Tokenizer, prompts, fmt: str = "task", max_new_tokens: int = 128,
                     greedy: bool = True, temperature: float = 1.0, top_k: int = None, generator=None,
                     batch_size: int = 64, statuses: bool = False, return_ids: bool = False):
    """Answers (strings) for a list of prompts (strings or Problems), in order.
    Batches prompts of similar length together. With statuses=True also returns
    each answer's status (see response_status); with return_ids=True also the
    generated ids (including the final <eos> when there is one)."""
    prompts = [p.prompt if isinstance(p, formal.Problem) else p for p in prompts]
    was_training = model.training
    model.eval()
    encoded = [prompt_ids_for(tokenizer, p, fmt) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    answers, stats, raw = [None] * len(prompts), [None] * len(prompts), [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        outs = model.generate([encoded[i] for i in chunk], max_new_tokens, eos_id=EOS, pad_id=PAD,
                              temperature=temperature, top_k=top_k, greedy=greedy, generator=generator)
        for i, out in zip(chunk, outs):
            body, status = response_status(out)
            answers[i] = tokenizer.decode(body)
            stats[i] = status
            raw[i] = out
    if was_training:
        model.train()
    result = [answers]
    if statuses:
        result.append(stats)
    if return_ids:
        result.append(raw)
    return result[0] if len(result) == 1 else tuple(result)


# ---------------------------------------------------------------- evaluation

def evaluate(model, tokenizer: Tokenizer, problems, fmt: str = "task", max_new_tokens: int = 128,
             batch_size: int = 64, records: bool = False):
    """Greedy accuracy judged by the checkers: {"<domain>": {"n", "correct",
    "accuracy"}, ..., "all": {...}}, plus "records" (one dict per problem) if
    records=True. An answer that is cut off or malformed counts as wrong."""
    answers, stats = generate_answers(model, tokenizer, problems, fmt=fmt, max_new_tokens=max_new_tokens,
                                      greedy=True, batch_size=batch_size, statuses=True)
    out, recs = {}, []
    for p, a, s in zip(problems, answers, stats):
        ok = s == "finished" and formal.verify(p.prompt, a)
        for key in (p.domain, "all"):
            r = out.setdefault(key, {"n": 0, "correct": 0})
            r["n"] += 1
            r["correct"] += int(ok)
        recs.append({"domain": p.domain, "prompt": p.prompt, "answer": a, "status": s, "ok": ok})
    for r in out.values():
        r["accuracy"] = r["correct"] / r["n"] if r["n"] else 0.0
    order = [d for d in formal.DOMAINS if d in out] + ["all"]
    out = {k: out[k] for k in order if k in out}
    if records:
        out["records"] = recs
    return out


def format_results(results: dict) -> str:
    parts = [f"{k} {v['accuracy']:.3f} ({v['correct']}/{v['n']})" for k, v in results.items() if k != "records"]
    return ", ".join(parts)
