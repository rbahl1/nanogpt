"""Hyperparameters for every stage, with size presets and command-line overrides.

Every stage script accepts `--preset {tiny,small,base}` and then `--<field> <value>`
for any field of its config class, e.g.

    python pretrain.py --preset small --max_iters 5000 --device cuda

`tiny` exists for tests and smoke runs (the whole pipeline in a minute or two
on a laptop CPU). `small` is the default; the whole pipeline takes about an
hour on a 4-core CPU. `base` is the nanoGPT "shakespeare-char" size and wants a
GPU.

The three post-training arms compared in REPORT.md (PPO against a learned
reward model, PPO against the checker, rejection-sampling fine-tuning) start
from the same SFT checkpoint and draw the same number of policy samples:
RLHFConfig.iters * rollout_batch = RFTConfig.rounds * prompts_per_round *
samples_per_prompt (9,600 in the small preset).
"""

import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class PretrainConfig:
    out: str = "out/pretrain.pt"
    # model (later stages inherit the architecture from the checkpoint)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False
    # data
    domains: str = "all"
    n_docs: int = 160_000
    # optimisation
    max_iters: int = 5000
    batch_size: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 20
    save_interval: int = 250      # iterations between resumable progress saves
    seed: int = 1337
    device: str = "auto"


@dataclass
class SFTConfig:
    init: str = "out/pretrain.pt"
    out: str = "out/sft.pt"
    domains: str = "all"
    n_examples: int = 80_000
    max_iters: int = 3000
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 50
    weight_decay: float = 0.0
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_problems: int = 75       # per domain, drawn from the val split
    max_new_tokens: int = 200
    save_interval: int = 250      # iterations between resumable progress saves
    seed: int = 2024
    device: str = "auto"


@dataclass
class RewardConfig:
    sft: str = "out/sft.pt"
    out: str = "out/reward.pt"
    preferences: str = "out/preferences.jsonl"   # where the checker-labelled comparisons are written
    human_preferences: str = ""   # JSONL of comparisons made by people, used instead of the checker's
    domains: str = "all"
    # preference collection
    n_prompts: int = 6000
    samples_per_prompt: int = 8
    temperature: float = 1.0
    max_new_tokens: int = 200
    include_reference: bool = True
    max_pairs_per_prompt: int = 8
    # reward model training
    max_iters: int = 3000
    batch_size: int = 64
    lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_iters: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_frac: float = 0.05
    eval_interval: int = 500
    save_interval: int = 250      # iterations between resumable progress saves
    seed: int = 7
    device: str = "auto"


@dataclass
class RLHFConfig:
    sft: str = "out/sft.pt"
    reward: str = "out/reward.pt"
    out: str = "out/rlhf.pt"
    reward_source: str = "rm"     # "rm": learned reward model (RLHF); "oracle": the checker itself
    advantage: str = "gae"        # "gae": learned critic (InstructGPT); "group": each answer vs its siblings (GRPO)
    group_size: int = 4           # answers sampled per prompt when advantage = "group"
    select_best: bool = False     # keep the iterate with the best val accuracy instead of the last one
    domains: str = "all"
    iters: int = 150
    rollout_batch: int = 64
    ppo_epochs: int = 2
    minibatch_size: int = 16
    lr: float = 5e-6
    critic_lr: float = 5e-5
    kl_coef: float = 0.1
    clip_range: float = 0.2
    vf_clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    max_new_tokens: int = 200
    temperature: float = 1.0
    score_clip: float = 3.0       # finished answers' normalised scores are clipped to [-score_clip, score_clip]
    no_eos_penalty: float = 1.0   # an unfinished answer scores -score_clip - no_eos_penalty, below every finished one
    grad_clip: float = 1.0
    eval_interval: int = 10
    eval_problems: int = 75       # per domain, drawn from the val split
    seed: int = 99
    device: str = "auto"


@dataclass
class RFTConfig:
    """Rejection-sampling fine-tuning (expert iteration): sample answers from
    the current policy on train-split prompts, keep the ones the checker
    accepts, fine-tune on them, repeat."""
    sft: str = "out/sft.pt"
    out: str = "out/rft.pt"
    domains: str = "all"
    rounds: int = 2
    prompts_per_round: int = 1200
    samples_per_prompt: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 200
    keep_per_prompt: int = 1      # distinct verified answers kept per prompt, shortest first
    replay_frac: float = 0.0      # fraction of each fine-tuning batch drawn from fresh SFT (reference) examples
                                  # (0: pure self-training on verified samples; needs epochs_per_round 0)
    epochs_per_round: float = 3.0  # passes over each round's verified examples (0: a fixed iters_per_round steps)
    iters_per_round: int = 300
    batch_size: int = 64
    lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_iters: int = 20
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_problems: int = 75       # per domain, drawn from the val split, logged after each round
    seed: int = 123
    device: str = "auto"


@dataclass
class EvalConfig:
    n: int = 500                  # problems per domain (greedy pass@1)
    split: str = "test"
    domains: str = "all"
    max_new_tokens: int = 200
    batch_size: int = 64
    passk_n: int = 0              # problems per domain for sampled pass@k (0: skip)
    passk_samples: int = 16       # samples per problem for pass@k
    temperature: float = 1.0      # for pass@k sampling
    lean: bool = False            # also compile every greedy answer with Lean 4
    seed: int = 31337
    json: str = ""                # optional path for machine-readable results
    device: str = "auto"


PRESETS = {
    "tiny": {
        PretrainConfig: dict(n_layer=2, n_head=2, n_embd=64, n_docs=4000, max_iters=150, batch_size=16,
                             warmup_iters=10, eval_interval=50, eval_iters=5),
        SFTConfig: dict(n_examples=2000, max_iters=100, batch_size=32, warmup_iters=10, eval_interval=50,
                        eval_problems=5, max_new_tokens=64),
        RewardConfig: dict(n_prompts=48, samples_per_prompt=2, max_new_tokens=64, max_iters=60, batch_size=16,
                           warmup_iters=5, eval_interval=30, val_frac=0.2, lr=1e-4),
        RLHFConfig: dict(iters=4, rollout_batch=16, ppo_epochs=2, minibatch_size=8, max_new_tokens=64,
                         eval_interval=2, eval_problems=4, lr=1e-5),
        RFTConfig: dict(rounds=2, prompts_per_round=16, samples_per_prompt=2, max_new_tokens=64,
                        epochs_per_round=2.0, batch_size=16, warmup_iters=2, eval_problems=4),
        EvalConfig: dict(n=10, max_new_tokens=64, batch_size=40, passk_n=2, passk_samples=4),
    },
    "small": {},
    "base": {
        PretrainConfig: dict(n_layer=6, n_head=6, n_embd=384, dropout=0.1, n_docs=500_000, max_iters=10_000,
                             batch_size=64, lr=6e-4, min_lr=6e-5, warmup_iters=200, eval_interval=500),
        SFTConfig: dict(n_examples=200_000, max_iters=5000, lr=1e-4, min_lr=1e-5),
        RewardConfig: dict(n_prompts=10_000, max_iters=3000, lr=3e-5, min_lr=3e-6),
        RLHFConfig: dict(iters=400, rollout_batch=128, minibatch_size=32, lr=2e-6, critic_lr=2e-5),
        RFTConfig: dict(rounds=4, prompts_per_round=3200, lr=3e-5, min_lr=3e-6),
        EvalConfig: dict(n=500),
    },
}


def _parse_bool(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def parse_config(cls, argv=None, positional: str = None):
    """Builds a `cls` from `--preset` plus `--field value` overrides.

    If `positional` is given, remaining positional arguments are returned under
    that name (evaluate.py uses it for its list of checkpoints).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="small", choices=sorted(PRESETS))
    known, _ = pre.parse_known_args(argv)

    defaults = dataclasses.asdict(cls())
    defaults.update(PRESETS[known.preset].get(cls, {}))

    parser = argparse.ArgumentParser(description=cls.__doc__, parents=[pre], allow_abbrev=False)
    for f in dataclasses.fields(cls):
        kind = _parse_bool if f.type in (bool, "bool") else {"int": int, "float": float, "str": str}.get(f.type, f.type)
        parser.add_argument(f"--{f.name}", type=kind, default=defaults[f.name])
    if positional:
        parser.add_argument(positional, nargs="+")
    # Intermixed, so checkpoints may come before, after or between options.
    args = vars(parser.parse_intermixed_args(argv) if positional else parser.parse_args(argv))
    extra = args.pop(positional) if positional else None
    preset = args.pop("preset")
    config = cls(**args)
    return (config, preset, extra) if positional else (config, preset)
