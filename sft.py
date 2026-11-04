"""Stage 2: supervised fine-tuning, declaration header -> proof.

    python sft.py --preset small --init out/pretrain.pt --out out/sft.pt

Each example is `<bos> header <ans> reference answer <eos>` for a train-split
problem; the loss covers the answer and the final <eos> only, so the model
learns to complete a header it is given, not to write headers.

Started from an SFT checkpoint instead of the pretrained one, this is the
"more SFT" control arm: the same training on fresh reference examples, chosen
by the data seed, e.g.

    python sft.py --init out/sft.pt --out out/sftplus_s1.pt --max_iters 600 --warmup_iters 20 --lr 1e-4 --seed 1001

Such checkpoints are recorded as stage "sft-continued".
"""

import time
from dataclasses import asdict

import torch

import data
import formal
from common import (check_fits, evaluate, finish_progress, format_results, get_device, load_checkpoint, load_progress,
                    lr_at, make_optimizer, save_checkpoint, save_progress, seed_everything, sft_batch)
from config import SFTConfig, parse_config

# The val problems every SFT run is scored on, whatever its own seed, so runs
# are compared on the same questions.
VAL_SEED = 0


def stage_after(init_stage: str) -> str:
    if init_stage == "pretrain":
        return "sft"
    if init_stage in ("sft", "sft-continued"):
        return "sft-continued"
    raise ValueError(f"sft.py starts from a pretrain or SFT checkpoint, not a {init_stage!r} one")


@torch.no_grad()
def reference_loss(model, examples, device: str, batch_size: int = 64) -> float:
    """Mean per-token loss on the reference answers of `examples`."""
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        x, y, mask = sft_batch(examples[start:start + batch_size], device)
        n = int((y != -1).sum())
        total += model(x, targets=y, attention_mask=mask)[1].item() * n
        count += n
    model.train()
    return total / max(count, 1)


def main(argv=None):
    config, preset = parse_config(SFTConfig, argv)
    seed_everything(config.seed)
    device = get_device(config.device)
    model, tokenizer, init = load_checkpoint(config.init, device, kind="policy")
    stage = stage_after(init["stage"])
    domains = formal.resolve(config.domains)

    problems = data.sample_problems(config.n_examples, "train", config.seed, domains)
    check_fits(tokenizer, problems, model.config.block_size)
    examples = [tokenizer.example_ids(p.prompt, p.reference) for p in problems]
    val_problems = data.sample_problems(config.eval_problems * len(domains), "val", VAL_SEED, domains, unique=True) \
        if config.eval_problems > 0 else []
    val_examples = [tokenizer.example_ids(p.prompt, p.reference) for p in val_problems]
    print(f"{stage}: from {config.init} ({init['stage']}), {len(examples):,} train examples, "
          f"{len(val_problems)} val problems, device {device}", flush=True)

    optimizer = make_optimizer(model, config.lr, config.weight_decay, config.beta2)
    gen = torch.Generator().manual_seed(config.seed)
    history, losses, start = [], [], 0
    run_config = asdict(config)
    state = load_progress(config.out, run_config)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        gen.set_state(state["gen"])
        history, losses, start = state["history"], state["losses"], state["iter"]
        print(f"resuming from iteration {start}", flush=True)
    started = time.time()
    model.train()
    for it in range(start, config.max_iters):
        lr = lr_at(it, config.lr, config.min_lr, config.warmup_iters, config.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = [examples[i] for i in torch.randint(len(examples), (config.batch_size,), generator=gen).tolist()]
        x, y, mask = sft_batch(batch, device)
        _, loss = model(x, targets=y, attention_mask=mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        losses.append(loss.item())

        if (it + 1) % config.eval_interval == 0 or it + 1 == config.max_iters:
            entry = {"iter": it + 1, "lr": lr, "train_loss": sum(losses) / len(losses),
                     "elapsed_s": time.time() - started}
            line = f"iter {it + 1:>6}  lr {lr:.2e}  train loss {entry['train_loss']:.4f}"
            if val_problems:
                entry["val_loss"] = reference_loss(model, val_examples, device)
                results = evaluate(model, tokenizer, val_problems, fmt="task", max_new_tokens=config.max_new_tokens)
                entry["val"] = results
                line += f"  val loss {entry['val_loss']:.4f}  val acc {format_results(results)}"
            history.append(entry)
            print(line + f"  {entry['elapsed_s']:.0f}s", flush=True)
            losses = []
        if config.save_interval > 0 and (it + 1) % config.save_interval == 0 and it + 1 < config.max_iters:
            save_progress(config.out, {"run_config": run_config, "iter": it + 1, "model": model.state_dict(),
                                       "optimizer": optimizer.state_dict(), "gen": gen.get_state(),
                                       "history": history, "losses": losses})

    save_checkpoint(config.out, model, tokenizer, stage=stage, run_config=run_config, history=history,
                    preset=preset, init={"path": config.init, "stage": init["stage"],
                                         "run_config": init.get("run_config", {})})
    finish_progress(config.out)
    print(f"saved {config.out}")
    return model, history


if __name__ == "__main__":
    main()
