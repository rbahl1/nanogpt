"""Stage 1: next-token pretraining on whole Lean 4 declarations.

    python pretrain.py --preset small --out out/pretrain.pt [--field value ...]

Each document is one declaration, `<bos> example ... := proof <eos>`, drawn
from the train split of the requested domains (data.pretrain_documents). The
model learns the language of the declarations, header and proof together, but
never sees the task format (`<ans>`); SFT teaches that. 1% of the documents are
held out for a validation loss.
"""

import hashlib
import time
from dataclasses import asdict

import torch

import data
from common import (finish_progress, get_device, load_progress, lr_at, make_optimizer, pad_batch, save_checkpoint,
                    save_progress, seed_everything)
from config import PretrainConfig, parse_config
from model import GPT, GPTConfig
from tokenizer import Tokenizer


def held_out(document: str) -> bool:
    """Whether a document is in the 1% kept for the validation loss. Chosen by
    a hash of the text, not at random: short declarations are often drawn more
    than once (a quarter of nat documents are repeats), and every copy must
    land on the same side for the validation loss to be on unseen text."""
    return int.from_bytes(hashlib.sha256(document.encode("utf-8")).digest()[:8], "big") % 100 == 0


def document_batch(seqs, device: str):
    """(x, y, mask) for next-token prediction on right-padded documents:
    targets are -1 wherever the next token is padding."""
    ids, mask = pad_batch(seqs)
    x, y = ids[:, :-1], ids[:, 1:].clone()
    y[~mask[:, 1:]] = -1
    return x.to(device), y.to(device), mask[:, :-1].to(device)


def draw(seqs, batch_size: int, generator: torch.Generator) -> list:
    return [seqs[i] for i in torch.randint(len(seqs), (batch_size,), generator=generator).tolist()]


@torch.no_grad()
def estimate_loss(model, seqs, batch_size: int, n_batches: int, device: str, seed: int) -> float:
    """Mean loss over n_batches random batches. The batches are the same at
    every call (fixed seed), so successive estimates differ only because the
    model did."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(n_batches):
        x, y, mask = document_batch(draw(seqs, batch_size, gen), device)
        losses.append(model(x, targets=y, attention_mask=mask)[1].item())
    model.train()
    return sum(losses) / len(losses)


def main(argv=None):
    config, preset = parse_config(PretrainConfig, argv)
    seed_everything(config.seed)
    device = get_device(config.device)
    tokenizer = Tokenizer()

    docs = data.pretrain_documents(config.n_docs, config.seed, config.domains)
    seqs = [tokenizer.document_ids(d) for d in docs]
    longest = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    if len(seqs[longest]) > config.block_size:
        raise ValueError(f"document longer than the context ({len(seqs[longest])} > {config.block_size} tokens): "
                         f"{docs[longest]}")
    is_val = [held_out(d) for d in docs]
    val = [s for s, v in zip(seqs, is_val) if v]
    train = [s for s, v in zip(seqs, is_val) if not v]
    if not val:
        raise ValueError(f"none of the {len(docs)} documents was held out for validation; raise n_docs")

    model = GPT(GPTConfig(block_size=config.block_size, vocab_size=tokenizer.vocab_size, n_layer=config.n_layer,
                          n_head=config.n_head, n_embd=config.n_embd, dropout=config.dropout, bias=config.bias))
    model.to(device)
    optimizer = make_optimizer(model, config.lr, config.weight_decay, config.beta2)
    print(f"pretrain: {model.num_params():,} parameters, {len(train):,} train / {len(val):,} val documents "
          f"(mean {sum(map(len, seqs)) / len(seqs):.0f} tokens), device {device}", flush=True)

    gen = torch.Generator().manual_seed(config.seed)
    history, start = [], 0
    run_config = asdict(config)
    state = load_progress(config.out, run_config)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        gen.set_state(state["gen"])
        history, start = state["history"], state["iter"]
        print(f"resuming from iteration {start}", flush=True)
    tokens, busy, started = 0, 0.0, time.time()
    model.train()
    for it in range(start, config.max_iters):
        lr = lr_at(it, config.lr, config.min_lr, config.warmup_iters, config.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        t0 = time.time()
        x, y, mask = document_batch(draw(train, config.batch_size, gen), device)
        _, loss = model(x, targets=y, attention_mask=mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        tokens += int(mask.sum())
        busy += time.time() - t0

        if (it + 1) % config.eval_interval == 0 or it + 1 == config.max_iters:
            train_loss = estimate_loss(model, train, config.batch_size, config.eval_iters, device, config.seed + 1)
            val_loss = estimate_loss(model, val, config.batch_size, config.eval_iters, device, config.seed + 2)
            entry = {"iter": it + 1, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
                     "tokens_per_s": tokens / max(busy, 1e-9), "elapsed_s": time.time() - started}
            history.append(entry)
            print(f"iter {it + 1:>6}  lr {lr:.2e}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"{entry['tokens_per_s']:,.0f} tok/s  {entry['elapsed_s']:.0f}s", flush=True)
            tokens, busy = 0, 0.0
        if config.save_interval > 0 and (it + 1) % config.save_interval == 0 and it + 1 < config.max_iters:
            save_progress(config.out, {"run_config": run_config, "iter": it + 1, "model": model.state_dict(),
                                       "optimizer": optimizer.state_dict(), "gen": gen.get_state(),
                                       "history": history})

    save_checkpoint(config.out, model, tokenizer, stage="pretrain", run_config=run_config, history=history,
                    preset=preset)
    finish_progress(config.out)
    print(f"saved {config.out}")
    return model, history


if __name__ == "__main__":
    main()
