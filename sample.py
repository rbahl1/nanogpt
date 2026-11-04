"""Shows what a checkpoint writes, with each answer's status and verdict.

    python sample.py out/sft.pt --prompt "example (p q : Prop) : p ∧ q → q ∧ p :="
    python sample.py out/rlhf.pt --n 3 --domains prop,nat          # random test problems
    python sample.py out/rlhf.pt --n 2 --samples 4 --temperature 0.8

With --prompt (repeatable) the model answers those headers; otherwise it
answers --n random problems per domain from --split, and their reference
answers are shown too. --temperature 0 (the default) decodes greedily. A
pretrain checkpoint is given `<bos> header ` (a document to continue), any
other `<bos> header <ans>`, as in evaluate.py.
"""

import argparse

import torch

import data
import formal
from common import generate_answers, get_device, load_checkpoint


def verdict(prompt: str, answer: str, status: str) -> str:
    if status != "finished":
        return status
    try:
        domain = formal.domain_for(prompt)
    except ValueError:
        return "finished, not a header any domain claims"
    return "finished, " + ("checks" if domain.verify(prompt, answer) else "does not check")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", action="append", default=[], help="a declaration header (repeatable)")
    parser.add_argument("--n", type=int, default=2, help="random problems per domain when no --prompt is given")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--split", default="test", choices=data.SPLITS)
    parser.add_argument("--samples", type=int, default=1, help="answers per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--format", default="auto", choices=("auto", "task", "document"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    model, tokenizer, ckpt = load_checkpoint(args.checkpoint, device, kind="policy")
    fmt = args.format if args.format != "auto" else ("document" if ckpt["stage"] == "pretrain" else "task")
    if args.prompt:
        items = [(p, None) for p in args.prompt]
    else:
        items = [(p.prompt, p.reference) for p in data.sample_problems(args.n * len(formal.resolve(args.domains)),
                                                                        args.split, args.seed, args.domains)]
    for prompt, _ in items:
        try:
            tokenizer.encode(prompt)
        except ValueError as err:
            parser.error(f"{prompt!r}: {err}")

    greedy = args.temperature <= 0
    samples = 1 if greedy else args.samples
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompts = [p for p, _ in items for _ in range(samples)]
    answers, statuses = generate_answers(model, tokenizer, prompts, fmt=fmt, max_new_tokens=args.max_new_tokens,
                                         greedy=greedy, temperature=args.temperature if not greedy else 1.0,
                                         top_k=args.top_k, generator=generator, statuses=True)
    decoding = "greedy" if greedy else f"{samples} samples at temperature {args.temperature}"
    print(f"{args.checkpoint}: stage {ckpt['stage']}, {fmt} format, {decoding}")
    for i, (prompt, reference) in enumerate(items):
        try:
            name = formal.domain_for(prompt).name
        except ValueError:
            name = "?"
        print(f"\n[{name}] {prompt}")
        for answer, status in zip(answers[i * samples:(i + 1) * samples], statuses[i * samples:(i + 1) * samples]):
            print(f"  -> {answer}\n     ({verdict(prompt, answer, status)})")
        if reference is not None:
            print(f"  reference: {reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
