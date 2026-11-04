"""Exports a policy checkpoint for the in-browser demo (web/index.html).

    python export_web.py out/rlhf.pt --out web/dist [--artifact]

writes two files into --out:

  model.json   the weights and what the page shows about them: the model
               config, alphabet, stage, parameter count, domains, example
               prompts from the test split, and every tensor as base64
               little-endian float32 with its shape (the tied lm_head.weight
               is left out; it is the token embedding). web/gpt.js reads this
               format, as reference/old_rlhf_model.json shows.
  index.html   web/index.html with web/gpt.js and web/checker.js inlined, so
               the page needs only model.json beside it. A standalone page
               starts with a doctype, charset, viewport and zero body margin;
               with --artifact it does not, since the host that publishes it
               supplies the document skeleton.

Serve the directory over HTTP (the page fetches model.json), e.g.
`python -m http.server -d web/dist`.
"""

import argparse
import base64
import json
import os

import numpy as np
import torch

import data
import formal
from common import load_checkpoint
from model import GPT, GPTConfig
from tokenizer import Tokenizer

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
TIED = "lm_head.weight"
# What the publisher's skeleton supplied besides the doctype: the page's own
# styles assume a zero body margin and a device-width viewport.
STANDALONE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
                   '<style>body{margin:0}</style>\n')


def encode_tensor(t: torch.Tensor) -> dict:
    array = t.detach().cpu().float().numpy().astype("<f4")
    return {"shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}


def decode_tensor(entry: dict) -> torch.Tensor:
    array = np.frombuffer(base64.b64decode(entry["data"]), dtype="<f4").reshape(entry["shape"])
    return torch.from_numpy(array.copy())


def bundle_for(model: GPT, tokenizer: Tokenizer, stage: str, run_config: dict, n_examples: int = 6,
               seed: int = 0) -> dict:
    """The model.json contents for a policy: n_examples test-split prompts per
    domain it was trained on (the page offers them as held-out examples)."""
    trained = formal.resolve(run_config.get("domains", "all"))
    examples = data.sample_problems(n_examples * len(trained), "test", seed, trained, unique=True)
    bundle = {
        "config": {k: getattr(model.config, k) for k in GPTConfig.__dataclass_fields__},
        "alphabet": tokenizer.alphabet,
        "stage": stage,
        "params": model.num_params(),
        "domains": list(formal.DOMAINS),
        "trained_domains": trained,
    }
    if "reward_source" in run_config:
        bundle["reward_source"] = run_config["reward_source"]
    bundle["examples"] = [{"domain": p.domain, "prompt": p.prompt} for p in examples]
    bundle["tensors"] = {k: encode_tensor(v) for k, v in model.state_dict().items() if k != TIED}
    return bundle


def model_from_bundle(bundle: dict) -> GPT:
    """The GPT a bundle describes (the inverse of bundle_for's tensors)."""
    model = GPT(GPTConfig(**bundle["config"]))
    state = {k: decode_tensor(v) for k, v in bundle["tensors"].items()}
    state[TIED] = state["transformer.wte.weight"]
    model.load_state_dict(state)
    return model.eval()


def strip_skeleton(html: str) -> str:
    """web/index.html was recovered from a published copy of the page, so it
    may still carry the skeleton the publisher wrapped around the page's own
    source (a first line `<!doctype html><html><head>...</head><body>` and a
    final `</body></html>`). Removes it, leaving the source as written."""
    first, sep, rest = html.partition("\n")
    tail = "</body></html>"
    if sep and first.lower().startswith("<!doctype html><html><head>") and first.endswith("<body>") \
            and rest.rstrip().endswith(tail):
        return rest.rstrip()[:-len(tail)].rstrip() + "\n"
    return html


def page(artifact: bool) -> str:
    """web/index.html with its two scripts inlined."""
    with open(os.path.join(WEB, "index.html"), encoding="utf-8") as f:
        html = strip_skeleton(f.read())
    for name in ("gpt.js", "checker.js"):
        tag = f'<script src="{name}"></script>'
        if html.count(tag) != 1:
            raise ValueError(f"web/index.html should contain {tag} exactly once")
        with open(os.path.join(WEB, name), encoding="utf-8") as f:
            code = f.read()
        if "</script" in code.lower():
            raise ValueError(f"web/{name} contains '</script', which would end the inlined script early")
        html = html.replace(tag, "<script>\n" + code.rstrip("\n") + "\n</script>")
    return html if artifact else STANDALONE_HEAD + html


def export(checkpoint: str, out_dir: str, artifact: bool = False, n_examples: int = 6, seed: int = 0) -> dict:
    model, tokenizer, ckpt = load_checkpoint(checkpoint, "cpu", kind="policy")
    bundle = bundle_for(model, tokenizer, ckpt["stage"], ckpt.get("run_config", {}), n_examples, seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page(artifact))
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a policy checkpoint (pretrain, SFT, RLHF, ...)")
    parser.add_argument("--out", default="web/dist", help="directory for model.json and index.html")
    parser.add_argument("--artifact", action="store_true",
                        help="no doctype/charset/viewport: the publishing host adds them")
    parser.add_argument("--examples", type=int, default=6, help="example prompts per trained domain")
    parser.add_argument("--seed", type=int, default=0, help="seed for drawing the example prompts")
    args = parser.parse_args(argv)
    bundle = export(args.checkpoint, args.out, args.artifact, args.examples, args.seed)
    size = os.path.getsize(os.path.join(args.out, "model.json"))
    print(f"wrote {args.out}/model.json ({size / 1e6:.1f} MB, {bundle['params']:,} parameters, stage "
          f"{bundle['stage']}) and {args.out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
