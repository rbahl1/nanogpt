"""The model: PyTorch against web/gpt.js on the old trained weights, the
key/value cache and padding against plain full forward passes, the reward
model's last-token read-out, the loss masks of pretraining and SFT batches, an
export with biases under web/gpt.js, and the parameter count."""

import json
import os
import subprocess

import pytest
import torch
import torch.nn.functional as F

from common import pad_batch, sft_batch
from config import PretrainConfig, parse_config
from export_web import bundle_for, model_from_bundle
from model import GPT, GPTConfig, RewardModel
from pretrain import document_batch
from tokenizer import ANS, BOS, EOS, PAD, Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(ROOT, "reference", "old_rlhf_model.json")


@pytest.fixture(scope="module")
def old_bundle():
    with open(OLD, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def old_model(old_bundle):
    return model_from_bundle(old_bundle)


@pytest.fixture(scope="module")
def tokenizer(old_bundle):
    return Tokenizer(old_bundle["alphabet"])


def task_prompts(bundle, tokenizer):
    return [tokenizer.prompt_ids(ex["prompt"]) for ex in bundle["examples"]]


def js_logits(node, bundle_path, ids):
    proc = subprocess.run([node, os.path.join(ROOT, "tests", "js_logits.js"), bundle_path, json.dumps(ids)],
                          capture_output=True, text=True, check=True)
    return torch.tensor(json.loads(proc.stdout))


def test_web_gpt_matches_pytorch(node, old_bundle, old_model, tokenizer):
    ids, _ = tokenizer.example_ids(old_bundle["examples"][1]["prompt"], "fun h a => h a")
    with torch.no_grad():
        expected = old_model(torch.tensor([ids]))[0][0]
    got = js_logits(node, OLD, ids)
    assert got.shape == expected.shape
    assert (got - expected).abs().max().item() < 1e-4


def test_export_reproduces_the_old_bundle(old_bundle, old_model, tokenizer):
    """Re-exporting the weights read from the old bundle gives the same bytes:
    export_web writes exactly the format web/gpt.js reads."""
    bundle = bundle_for(old_model, tokenizer, "rlhf", {"domains": "nat,linear", "reward_source": "rm"}, n_examples=1)
    assert bundle["config"] == old_bundle["config"]
    assert bundle["params"] == old_bundle["params"]
    assert list(bundle["tensors"]) == list(old_bundle["tensors"])
    assert bundle["tensors"] == old_bundle["tensors"]
    assert list(bundle)[:-2] == [k for k in old_bundle if k not in ("examples", "tensors")]


def full_forward_greedy(model, prompt, max_new_tokens):
    ids = list(prompt)
    out = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            nxt = int(model(torch.tensor([ids]))[0][0, -1].argmax())
            out.append(nxt)
            if nxt == EOS:
                break
            ids.append(nxt)
    return out


def test_kv_cache_greedy_equals_full_forward(old_bundle, old_model, tokenizer):
    for prompt in task_prompts(old_bundle, tokenizer)[:3]:
        cached = old_model.generate([prompt], 32, eos_id=EOS, greedy=True)[0]
        assert cached == full_forward_greedy(old_model, prompt, 32)
        assert len(cached) > 3


def test_left_padded_batch_generates_like_single_prompts(old_bundle, old_model, tokenizer):
    prompts = task_prompts(old_bundle, tokenizer)[:8]
    assert len({len(p) for p in prompts}) > 3          # the batch really is padded
    batch = old_model.generate(prompts, 48, eos_id=EOS, greedy=True)
    alone = [old_model.generate([p], 48, eos_id=EOS, greedy=True)[0] for p in prompts]
    assert batch == alone
    # and the trained model does finish its answers
    assert sum(EOS in out for out in batch) >= 6


def test_each_prompt_gets_its_own_answer_budget():
    """A prompt may be answered with up to min(max_new_tokens, block_size - its
    own length) tokens, whatever else is in the batch. A long neighbour must
    not cut a short prompt's answer off: RLHF samples its 64 rollout prompts as
    one batch, and fol prompts run to 162 of the 256 positions, which shrank
    almost every rollout answer's budget from 128 to about 105 tokens while
    RFT (length-sorted batches) and evaluation kept 128."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(block_size=32, vocab_size=20, n_layer=2, n_head=2, n_embd=16)).eval()
    short, long = [1, 5, 6], [1] + [7] * 24
    never = -1                       # an <eos> the model cannot emit: every answer runs to its budget
    alone = [model.generate([p], 16, eos_id=never, greedy=True)[0] for p in (short, long)]
    assert [len(a) for a in alone] == [16, 32 - 25]
    assert model.generate([short, long], 16, eos_id=never, greedy=True) == alone
    assert model.generate([long, short], 16, eos_id=never, greedy=True) == alone[::-1]
    # a prompt that fills the context gets no answer, and does not stop the others
    full = [1] * 32
    assert model.generate([short, full], 16, eos_id=never, greedy=True) == [alone[0], []]


def test_reward_score_reads_the_last_real_token():
    torch.manual_seed(0)
    rm = RewardModel(GPTConfig(block_size=32, vocab_size=20, n_layer=2, n_head=2, n_embd=16)).eval()
    seqs = [[1, 5, 6, 7, 8, 9, 2], [1, 4, 2], [1, 7, 7, 7, 7, 2]]
    with torch.no_grad():
        alone = torch.stack([rm.score(torch.tensor([s]))[0] for s in seqs])
        for left in (False, True):
            ids, mask = pad_batch(seqs, left=left)
            assert torch.allclose(rm.score(ids, attention_mask=mask), alone, atol=1e-5)
        # the score is the per-position value at the last real token
        values = rm.values(torch.tensor([seqs[0]]))[0]
        assert torch.allclose(alone[0], values[-1])


def test_loss_ignores_minus_one_targets():
    torch.manual_seed(0)
    model = GPT(GPTConfig(block_size=32, vocab_size=20, n_layer=2, n_head=2, n_embd=16)).eval()
    seqs = [[1, 5, 6, 7, 8, 9, 2], [1, 4, 11, 2]]
    with torch.no_grad():
        per_seq = []
        for s in seqs:
            logits = model(torch.tensor([s[:-1]]))[0][0]
            per_seq.append(F.cross_entropy(logits, torch.tensor(s[1:]), reduction="sum"))
        expected = sum(per_seq) / sum(len(s) - 1 for s in seqs)
        for pad in (0, 13):                             # what the padding holds must not matter
            ids, mask = pad_batch(seqs, pad=pad)
            x, y = ids[:, :-1], ids[:, 1:].clone()
            y[~mask[:, 1:]] = -1
            _, loss = model(x, targets=y, attention_mask=mask[:, :-1])
            assert torch.allclose(loss, expected, atol=1e-6)


def test_sft_batch_trains_on_the_answer_and_eos_only():
    """Targets are the answer's tokens and the final <eos>, each predicted
    from the token before it (the first from <ans>); the header, <ans> itself
    and the padding are -1."""
    tok = Tokenizer()
    long_ = tok.example_ids("example : 2 + 3 =", "5 := by decide")
    short = tok.example_ids("example : 1 =", "1")
    x, y, mask = sft_batch([long_, short], "cpu")
    for row, (ids, n_prompt) in enumerate([long_, short]):
        answer = ids[n_prompt:]
        assert ids[n_prompt - 1] == ANS and answer[-1] == EOS
        assert x[row, :len(ids) - 1].tolist() == ids[:-1] and mask[row].sum() == len(ids) - 1
        trained = (y[row] != -1).nonzero().squeeze(1).tolist()
        assert trained == list(range(n_prompt - 1, len(ids) - 1))
        assert y[row, trained].tolist() == answer
        assert x[row, n_prompt - 1].item() == ANS
    assert (x[1, len(short[0]) - 1:] == PAD).all() and not mask[1, len(short[0]) - 1:].any()


def test_document_batch_trains_on_every_next_token_up_to_eos():
    tok = Tokenizer()
    docs = [tok.document_ids("example : 2 + 3 = 5 := by decide"), tok.document_ids("example : 1 = 1 := rfl")]
    x, y, mask = document_batch(docs, "cpu")
    for row, ids in enumerate(docs):
        n = len(ids) - 1
        assert x[row, :n].tolist() == ids[:-1] and x[row, 0] == BOS
        assert y[row, :n].tolist() == ids[1:] and y[row, n - 1] == EOS
        assert (y[row, n:] == -1).all()
        # the mask marks x's real tokens (a shorter row's <eos> is one; nothing it feeds is trained on)
        assert mask[row].tolist() == [t < len(ids) for t in range(x.shape[1])]


def test_export_with_biases_runs_under_web_gpt(node, tmp_path):
    """bias=True adds LayerNorm and Linear biases, which web/gpt.js reads by
    name; the exported model must still give PyTorch's logits."""
    torch.manual_seed(1)
    tok = Tokenizer()
    model = GPT(GPTConfig(block_size=48, vocab_size=tok.vocab_size, n_layer=2, n_head=4, n_embd=32, bias=True))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.normal_(0.0, 0.5)                     # nonzero, so a dropped bias would show
    bundle = bundle_for(model, tok, "sft", {"domains": "nat"}, n_examples=1)
    assert any(k.endswith("ln_1.bias") for k in bundle["tensors"])
    path = tmp_path / "model.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    ids, _ = tok.example_ids(bundle["examples"][0]["prompt"], "12 := by decide")
    with torch.no_grad():
        expected = model.eval()(torch.tensor([ids]))[0][0]
    assert (js_logits(node, str(path), ids) - expected).abs().max().item() < 1e-4


def test_small_preset_has_the_old_parameter_count(old_bundle):
    config, _ = parse_config(PretrainConfig, ["--preset", "small"])
    model = GPT(GPTConfig(block_size=config.block_size, vocab_size=Tokenizer().vocab_size, n_layer=config.n_layer,
                          n_head=config.n_head, n_embd=config.n_embd, dropout=config.dropout, bias=config.bias))
    assert model.num_params() == old_bundle["params"] == 803072
