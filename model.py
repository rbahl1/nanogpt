"""GPT-2 style transformer (the nanoGPT architecture), plus a scalar-head
variant used as the reward model and the PPO critic.

Additions over nanoGPT, all needed for post-training:
  * an attention mask, so left-padded batches of prompts can be run together
    (position ids count real tokens only, so padding does not shift them);
  * a key/value cache for generation;
  * `RewardModel`, the same trunk with a scalar head.

web/gpt.js reimplements `GPT.forward` for one unpadded sequence;
tests/test_model.py checks that the two agree.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 121
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch's cannot drop only the bias)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, allowed, past=None):
        """x: (B, T, C). allowed: bool (B, 1, T, S) saying which of the S keys
        (cached ones first) each query may attend to. past: (k, v) from earlier
        steps or None. Returns (y, (k, v)) with the cache extended by x."""
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        heads = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # noqa: E731
        q, k, v = heads(q), heads(k), heads(v)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, allowed, past=None):
        a, cache = self.attn(self.ln_1(x), allowed, past)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, cache


def _allowed(key_mask: torch.Tensor, n_query: int) -> torch.Tensor:
    """(B, 1, Tq, S) attention pattern for the last `n_query` positions of a
    sequence whose S keys have padding mask `key_mask` (B, S): causal, never a
    padding key, and every query may see itself (a padding query must attend
    to something, or softmax returns NaN; its output is never used)."""
    B, S = key_mask.shape
    q_pos = torch.arange(S - n_query, S, device=key_mask.device)
    k_pos = torch.arange(S, device=key_mask.device)
    causal = k_pos[None, :] <= q_pos[:, None]                       # (Tq, S)
    allowed = causal[None] & key_mask[:, None, :].bool()             # (B, Tq, S)
    allowed = allowed | (k_pos[None, :] == q_pos[:, None])[None]
    return allowed[:, None]


def _positions(mask: torch.Tensor) -> torch.Tensor:
    """Position ids that count real tokens only, so left padding does not
    shift them; padding positions get 0."""
    return (mask.long().cumsum(dim=1) - 1).clamp(min=0)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Parameters, counting the tied embedding once and excluding positions."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel()

    def _trunk(self, idx, attention_mask=None, past=None, positions=None):
        """Runs the blocks over idx (B, T). attention_mask (B, S) covers the
        cached keys and idx (S = cached + T); defaults to all ones. Returns
        (hidden after ln_f (B, T, C), new cache)."""
        B, T = idx.shape
        n_past = 0 if past is None else past[0][0].shape[2]
        if attention_mask is None:
            attention_mask = torch.ones(B, n_past + T, dtype=torch.bool, device=idx.device)
        attention_mask = attention_mask.bool()
        if positions is None:
            positions = _positions(attention_mask)[:, n_past:]
        if int(positions.max()) >= self.config.block_size:
            raise ValueError(f"sequence longer than the context ({self.config.block_size} tokens)")
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(positions))
        allowed = _allowed(attention_mask, T)
        cache = []
        for i, block in enumerate(self.transformer.h):
            x, kv = block(x, allowed, None if past is None else past[i])
            cache.append(kv)
        return self.transformer.ln_f(x), cache

    def hidden_states(self, idx, attention_mask=None):
        return self._trunk(idx, attention_mask)[0]

    def forward(self, idx, targets=None, attention_mask=None):
        """Logits (B, T, V), and the mean cross-entropy over targets != -1 if
        targets are given."""
        h, _ = self._trunk(idx, attention_mask)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens: int, eos_id: int, pad_id: int = 0, temperature: float = 1.0,
                 top_k: int = None, greedy: bool = False, generator: torch.Generator = None):
        """Continues each prompt (a list of token ids) until it emits eos_id or
        has max_new_tokens new tokens, or its context is full. Prompts are
        left-padded and run as one batch with a key/value cache. Returns one
        list of new token ids per prompt, including the eos_id if emitted.

        Each prompt has its own budget, min(max_new_tokens, block_size - its
        length): positions count real tokens only, so a short prompt may run
        past the point where a long one in the same batch must stop, and what
        a prompt can produce does not depend on its neighbours."""
        device = self.transformer.wte.weight.device
        B = len(prompts)
        L = max(len(p) for p in prompts)
        budgets = torch.tensor([min(max_new_tokens, self.config.block_size - len(p)) for p in prompts], device=device)
        budget = int(budgets.max())
        if budget <= 0:
            return [[] for _ in prompts]
        idx = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        for i, p in enumerate(prompts):
            idx[i, L - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
            mask[i, L - len(p):] = True
        h, past = self._trunk(idx, mask)
        logits = self.lm_head(h[:, -1, :])
        next_pos = mask.long().sum(dim=1)                     # (B,) position of the next token
        out = [[] for _ in prompts]
        done = budgets <= 0
        for step in range(budget):
            if greedy:
                nxt = logits.argmax(dim=-1)
            else:
                scaled = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled.float(), dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            for i in range(B):
                if not done[i]:
                    out[i].append(int(nxt[i]))
            done = done | (nxt == eos_id) | (budgets <= step + 1)
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.bool, device=device)], dim=1)
            # Finished rows are still fed (their outputs are dropped); keep
            # their positions inside the context. Unfinished rows never reach it.
            positions = next_pos.clamp(max=self.config.block_size - 1)[:, None]
            h, past = self._trunk(nxt[:, None], mask, past, positions=positions)
            next_pos = next_pos + 1
            logits = self.lm_head(h[:, -1, :])
        return out


class RewardModel(nn.Module):
    """A GPT trunk with a scalar head: `values` gives one value per position
    (the PPO critic), `score` the value at each sequence's last real token
    (the reward model)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.gpt = GPT(config)
        self.head = nn.Linear(config.n_embd, 1)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        torch.nn.init.zeros_(self.head.bias)

    def values(self, idx, attention_mask=None):
        return self.head(self.gpt.hidden_states(idx, attention_mask)).squeeze(-1)

    def score(self, idx, attention_mask=None):
        """Value at the last real token of each row. Works with left or right
        padding: it takes the last position whose mask is set."""
        v = self.values(idx, attention_mask)
        if attention_mask is None:
            return v[:, -1]
        T = attention_mask.shape[1]
        last = T - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1)
        return v.gather(1, last[:, None]).squeeze(1)
