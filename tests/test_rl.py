"""The post-training stages: the PPO maths (GAE, whitening, the clipped
surrogate, one update), answer scores and the folded critic, RFT selection,
Bradley-Terry training and preference loading, and tiny end-to-end runs of
reward.py, rlhf.py and rft.py on the nat and linear domains (the policy is a
tiny GPT trained here for a few hundred steps on reference answers).

The end-to-end runs are watched by spies for what would silently spoil the
comparison of the arms: the reference and reward model must stay frozen and
apart from the policy and critic, each optimiser must hold its own model's
parameters only, the reward model must never train on its held-out prompts,
RFT must train on verified answers to fresh prompts only, the samples drawn
must be the budget, seeds must control runs, and every run must be scored
on the same val problems."""

import collections
import dataclasses
import json
import os
import random

import pytest
import torch

import common
import data
import formal
import reward
import rft
import rlhf
import sft
from config import RewardConfig, RFTConfig, RLHFConfig, parse_config
from model import GPT, GPTConfig, RewardModel
from tokenizer import EOS, SPECIAL_TOKENS, Tokenizer

DOMAINS = "nat,linear"
TOK = Tokenizer()


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    # The machine is shared, and with busy cores PyTorch's spinning worker
    # threads slow models this small down by orders of magnitude.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny_config():
    return GPTConfig(vocab_size=TOK.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=32)


# --------------------------------------------------------------- the maths

def test_gae_matches_hand_computation():
    # Row 0: an episode at positions 1..3; row 1: at positions 2..3. Rewards
    # and values off the mask (the 9s) must not leak in.
    rewards = torch.tensor([[9.0, 0.0, 0.0, 1.0, 9.0], [9.0, 9.0, -1.0, 0.0, 9.0]])
    values = torch.tensor([[9.0, 0.5, 0.2, 0.4, 9.0], [9.0, 9.0, 0.3, 0.1, 9.0]])
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 0]], dtype=torch.bool)
    adv, ret = rlhf.gae(rewards, values, mask, gamma=1.0, lam=0.5)
    # Row 0: deltas (0 + 0.2 - 0.5, 0 + 0.4 - 0.2, 1 - 0.4) = (-0.3, 0.2, 0.6);
    # A3 = 0.6, A2 = 0.2 + 0.5 * 0.6 = 0.5, A1 = -0.3 + 0.5 * 0.5 = -0.05.
    # Row 1: deltas (-1 + 0.1 - 0.3, 0 - 0.1) = (-1.2, -0.1); A = (-1.25, -0.1).
    assert torch.allclose(adv, torch.tensor([[0, -0.05, 0.5, 0.6, 0], [0, 0, -1.25, -0.1, 0]]), atol=1e-6)
    assert torch.allclose(ret, torch.tensor([[0, 0.45, 0.7, 1.0, 0], [0, 0, -0.95, 0.0, 0]]), atol=1e-6)
    # With lam = 1 the return is the discounted sum of rewards.
    adv, ret = rlhf.gae(rewards, values, mask, gamma=0.9, lam=1.0)
    assert torch.allclose(ret[0, 1:4], torch.tensor([0.81, 0.9, 1.0]), atol=1e-6)
    assert torch.allclose(adv[0, 1:4], torch.tensor([0.31, 0.7, 0.6]), atol=1e-6)


def test_masked_whiten():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 9, generator=gen) * 5 + 3
    mask = torch.rand(4, 9, generator=gen) < 0.6
    w = rlhf.masked_whiten(x, mask)
    on = w[mask]
    assert abs(on.mean().item()) < 1e-5
    assert abs(on.std(unbiased=False).item() - 1) < 1e-4
    assert (w[~mask] == 0).all()
    # Values off the mask do not influence the result.
    x2 = x.clone()
    x2[~mask] = 1e6
    assert torch.allclose(rlhf.masked_whiten(x2, mask), w)


def test_clipped_surrogate():
    eps = 0.2
    mask = torch.tensor([[True]])

    def run(adv, ratio):
        logp = torch.tensor([[float(torch.log(torch.tensor(ratio)))]], requires_grad=True)
        loss, clip = rlhf.clipped_surrogate(logp, torch.zeros(1, 1), torch.tensor([[adv]]), mask, eps)
        loss.backward()
        return loss.item(), logp.grad.item(), clip.item()

    # Positive advantage: no gain from pushing the ratio past 1 + eps...
    loss, grad, clip = run(2.0, 1.5)
    assert loss == pytest.approx(-2.0 * 1.2) and grad == 0 and clip == 1
    # ...but a ratio below 1 - eps is not clipped (the pessimistic bound).
    loss, grad, clip = run(2.0, 0.5)
    assert loss == pytest.approx(-2.0 * 0.5) and grad < 0 and clip == 0
    # Negative advantage: no gain from pushing the ratio below 1 - eps...
    loss, grad, clip = run(-1.0, 0.5)
    assert loss == pytest.approx(1.0 * 0.8) and grad == 0 and clip == 1
    # ...but a ratio above 1 + eps is still penalised in full.
    loss, grad, clip = run(-1.0, 1.5)
    assert loss == pytest.approx(1.0 * 1.5) and grad > 0 and clip == 0
    # Inside the trust region the loss is -A * ratio.
    loss, grad, clip = run(1.0, 1.1)
    assert loss == pytest.approx(-1.1) and grad == pytest.approx(-1.1) and clip == 0
    # Masked positions do not count.
    logp = torch.tensor([[0.0, 5.0]])
    loss, clip = rlhf.clipped_surrogate(logp, torch.zeros(1, 2), torch.ones(1, 2), torch.tensor([[True, False]]), eps)
    assert loss.item() == pytest.approx(-1.0) and clip.item() == 0


def test_value_loss_clips_the_move():
    old, ret, mask = torch.zeros(1, 2), torch.tensor([[1.0, 1.0]]), torch.ones(1, 2, dtype=torch.bool)
    # A value that moved past the target by more than vf_clip: the clipped
    # error is the larger one, so the gradient still pulls it back.
    v = torch.tensor([[0.1, 3.0]], requires_grad=True)
    loss = rlhf.value_loss(v, old, ret, mask, vf_clip=0.2)
    loss.backward()
    assert loss.item() == pytest.approx(0.5 * ((0.9 ** 2) + (2.0 ** 2)) / 2)
    assert v.grad[0, 0] < 0 and v.grad[0, 1] > 0


def sequence_batch(policy, prompt, answers, advantages):
    """A PPO batch laid out as rlhf.rollout lays it out, for fixed answers."""
    head = TOK.prompt_ids(prompt)
    seqs = [head + TOK.encode(a) + [EOS] for a in answers]
    ids, mask = common.pad_batch(seqs)
    act = torch.zeros(len(seqs), ids.shape[1] - 1, dtype=torch.bool)
    for i, s in enumerate(seqs):
        act[i, len(head) - 1:len(s) - 1] = True
    with torch.no_grad():
        old = rlhf.action_logprobs(policy, ids, mask, 1.0) * act
    return {"ids": ids, "mask": mask, "action_mask": act, "old_logp": old, "ref_logp": old.clone(),
            "advantages": torch.tensor(advantages)[:, None] * act}


def answer_logprob(policy, batch):
    with torch.no_grad():
        return (rlhf.action_logprobs(policy, batch["ids"], batch["mask"], 1.0) * batch["action_mask"]).sum(dim=1)


def test_ppo_update_favours_positive_advantages():
    torch.manual_seed(0)
    policy = GPT(tiny_config()).eval()
    prompt = "example : 2 + 3 ="
    batch = sequence_batch(policy, prompt, ["5 := by decide", "6 := by decide", "5 := rfl", "4 := rfl"],
                           [1.0, -1.0, 1.0, -1.0])
    before = answer_logprob(policy, batch)
    cfg = RLHFConfig(advantage="group", kl_coef=0.0, ppo_epochs=4, minibatch_size=2)
    opt = common.make_optimizer(policy, 3e-3, 0.0)
    stats = rlhf.ppo_update(policy, batch, cfg, opt, torch.Generator().manual_seed(0))
    gain = answer_logprob(policy, batch) - before
    # Rows 0 and 1 share the suffix " := by decide", so row 1 may rise a little
    # with row 0; what PPO guarantees is the relative move.
    assert gain[[0, 2]].min() > gain[[1, 3]].max(), gain
    assert gain[0] > 0 and gain[3] < 0, gain
    assert {"policy_loss", "clip_frac", "k3", "approx_kl"} <= set(stats)
    # The ratios are against the batch's old_logp, fixed at sampling time: had
    # they been recomputed before each step, every ratio would be 1, nothing
    # would ever be clipped and approx_kl would stay 0.
    assert stats["clip_frac"] > 0 and stats["approx_kl"] > 0, stats


def test_ppo_update_trains_the_critic_toward_returns():
    torch.manual_seed(1)
    policy = GPT(tiny_config()).eval()
    critic = RewardModel(tiny_config()).eval()
    batch = sequence_batch(policy, "example : 2 + 3 =", ["5 := by decide", "6 := by decide"], [1.0, -1.0])
    act = batch["action_mask"]
    with torch.no_grad():
        batch["old_values"] = critic.values(batch["ids"], batch["mask"])[:, :-1] * act
    batch["returns"] = torch.tensor([[1.0], [-1.0]]) * act
    def error():
        return ((critic.values(batch["ids"], batch["mask"])[:, :-1] - batch["returns"]) ** 2)[act].mean().item()

    start = error()
    cfg = RLHFConfig(advantage="gae", ppo_epochs=20, minibatch_size=2, vf_clip=10.0)
    stats = rlhf.ppo_update(policy, batch, cfg, common.make_optimizer(policy, 1e-4, 0.0),
                            torch.Generator().manual_seed(0), critic, common.make_optimizer(critic, 1e-2, 0.0))
    with torch.no_grad():
        assert error() < 0.5 * start
    assert "value_loss" in stats


# --------------------------------------------------- scores and the critic

def test_folded_critic_values_are_normalised_rm_scores():
    torch.manual_seed(2)
    rm = RewardModel(tiny_config()).eval()
    with torch.no_grad():
        rm.head.bias.fill_(0.3)
    mean, std = 0.7, 2.5
    critic = rlhf.critic_from_reward_model(rm, mean, std)
    seqs = [TOK.example_ids("example : 2 + 3 =", "5 := by decide")[0], TOK.example_ids("example : 1 =", "1 := rfl")[0]]
    ids, mask = common.pad_batch(seqs)
    with torch.no_grad():
        values = critic.values(ids, mask)
        last = torch.tensor([len(s) - 1 for s in seqs])
        at_last = values.gather(1, last[:, None]).squeeze(1)
        expected = (rm.score(ids, mask) - mean) / std
    assert torch.allclose(at_last, expected, atol=1e-5)
    assert rm.head.bias.item() == pytest.approx(0.3)   # the reward model itself is untouched


@pytest.mark.parametrize("source", ["rm", "oracle"])
def test_unfinished_answers_score_below_every_finished_one(source):
    gen = torch.Generator().manual_seed(3)
    n = 400
    finished = torch.rand(n, generator=gen) < 0.5
    correct = finished & (torch.rand(n, generator=gen) < 0.5)
    raw = torch.randn(n, generator=gen) * 100       # including scores far outside the clip range
    raw[~finished] = 1e6                            # whatever an unfinished answer's raw score would be
    for penalty in (1.0, 0.01):
        cfg = RLHFConfig(reward_source=source, score_clip=3.0, no_eos_penalty=penalty)
        scores = rlhf.answer_scores(cfg, finished.tolist(), correct.tolist(), raw, reward_mean=5.0, reward_std=2.0)
        assert scores[~finished].max() < scores[finished].min()
        assert scores[~finished].max() == pytest.approx(rlhf.score_floor(cfg) - penalty)
    if source == "oracle":
        assert set(scores[finished].tolist()) == {1.0, -1.0}
        assert (scores[correct] == 1).all()
    else:
        assert scores[finished].abs().max() <= 3.0


def test_config_must_keep_unfinished_answers_below_finished_ones():
    # answer_scores puts an unfinished answer at -score_clip - no_eos_penalty:
    # below every finished score only if both are positive. A zero penalty
    # ties a cut-off answer with the worst finished one, a negative one pays
    # for running on, and score_clip <= 0 erases the reward model's signal
    # (every finished answer would score the same).
    for bad in (dict(no_eos_penalty=0.0), dict(no_eos_penalty=-1.0), dict(score_clip=0.0), dict(score_clip=-3.0)):
        with pytest.raises(ValueError, match="no_eos_penalty|score_clip"):
            rlhf.check_config(RLHFConfig(**bad))
    rlhf.check_config(RLHFConfig())


def test_rollout_prompts_rotate_through_domains():
    domains = ["nat", "linear"]
    cfg = RLHFConfig(iters=6, rollout_batch=9, seed=5)
    schedule = rlhf.rollout_problems(cfg, domains)
    assert len(schedule) == 6 and all(len(b) == 9 for b in schedule)
    for b in schedule:
        counts = [sum(p.domain == d for p in b) for d in domains]
        assert max(counts) - min(counts) <= 1
    assert sum(p.domain == "nat" for b in schedule for p in b) == 27
    keys = [data.split_key(p.prompt) for b in schedule for p in b]
    assert len(set(keys)) == len(keys) and all(data.split_of_key(k) == "train" for k in keys)
    group = rlhf.rollout_problems(RLHFConfig(iters=3, rollout_batch=12, advantage="group", group_size=4), domains)
    for b in group:
        assert len(b) == 12 and all(b[i] == b[i - i % 4] for i in range(12))
        assert len({p.prompt for p in b}) == 3
    # Seeds are replicates: another seed, other prompts.
    other = rlhf.rollout_problems(dataclasses.replace(cfg, seed=6), domains)
    assert not {p.prompt for b in schedule for p in b} & {p.prompt for b in other for p in b}


# ------------------------------------------------------------------ RFT

def test_rft_selection():
    prompt = "example : 2 + 3 ="
    answers = ["5 := by decide", "5 := rfl", "6 := rfl", "5 := rfl", "5 := by simp", "5 := rfl", "5 := by rfl"]
    statuses = ["finished", "cut off", "finished", "finished", "finished", "finished", "malformed"]
    assert rft.select_verified(prompt, answers, statuses, 2) == ["5 := rfl", "5 := by simp"]
    assert rft.select_verified(prompt, answers, statuses, 10) == ["5 := rfl", "5 := by simp", "5 := by decide"]
    assert rft.select_verified(prompt, ["6 := rfl", "5 := rfl"], ["finished", "cut off"], 1) == []


def test_rft_rounds_never_reuse_a_prompt():
    cfg = RFTConfig(rounds=3, prompts_per_round=7, iters_per_round=4, epochs_per_round=0, batch_size=4, replay_frac=0.5,
                    seed=9)
    rounds, replay = rft.round_problems(cfg, ["nat", "linear"])
    assert [len(r) for r in rounds] == [7, 7, 7] and len(replay) == 3 * 4 * 2
    keys = [data.split_key(p.prompt) for r in rounds for p in r] + [data.split_key(p.prompt) for p in replay]
    assert len(set(keys)) == len(keys) and all(data.split_of_key(k) == "train" for k in keys)
    for r in rounds:
        counts = [sum(p.domain == d for p in r) for d in ("nat", "linear")]
        assert max(counts) - min(counts) <= 1
    other, _ = rft.round_problems(dataclasses.replace(cfg, seed=10), ["nat", "linear"])
    assert other != rounds


def test_rft_fine_tune_learns_its_examples():
    torch.manual_seed(4)
    policy = GPT(tiny_config())
    examples = [TOK.example_ids("example : 2 + 3 =", "5 := by decide")]
    replay = [TOK.example_ids("example : 1 + 1 =", "2 := rfl")] * 80
    cfg = RFTConfig(iters_per_round=40, epochs_per_round=0, batch_size=4, replay_frac=0.5, lr=1e-2, min_lr=1e-3,
                    warmup_iters=2)
    losses = rft.fine_tune(policy, examples, replay, cfg, "cpu", torch.Generator().manual_seed(0))
    assert len(losses) == 40 and losses[-1] < 0.2 * losses[0]
    assert not policy.training


# --------------------------------------------------------------- the RM

def test_bradley_terry_separates_synthetic_preferences():
    torch.manual_seed(5)
    rng = random.Random(0)
    problems = data.sample_problems(60, "train", 11, DOMAINS)

    def pair(p):
        # Chosen answers end in a digit, rejected ones in "?".
        n = rng.randint(0, 99)
        return {"prompt": p.prompt, "chosen": f"{n} := by decide {rng.randint(0, 9)}",
                "rejected": f"{n} := by decide ?", "domain": p.domain}

    train, val = [pair(p) for p in problems[:45]], [pair(p) for p in problems[45:]]
    rm = RewardModel(GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_head=2, n_embd=32))
    cfg = RewardConfig(max_iters=60, batch_size=16, lr=3e-3, min_lr=3e-4, warmup_iters=5, eval_interval=30, seed=0)
    history = reward.train_reward_model(rm, TOK, train, val, cfg, "cpu")
    assert [h["iter"] for h in history] == [30, 60]
    assert history[-1]["val_accuracy"] >= 0.95
    assert history[-1]["val_loss"] < 0.3


def test_comparisons_follow_the_checker():
    p = formal.Problem("nat", "example : 2 + 3 =", "5 := by decide")
    answers = reward.candidates(p, [("5 := rfl", "finished"), ("6 := rfl", "finished"), ("5 := rfl", "finished"),
                                    ("7 := rfl", "cut off"), ("8 := rfl", "malformed")], True, TOK, 256)
    assert answers == ["5 := rfl", "6 := rfl", "5 := by decide"]
    pairs = reward.comparisons(p, answers, 10, random.Random(0))
    # nat has no preference between two correct answers: only correct vs wrong.
    assert sorted((q["chosen"], q["rejected"]) for q in pairs) == [("5 := by decide", "6 := rfl"),
                                                                    ("5 := rfl", "6 := rfl")]
    assert all(q["domain"] == "nat" and q["prompt"] == p.prompt for q in pairs)
    assert len(reward.comparisons(p, answers, 1, random.Random(0))) == 1


def test_held_out_is_per_prompt_and_respelling_invariant():
    assert reward.held_out("example : (1 + 2) + 3 =", 0.5) == reward.held_out("example : 1 + 2 + 3 =", 0.5)
    prompts = [p.prompt for p in data.sample_problems(2000, "train", 3, DOMAINS, unique=True)]
    frac = sum(reward.held_out(q, 0.2) for q in prompts) / len(prompts)
    assert 0.17 < frac < 0.23
    assert not any(reward.held_out(q, 0.0) for q in prompts)


def test_human_preferences_loading(tmp_path):
    train = data.sample_problems(2, "train", 0, "nat", unique=True)
    val = data.sample_problems(1, "val", 0, "nat", unique=True)[0]
    test = data.sample_problems(1, "test", 0, "linear", unique=True)[0]
    rows = [
        {"prompt": train[0].prompt, "chosen": train[0].reference, "rejected": "0 := rfl", "domain": "anything"},
        {"prompt": train[1].prompt, "chosen": "1 := rfl", "rejected": "1 := rfl"},                       # tie
        {"prompt": "theorem t : True :=", "chosen": "trivial", "rejected": "rfl"},                     # unclaimed
        {"prompt": val.prompt, "chosen": val.reference, "rejected": "0 := rfl"},                        # val split
        {"prompt": test.prompt, "chosen": test.reference, "rejected": "0 := rfl"},                      # test split
        {"prompt": train[1].prompt, "chosen": "x" * 300, "rejected": "1 := rfl"},                      # too long
    ]
    path = tmp_path / "prefs.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:3]) + "\n\n"
                    + "\n".join(json.dumps(r) for r in rows[3:]) + "\n", encoding="utf-8")
    pairs, skipped = reward.load_human_preferences(str(path), TOK, 256)
    assert pairs == [{"prompt": train[0].prompt, "chosen": train[0].reference, "rejected": "0 := rfl", "domain": "nat"}]
    assert skipped == {"tie": 1, "unclaimed": 1, "not train": 2, "too long": 1}

    good = json.dumps(rows[0])
    for bad, line in [("{not json", 2), (json.dumps({"prompt": train[0].prompt, "chosen": "1"}), 2),
                      (json.dumps(["a", "b", "c"]), 2), (json.dumps({"prompt": train[0].prompt, "chosen": 5,
                                                                      "rejected": "1"}), 2),
                      (json.dumps({"prompt": train[0].prompt, "chosen": "5 := décidé", "rejected": "1"}), 3)]:
        path = tmp_path / "bad.jsonl"
        lines = [good, bad] if line == 2 else [good, "", bad]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=f"bad.jsonl:{line}:"):
            reward.load_human_preferences(str(path), TOK, 256)


def test_auc():
    assert reward.auc([0.1, 0.9, 0.5, 0.3], [False, True, True, False]) == 1.0
    assert reward.auc([0.1, 0.9], [True, False]) == 0.0
    assert reward.auc([0.5, 0.5, 0.5], [True, False, False]) == 0.5
    assert reward.auc([0.1, 0.2], [True, True]) is None


# ------------------------------------------------------- end to end, tiny

@pytest.fixture(scope="module")
def sft_path(tmp_path_factory):
    """A tiny policy (the tiny preset's architecture) trained for 200 steps on
    nat and linear references: it finishes nearly every answer and gets a few
    percent right, so comparisons and verified samples exist."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, n_layer=2, n_head=2, n_embd=64))
    examples = [TOK.example_ids(p.prompt, p.reference) for p in data.sample_problems(4000, "train", 5, DOMAINS)]
    opt = common.make_optimizer(model, 3e-3, 0.0)
    rng = random.Random(0)
    for it in range(200):
        for g in opt.param_groups:
            g["lr"] = common.lr_at(it, 3e-3, 3e-4, 10, 200)
        x, y, mask = common.sft_batch(rng.sample(examples, 32), "cpu")
        _, loss = model(x, targets=y, attention_mask=mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    path = str(tmp_path_factory.mktemp("rl") / "sft.pt")
    common.save_checkpoint(path, model, TOK, stage="sft")
    return path


def test_rollout_batch(sft_path):
    """rollout() lays out the batch as documented, scores exactly `<bos> prompt
    <ans> answer <eos>` with the reward model, and puts the score on the last
    answer token (gae) or z-scores it within each group (group)."""
    policy, _, _ = common.load_checkpoint(sft_path, kind="policy")
    ref_policy, _, _ = common.load_checkpoint(sft_path, kind="policy")
    torch.manual_seed(6)
    rm = RewardModel(policy.config).eval()
    problems = data.sample_problems(32, "train", 0, DOMAINS, unique=True)
    # The normalisation in the units of this reward model's scores, so that
    # most normalised scores fall inside the clip range: were they all clipped,
    # a missing or doubled normalisation would go unseen.
    on_references = reward.score(rm, TOK, [(p.prompt, p.reference) for p in problems], "cpu")
    mean, std = on_references.mean().item(), on_references.std().item()
    critic = rlhf.critic_from_reward_model(rm, mean, std)
    # 16 new tokens: most answers finish, the rest are cut off.
    # gamma = lam = 1 and policy == reference (no KL reward): every answer
    # token's return is the answer's score.
    cfg = RLHFConfig(reward_source="rm", advantage="gae", max_new_tokens=16, gamma=1.0, lam=1.0, score_clip=3.0,
                     no_eos_penalty=0.5)
    batch, stats = rlhf.rollout(policy.eval(), ref_policy.eval(), TOK, problems, cfg,
                                torch.Generator().manual_seed(0), critic, rm, mean, std)
    act = batch["action_mask"]
    finished = []
    for i, p in enumerate(problems):
        seq = batch["ids"][i][batch["mask"][i]].tolist()
        head = TOK.prompt_ids(p.prompt)
        assert seq[:len(head)] == head
        actions = seq[len(head):]
        assert act[i].nonzero().squeeze(1).tolist() == list(range(len(head) - 1, len(seq) - 1))
        answer, status = common.response_status(actions)
        finished.append(status == "finished")
        if status == "finished":
            expected = (reward.score(rm, TOK, [(p.prompt, TOK.decode(answer))], "cpu") - mean) / std
            assert batch["scores"][i].item() == pytest.approx(expected.clamp(-3, 3).item(), abs=1e-4)
        else:
            assert status == "cut off" and len(actions) == 16
            assert batch["scores"][i].item() == pytest.approx(-3.5)
    assert 0 < sum(finished) < len(problems), finished
    inside = (batch["scores"].abs() < 3).sum().item()
    assert inside >= sum(finished) // 2, batch["scores"]
    assert stats["finished"] == sum(finished) / len(problems)
    assert torch.allclose(batch["old_logp"], batch["ref_logp"]) and abs(stats["kl"]) < 1e-6
    assert (batch["old_logp"][~act] == 0).all() and (batch["old_logp"][act] < 0).any()
    assert torch.allclose(batch["returns"], batch["scores"][:, None] * act, atol=1e-5)
    adv = batch["advantages"][act]
    assert abs(adv.mean().item()) < 1e-5 and abs(adv.std(unbiased=False).item() - 1) < 1e-3

    group = RLHFConfig(reward_source="oracle", advantage="group", group_size=4, max_new_tokens=14)
    problems = [p for p in problems[:8] for _ in range(4)]
    batch, stats = rlhf.rollout(policy, ref_policy, TOK, problems, group, torch.Generator().manual_seed(1))
    scores = batch["scores"].view(8, 4)
    z = ((scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)).reshape(-1)
    assert torch.allclose(batch["advantages"], z[:, None] * batch["action_mask"])
    assert "returns" not in batch and stats["rm_score"] is None


def test_kl_counts_answer_tokens_only(sft_path):
    """With a reference that differs from the policy on every token, prompt
    included, ref_logp, the logged KL and the per-token KL penalty in the
    rewards cover exactly the answer tokens, and the score is added once, at
    the end (test_rollout_batch cannot see this: its reference is the policy)."""
    policy, _, _ = common.load_checkpoint(sft_path, kind="policy")
    ref_policy, _, _ = common.load_checkpoint(sft_path, kind="policy")
    torch.manual_seed(7)
    with torch.no_grad():
        for p in ref_policy.parameters():
            p.add_(0.05 * torch.randn_like(p))
    critic, _, _ = common.load_checkpoint(sft_path, kind="reward")
    torch.nn.init.zeros_(critic.head.weight)          # values 0: a return is the plain sum of rewards from t on
    problems = data.sample_problems(16, "train", 1, DOMAINS, unique=True)
    cfg = RLHFConfig(reward_source="oracle", advantage="gae", max_new_tokens=14, gamma=1.0, lam=1.0, kl_coef=0.3)
    batch, stats = rlhf.rollout(policy.eval(), ref_policy.eval(), TOK, problems, cfg, torch.Generator().manual_seed(0),
                                critic.eval())
    act, real = batch["action_mask"], batch["mask"][:, 1:]
    with torch.no_grad():
        log_ratio = (rlhf.action_logprobs(policy, batch["ids"], batch["mask"], 1.0)
                     - rlhf.action_logprobs(ref_policy, batch["ids"], batch["mask"], 1.0))
    assert log_ratio[real & ~act].abs().max() > 1e-2   # the prompt's tokens differ too, and must not count
    assert (batch["ref_logp"][~act] == 0).all()
    assert stats["kl"] == pytest.approx((log_ratio * act).sum(dim=1).mean().item(), abs=1e-5)
    to_go = (log_ratio * act).flip(1).cumsum(dim=1).flip(1)
    assert torch.allclose(batch["returns"], (batch["scores"][:, None] - cfg.kl_coef * to_go) * act, atol=1e-4)


@pytest.fixture(scope="module")
def reward_run(sft_path):
    """reward.py on the tiny preset, recording the comparisons it trains and
    validates the reward model on."""
    out_dir = os.path.dirname(sft_path)
    out, prefs = os.path.join(out_dir, "reward.pt"), os.path.join(out_dir, "preferences.jsonl")
    seen = {}
    train = reward.train_reward_model

    def spy(rm, tokenizer, train_pairs, val_pairs, *args):
        seen.update(train=train_pairs, val=val_pairs)
        return train(rm, tokenizer, train_pairs, val_pairs, *args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(reward, "train_reward_model", spy)
        reward.main(["--preset", "tiny", "--domains", DOMAINS, "--sft", sft_path, "--out", out, "--preferences",
                     prefs, "--device", "cpu"])
    return out, prefs, seen


def test_reward_end_to_end(reward_run):
    out, prefs, seen = reward_run
    ckpt = torch.load(out, weights_only=True)
    assert ckpt["kind"] == "reward" and ckpt["stage"] == "reward"
    assert ckpt["reward_std"] > 0 and isinstance(ckpt["reward_mean"], float)
    assert 0 <= ckpt["val_accuracy"] <= 1
    assert ckpt["auc"] is None or 0 <= ckpt["auc"] <= 1
    assert [h["iter"] for h in ckpt["history"]] == [30, 60]
    rm, _, _ = common.load_checkpoint(out, kind="reward")
    with open(prefs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    assert pairs and len(pairs) == ckpt["n_comparisons"]
    per_prompt = {}
    for q in pairs:
        assert set(q) == {"prompt", "chosen", "rejected", "domain"} and q["domain"] in ("nat", "linear")
        assert formal.prefer(q["prompt"], q["chosen"], q["rejected"]) == 1
        assert data.split_of(q["prompt"]) == "train"
        per_prompt[q["prompt"]] = per_prompt.get(q["prompt"], 0) + 1
    assert max(per_prompt.values()) <= RewardConfig().max_pairs_per_prompt
    # The reward model was trained on the comparisons of the other prompts only:
    # every comparison is on exactly one side, by its prompt's hold-out hash.
    train_keys = {data.split_key(q["prompt"]) for q in seen["train"]}
    val_keys = {data.split_key(q["prompt"]) for q in seen["val"]}
    assert seen["val"] and not train_keys & val_keys
    assert all(reward.held_out(q["prompt"], 0.2) for q in seen["val"])
    assert sorted(map(json.dumps, seen["train"] + seen["val"])) == sorted(map(json.dumps, pairs))
    assert ckpt["n_val_comparisons"] == len(seen["val"])


def parameter_ids(*models) -> set:
    return {id(p) for m in models if m is not None for p in m.parameters()}


def optimised_ids(optimizer) -> set:
    return {id(p) for g in optimizer.param_groups for p in g["params"]} if optimizer is not None else set()


@pytest.mark.parametrize("source,advantage", [("rm", "gae"), ("oracle", "gae"), ("oracle", "group")])
def test_rlhf_end_to_end(sft_path, reward_run, tmp_path, monkeypatch, source, advantage):
    out = str(tmp_path / f"{source}_{advantage}.pt")
    stale = str(tmp_path / f"{source}_{advantage}_last.pt")
    with open(stale, "w") as f:
        f.write("an earlier run's last iterate")
    select_best = source == "oracle" and advantage == "gae"
    evals = advantage == "gae"           # the group run checks that eval_problems 0 skips evaluation
    # Spies: the models every rollout uses, the optimisers every update steps,
    # and the prompt of every sampled answer (evaluation is greedy and goes
    # through common's generate_answers, so it is not counted).
    seen = {"sampled": []}
    rollout, ppo_update, generate = rlhf.rollout, rlhf.ppo_update, rlhf.generate_answers

    def spy_rollout(policy, ref_policy, tokenizer, problems, config, generator=None, critic=None, reward_model=None,
                    *rest):
        seen.update(policy=policy, ref=ref_policy, critic=critic, rm=reward_model)
        return rollout(policy, ref_policy, tokenizer, problems, config, generator, critic, reward_model, *rest)

    def spy_update(policy, batch, config, policy_opt, generator=None, critic=None, critic_opt=None):
        assert optimised_ids(policy_opt) == parameter_ids(policy)
        assert optimised_ids(critic_opt) == parameter_ids(critic)
        return ppo_update(policy, batch, config, policy_opt, generator, critic, critic_opt)

    def spy_generate(model, tokenizer, prompts, **kwargs):
        seen["sampled"] += [p.prompt for p in prompts]
        return generate(model, tokenizer, prompts, **kwargs)

    monkeypatch.setattr(rlhf, "rollout", spy_rollout)
    monkeypatch.setattr(rlhf, "ppo_update", spy_update)
    monkeypatch.setattr(rlhf, "generate_answers", spy_generate)
    rlhf.main(["--preset", "tiny", "--domains", DOMAINS, "--sft", sft_path, "--reward", reward_run[0], "--out", out,
               "--reward_source", source, "--advantage", advantage, "--seed", "3", "--device", "cpu",
               "--select_best", str(select_best)] + ([] if evals else ["--eval_problems", "0"]))
    ckpt = torch.load(out, weights_only=True)
    assert ckpt["kind"] == "policy" and ckpt["stage"] == ("rlhf" if source == "rm" else "ppo-checker")
    assert ckpt["run_config"]["reward_source"] == source and ckpt["run_config"]["advantage"] == advantage
    with open(str(tmp_path / f"{source}_{advantage}.json"), encoding="utf-8") as f:
        log = json.load(f)
    assert log["policy_samples"] == 4 * 16
    assert [e["iteration"] for e in log["iterations"]] == [1, 2, 3, 4]
    assert [e["iteration"] for e in log["evals"]] == ([0, 2, 4] if evals else [])
    for e in log["iterations"]:
        assert {"score", "rm_score", "kl", "accuracy", "finished", "accuracy_by_domain", "policy_loss", "clip_frac",
                "seconds"} <= set(e)
        assert set(e["accuracy_by_domain"]) == {"nat", "linear"}
        assert e["rm_score"] is not None              # the reward model is logged in both modes
        assert ("value_loss" in e) == (advantage == "gae")
    if evals:
        assert set(log["evals"][0]["results"]) == {"nat", "linear", "all"}
        assert log["evals"][0]["results"]["all"]["n"] == 8
    else:
        assert ckpt["val"] is None
    if select_best:
        assert os.path.exists(stale)
        assert torch.load(stale, weights_only=True)["iteration"] == 4
        assert ckpt["iteration"] == log["selected_iteration"]
        best = max(e["results"]["all"]["accuracy"] for e in log["evals"])
        assert ckpt["val"]["all"]["accuracy"] == best
    else:
        assert not os.path.exists(stale)
        assert ckpt["iteration"] == 4

    # Four models with no parameter in common (the critic is a copy of the
    # reward model, not the reward model); the reference and the reward model
    # end the run bit for bit as loaded, and the KL starts at 0.
    models = [seen[k] for k in ("policy", "ref", "critic", "rm") if seen[k] is not None]
    assert len(models) == (4 if advantage == "gae" else 3)
    assert len(parameter_ids(*models)) == sum(len(parameter_ids(m)) for m in models)
    for model, path in ((seen["ref"], sft_path), (seen["rm"], reward_run[0])):
        saved = torch.load(path, weights_only=True)["state_dict"]
        assert all(torch.equal(v, saved[k]) for k, v in model.state_dict().items())
    assert abs(log["iterations"][0]["kl"]) < 1e-6
    # The budget is what was drawn: distinct train prompts, or groups of 4.
    assert len(seen["sampled"]) == log["policy_samples"]
    assert all(data.split_of(p) == "train" for p in seen["sampled"])
    counts = set(collections.Counter(seen["sampled"]).values())
    assert counts == ({1} if advantage == "gae" else {4})


def test_rft_end_to_end(sft_path, tmp_path, monkeypatch):
    sampled, trained = [], []
    generate, fine_tune = rft.generate_answers, rft.fine_tune

    def spy_generate(model, tokenizer, prompts, **kwargs):
        sampled.append([p.prompt for p in prompts])
        return generate(model, tokenizer, prompts, **kwargs)

    def spy_fine_tune(policy, examples, replay, *args):
        trained.append((len(sampled), examples, replay))
        return fine_tune(policy, examples, replay, *args)

    monkeypatch.setattr(rft, "generate_answers", spy_generate)
    monkeypatch.setattr(rft, "fine_tune", spy_fine_tune)
    out = str(tmp_path / "rft.pt")
    rft.main(["--preset", "tiny", "--domains", DOMAINS, "--sft", sft_path, "--out", out, "--seed", "1",
              "--prompts_per_round", "32", "--samples_per_prompt", "4", "--replay_frac", "0.25", "--epochs_per_round", "0",
              "--iters_per_round", "10", "--device", "cpu"])
    ckpt = torch.load(out, weights_only=True)
    assert ckpt["kind"] == "policy" and ckpt["stage"] == "rft" and ckpt["policy_samples"] == 2 * 32 * 4
    with open(str(tmp_path / "rft.json"), encoding="utf-8") as f:
        log = json.load(f)
    assert log["policy_samples"] == log["policy_samples_drawn"] == 2 * 32 * 4
    assert [r["round"] for r in log["rounds"]] == [1, 2]
    assert [e["round"] for e in log["evals"]] == [0, 1, 2]
    for r in log["rounds"]:
        assert set(r["yield"]) == {"nat", "linear"} and all(0 <= y <= 1 for y in r["yield"].values())
        assert r["examples"] <= 32 * RFTConfig().keep_per_prompt
    assert any(r["train_loss"] is not None for r in log["rounds"])

    # Each round sampled 4 answers to each of 32 train prompts no other round used.
    assert len(sampled) == 2 and sum(map(len, sampled)) == log["policy_samples_drawn"]
    keys = [{data.split_key(p) for p in round_prompts} for round_prompts in sampled]
    assert all(len(k) == 32 for k in keys) and not keys[0] & keys[1]
    assert all(set(collections.Counter(r).values()) == {4} and data.split_of(r[0]) == "train" for r in sampled)
    # Fine-tuning saw only finished answers the checker accepts, to that round's
    # prompts, at most keep_per_prompt each; replay is reference answers to
    # prompts no round sampled.
    assert trained
    n = len(SPECIAL_TOKENS)
    for r, examples, replay in trained:
        prompts = []
        for ids, n_prompt in examples:
            prompt, answer = TOK.decode(ids[:n_prompt]), ids[n_prompt:-1]
            assert ids[:n_prompt] == TOK.prompt_ids(prompt) and ids[-1] == EOS and all(t >= n for t in answer)
            assert prompt in sampled[r - 1] and formal.verify(prompt, TOK.decode(answer))
            prompts.append(prompt)
        assert max(collections.Counter(prompts).values()) <= RFTConfig().keep_per_prompt
        for ids, n_prompt in replay:
            prompt = TOK.decode(ids[:n_prompt])
            assert not any(prompt in s for s in sampled) and formal.verify(prompt, TOK.decode(ids[n_prompt:-1]))


def test_seed_controls_each_run_but_never_the_val_set(sft_path, reward_run, tmp_path, monkeypatch):
    """The same seed gives the same weights and another seed other weights,
    for PPO (with the reward model and the critic) and for RFT; and every
    evaluation of every run, whatever its seed, is on sft.py's val problems."""
    val_sets = []
    for module in (rlhf, rft):
        def spy(model, tokenizer, problems, _evaluate=module.evaluate, **kwargs):
            val_sets.append([p.prompt for p in problems])
            return _evaluate(model, tokenizer, problems, **kwargs)
        monkeypatch.setattr(module, "evaluate", spy)

    def weights(main, seed, name, *extra):
        out = str(tmp_path / f"{name}.pt")
        main(["--preset", "tiny", "--domains", DOMAINS, "--sft", sft_path, "--out", out, "--seed", str(seed),
              "--device", "cpu", *extra])
        return torch.load(out, weights_only=True)["state_dict"]

    def same(a, b):
        return all(torch.equal(a[k], b[k]) for k in a)

    ppo = ("--reward", reward_run[0], "--reward_source", "rm", "--iters", "2", "--eval_interval", "0")
    a, b, c = (weights(rlhf.main, s, f"ppo{i}", *ppo) for i, s in enumerate((3, 3, 4)))
    assert same(a, b) and not same(a, c)
    one_round = ("--rounds", "1", "--prompts_per_round", "16")
    a, b, c = (weights(rft.main, s, f"rft{i}", *one_round) for i, s in enumerate((1, 1, 2)))
    assert same(a, b) and not same(a, c)
    assert rlhf.VAL_SEED == rft.VAL_SEED == sft.VAL_SEED
    expected = [p.prompt for p in data.sample_problems(4 * 2, "val", sft.VAL_SEED, DOMAINS, unique=True)]
    assert len(val_sets) == 12 and all(v == expected for v in val_sets)


@pytest.mark.parametrize("preset", ["tiny", "small", "base"])
def test_arms_share_the_sample_budget_and_the_sampling(preset):
    """PPO and RFT draw the same number of policy samples, the same way, and are
    evaluated alike; the reward model's normalisation comes from SFT samples
    drawn as PPO draws its first ones."""
    ppo, _ = parse_config(RLHFConfig, ["--preset", preset])
    rs, _ = parse_config(RFTConfig, ["--preset", preset])
    rm, _ = parse_config(RewardConfig, ["--preset", preset])
    assert ppo.iters * ppo.rollout_batch == rs.rounds * rs.prompts_per_round * rs.samples_per_prompt
    for name in ("domains", "temperature", "max_new_tokens", "eval_problems"):
        assert getattr(ppo, name) == getattr(rs, name), name
    assert (rm.temperature, rm.max_new_tokens, rm.domains) == (ppo.temperature, ppo.max_new_tokens, ppo.domains)


@pytest.mark.parametrize("main", [reward.main, rlhf.main, rft.main])
def test_greedy_sampling_is_rejected(main, tmp_path):
    # Comparisons, PPO and rejection sampling all need distinct samples; the
    # check comes before any checkpoint is read.
    with pytest.raises(ValueError, match="temperature"):
        main(["--preset", "tiny", "--temperature", "0", "--sft", str(tmp_path / "missing.pt"), "--device", "cpu"])


def test_rft_steps_scale_with_the_verified_set():
    import rft
    cfg = RFTConfig(epochs_per_round=3.0, batch_size=64)
    assert rft.steps_for(cfg, 600) == 29 and rft.steps_for(cfg, 1) == 1 and rft.steps_for(cfg, 0) == 1
    assert rft.steps_for(RFTConfig(epochs_per_round=0, iters_per_round=17), 600) == 17
