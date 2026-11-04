"""An interrupted pretrain or SFT run, rerun with the same configuration,
continues from its last progress save and ends with exactly the weights of an
uninterrupted run."""

import os

import pytest
import torch

import common
import pretrain
import sft


class Interrupted(Exception):
    pass


def _interrupt_after_first_save(monkeypatch, module):
    real = common.save_progress

    def save_then_die(out, state):
        real(out, state)
        raise Interrupted

    monkeypatch.setattr(module, "save_progress", save_then_die)


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


ARGS = ["--preset", "tiny", "--domains", "nat,linear", "--device", "cpu", "--max_iters", "60", "--save_interval", "25",
        "--eval_interval", "30"]


def test_pretrain_resumes_exactly(tmp_path, monkeypatch):
    whole = str(tmp_path / "whole.pt")
    pretrain.main(ARGS + ["--out", whole])
    broken = str(tmp_path / "broken.pt")
    with monkeypatch.context() as m:
        _interrupt_after_first_save(m, pretrain)
        with pytest.raises(Interrupted):
            pretrain.main(ARGS + ["--out", broken])
    assert os.path.exists(common.progress_path(broken)) and not os.path.exists(broken)
    pretrain.main(ARGS + ["--out", broken])
    assert not os.path.exists(common.progress_path(broken))
    assert _same(_weights(whole), _weights(broken))


def test_sft_resumes_exactly(tmp_path, monkeypatch):
    init = str(tmp_path / "pre.pt")
    pretrain.main(ARGS + ["--out", init, "--save_interval", "0"])
    args = ["--preset", "tiny", "--domains", "nat,linear", "--device", "cpu", "--init", init, "--max_iters", "60",
            "--save_interval", "25", "--eval_interval", "30", "--eval_problems", "2"]
    whole = str(tmp_path / "whole.pt")
    sft.main(args + ["--out", whole])
    broken = str(tmp_path / "broken.pt")
    with monkeypatch.context() as m:
        _interrupt_after_first_save(m, sft)
        with pytest.raises(Interrupted):
            sft.main(args + ["--out", broken])
    sft.main(args + ["--out", broken])
    assert _same(_weights(whole), _weights(broken))


def test_progress_from_another_configuration_is_ignored(tmp_path):
    out = str(tmp_path / "x.pt")
    common.save_progress(out, {"run_config": {"lr": 1.0}, "iter": 5})
    assert common.load_progress(out, {"lr": 2.0}) is None
    assert common.load_progress(out, {"lr": 1.0})["iter"] == 5


def test_reward_resumes_exactly(tmp_path, monkeypatch):
    import reward
    pre, sft_ckpt = str(tmp_path / "pre.pt"), str(tmp_path / "sft.pt")
    pretrain.main(ARGS + ["--out", pre, "--save_interval", "0"])
    sft.main(["--preset", "tiny", "--domains", "nat,linear", "--device", "cpu", "--init", pre, "--out", sft_ckpt,
              "--max_iters", "40", "--eval_problems", "0", "--save_interval", "0"])
    args = ["--preset", "tiny", "--domains", "nat,linear", "--device", "cpu", "--sft", sft_ckpt, "--max_iters", "60",
            "--save_interval", "25", "--eval_interval", "30", "--preferences", ""]
    whole = str(tmp_path / "whole.pt")
    reward.main(args + ["--out", whole])
    broken = str(tmp_path / "broken.pt")
    with monkeypatch.context() as m:
        _interrupt_after_first_save(m, reward)
        with pytest.raises(Interrupted):
            reward.main(args + ["--out", broken])
    assert os.path.exists(broken + ".samples.json")
    reward.main(args + ["--out", broken])
    assert not os.path.exists(broken + ".samples.json") and not os.path.exists(common.progress_path(broken))
    assert _same(_weights(whole), _weights(broken))
