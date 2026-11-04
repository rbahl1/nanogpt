# Trained checkpoints

The 15 checkpoints of the experiment in REPORT.md (weights and metadata only, no
optimiser state; about 3.4 MB each):

| file | what |
|------|------|
| `pretrain.pt` | pretrained on whole Lean 4 declarations (5,000 steps) |
| `sft.pt` | supervised fine-tuning from `pretrain.pt` (3,000 steps) |
| `reward.pt` | the Bradley-Terry reward model (SFT trunk + scalar head) |
| `rlhf_s{1,2,3}.pt` | PPO against the reward model, seeds 1-3 |
| `ppochecker_s{1,2,3}.pt` | PPO against the checker, seeds 1-3 |
| `rft_s{1,2,3}.pt` | rejection-sampling fine-tuning, seeds 1-3 |
| `sftplus_s{1,2,3}.pt` | the control: 600 more SFT steps, seeds 1-3 |

Load one with `common.load_checkpoint(path, "cpu")`, or try it:

    python sample.py checkpoints/sft.pt "example (p q : Prop) : p ∧ q → q ∧ p :="
    python evaluate.py --preset small checkpoints/sft.pt --n 100
    python export_web.py checkpoints/sft.pt --out web/dist   # the in-browser chat page

`results/preferences.jsonl` holds the reward model's 26,783 checker-labelled comparisons.
