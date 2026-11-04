## Test accuracy (lean verdicts; prop 500, fol 500, nat 500, linear 500 problems)

| checkpoint | prop | fol | nat | linear | all | Δ all vs SFT [95% CI] | p |
|---|--:|--:|--:|--:|--:|--:|--:|
| pretrained | 15.2 | 22.4 | 9.2 | 9.0 | 14.0 | | |
| SFT | 32.4 | 77.6 | 15.8 | 13.0 | 34.7 | | |
| RLHF (PPO vs reward model) (3 seeds) | 30.1 ± 1.8 | 75.1 ± 0.6 | 14.4 ± 0.4 | 11.5 ± 0.8 | 32.8 ± 0.2 | -1.9 [-3.2, -0.7] | 0.0029 |
| PPO vs checker (3 seeds) | 32.3 ± 0.8 | 77.9 ± 0.8 | 14.5 ± 0.3 | 12.4 ± 1.7 | 34.3 ± 0.4 | -0.4 [-1.5, +0.6] | 0.43 |
| rejection-sampling FT (3 seeds) | 30.2 ± 0.5 | 73.8 ± 1.6 | 13.6 ± 0.7 | 11.3 ± 3.2 | 32.2 ± 1.1 | -2.5 [-3.8, -1.2] | 0.00015 |
| more SFT (control) (3 seeds) | 32.5 ± 1.2 | 81.1 ± 0.3 | 15.1 ± 1.0 | 13.0 ± 1.1 | 35.4 ± 0.2 | +0.7 [-0.4, +1.9] | 0.21 |

Per checkpoint (finished answers, mean answer length, checker vs Lean disagreements):

| checkpoint | all | finished | answer chars | checker true, Lean false | checker false, Lean true |
|---|--:|--:|--:|--:|--:|
| pretrain | 14.0 | 99.9 | 27.5 | 0 | 0 |
| sft | 34.7 | 100.0 | 22.2 | 0 | 0 |
| rlhf_s1 | 32.6 | 100.0 | 22.2 | 0 | 0 |
| rlhf_s2 | 33.0 | 100.0 | 22.3 | 0 | 1 |
| rlhf_s3 | 32.7 | 100.0 | 22.0 | 0 | 1 |
| ppochecker_s1 | 34.5 | 100.0 | 22.4 | 0 | 0 |
| ppochecker_s2 | 33.9 | 100.0 | 22.2 | 0 | 0 |
| ppochecker_s3 | 34.4 | 100.0 | 22.1 | 0 | 0 |
| rft_s1 | 33.5 | 100.0 | 21.7 | 0 | 1 |
| rft_s2 | 31.6 | 100.0 | 21.7 | 0 | 1 |
| rft_s3 | 31.6 | 100.0 | 21.7 | 0 | 0 |
| sftplus_s1 | 35.1 | 100.0 | 22.4 | 0 | 0 |
| sftplus_s2 | 35.6 | 100.0 | 22.7 | 0 | 0 |
| sftplus_s3 | 35.5 | 100.0 | 22.5 | 0 | 0 |

## Reward model

- comparisons: 26783 (1161 held out)
- held-out pairwise accuracy: 55.8%
- AUC of the score for checker correctness (held-out prompts): 0.617 (prop 0.584, fol 0.656, nat 0.626, linear 0.550)
- step 500: train loss 0.657, held-out loss 0.725, held-out accuracy 51.5%
- step 1000: train loss 0.570, held-out loss 0.768, held-out accuracy 54.2%
- step 1500: train loss 0.483, held-out loss 0.851, held-out accuracy 54.8%
- step 2000: train loss 0.406, held-out loss 0.922, held-out accuracy 54.7%
- step 2500: train loss 0.352, held-out loss 0.992, held-out accuracy 53.8%
- step 3000: train loss 0.324, held-out loss 1.024, held-out accuracy 55.8%

## PPO runs

| run | rollout acc, first 10 iters | last 10 iters | score first 10 | last 10 | KL last 10 | val acc start → end |
|---|--:|--:|--:|--:|--:|--:|
| ppochecker_s1 | 30.9 | 25.8 | -0.381 | -0.484 | 0.109 | 32.0 → 34.0 |
| ppochecker_s2 | 28.1 | 29.2 | -0.438 | -0.416 | 0.121 | 32.0 → 36.0 |
| ppochecker_s3 | 31.1 | 30.8 | -0.378 | -0.384 | 0.103 | 32.0 → 34.0 |
| rlhf_s1 | 28.7 | 25.3 | -0.044 | -0.029 | 0.356 | 32.0 → 33.7 |
| rlhf_s2 | 29.4 | 27.0 | -0.022 | -0.027 | 0.364 | 32.0 → 32.7 |
| rlhf_s3 | 29.4 | 29.5 | -0.049 | +0.011 | 0.266 | 32.0 → 34.3 |

## Rejection-sampling fine-tuning rounds

- rft_s1 round 1: 609 verified examples (yield prop 0.42, fol 0.93, nat 0.36, linear 0.31), 29 steps, val accuracy 34.0%
- rft_s1 round 2: 582 verified examples (yield prop 0.42, fol 0.88, nat 0.33, linear 0.31), 28 steps, val accuracy 34.3%
- rft_s2 round 1: 606 verified examples (yield prop 0.43, fol 0.92, nat 0.36, linear 0.31), 29 steps, val accuracy 31.3%
- rft_s2 round 2: 543 verified examples (yield prop 0.39, fol 0.86, nat 0.27, linear 0.29), 26 steps, val accuracy 32.0%
- rft_s3 round 1: 622 verified examples (yield prop 0.43, fol 0.89, nat 0.33, linear 0.42), 30 steps, val accuracy 32.3%
- rft_s3 round 2: 553 verified examples (yield prop 0.42, fol 0.88, nat 0.27, linear 0.28), 26 steps, val accuracy 32.7%

## SFT validation accuracy

- step 500: prop 10.7, fol 50.7, nat 10.7, linear 2.7, all 18.7
- step 1000: prop 22.7, fol 68.0, nat 6.7, linear 5.3, all 25.7
- step 1500: prop 18.7, fol 77.3, nat 10.7, linear 5.3, all 28.0
- step 2000: prop 25.3, fol 77.3, nat 10.7, linear 8.0, all 30.3
- step 2500: prop 29.3, fol 82.7, nat 13.3, linear 6.7, all 33.0
- step 3000: prop 25.3, fol 80.0, nat 18.7, linear 4.0, all 32.0
