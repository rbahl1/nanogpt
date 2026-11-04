
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
