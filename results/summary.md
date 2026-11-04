
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
| ppochecker_s1 | 30.9 | 27.5 | -0.381 | -0.452 | 0.101 | 32.0 → 33.7 |
| rlhf_s1 | 28.7 | 27.7 | -0.044 | -0.069 | 0.213 | 32.0 → 32.3 |

## SFT validation accuracy

- step 500: prop 10.7, fol 50.7, nat 10.7, linear 2.7, all 18.7
- step 1000: prop 22.7, fol 68.0, nat 6.7, linear 5.3, all 25.7
- step 1500: prop 18.7, fol 77.3, nat 10.7, linear 5.3, all 28.0
- step 2000: prop 25.3, fol 77.3, nat 10.7, linear 8.0, all 30.3
- step 2500: prop 29.3, fol 82.7, nat 13.3, linear 6.7, all 33.0
- step 3000: prop 25.3, fol 80.0, nat 18.7, linear 4.0, all 32.0
