# One-off analysis scripts

The scripts the report's deeper analyses came from: per-problem gains and losses, error
types, reward-model rescoring of test answers, matched-prompt PPO baselines, the
validation-versus-test permutation test, the two-level (problems and seeds) bootstrap,
pass@k comparisons, the Lean audit and the split-integrity checks. They were written
during the analysis and are kept as they ran, not polished: they read `results/` and
`out/` (the checkpoints are not in the repository), and some write intermediate files
to a scratch directory named at their top, which you will need to change to rerun
them. `stats.py` and `analyze.py` in the repository root produce the main table and
the figures.

`mystats_out.json`, `passk_table.json` and `integrity.json` are outputs of these
scripts that the report quotes.
