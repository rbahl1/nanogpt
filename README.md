# nanoGPT: a GPT that has only ever read Lean 4

A small character-level GPT (the nanoGPT architecture, 0.8M parameters) taken
through the modern training pipeline on nothing but Lean 4 declarations:

1. **Pretraining**: next-token prediction on whole Lean 4 declarations.
2. **Supervised fine-tuning**: complete a declaration from its header,
   `<bos> header <ans> proof <eos>`, with the loss on the proof only.
3. **Post-training**, compared head to head from the same SFT checkpoint with
   the same number of policy samples:
   * **RLHF**: a Bradley-Terry reward model trained on comparisons of the SFT
     model's own answers, then PPO against it with a KL penalty to the SFT model;
   * **PPO against the checker** itself (RL from verifier feedback);
   * **rejection-sampling fine-tuning** (expert iteration): keep the sampled
     answers that check, fine-tune on them;
   * a control: **more SFT** on fresh reference proofs.

**Only formal data.** Every pretraining document, SFT target, compared answer
and evaluation problem is Lean 4 source; there is no natural language anywhere
in the pipeline. **No safety training**: no refusal data, no harmlessness
preferences, no content filters, no safety term in any reward. Rewards and
preferences measure one thing: whether the Lean checks (and, for proofs, how
short it is).

**[REPORT.md](REPORT.md)** describes the experiment, its measurements and what
they show.

## The data: four Lean 4 domains

A problem is a Lean 4 declaration cut in two: the model sees the header and
writes the rest, and `header + " " + answer` must compile.

| domain | the model sees | and writes |
|--------|----------------|------------|
| `prop` | `example (p q s : Prop) : p ∧ (¬p ∧ ¬q) ∧ s → ¬p :=` | `fun h0 => h0.2.1.1` |
| `fol` | `example (α : Type) (P R : α → Prop) (q : Prop) : (∃ x, P x) → (∀ x, P x → q ∧ R x) → ∃ x, q ∧ R x :=` | `fun ⟨x, h0⟩ h1 => ⟨x, h1 x h0⟩` |
| `nat` | `example : (10 - 10 * 6) % 15 + 7 =` | `7 := by decide` |
| `linear` | `example (x : Int) (h : 7 * x - 14 = x + 10) : x =` | `4 := by omega` |

* **prop**: intuitionistic propositional logic; proofs are terms or tactic
  blocks, and the reference is the shortest proof a proof search finds.
* **fol**: first-order logic over a type `α` with predicates, a relation,
  propositions and individuals, proved in term mode (102 inference schemas,
  instantiated with random formulas, then shortened by search).
* **nat**: closed `Nat` arithmetic as Lean evaluates it: truncated subtraction
  (`3 - 7 = 0`), floor division, `n / 0 = 0`, `n % 0 = n`.
* **linear**: one-variable linear equations over `Int`, closed by `omega`.

Each domain pairs a generator with a **checker**: a Python implementation of
the fragment of Lean 4 the domain uses, fast enough to judge every sample
during training. A checker must be *sound with respect to Lean*: it may reject
a proof Lean would accept, never the reverse. `lean_oracle.py` compiles
declarations with a real Lean 4 toolchain; the tests use it to check that every
generated declaration compiles and that every answer a checker accepts,
including hundreds of thousands of fuzzed mutants, compiles too, and the
evaluation compiles every answer every model gives.

Problems are split into train/val/test by a hash of a canonical form of the
header (atoms renamed in order of appearance, bound variables normalised,
commuted operands and swapped sides of an equation identified), so a renamed
or respelled copy of a test problem never sits in the training data.

## The pipeline

| stage | script | starts from | data | objective |
|-------|--------|-------------|------|-----------|
| pretraining | `pretrain.py` | random init | 160k whole declarations | next-token cross-entropy |
| SFT | `sft.py` | `pretrain.pt` | 80k (header, reference proof) pairs | cross-entropy on the proof and `<eos>` |
| reward model | `reward.py` | `sft.pt` trunk + scalar head | comparisons of SFT samples, labelled by the checker | Bradley-Terry |
| RLHF | `rlhf.py --reward_source rm` | `sft.pt` (policy, frozen reference), `reward.pt` (reward, critic) | train-split headers | clipped PPO on `RM score − β·KL`, GAE |
| PPO vs checker | `rlhf.py --reward_source oracle` | `sft.pt` | train-split headers | clipped PPO on `±1 from the checker − β·KL` |
| RFT | `rft.py` | `sft.pt` | the policy's own samples that check | cross-entropy |
| more SFT | `sft.py --init out/sft.pt` | `sft.pt` | fresh reference proofs | cross-entropy |
| evaluation | `evaluate.py`, `stats.py` | any checkpoints | 2,000 test problems | greedy pass@1 (checker and Lean), sampled pass@k, paired statistics |

`./run_experiment.sh` runs all of it (four arms, three seeds each) on a 4-core
CPU in a few hours. Every long stage saves its full state every 250 steps and
resumes exactly when rerun.

## Usage

```bash
pip install -r requirements.txt
export LEAN=/path/to/lean            # optional: Lean 4 (core only, any recent release)

./run_experiment.sh out results      # the whole experiment

# or stage by stage
python pretrain.py --preset small --out out/pretrain.pt
python sft.py      --preset small --init out/pretrain.pt --out out/sft.pt
python reward.py   --preset small --sft out/sft.pt --out out/reward.pt
python rlhf.py     --preset small --sft out/sft.pt --reward out/reward.pt --out out/rlhf.pt
python rft.py      --preset small --sft out/sft.pt --out out/rft.pt
python evaluate.py --preset small out/sft.pt out/rlhf.pt out/rft.pt --lean true --json results/eval.json
python stats.py results/eval.json --baseline sft

python sample.py out/rlhf.pt "example (p q : Prop) : p ∧ q → q ∧ p :="
python export_web.py out/rlhf.pt --out web/dist && python -m http.server -d web/dist   # in-browser chat page
```

Presets (`config.py`): `tiny` runs everything in a minute or two for smoke
tests; `small` (the default: 4 layers × 128 wide, 0.8M parameters) is the one
the report uses; `base` (6 × 384, 10.7M parameters) wants a GPU. Any config
field can be overridden on the command line (`--max_iters 5000 --domains prop,nat`).

## Files

| file | contents |
|------|----------|
| `model.py` | GPT with padding-aware attention and a key/value cache; `RewardModel` (GPT + scalar head), the reward model and the PPO critic |
| `tokenizer.py` | character-level tokenizer over ASCII plus the Lean 4 symbols the domains use |
| `formal/` | the four domains (generators, checkers, canonical split keys); `formal.verify` and `formal.prefer` |
| `lean_oracle.py` | compiles batches of declarations with a real Lean 4 toolchain |
| `data.py` | the canonical-form train/val/test split and problem sampling |
| `common.py` | checkpoints, resumable progress, batching, the learning-rate schedule, sampling, checker-verified evaluation |
| `config.py` | per-stage hyperparameters and presets |
| `pretrain.py`, `sft.py`, `reward.py`, `rlhf.py`, `rft.py` | the training stages |
| `evaluate.py`, `stats.py` | evaluation (checker and Lean verdicts, pass@k) and paired statistics (Wilson CIs, paired bootstrap, McNemar, seed pooling) |
| `run_experiment.sh` | the experiment in REPORT.md |
| `sample.py`, `export_web.py`, `web/` | sampling from a checkpoint; the in-browser chat page (`web/gpt.js` reproduces the PyTorch forward pass, `web/checker.js` ports the checkers) |
| `results/` | the experiment's evaluation output |
| `tests/` | checker soundness (fuzzing against Lean), differential tests of the Python and JavaScript checkers, the model and its cache, PPO and RFT mechanics, resumption, statistics, a tiny end-to-end run |

Run the tests with `python -m pytest -q` (about 10 minutes on 4 cores; the
Lean tests are skipped without a toolchain).

## How this was built

An earlier version of this project was lost with the machine it lived on,
except for its published chat page. The page held the JavaScript model and
the JavaScript ports of the four checkers, and these became the specification
for the rebuild: `model.py` reproduces `web/gpt.js` to 1e-5 on the old trained
weights (`reference/old_rlhf_model.json`), and the Python checkers agree with
`web/checker.js` on every one of several hundred thousand differential cases.
Each component was then reviewed adversarially before training. Among the
findings, each reproduced and fixed with a regression test:

* two places where the old checkers accepted proofs Lean 4 rejects: formulas
  separated by non-space whitespace (fol), and headers or `trivial` goals deep
  enough to exhaust Lean's recursion limit (prop);
* train/test leakage through respellings: about a fifth of nat and a seventh
  of linear test problems had a commuted or side-swapped twin in training;
* batched generation giving every prompt the budget of the longest one, which
  capped PPO answers at about 105 tokens while other arms got the full budget;
* `lean_oracle.py` blaming errors Lean reports at the start of the file on the
  first declaration of a batch.
