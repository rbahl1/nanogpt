#!/usr/bin/env bash
# The experiment in REPORT.md, end to end, on a 4-core CPU:
#
#   pretrain -> SFT -> four post-training arms from the same SFT checkpoint,
#   three seeds each, then one evaluation of every checkpoint on the same
#   2,000 held-out test problems, every answer compiled with Lean 4.
#
# Arms (equal policy-sample budget for the three that sample: 9,600 answers):
#   rlhf_s*        PPO against a learned reward model (RLHF)
#   ppochecker_s*  PPO against the checker itself
#   rft_s*         rejection-sampling fine-tuning: keep verified samples, fine-tune
#   sftplus_s*     control: 600 more SFT steps on fresh reference proofs
#                  (the same number of optimiser steps and batch size as RFT)
#
# Usage: LEAN=/path/to/lean ./run_experiment.sh [out_dir] [results_dir]
set -euo pipefail
OUT=${1:-out}
RES=${2:-results}
SEEDS=${SEEDS:-"1 2 3"}
JOBS=${JOBS:-2}                      # arms trained concurrently
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$((4 / JOBS))}
mkdir -p "$OUT" "$RES" "$OUT/logs"

run() {  # run <log name> <command...>: skip if the output exists, log to file
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] $name: $*"
  "$@" > "$OUT/logs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] $name done"
}

[ -f "$OUT/pretrain.pt" ] || run pretrain env OMP_NUM_THREADS=4 python pretrain.py --preset small --out "$OUT/pretrain.pt"
[ -f "$OUT/sft.pt" ]      || run sft env OMP_NUM_THREADS=4 python sft.py --preset small --init "$OUT/pretrain.pt" --out "$OUT/sft.pt"
[ -f "$OUT/reward.pt" ]   || run reward env OMP_NUM_THREADS=4 python reward.py --preset small --sft "$OUT/sft.pt" \
                               --out "$OUT/reward.pt" --preferences "$OUT/preferences.jsonl"

arm() {  # arm <name> <seed>
  local name=$1 s=$2 dest="$OUT/${1}_s${2}.pt"
  [ -f "$dest" ] && return 0
  case $name in
    rlhf)       run "${name}_s$s" python rlhf.py --preset small --sft "$OUT/sft.pt" --reward "$OUT/reward.pt" \
                  --reward_source rm --seed "$s" --out "$dest" ;;
    ppochecker) run "${name}_s$s" python rlhf.py --preset small --sft "$OUT/sft.pt" --reward "$OUT/reward.pt" \
                  --reward_source oracle --seed "$s" --out "$dest" ;;
    rft)        run "${name}_s$s" python rft.py --preset small --sft "$OUT/sft.pt" --seed "$s" --out "$dest" ;;
    sftplus)    run "${name}_s$s" python sft.py --preset small --init "$OUT/sft.pt" --out "$dest" \
                  --max_iters 600 --warmup_iters 20 --lr 1e-4 --min_lr 1e-5 --eval_interval 300 --seed $((1000 + s)) ;;
  esac
}

jobs_list=()
for s in $SEEDS; do for name in rlhf ppochecker rft sftplus; do jobs_list+=("$name $s"); done; done
running=0
for j in "${jobs_list[@]}"; do
  arm $j &
  running=$((running + 1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
done
wait

ckpts=("$OUT/pretrain.pt" "$OUT/sft.pt")
for name in rlhf ppochecker rft sftplus; do for s in $SEEDS; do ckpts+=("$OUT/${name}_s${s}.pt"); done; done

# Greedy accuracy on 500 test problems per domain, every answer compiled with Lean.
run eval env OMP_NUM_THREADS=4 python evaluate.py --preset small "${ckpts[@]}" --n 500 --lean true --json "$RES/eval.json"
# Sampled pass@k (16 samples) on 100 test problems per domain, for SFT and seed 1 of each arm.
run passk env OMP_NUM_THREADS=4 python evaluate.py --preset small "$OUT/sft.pt" "$OUT/rlhf_s1.pt" "$OUT/ppochecker_s1.pt" \
  "$OUT/rft_s1.pt" "$OUT/sftplus_s1.pt" --n 100 --passk_n 100 --passk_samples 16 --json "$RES/passk.json"
python stats.py "$RES/eval.json" --baseline sft > "$RES/stats.txt"
cat "$RES/stats.txt"
