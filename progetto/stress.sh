#!/usr/bin/env bash
set -euo pipefail

# --- parametri ---
maxN=24          # parità da 10 a maxN (step 2)
reps=20          # ripetizioni per stima varianza
thr_list=(1 20)  # confrontiamo sequenziale vs 20-thread
out=results_speed.csv

echo "func,vars,threads,rep,nodes,ms" > "$out"

for n in $(seq 10 2 $maxN); do
  for thr in "${thr_list[@]}"; do
      export OMP_NUM_THREADS="$thr"
      for r in $(seq 0 $((reps-1))); do
          ./run_stress "$n" "$thr" >> "$out"
      done
  done
done
