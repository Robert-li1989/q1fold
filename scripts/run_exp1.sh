#!/usr/bin/env bash
# =============================================================================
# Experiment 1 -- distance-bucketed AP: Q1Fold (v20, quantum) vs the Table 2
# competitive classical extractors (x10, x33, x100, x333, x1000), 5 seeds each,
# plus the RNAfold (McCaskill) reference on the same sequences.
#
# Evaluation only (no training). Each (model, seed) is evaluated once and its
# per-sequence histograms saved; finished runs are skipped on re-run, so it is
# safe to Ctrl-C and restart.
#
#   bash run_exp1.sh                       # everything
#   MODELS="quantum competitive" bash run_exp1.sh   # subset (e.g. per GPU)
#   GPU=1 MODELS="competitive_x333 competitive_x1000" RUN_RNAFOLD=0 bash run_exp1.sh
#
# When everything has finished, paste me:  $OUT/summary/summary.txt
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------- EDIT THESE
BASE=~/Experiments/20260202_Q1fold_polish
REPO=/home/ad/z9li/Experiments/20260202_Q1fold_polish          # folder that CONTAINS the q1fold/ package
TEST=/home/ad/z9li/Experiments/20260202_Q1fold_polish/data/bpRNA-1m/TS0.csv                  # the same TS0 used for Table 1 / Table 2
CFG=$BASE/multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json
SCRIPTS=$BASE                                # where eval_buckets.py / rnafold_probe.py live
# ---------------------------------------------------------------------------

T2=$BASE/table2_scaling_r2
V20=$BASE/results/v20_multiseed
OUT=${OUT:-$BASE/exp1_distance}
GPU=${GPU:-0}
DEVICE=cuda
SEEDS=(42 123 456 789 1011)
MODELS=${MODELS:-"quantum competitive competitive_x33 competitive_x100 competitive_x333 competitive_x1000"}
RUN_RNAFOLD=${RUN_RNAFOLD:-1}
RNAFOLD_WORKERS=${RNAFOLD_WORKERS:-16}
BOOT=${BOOT:-2000}

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT/hist" "$OUT/logs" "$OUT/summary" "$OUT/cache"

for f in "$REPO/q1fold/__init__.py" "$TEST" "$CFG" "$SCRIPTS/eval_buckets.py" "$SCRIPTS/rnafold_probe.py"; do
  [[ -e "$f" ]] || { echo "MISSING: $f  -- edit the paths at the top of this script"; exit 1; }
done

# expected test f1_post from the training-side logs, for the load sanity check
expected_f1() {
python - "$1" <<'EOF'
import sys, json, csv
p = sys.argv[1]
try:
    if p.endswith(".json"):
        print(f"{json.load(open(p))['test_f1']:.3f}")
    else:
        row = next(csv.DictReader(open(p)))
        row = {k.strip(): v for k, v in row.items()}
        print(f"{float(row['f1_post']):.3f}")
except Exception:
    print("n/a")
EOF
}

run_eval() {   # name seed weights expected_file [extra eval_buckets args...]
  local name=$1 seed=$2 weights=$3 expect=$4; shift 4
  local npz="$OUT/hist/${name}_seed${seed}.npz" log="$OUT/logs/${name}_seed${seed}.log"
  if [[ -f "$npz" ]]; then echo "[skip] $name seed $seed (done)"; return; fi
  if [[ ! -f "$weights" ]]; then echo "[MISSING] $weights"; return; fi
  echo "[run ] $name seed $seed  ($(date +%H:%M))"
  if ! python "$SCRIPTS/eval_buckets.py" \
        --repo-root "$REPO" --test-file "$TEST" --config "$CFG" \
        --weights "$weights" --name "$name" --run-seed "$seed" \
        --device "$DEVICE" --batch-size 4 --max-length 512 \
        --cache-dir "$OUT/cache" --save-hist "$npz" "$@" > "$log" 2>&1; then
    echo "  FAILED -- see $log"; tail -5 "$log"; return
  fi
  local got
  got=$(grep -oP 'f1_post=\K[0-9.]+' "$log" | head -1)
  echo "  f1_post here: $got   |   from training logs: $(expected_f1 "$expect")"
}

for g in $MODELS; do
  for s in "${SEEDS[@]}"; do
    if [[ $g == quantum ]]; then
      run_eval quantum "$s" "$V20/seed$s/weights.pmt" "$V20/seed$s/test_metrics.csv"
    else
      rj="$T2/$g/seed$s/result.json"
      [[ -f "$rj" ]] || { echo "[MISSING] $rj"; continue; }
      hidden=$(python -c "import json;print(json.load(open('$rj'))['competitive_hidden'])")
      run_eval "$g" "$s" "$T2/$g/seed$s/weights.pmt" "$rj" \
        --set extractor_type=competitive_classical \
        --set competitive_hidden="$hidden" \
        --set extractor_seed="$s"
    fi
  done
done

# ---- RNAfold reference on the same sequences (CPU only) -----------------------
if [[ $RUN_RNAFOLD == 1 && ! -f "$OUT/hist/RNAfold_seed0.npz" ]]; then
  echo "[run ] RNAfold (McCaskill bpp, $RNAFOLD_WORKERS workers)  ($(date +%H:%M))"
  python "$SCRIPTS/rnafold_probe.py" --test-file "$TEST" --repo-root "$REPO" \
      --max-len 512 --workers "$RNAFOLD_WORKERS" \
      --save-hist "$OUT/hist/RNAfold_seed0.npz" --out "$OUT/rnafold" \
      > "$OUT/logs/RNAfold.log" 2>&1 \
    && echo "  done -- table in $OUT/logs/RNAfold.log" \
    || { echo "  RNAfold FAILED -- see $OUT/logs/RNAfold.log"; tail -5 "$OUT/logs/RNAfold.log"; }
fi

# ---- Aggregate (CPU, ~1-3 min) ------------------------------------------------
echo "[aggregate] bootstrap B=$BOOT"
python "$SCRIPTS/eval_buckets.py" --aggregate "$OUT/hist" --ref quantum \
    --bootstrap "$BOOT" --out "$OUT/summary" 2>&1 | tee "$OUT/summary/summary.txt"
{ echo; echo "===== RNAfold table ====="; sed -n '/^RNAfold vs annotation/,$p' "$OUT/logs/RNAfold.log" 2>/dev/null; } \
    | tee -a "$OUT/summary/summary.txt"

echo
echo "Paste me: $OUT/summary/summary.txt"
echo "Figure:   $OUT/summary/distance_buckets.pdf"
