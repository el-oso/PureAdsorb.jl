#!/usr/bin/env bash
# Runs kUPS's shipped `examples/mcmc_rigid.yaml` GCMC case as a reference oracle for task 9's
# cross-code validation, plus a memory-ceiling sweep. PureAdsorb never calls this script; kUPS is
# a one-off checkout with its own `uv`-managed environment outside this repo, the sole approved
# exception to the project's no-Python rule.
#
# `mcmc_rigid.yaml` labels its pressure "10_000  # Pa (10 bar)", which is wrong: 1e4 Pa is
# 0.1 bar. It also sets translation_prob/rotation_prob/reinsertion_prob to 0, leaving
# exchange_prob at its RunConfig default (0.5) as the only nonzero move weight, so every cycle is
# an exchange attempt regardless of that default's absolute value. Modes:
#
#   main     -- runs mcmc_rigid.yaml UNCHANGED (the labelled-vs-actual pressure and the
#               100%-exchange move mix are exactly what task 9 must match, not correct), then a
#               second short NVT-only run of the same host at init_adsorbates: [0] to get kUPS's
#               host-host energy baseline (kUPS reports the FULL system energy; PureAdsorb's
#               total_energy never does, since that term is a constant that cancels). Both are
#               analyzed with `analyze_mcmc_file` (n_blocks=None, its own automatic block count).
#   nscale   -- N independent copies of the SAME mcmc_rigid.yaml case (short num_cycles, no
#               warmup) batched into one run, nsys doubling from 1 until a genuine failure
#               (OOM), to find kUPS's GCMC memory ceiling.
#
# Usage: KUPS=~/src/kups bench/run_kups_gcmc.sh main
#        KUPS=~/src/kups bench/run_kups_gcmc.sh nscale [num_cycles] [max_nsys]
set -uo pipefail

KUPS="${KUPS:-$HOME/src/kups}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE="$(date +%Y%m%d)"
LOGDIR="$HERE/results/logs"
mkdir -p "$HERE/results" "$LOGDIR"
HOST="${PA_HOST:-$(hostname)}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
KUPS_COMMIT="$(git -C "$KUPS" rev-parse HEAD)"
EXDIR="$KUPS/examples"

# Extracts one system's analyze_mcmc_file() result (system id 0 only -- every mode here runs
# single- or identical-system batches) as JSON: {energy:{mean,sem,n_blocks}, loading:{...},
# heat_of_adsorption:{...}} (kUPS's own sign convention, NOT negated -- task 9's own comment
# negates it when comparing to PureAdsorb's Widom-convention q_st).
analyze_to_json() {
    local h5="$1"
    ( cd "$KUPS" && JULIA_GUARD=off uv run python -c "
import json, sys
from kups.application.mcmc.analysis import analyze_mcmc_file
r = analyze_mcmc_file('$h5')[0]
def ba(x):
    return {'mean': float(x.mean if x.mean.ndim == 0 else x.mean[0]),
            'sem': float(x.sem if x.sem.ndim == 0 else x.sem[0]),
            'n_blocks': int(x.n_blocks)}
print(json.dumps({'energy': ba(r.energy), 'loading': ba(r.loading),
                   'heat_of_adsorption': ba(r.heat_of_adsorption)}))
"
    )
}

mode="${1:-main}"

if [ "$mode" = "main" ]; then
    OUT="$HERE/results/kups_gcmc_main_${HOST}_f64_${DATE}.json"
    MAIN_H5="$EXDIR/mcmc_rigid.h5"
    BASE_YAML="$EXDIR/mcmc_rigid_baseline.yaml"
    BASE_H5="$EXDIR/mcmc_rigid_baseline.h5"

    echo "kUPS checkout: $KUPS  commit: $KUPS_COMMIT  host: $HOST  gpu: $GPU_NAME"
    echo "== main: examples/mcmc_rigid.yaml, unchanged =="
    rm -f "$MAIN_H5"
    ( cd "$EXDIR" && JULIA_GUARD=off uv run kups_mcmc_rigid mcmc_rigid.yaml ) \
        > "$LOGDIR/gcmc_main_${DATE}.log" 2>&1
    if [ $? -ne 0 ] && ! grep -q "Done\.$" "$LOGDIR/gcmc_main_${DATE}.log"; then
        echo "kUPS main GCMC run failed:" >&2; tail -n 40 "$LOGDIR/gcmc_main_${DATE}.log" >&2; exit 1
    fi
    main_json="$(analyze_to_json "$MAIN_H5")"
    echo "main: $main_json"

    echo "== baseline: same host, init_adsorbates=[0], exchange_prob=0 =="
    cat > "$BASE_YAML" <<'EOF'
adsorbates:
  - !import ["adsorbate/co2.yaml"]

hosts:
  - cif_file: host/RUBTAK.cif
    pressure: 10_000   # Pa
    temperature: 298.15   # K
    init_adsorbates: [0]
    cell_replication: 3

lj: !import ["lennard_jones/trappe.yaml"]

ewald:
  real_cutoff: 12.0
  precision: 1.e-6

run:
  out_file: mcmc_rigid_baseline.h5
  num_cycles: 100
  num_warmup_cycles: 0
  min_cycle_length: 1
  seed: 42
  translation_prob: 1
  rotation_prob: 0
  reinsertion_prob: 0
  exchange_prob: 0
EOF
    rm -f "$BASE_H5"
    ( cd "$EXDIR" && JULIA_GUARD=off uv run kups_mcmc_rigid mcmc_rigid_baseline.yaml ) \
        > "$LOGDIR/gcmc_baseline_${DATE}.log" 2>&1
    if [ $? -ne 0 ] && ! grep -q "Done\.$" "$LOGDIR/gcmc_baseline_${DATE}.log"; then
        echo "kUPS baseline run failed:" >&2; tail -n 40 "$LOGDIR/gcmc_baseline_${DATE}.log" >&2; exit 1
    fi
    baseline_json="$(analyze_to_json "$BASE_H5")"
    echo "baseline: $baseline_json"
    rm -f "$MAIN_H5" "$BASE_H5" "$BASE_YAML"

    jq -n --arg host "$HOST" --arg gpu "$GPU_NAME" --arg kups_commit "$KUPS_COMMIT" \
        --arg date "$(date -Iseconds)" --argjson main "$main_json" --argjson baseline "$baseline_json" \
        '{meta: {host: $host, gpu: $gpu, backend: "kups-jax", precision: "f64", kups_commit: $kups_commit,
                 date: $date,
                 config: "examples/mcmc_rigid.yaml unchanged: RUBTAK 3x3x3, CO2, 298.15K, pressure=1e4 Pa (labelled 10 bar in the yaml, actually 0.1 bar), 100% exchange (other three move probs are 0), num_cycles=10000, num_warmup_cycles=1000, min_cycle_length=20, seed=42",
                 baseline_config: "same host, init_adsorbates=[0], exchange_prob=0, 100 cycles, gives U_host-host (kUPS reports the FULL system energy; PureAdsorb never does)"},
          main: $main, baseline: $baseline}' > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
    echo "wrote $OUT"

elif [ "$mode" = "nscale" ]; then
    NUM_CYCLES="${2:-200}"
    MAX_NSYS="${3:-64}"
    OUT="$HERE/results/kups_gcmc_nscale_${HOST}_f64_${DATE}.json"
    echo "kUPS checkout: $KUPS  commit: $KUPS_COMMIT  host: $HOST  gpu: $GPU_NAME"
    samples=()
    failure_json="null"
    nsys=1
    while [ "$nsys" -le "$MAX_NSYS" ]; do
        yaml="$EXDIR/gcmc_nscale_${nsys}.yaml"
        h5="$EXDIR/gcmc_nscale_${nsys}.h5"
        {
            echo 'adsorbates:'
            echo '  - !import ["adsorbate/co2.yaml"]'
            echo ''
            echo 'hosts:'
            for _ in $(seq 1 "$nsys"); do
                echo '  - {cif_file: host/RUBTAK.cif, pressure: 10_000, temperature: 298.15, init_adsorbates: [0], cell_replication: 3}'
            done
            echo ''
            echo 'lj: !import ["lennard_jones/trappe.yaml"]'
            echo ''
            echo 'ewald:'
            echo '  real_cutoff: 12.0'
            echo '  precision: 1.e-6'
            echo ''
            echo 'run:'
            echo "  out_file: $(basename "$h5")"
            echo "  num_cycles: $NUM_CYCLES"
            echo '  num_warmup_cycles: 0'
            echo '  min_cycle_length: 20'
            echo '  seed: 42'
            echo '  translation_prob: 0'
            echo '  rotation_prob: 0'
            echo '  reinsertion_prob: 0'
            echo '  exchange_prob: 1'
        } > "$yaml"
        echo "== nscale: nsys=$nsys num_cycles=$NUM_CYCLES =="
        rm -f "$h5"
        t0=$(date +%s.%N)
        ( cd "$EXDIR" && JULIA_GUARD=off uv run kups_mcmc_rigid "$(basename "$yaml")" ) \
            > "$LOGDIR/gcmc_nscale_nsys${nsys}_${DATE}.log" 2>&1
        status=$?
        t1=$(date +%s.%N)
        rm -f "$yaml" "$h5"
        log="$LOGDIR/gcmc_nscale_nsys${nsys}_${DATE}.log"
        if [ "$status" -ne 0 ] && ! grep -q "Done\.$" "$log"; then
            if grep -qi "RESOURCE_EXHAUSTED\|out of memory\|OOM" "$log"; then
                err="$(grep -i "RESOURCE_EXHAUSTED\|Try setting\|to allocate" "$log" | head -5 | tr '\n' ' ')"
                echo "  nsys=$nsys OOM: $err"
                failure_json="$(jq -n --argjson nsys "$nsys" --arg error "$err" '{nsys: $nsys, error: $error}')"
                break
            else
                echo "kUPS nscale run failed (exit $status), not a known OOM signature: $log" >&2
                tail -n 40 "$log" >&2
                exit 1
            fi
        fi
        wall=$(echo "$t1 - $t0" | bc)
        echo "  nsys=$nsys ok: ${wall}s"
        samples+=("{\"nsys\":$nsys,\"num_cycles\":$NUM_CYCLES,\"wall_s\":$wall}")
        nsys=$((nsys * 2))
    done
    samples_json="$(printf '%s,' "${samples[@]}")"; samples_json="[${samples_json%,}]"
    jq -n --arg host "$HOST" --arg gpu "$GPU_NAME" --arg kups_commit "$KUPS_COMMIT" \
        --arg date "$(date -Iseconds)" --argjson num_cycles "$NUM_CYCLES" \
        --argjson samples "$samples_json" --argjson failure "$failure_json" \
        '{meta: {host: $host, gpu: $gpu, backend: "kups-jax", precision: "f64", kups_commit: $kups_commit,
                 date: $date,
                 config: "N independent copies of examples/mcmc_rigid.yaml (RUBTAK 3x3x3 + CO2, 100% exchange), num_warmup_cycles=0"},
          num_cycles: $num_cycles, samples: $samples, first_failure: $failure}' > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
    echo "wrote $OUT"
else
    echo "unknown mode: $mode (expected main|nscale)" >&2
    exit 1
fi
