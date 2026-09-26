#!/usr/bin/env bash
# Stopwatch measurement of kUPS's canonical (NVT) Monte Carlo throughput: the shipped
# examples/nvt_co2_pressure_test.yaml case (50 CO2 in a 30 A cubic box, host/empty.cif's single
# non-interacting dummy site, exchange_prob: 0). PureAdsorb never calls this script; kUPS is a
# one-off checkout with its own `uv`-managed environment outside this repo, the sole approved
# exception to the project's no-Python rule.
#
# A kUPS "cycle" repeats its propagator max(particle_count_over_all_systems, min_cycle_length)
# times, one shared scalar for the whole batch (`mcmc_rigid.py`: `repetitions =
# max(groups.data.system.counts.data.max(), config.min_cycle_length)`). With every host entry
# set to init_adsorbates: [50] and min_cycle_length: 1, that scalar is 50 regardless of how many
# systems are batched, so one cycle is 50 move attempts *per system*: total move attempts for N
# systems over C cycles is N*C*50.
#
# Two modes:
#   timing  -- nsys=1, sweep num_cycles (warmup=0, isolating the linear cost from the one-time
#              JAX compile/startup) to fit t = intercept + nmoves/rate.
#   nscale  -- fixed num_cycles, sweep nsys over powers of two until a genuine failure (OOM or
#              anything other than the known post-hoc block-average crash), to get cost/move
#              against N and the memory ceiling.
#
# A kUPS process that exits nonzero after printing "Done." (simulation.py's own log line,
# emitted once the timed run() call returns and the HDF5 writer has stopped) failed only in its
# convenience call to analyze_mcmc_file's block-average post-processing (too few cycles to form
# a block; observed signatures: "OverflowError: cannot convert float infinity to integer" and,
# for very few cycles, "ValueError: Need at least one array to stack") -- the timed work already
# completed and the failure is not counted against the run. Any failure WITHOUT "Done." in the
# log (e.g. a JAX RESOURCE_EXHAUSTED trying to allocate N GiB) is real and aborts the sweep.
#
# Usage: KUPS=~/src/kups bench/run_kups_nvt.sh timing [num_cycles_csv] [reps]
#        KUPS=~/src/kups bench/run_kups_nvt.sh nscale [num_cycles] [reps] [max_nsys]
set -uo pipefail

KUPS="${KUPS:-$HOME/src/kups}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE="$(date +%Y%m%d)"
LOGDIR="$HERE/results/logs"
mkdir -p "$HERE/results" "$LOGDIR"
HOST="${PA_HOST:-$(hostname)}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
KUPS_COMMIT="$(git -C "$KUPS" rev-parse HEAD)"
GOV_FILE=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
GOVERNOR="$(cat "$GOV_FILE" 2>/dev/null || echo unknown)"
EMPTY_CIF="$KUPS/examples/host/empty.cif"

# Writes an NVT config with $1 independent host boxes (each 50 CO2 in a 30 A cubic box, the
# shipped nvt_co2_pressure_test.yaml case) to $2, num_cycles=$3, num_warmup_cycles=$4, HDF5
# output at $5.
gen_yaml() {
    local n="$1" out_yaml="$2" num_cycles="$3" num_warmup="$4" out_h5="$5"
    {
        cat <<EOF
adsorbates:
  - critical_temperature: 303.75
    critical_pressure: 7_840_000
    acentric_factor: 0.22394
    positions:
      - [0, 0, 0]
      - [-1.16, 0, 0]
      - [1.16, 0, 0]
    symbols:
      - "C_co2"
      - "O_co2"
      - "O_co2"
    charges: [0.7, -0.35, -0.35]

hosts:
EOF
        for _ in $(seq 1 "$n"); do
            printf '  - {cif_file: %s, pressure: 10_000, temperature: 298.15, init_adsorbates: [50], cell_replication: 1}\n' "$EMPTY_CIF"
        done
        cat <<EOF
lj:
  cutoff: 12.0
  tail_correction: true
  mixing_rule: lorentz_berthelot
  parameters:
    O_co2: [3.05, 0.006807690966401598]
    C_co2: [2.8, 0.0023266791910486473]
    X1: [null, null]

ewald:
  real_cutoff: 12.0
  precision: 1.e-6

run:
  out_file: $out_h5
  num_cycles: $num_cycles
  num_warmup_cycles: $num_warmup
  min_cycle_length: 1
  seed: 42
  exchange_prob: 0

max_num_adsorbates: 100
compute_stress: false
EOF
    } > "$out_yaml"
}

# Runs one kUPS invocation, times the whole process, logs to $2. Sets RUN_STATUS to "ok" or
# "oom" (with RUN_ERROR set to the matched line) and RUN_TIME_S to the wall time. Aborts the
# script for any failure that is neither success nor the known post-hoc analysis crash.
run_once() {
    local yaml="$1" log="$2"
    local t0 t1
    t0=$(date +%s.%N)
    ( cd "$KUPS" && JULIA_GUARD=off uv run kups_mcmc_rigid "$yaml" ) > "$log" 2>&1
    local status=$?
    t1=$(date +%s.%N)
    RUN_TIME_S=$(echo "$t1 - $t0" | bc)
    RUN_STATUS="ok"
    RUN_ERROR=""
    if [ "$status" -ne 0 ]; then
        if grep -q "Done\.$" "$log"; then
            : # timed run() completed; only the post-hoc block-average analysis crashed
        elif grep -qi "RESOURCE_EXHAUSTED\|out of memory\|OOM" "$log"; then
            RUN_STATUS="oom"
            RUN_ERROR="$(grep -i "RESOURCE_EXHAUSTED\|Try setting\|to allocate" "$log" | head -5 | tr '\n' ' ')"
        else
            echo "kUPS run failed (exit $status), and it is not the known post-hoc analysis crash nor an OOM: $log" >&2
            tail -n 40 "$log" >&2
            exit 1
        fi
    fi
}

mode="${1:-timing}"

if [ "$mode" = "timing" ]; then
    CYCLES_CSV="${2:-2000,10000,20000}"
    REPS="${3:-3}"
    OUT="$HERE/results/kups_nvt_timing_${HOST}_f64_${DATE}.json"
    echo "kUPS checkout: $KUPS  commit: $KUPS_COMMIT  host: $HOST  gpu: $GPU_NAME"
    samples=()
    IFS=',' read -ra CYCLES <<< "$CYCLES_CSV"
    for nc in "${CYCLES[@]}"; do
        yaml="$(mktemp --suffix=.yaml)"
        h5="$(mktemp -u --suffix=.h5)"
        gen_yaml 1 "$yaml" "$nc" 0 "$h5"
        moves=$((1 * nc * 50))
        echo "== timing: nsys=1 num_cycles=$nc (moves=$moves) x$REPS reps =="
        times=()
        last_h5=""
        for i in $(seq 1 "$REPS"); do
            run_once "$yaml" "$LOGDIR/nvt_timing_nc${nc}_rep${i}_${DATE}.log"
            if [ "$RUN_STATUS" != "ok" ]; then
                echo "unexpected non-ok status during timing sweep: $RUN_STATUS $RUN_ERROR" >&2
                exit 1
            fi
            times+=("$RUN_TIME_S")
            last_h5="$h5"
            echo "  rep $i: ${RUN_TIME_S}s"
        done
        rm -f "$yaml"
        times_json="$(printf '%s,' "${times[@]}")"; times_json="[${times_json%,}]"
        samples+=("{\"nsys\":1,\"num_cycles\":$nc,\"num_warmup_cycles\":0,\"cycle_length\":50,\"nmoves\":$moves,\"times_s\":$times_json}")
        rm -f "$last_h5"
    done
    samples_json="$(printf '%s,' "${samples[@]}")"; samples_json="[${samples_json%,}]"
    jq -n \
        --arg host "$HOST" --arg gpu "$GPU_NAME" --arg kups_commit "$KUPS_COMMIT" \
        --arg governor "$GOVERNOR" --arg date "$(date -Iseconds)" --argjson reps "$REPS" \
        --arg cycles_csv "$CYCLES_CSV" --argjson samples "$samples_json" \
        '{meta: {host: $host, gpu: $gpu, backend: "kups-jax", precision: "f64",
                 kups_commit: $kups_commit, cpu_governor: $governor, date: $date,
                 config: "50 CO2, 30A cubic box, host/empty.cif, exchange_prob=0, num_warmup_cycles=0, nsys=1",
                 num_cycles_csv: $cycles_csv, reps: $reps},
          samples: $samples}' > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
    echo "wrote $OUT"

elif [ "$mode" = "nscale" ]; then
    NUM_CYCLES="${2:-200}"
    REPS="${3:-2}"
    MAX_NSYS="${4:-64}"
    OUT="$HERE/results/kups_nvt_nscale_${HOST}_f64_${DATE}.json"
    echo "kUPS checkout: $KUPS  commit: $KUPS_COMMIT  host: $HOST  gpu: $GPU_NAME"
    samples=()
    failure_json="null"
    nsys=1
    while [ "$nsys" -le "$MAX_NSYS" ]; do
        yaml="$(mktemp --suffix=.yaml)"
        h5="$(mktemp -u --suffix=.h5)"
        gen_yaml "$nsys" "$yaml" "$NUM_CYCLES" 0 "$h5"
        moves=$((nsys * NUM_CYCLES * 50))
        echo "== nscale: nsys=$nsys num_cycles=$NUM_CYCLES (moves=$moves) x$REPS reps =="
        times=()
        failed=0
        for i in $(seq 1 "$REPS"); do
            run_once "$yaml" "$LOGDIR/nvt_nscale_nsys${nsys}_rep${i}_${DATE}.log"
            if [ "$RUN_STATUS" = "oom" ]; then
                echo "  nsys=$nsys OOM: $RUN_ERROR"
                failure_json="$(jq -n --argjson nsys "$nsys" --arg error "$RUN_ERROR" \
                    '{nsys: $nsys, error: $error}')"
                failed=1
                break
            fi
            times+=("$RUN_TIME_S")
            echo "  rep $i: ${RUN_TIME_S}s"
        done
        rm -f "$yaml" "$h5"
        if [ "$failed" -eq 1 ]; then
            break
        fi
        times_json="$(printf '%s,' "${times[@]}")"; times_json="[${times_json%,}]"
        samples+=("{\"nsys\":$nsys,\"num_cycles\":$NUM_CYCLES,\"num_warmup_cycles\":0,\"cycle_length\":50,\"nmoves\":$moves,\"times_s\":$times_json}")
        nsys=$((nsys * 2))
    done
    samples_json="$(printf '%s,' "${samples[@]}")"; samples_json="[${samples_json%,}]"
    jq -n \
        --arg host "$HOST" --arg gpu "$GPU_NAME" --arg kups_commit "$KUPS_COMMIT" \
        --arg governor "$GOVERNOR" --arg date "$(date -Iseconds)" --argjson reps "$REPS" \
        --argjson num_cycles "$NUM_CYCLES" --argjson samples "$samples_json" --argjson failure "$failure_json" \
        '{meta: {host: $host, gpu: $gpu, backend: "kups-jax", precision: "f64",
                 kups_commit: $kups_commit, cpu_governor: $governor, date: $date,
                 config: "N independent 50-CO2/30A-cubic-box systems, host/empty.cif, exchange_prob=0, num_warmup_cycles=0",
                 num_cycles: $num_cycles, reps: $reps},
          samples: $samples, first_failure: $failure}' > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
    echo "wrote $OUT"
else
    echo "unknown mode: $mode (expected timing|nscale)" >&2
    exit 1
fi
