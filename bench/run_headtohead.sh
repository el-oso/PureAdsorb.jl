#!/usr/bin/env bash
# Head-to-head Widom throughput: kUPS (JAX) vs PureAdsorb, same GPU, same physical case
# (RUBTAK 3x3x3 + CO2, 298.15 K, real-space cutoff 12 A, Ewald precision 1e-6, empty host, no
# displacement moves), same total insertion count, both float64 (kUPS forces jax_enable_x64;
# see bench/kups_widom_reference.yaml). Alternates a kUPS block with a PureAdsorb block for each
# grid point, so neither code is measured entirely cold or entirely GPU-warmed relative to the
# other.
#
# `ninsert` below is the TOTAL insertions across the batch (PureAdsorb's convention): kUPS
# evaluates `num_widom_per_cycle` insertions per system per cycle in parallel across its
# batched systems, so its per-system count is ninsert/nsys, set as num_widom_per_cycle with
# num_cycles=1 (one compiled while_loop per run, avoiding Python-loop overhead on the kUPS
# side that the default MC-driver cycle structure would add).
#
# nsys=64 (and 32, 16, 8) run out of GPU memory on the 6 GB RTX 3050 while building the batched
# state, before any cycle runs (JaxRuntimeError: RESOURCE_EXHAUSTED, trying to allocate 51.26 /
# 23.82 / 10.08 / 5.10 GiB respectively) -- confirmed by hand for the 10000-insertion point.
# nsys=4 is the largest batch that fits. The default grid below uses nsys=4 in place of 64.
#
# A `num_cycles=1` kUPS run always exits 1: after finishing its (correctly timed) cycle and
# writing the HDF5 output, `main()`'s convenience call to `analyze_widom_file` runs
# `optimal_block_average`, which needs at least 8 cycles (`block_transform`'s `min_blocks=4`
# needs `n_samples // 2 >= 4`) and raises `OverflowError: cannot convert float infinity to
# integer` on fewer. This happens after the timed work completes, so it does not invalidate a
# sample; `run_kups` below distinguishes it from a real failure (an OOM, or anything else) by
# its exact traceback signature and treats only that one signature as non-fatal.
#
# kUPS prints no insertion count, and its HDF5 output is zstd-compressed (filter id 32015) with
# no zstd HDF5 plugin on this host, so `h5dump`/`h5ls` cannot decode `n_samples` either -- this
# script instead checks that the run's tqdm progress bar reports completing its one requested
# cycle ("1/1"), plus the exit-status check above; there is no independent count of insertions
# actually performed.
#
# Usage: KUPS=~/src/kups REPS=5 bench/run_headtohead.sh
set -uo pipefail

KUPS="${KUPS:-$HOME/src/kups}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
REPS="${REPS:-5}"
GRID="${HTH_GRID:-1:10000 1:100000 1:1000000 4:10000 4:100000 4:1000000}"
DATE="$(date +%Y%m%d)"
LOGDIR="$HERE/results/logs"
mkdir -p "$HERE/results" "$LOGDIR"
GOV_FILE=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
GOVERNOR="$(cat "$GOV_FILE" 2>/dev/null || echo unknown)"
if [ -w "$GOV_FILE" ]; then
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f"; done
    echo "CPU governor set to performance for this run (was: $GOVERNOR)"
    trap 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "'"$GOVERNOR"'" > "$f"; done' EXIT
else
    echo "CPU governor not writable without sudo; staying at '$GOVERNOR' (recorded in meta)"
fi

# PA_HOST distinguishes multiple GPUs on the same physical host in result filenames (e.g.
# "neuromancer4070" vs "neuromancer" for an earlier card in the same eGPU enclosure slot),
# matching widom_bench.jl's own PA_HOST convention; it defaults to the bare hostname.
HOST="${PA_HOST:-$(hostname)}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
KUPS_OUT="$HERE/results/kups_widom_timing_${HOST}_f64_${DATE}.json"
KUPS_COMMIT="$(git -C "$KUPS" rev-parse HEAD)"
JAX_VERSION="$(cd "$KUPS" && JULIA_GUARD=off uv run python -c 'import jax; print(jax.__version__)')"

# Runs one kUPS invocation, logs it to $2, and aborts the whole script on any failure except
# the num_cycles=1 analyze_widom_file OverflowError documented above.
run_kups() {
    local yaml="$1" log="$2"
    ( cd "$KUPS" && JULIA_GUARD=off uv run kups_mcmc_widom "$yaml" ) > "$log" 2>&1
    local status=$?
    if [ "$status" -ne 0 ]; then
        if grep -q "OverflowError: cannot convert float infinity to integer" "$log" \
            && grep -q "analyze_widom_file" "$log" \
            && ! grep -q "RESOURCE_EXHAUSTED" "$log"; then
            : # known post-hoc analysis-only failure; the timed run already completed
        else
            echo "kUPS run failed (exit $status): $log" >&2
            tail -n 30 "$log" >&2
            exit 1
        fi
    fi
    if ! grep -qE '100%\|.*\| 1/1 ' "$log"; then
        echo "kUPS run did not report completing its single requested cycle: $log" >&2
        exit 1
    fi
}

kups_samples=()
for point in $GRID; do
    nsys="${point%%:*}"
    ninsert="${point##*:}"
    per_system=$((ninsert / nsys))

    yaml="$(mktemp --suffix=.yaml)"
    h5="$(mktemp -u --suffix=.h5)"
    {
        echo "adsorbates:"
        echo "  - !import [\"$KUPS/examples/adsorbate/co2.yaml\"]"
        echo "hosts:"
        for _ in $(seq 1 "$nsys"); do
            printf '  - {cif_file: %s/examples/host/RUBTAK.cif, pressure: 100_000, temperature: 298.15, init_adsorbates: [0], cell_replication: 3}\n' "$KUPS"
        done
        echo "lj: !import [\"$KUPS/examples/lennard_jones/trappe.yaml\"]"
        echo "ewald: {real_cutoff: 12.0, precision: 1.e-6}"
        printf 'run: {out_file: %s, num_cycles: 1, num_warmup_cycles: 0, num_displacements_per_cycle: 0, num_widom_per_cycle: %d, translation_prob: 0.333, rotation_prob: 0.333, reinsertion_prob: 0.333, seed: 42}\n' "$h5" "$per_system"
        echo "max_num_adsorbates: 200"
    } > "$yaml"

    echo "== kUPS: nsys=$nsys ninsert=$ninsert (num_widom_per_cycle=$per_system, num_cycles=1) =="
    run_kups "$yaml" "$LOGDIR/kups_nsys${nsys}_ninsert${ninsert}_warmup_${DATE}.log"
    times=()
    for i in $(seq 1 "$REPS"); do
        t0=$(date +%s.%N)
        run_kups "$yaml" "$LOGDIR/kups_nsys${nsys}_ninsert${ninsert}_rep${i}_${DATE}.log"
        t1=$(date +%s.%N)
        dt=$(echo "$t1 - $t0" | bc)
        times+=("$dt")
        echo "  kUPS rep $i: ${dt}s"
    done
    rm -f "$yaml" "$h5"
    times_json="$(printf '%s,' "${times[@]}")"
    times_json="[${times_json%,}]"
    kups_samples+=("{\"nsys\":$nsys,\"ninsert\":$ninsert,\"backend\":\"kups-jax\",\"times_s\":$times_json}")

    echo "== PureAdsorb CUDA f64: nsys=$nsys ninsert=$ninsert =="
    ( cd "$REPO" && PA_BACKEND=cuda PA_PRECISION=f64 PA_GRID="$nsys:$ninsert" PA_REPS="$REPS" PA_HOST="$HOST" \
        julia --project=bench/gpu bench/widom_bench.jl )
done

new_samples_json="$(printf '%s,' "${kups_samples[@]}")"
new_samples_json="[${new_samples_json%,}]"

# Merge with any existing file for today: a nsys value covered by this run replaces all of that
# nsys's prior entries (e.g. a corrected nsys after a re-run), other nsys values are kept as-is.
if [ -f "$KUPS_OUT" ]; then
    prior_samples_json="$(jq -c '.samples' "$KUPS_OUT")"
else
    prior_samples_json="[]"
fi

jq -n \
    --arg host "$HOST" --arg gpu "$GPU_NAME" --arg backend "kups-jax" \
    --arg precision "f64" --arg jax "$JAX_VERSION" --arg kups_commit "$KUPS_COMMIT" \
    --arg governor "$GOVERNOR" --arg date "$(date -Iseconds)" --arg grid "$GRID" --argjson reps "$REPS" \
    --argjson prior "$prior_samples_json" --argjson new "$new_samples_json" \
    '($new | map(.nsys)) as $new_nsys |
     {meta: {host: $host, gpu: $gpu, backend: $backend, precision: $precision, jax_version: $jax,
             kups_commit: $kups_commit, cpu_governor: $governor, date: $date, grid: $grid, reps: $reps},
      samples: (($prior | map(. as $s | select(($new_nsys | index($s.nsys)) | not))) + $new)}' \
    > "$KUPS_OUT.tmp" && mv "$KUPS_OUT.tmp" "$KUPS_OUT"
echo "wrote $KUPS_OUT"
