#!/usr/bin/env bash
# Runs the kUPS reference Widom simulation (bench/kups_widom_reference.yaml) and logs its
# output. This is documentation of an external measurement: PureAdsorb never calls this script,
# and nothing here is a dependency of the package. kUPS itself is a one-off checkout with its
# own `uv`-managed environment outside this repo (the narrow, user-approved exception to the
# project's no-Python rule); see docs/superpowers/plans/2026-09-05-milestone-a-widom.md Task 11
# for that approval's scope.
#
# Usage: KUPS=~/src/kups bench/run_kups.sh
set -euo pipefail

KUPS="${KUPS:-$HOME/src/kups}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/results"
LOG="$HERE/results/kups_widom_$(hostname)_$(date +%Y%m%d).log"

echo "kUPS checkout: $KUPS"
echo "kUPS commit:   $(git -C "$KUPS" rev-parse HEAD)"
echo "JAX version:   $(cd "$KUPS" && JULIA_GUARD=off uv run python -c 'import jax; print(jax.__version__)')"
echo "log:           $LOG"

(
    cd "$KUPS"
    JULIA_GUARD=off uv run kups_mcmc_widom "$HERE/kups_widom_reference.yaml"
) 2>&1 | tee "$LOG"
