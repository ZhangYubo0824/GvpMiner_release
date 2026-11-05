#!/usr/bin/env bash
# First-pass guarded native sampling; never launches the legacy timeout pass.
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$BASELINE_ROOT/../.." && pwd)"
cd -- "$BASELINE_ROOT"

resume_args=()
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 1 || "$1" != "--resume" ]]; then
    printf 'Usage: bash run_all_families.sh [--resume]\n' >&2
    exit 2
  fi
  resume_args=(--resume)
fi

MOTIFAE_RESULTS="${MOTIFAE_RESULTS:-$PACKAGE_ROOT/work/results/baselines/motifae}"
MOTIFAE_RUN="${MOTIFAE_RUN:-guarded_first_all}"
: "${MOTIFAE_GIBBS:?Set MOTIFAE_GIBBS to your licensed GibbsCluster 2.0 installation}"
MOTIFAE_FASTA="${MOTIFAE_FASTA:-$PACKAGE_ROOT/data/reference/preprocessed/by_family}"
MOTIFAE_QUAL="${MOTIFAE_QUAL:-$MOTIFAE_RESULTS/alignment/_qualification/guarded_first}"
PYTHON="${PYTHON:-python}"
mkdir -p -- "$MOTIFAE_RESULTS"

"$PYTHON" scripts/baseline_adapters/motifae_family_alignment.py preflight \
  --discovery-root "$MOTIFAE_RESULTS/discovery" --fasta-dir "$MOTIFAE_FASTA" \
  --select-run latest --gibbs-root "$MOTIFAE_GIBBS" \
  --report "$MOTIFAE_RESULTS/preflight_${MOTIFAE_RUN}.json"

"$PYTHON" scripts/baseline_adapters/motifae_family_alignment.py qualify \
  --gibbs-root "$MOTIFAE_GIBBS" --output-dir "$MOTIFAE_QUAL" --timeout 300 --reuse

"$PYTHON" -u scripts/baseline_adapters/motifae_family_alignment.py run \
  --discovery-root "$MOTIFAE_RESULTS/discovery" --fasta-dir "$MOTIFAE_FASTA" \
  --select-run latest --alignment-root "$MOTIFAE_RESULTS/alignment" \
  --gibbs-root "$MOTIFAE_GIBBS" --qualification-dir "$MOTIFAE_QUAL" \
  --run-name "$MOTIFAE_RUN" \
  --job-timeout 300 --heartbeat 15 "${resume_args[@]}" \
  2>&1 | tee -a "$MOTIFAE_RESULTS/${MOTIFAE_RUN}.log"

printf 'Alignment completed. Family pointer: %s\n' \
  "$MOTIFAE_RESULTS/alignment/_runs/$MOTIFAE_RUN/family_runs.json"
