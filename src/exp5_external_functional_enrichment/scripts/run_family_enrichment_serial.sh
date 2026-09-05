#!/usr/bin/env bash
#chmod +x \
#  src/exp5_external_functional_enrichment/scripts/run_family_enrichment_serial.sh
#
#src/exp5_external_functional_enrichment/scripts/run_family_enrichment_serial.sh

set -u

ROOT="results6/exp5_external_functional_enrichment"

R_SCRIPT="$ROOT/clusterprofiler/run_clusterprofiler_enricher.R"
TERM2NAME="$ROOT/annotations/term2name.csv"
MOTIF_DIR="$ROOT/clusterprofiler/family_method_sets"
TERM2GENE_DIR="$ROOT/clusterprofiler/family_term2gene"
OUT_DIR="$ROOT/clusterprofiler/batch_results"

mkdir -p "$OUT_DIR"

families=(
  GvpA GvpC GvpD GvpF GvpG GvpH
  GvpJ GvpL GvpM GvpN GvpO GvpQ
  GvpT GvpU GvpV
)

if [[ -n "${METHODS:-}" ]]; then
    read -r -a methods <<< "$METHODS"
else
    methods=(M2 M3 MEME GLAM2 STREME SLiMFinder DiMotif MotifAE)
fi

for family in "${families[@]}"; do
    for method in "${methods[@]}"; do
        motif_sets="$MOTIF_DIR/motif_sets_${family}_${method}.csv"
        term2gene="$TERM2GENE_DIR/term2gene_${family}.csv"
        output="$OUT_DIR/${family}_${method}.csv"
        log="$OUT_DIR/${family}_${method}.log"

        if [[ -s "$output" ]]; then
            echo "SKIP $family $method: output exists"
            continue
        fi

        if [[ ! -s "$motif_sets" ]]; then
            echo "ERROR missing motif sets: $motif_sets"
            exit 1
        fi

        if [[ ! -s "$term2gene" ]]; then
            echo "ERROR missing TERM2GENE: $term2gene"
            exit 1
        fi

        echo "START $(date '+%F %T') $family $method"

        start_time=$(date +%s)

        Rscript "$R_SCRIPT" \
          --term2gene="$term2gene" \
          --term2name="$TERM2NAME" \
          --motif-sets="$motif_sets" \
          --output="$output" \
          > "$log" 2>&1

        status=$?
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        if [[ "$status" -eq 0 && -s "$output" ]]; then
            echo \
              "DONE $(date '+%F %T') $family $method " \
              "elapsed=${elapsed}s"
        else
            echo \
              "FAIL $(date '+%F %T') $family $method " \
              "status=$status elapsed=${elapsed}s"
            tail -n 50 "$log"
            exit "$status"
        fi
    done
done
