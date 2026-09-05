#!/usr/bin/env bash

#chmod +x \
#  src/exp5_external_functional_enrichment/scripts/run_cluster_batches.sh
#
#src/exp5_external_functional_enrichment/scripts/run_cluster_batches.sh GvpK

set -u

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 FAMILY"
    exit 1
fi

FAMILY="$1"
if [[ -n "${METHODS:-}" ]]; then
    read -r -a methods <<< "$METHODS"
else
    methods=(M2 M3 MEME GLAM2 STREME SLiMFinder DiMotif MotifAE)
fi

ROOT="results6/exp5_external_functional_enrichment"

R_SCRIPT="$ROOT/clusterprofiler/run_clusterprofiler_enricher.R"
TERM2GENE="$ROOT/clusterprofiler/family_term2gene/term2gene_${FAMILY}.csv"
TERM2NAME="$ROOT/annotations/term2name.csv"
BATCH_DIR="$ROOT/clusterprofiler/cluster_batches/${FAMILY}"
OUT_DIR="$ROOT/clusterprofiler/batch_results/${FAMILY}_batches"

mkdir -p "$OUT_DIR"

if [[ ! -s "$TERM2GENE" ]]; then
    echo "ERROR missing TERM2GENE: $TERM2GENE"
    exit 1
fi

for method in "${methods[@]}"; do
    found=0

    for input in \
      "$BATCH_DIR"/motif_sets_"${FAMILY}"_"${method}"_batch_*.csv
    do
        if [[ ! -e "$input" ]]; then
            continue
        fi

        found=1

        base=$(basename "$input" .csv)
        output="$OUT_DIR/${base}.csv"
        log="$OUT_DIR/${base}.log"

        if [[ -s "$output" ]]; then
            echo "SKIP $base"
            continue
        fi

        echo "START $(date '+%F %T') $base"

        start_time=$(date +%s)

        Rscript "$R_SCRIPT" \
          --term2gene="$TERM2GENE" \
          --term2name="$TERM2NAME" \
          --motif-sets="$input" \
          --output="$output" \
          > "$log" 2>&1

        status=$?
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        if [[ "$status" -eq 0 && -s "$output" ]]; then
            echo \
              "DONE $(date '+%F %T') $base " \
              "elapsed=${elapsed}s"
        else
            echo \
              "FAIL $(date '+%F %T') $base " \
              "status=$status elapsed=${elapsed}s"
            tail -n 50 "$log"
            exit "$status"
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        echo "ERROR no batches found for $FAMILY $method"
        exit 1
    fi
done
