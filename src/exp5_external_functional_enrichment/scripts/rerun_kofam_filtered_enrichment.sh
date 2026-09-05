#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  rerun_kofam_filtered_enrichment.sh \
    --exp5-root PATH \
    [--output-root PATH] \
    [--interproscan-tsv PATH] \
    [--eggnog-annotations PATH] \
    [--kofamscan-tsv PATH] \
    [--include-interpro-pathways] \
    [--methods "M2 M3 ..."] \
    [--method-motif-root METHOD=PATH]... \
    [--catalog PATH]... \
    [--random-seed INTEGER] \
    [--random-iterations INTEGER] \
    [--python-bin COMMAND] \
    [--rscript-bin COMMAND] \
    [--skip-support-analysis]

The original Exp5 directory is read-only input. Corrected annotations,
enrichment results, and support-analysis tables are written to a separate
output directory. InterProScan Pathway rows are excluded unless explicitly
enabled. Work is skipped only when outputs are non-empty and their content
signature matches current inputs, parameters, and code.

This script preserves the existing clusterProfiler p.adjust <= alpha rule and
does not add a multiple-comparison correction.
EOF
}

EXP5_ROOT=""
OUTPUT_ROOT=""
INTERPROSCAN_TSV=""
EGGNOG_ANNOTATIONS=""
KOFAMSCAN_TSV=""
PYTHON_BIN="python3"
RSCRIPT_BIN="Rscript"
INCLUDE_INTERPRO_PATHWAYS=0
BUILD_SUPPORT=1
RANDOM_SEED=20260714
RANDOM_ITERATIONS=1000
CATALOG_PATHS=()
METHODS_EXPLICIT=0
declare -A METHOD_MOTIF_ROOT_OVERRIDES=()
declare -A METHOD_MOTIF_ROOTS=()

if [[ -n "${METHODS:-}" ]]; then
    read -r -a METHODS_ARRAY <<< "$METHODS"
    METHODS_EXPLICIT=1
else
    METHODS_ARRAY=(M2 M3 MEME GLAM2 STREME SLiMFinder DiMotif MotifAE)
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --exp5-root)
            EXP5_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --interproscan-tsv)
            INTERPROSCAN_TSV="$2"
            shift 2
            ;;
        --eggnog-annotations)
            EGGNOG_ANNOTATIONS="$2"
            shift 2
            ;;
        --kofamscan-tsv)
            KOFAMSCAN_TSV="$2"
            shift 2
            ;;
        --include-interpro-pathways)
            INCLUDE_INTERPRO_PATHWAYS=1
            shift
            ;;
        --methods)
            read -r -a METHODS_ARRAY <<< "$2"
            METHODS_EXPLICIT=1
            shift 2
            ;;
        --method-motif-root)
            if [[ "$2" != *=* ]]; then
                echo "--method-motif-root must use METHOD=PATH syntax" >&2
                exit 2
            fi
            method_name="${2%%=*}"
            method_path="${2#*=}"
            if [[ -z "$method_name" || -z "$method_path" ]]; then
                echo "--method-motif-root must use non-empty METHOD=PATH values" >&2
                exit 2
            fi
            METHOD_MOTIF_ROOT_OVERRIDES["$method_name"]="$method_path"
            shift 2
            ;;
        --catalog)
            CATALOG_PATHS+=("$2")
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --random-iterations)
            RANDOM_ITERATIONS="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --rscript-bin)
            RSCRIPT_BIN="$2"
            shift 2
            ;;
        --skip-support-analysis|--skip-status)
            BUILD_SUPPORT=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$EXP5_ROOT" ]]; then
    echo "--exp5-root is required" >&2
    usage >&2
    exit 2
fi
if ! [[ "$RANDOM_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "--random-seed must be an integer" >&2
    exit 2
fi
if ! [[ "$RANDOM_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--random-iterations must be a positive integer" >&2
    exit 2
fi

EXP5_ROOT="${EXP5_ROOT%/}"
if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="${EXP5_ROOT}_corrected"
fi
OUTPUT_ROOT="${OUTPUT_ROOT%/}"

if ! command -v realpath >/dev/null 2>&1; then
    echo "Required command was not found: realpath" >&2
    exit 1
fi
if [[ ! -d "$EXP5_ROOT" ]]; then
    echo "Exp5 input root does not exist: $EXP5_ROOT" >&2
    exit 1
fi
EXP5_ROOT="$(realpath -e "$EXP5_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
case "$OUTPUT_ROOT/" in
    "$EXP5_ROOT/"*)
        echo "Output root must be separate from and outside the Exp5 input root" >&2
        exit 2
        ;;
esac
case "$EXP5_ROOT/" in
    "$OUTPUT_ROOT/"*)
        echo "Output root must be separate from and not contain the Exp5 input root" >&2
        exit 2
        ;;
esac

if [[ -z "$INTERPROSCAN_TSV" ]]; then
    INTERPROSCAN_TSV="$EXP5_ROOT/interproscan/all_gvp_proteins.fasta.tsv"
fi
if [[ -z "$EGGNOG_ANNOTATIONS" ]]; then
    EGGNOG_ANNOTATIONS="$EXP5_ROOT/eggnog/gvp_eggnog.emapper.annotations"
fi
if [[ -z "$KOFAMSCAN_TSV" ]]; then
    KOFAMSCAN_TSV="$EXP5_ROOT/kofamscan/gvp_kofam.tsv"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_PY="$PROJECT_ROOT/src/exp5_external_functional_enrichment/run.py"
PACKAGE_DIR="$PROJECT_ROOT/src/exp5_external_functional_enrichment/package"
SPLIT_TERM2GENE="$SCRIPT_DIR/split_term2gene_by_family.py"
MERGE_RESULTS="$SCRIPT_DIR/merge_clusterprofiler_results.py"
SUPPORT_SCRIPT="$SCRIPT_DIR/build_fair_support_tables.py"

SEQUENCE_INDEX="$EXP5_ROOT/input/sequence_index.csv"

ANNOTATION_DIR="$OUTPUT_ROOT/annotations"
CLUSTERPROFILER_DIR="$OUTPUT_ROOT/clusterprofiler"
FAMILY_TERM2GENE_DIR="$CLUSTERPROFILER_DIR/family_term2gene"
BATCH_RESULTS_DIR="$CLUSTERPROFILER_DIR/batch_results"
SUPPORT_OUTPUT_DIR="$OUTPUT_ROOT/support_analysis"
R_SCRIPT="$CLUSTERPROFILER_DIR/run_clusterprofiler_enricher.R"

require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Required file is missing or empty: $1" >&2
        exit 1
    fi
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command was not found: $1" >&2
        exit 1
    fi
}

content_signature() {
    local label="$1"
    shift
    {
        printf 'signature_schema=1\nlabel=%s\n' "$label"
        local component
        for component in "$@"; do
            if [[ -f "$component" ]]; then
                printf 'file=%s\t' "$component"
                sha256sum "$component" | awk '{print $1}'
            else
                printf 'value=%s\n' "$component"
            fi
        done
    } | sha256sum | awk '{print $1}'
}

signature_matches() {
    local signature_file="$1"
    local expected="$2"
    [[ -s "$signature_file" ]] && [[ "$(<"$signature_file")" == "$expected" ]]
}

write_signature() {
    local signature_file="$1"
    local signature="$2"
    local temporary="${signature_file}.tmp.$$"
    printf '%s\n' "$signature" > "$temporary"
    mv "$temporary" "$signature_file"
}

REGULAR_FAMILIES=(
    GvpA GvpC GvpD GvpF GvpG GvpH GvpJ
    GvpL GvpM GvpO GvpQ GvpT GvpU GvpV
)
ALL_FAMILIES=("${REGULAR_FAMILIES[@]}" GvpK GvpN)

family_term_outputs_complete() {
    local family
    for family in "${ALL_FAMILIES[@]}"; do
        if [[ ! -s "$FAMILY_TERM2GENE_DIR/term2gene_${family}.csv" ]]; then
            return 1
        fi
    done
}

method_root_has_inventory() {
    local method="$1"
    local root="$2"
    local family
    local batch
    local -a batch_files=()
    for family in "${REGULAR_FAMILIES[@]}"; do
        if [[ ! -s "$root/batches/motif_sets_${family}_${method}.csv" ]]; then
            return 1
        fi
    done
    for family in GvpK GvpN; do
        mapfile -t batch_files < <(
            compgen -G \
                "$root/cluster_batches/$family/motif_sets_${family}_${method}_batch_*.csv" \
                | sort
        )
        if [[ "${#batch_files[@]}" -eq 0 ]]; then
            return 1
        fi
        for batch in "${batch_files[@]}"; do
            if [[ ! -s "$batch" ]]; then
                return 1
            fi
        done
    done
}

REQUESTED_METHODS=("${METHODS_ARRAY[@]}")
METHODS_ARRAY=()
for method in "${REQUESTED_METHODS[@]}"; do
    case "$method" in
        M2|M3|MEME|GLAM2|STREME|SLiMFinder|DiMotif|MotifAE) ;;
        *)
            echo "Unsupported motif method: $method" >&2
            exit 2
            ;;
    esac
    candidate_root=""
    if [[ -n "${METHOD_MOTIF_ROOT_OVERRIDES[$method]:-}" ]]; then
        candidate_root="${METHOD_MOTIF_ROOT_OVERRIDES[$method]}"
        if [[ -d "$candidate_root" ]]; then
            candidate_root="$(realpath -e "$candidate_root")"
        fi
    else
        for possible_root in \
            "$EXP5_ROOT/$method/clusterprofiler" \
            "$EXP5_ROOT/clusterprofiler"
        do
            if [[ -d "$possible_root" ]]; then
                possible_root="$(realpath -e "$possible_root")"
                if method_root_has_inventory "$method" "$possible_root"; then
                    candidate_root="$possible_root"
                    break
                fi
            fi
        done
    fi
    if [[ -n "$candidate_root" ]] && \
       method_root_has_inventory "$method" "$candidate_root"; then
        METHOD_MOTIF_ROOTS["$method"]="$candidate_root"
        METHODS_ARRAY+=("$method")
        echo "Motif inventory: $method -> $candidate_root"
    elif [[ "$METHODS_EXPLICIT" -eq 1 ]]; then
        echo "Incomplete motif-set inventory for explicitly requested method: $method" >&2
        echo "Use --method-motif-root $method=/path/to/clusterprofiler" >&2
        exit 1
    else
        echo "SKIP unavailable method inventory: $method" >&2
    fi
done
if [[ "${#METHODS_ARRAY[@]}" -eq 0 ]]; then
    echo "No complete method motif inventories were found" >&2
    exit 1
fi
echo "Selected methods: ${METHODS_ARRAY[*]}"

require_command "$PYTHON_BIN"
require_command "$RSCRIPT_BIN"
require_command sha256sum
require_file "$RUN_PY"
require_file "$SPLIT_TERM2GENE"
require_file "$MERGE_RESULTS"
require_file "$SUPPORT_SCRIPT"
require_file "$PACKAGE_DIR/annotations.py"
require_file "$PACKAGE_DIR/cli.py"
require_file "$PACKAGE_DIR/clusterprofiler.py"
require_file "$PACKAGE_DIR/support_analysis.py"
require_file "$PACKAGE_DIR/reproducibility.py"
require_file "$INTERPROSCAN_TSV"
require_file "$EGGNOG_ANNOTATIONS"
require_file "$KOFAMSCAN_TSV"
require_file "$SEQUENCE_INDEX"
for catalog in "${CATALOG_PATHS[@]}"; do
    require_file "$catalog"
done

mkdir -p \
    "$ANNOTATION_DIR" \
    "$FAMILY_TERM2GENE_DIR" \
    "$BATCH_RESULTS_DIR" \
    "$SUPPORT_OUTPUT_DIR"

echo "Original Exp5 input: $EXP5_ROOT"
echo "Corrected output:     $OUTPUT_ROOT"
echo "InterPro pathways:    $INCLUDE_INTERPRO_PATHWAYS"
echo "Random seed:          $RANDOM_SEED"

R_SCRIPT_SIGNATURE_FILE="${R_SCRIPT}.signature"
r_script_signature="$(content_signature \
    clusterprofiler-r-script-v2 \
    "$RUN_PY" \
    "$PACKAGE_DIR/cli.py" \
    "$PACKAGE_DIR/clusterprofiler.py")"
if [[ -s "$R_SCRIPT" ]] && \
   signature_matches "$R_SCRIPT_SIGNATURE_FILE" "$r_script_signature"; then
    echo "SKIP R script generation: matching content signature"
else
    "$PYTHON_BIN" "$RUN_PY" write-r-script --output "$R_SCRIPT"
    require_file "$R_SCRIPT"
    write_signature "$R_SCRIPT_SIGNATURE_FILE" "$r_script_signature"
fi

ANNOTATION_SIGNATURE_FILE="$OUTPUT_ROOT/.annotations.signature"
annotation_signature="$(content_signature \
    annotation-v2 \
    "$INTERPROSCAN_TSV" \
    "$EGGNOG_ANNOTATIONS" \
    "$KOFAMSCAN_TSV" \
    "$RUN_PY" \
    "$PACKAGE_DIR/annotations.py" \
    "$PACKAGE_DIR/cli.py" \
    "$PACKAGE_DIR/reproducibility.py" \
    "include_interpro_pathways=$INCLUDE_INTERPRO_PATHWAYS")"
annotation_outputs=(
    "$ANNOTATION_DIR/functional_annotations_long.csv"
    "$ANNOTATION_DIR/term2gene.csv"
    "$ANNOTATION_DIR/term2name.csv"
    "$ANNOTATION_DIR/annotation_namespace_counts.csv"
    "$ANNOTATION_DIR/annotation_filter_counts.csv"
    "$ANNOTATION_DIR/annotation_run_manifest.json"
)
annotation_complete=1
for output in "${annotation_outputs[@]}"; do
    if [[ ! -s "$output" ]]; then
        annotation_complete=0
    fi
done
if [[ "$annotation_complete" -eq 1 ]] && \
   signature_matches "$ANNOTATION_SIGNATURE_FILE" "$annotation_signature"; then
    echo "SKIP annotation parsing: matching content signature"
else
    echo "Parsing annotations with InterProScan Pathway filtering contract"
    parse_arguments=(
        parse-annotations
        --interproscan-tsv "$INTERPROSCAN_TSV"
        --eggnog-annotations "$EGGNOG_ANNOTATIONS"
        --kofamscan-tsv "$KOFAMSCAN_TSV"
        --output-dir "$ANNOTATION_DIR"
    )
    if [[ "$INCLUDE_INTERPRO_PATHWAYS" -eq 1 ]]; then
        parse_arguments+=(--include-interpro-pathways)
    fi
    "$PYTHON_BIN" "$RUN_PY" "${parse_arguments[@]}"
    for output in "${annotation_outputs[@]}"; do
        require_file "$output"
    done
    write_signature "$ANNOTATION_SIGNATURE_FILE" "$annotation_signature"
fi

TERM_SPLIT_SIGNATURE_FILE="$OUTPUT_ROOT/.family_term2gene.signature"
term_split_signature="$(content_signature \
    family-term2gene-v2 \
    "$ANNOTATION_DIR/term2gene.csv" \
    "$SEQUENCE_INDEX" \
    "$SPLIT_TERM2GENE")"
if signature_matches "$TERM_SPLIT_SIGNATURE_FILE" "$term_split_signature" && \
   family_term_outputs_complete; then
    echo "SKIP family TERM2GENE split: matching content signature"
else
    echo "Splitting corrected TERM2GENE by Gvp family"
    "$PYTHON_BIN" "$SPLIT_TERM2GENE" \
        --sequence-index "$SEQUENCE_INDEX" \
        --term2gene "$ANNOTATION_DIR/term2gene.csv" \
        --output-dir "$FAMILY_TERM2GENE_DIR"
    for family in "${ALL_FAMILIES[@]}"; do
        require_file "$FAMILY_TERM2GENE_DIR/term2gene_${family}.csv"
    done
    write_signature "$TERM_SPLIT_SIGNATURE_FILE" "$term_split_signature"
fi

run_enrichment() {
    local motif_sets="$1"
    local term2gene="$2"
    local output="$3"
    local log="$4"
    local temp_output="${output}.tmp.$$"
    local signature_file="${output}.signature"
    local expected_signature

    require_file "$motif_sets"
    require_file "$term2gene"
    expected_signature="$(content_signature \
        clusterprofiler-enricher-v2 \
        "$motif_sets" \
        "$term2gene" \
        "$ANNOTATION_DIR/term2name.csv" \
        "$R_SCRIPT" \
        "significance=clusterProfiler-p.adjust")"
    if [[ -s "$output" ]] && signature_matches "$signature_file" "$expected_signature"; then
        echo "SKIP matching result: $output"
        return
    fi

    echo "START $(date '+%F %T') $(basename "$output")"
    if "$RSCRIPT_BIN" "$R_SCRIPT" \
        --term2gene="$term2gene" \
        --term2name="$ANNOTATION_DIR/term2name.csv" \
        --motif-sets="$motif_sets" \
        --output="$temp_output" \
        >"$log" 2>&1
    then
        require_file "$temp_output"
        mv "$temp_output" "$output"
        write_signature "$signature_file" "$expected_signature"
        echo "DONE  $(date '+%F %T') $(basename "$output")"
    else
        echo "FAILED $(basename "$output"); see $log" >&2
        tail -n 50 "$log" >&2 || true
        exit 1
    fi
}

EXPECTED_TOP_LEVEL_RESULTS=()
EXPECTED_MOTIF_INPUTS=()
for family in "${REGULAR_FAMILIES[@]}"; do
    term2gene="$FAMILY_TERM2GENE_DIR/term2gene_${family}.csv"
    for method in "${METHODS_ARRAY[@]}"; do
        method_root="${METHOD_MOTIF_ROOTS[$method]}"
        family_method_output="$BATCH_RESULTS_DIR/${family}_${method}.csv"
        EXPECTED_TOP_LEVEL_RESULTS+=("$family_method_output")
        EXPECTED_MOTIF_INPUTS+=(
            "$method_root/batches/motif_sets_${family}_${method}.csv"
        )
        run_enrichment \
            "$method_root/batches/motif_sets_${family}_${method}.csv" \
            "$term2gene" \
            "$family_method_output" \
            "$BATCH_RESULTS_DIR/${family}_${method}.log"
    done
done

for family in GvpK GvpN; do
    term2gene="$FAMILY_TERM2GENE_DIR/term2gene_${family}.csv"
    for method in "${METHODS_ARRAY[@]}"; do
        method_root="${METHOD_MOTIF_ROOTS[$method]}"
        family_output_dir="$BATCH_RESULTS_DIR/${family}_batches"
        mkdir -p "$family_output_dir"
        found=0
        batch_outputs=()
        for motif_sets in \
            "$method_root/cluster_batches/$family"/motif_sets_"${family}"_"${method}"_batch_*.csv
        do
            if [[ ! -e "$motif_sets" ]]; then
                continue
            fi
            found=1
            EXPECTED_MOTIF_INPUTS+=("$motif_sets")
            base="$(basename "$motif_sets" .csv)"
            batch_output="$family_output_dir/${base}.csv"
            batch_outputs+=("$batch_output")
            run_enrichment \
                "$motif_sets" \
                "$term2gene" \
                "$batch_output" \
                "$family_output_dir/${base}.log"
        done
        if [[ "$found" -eq 0 ]]; then
            echo "No cluster batches found for $family $method" >&2
            exit 1
        fi

        merged_output="$BATCH_RESULTS_DIR/${family}_${method}.csv"
        EXPECTED_TOP_LEVEL_RESULTS+=("$merged_output")
        merged_signature_file="${merged_output}.signature"
        merged_signature="$(content_signature \
            family-method-merge-v2 \
            "$MERGE_RESULTS" \
            "${batch_outputs[@]}")"
        if [[ -s "$merged_output" ]] && \
           signature_matches "$merged_signature_file" "$merged_signature"; then
            echo "SKIP matching merged result: $merged_output"
        else
            merged_temp="${merged_output}.tmp.$$"
            merge_arguments=(--output "$merged_temp")
            for batch_output in "${batch_outputs[@]}"; do
                merge_arguments+=(--input-file "$batch_output")
            done
            "$PYTHON_BIN" "$MERGE_RESULTS" "${merge_arguments[@]}"
            require_file "$merged_temp"
            mv "$merged_temp" "$merged_output"
            write_signature "$merged_signature_file" "$merged_signature"
        fi
    done
done

echo "Merging the exact selected family-method enrichment inventory"
for output in "${EXPECTED_TOP_LEVEL_RESULTS[@]}"; do
    require_file "$output"
done
ALL_RESULTS="$CLUSTERPROFILER_DIR/clusterprofiler_enrichment.csv"
ALL_RESULTS_SIGNATURE_FILE="${ALL_RESULTS}.signature"
all_results_signature="$(content_signature \
    all-results-merge-v2 \
    "$MERGE_RESULTS" \
    "methods=${METHODS_ARRAY[*]}" \
    "${EXPECTED_TOP_LEVEL_RESULTS[@]}")"
if [[ -s "$ALL_RESULTS" ]] && \
   signature_matches "$ALL_RESULTS_SIGNATURE_FILE" "$all_results_signature"; then
    echo "SKIP all-results merge: matching content signature"
else
    ALL_RESULTS_TEMP="${ALL_RESULTS}.tmp.$$"
    all_merge_arguments=(--output "$ALL_RESULTS_TEMP")
    for family_method_output in "${EXPECTED_TOP_LEVEL_RESULTS[@]}"; do
        all_merge_arguments+=(--input-file "$family_method_output")
    done
    "$PYTHON_BIN" "$MERGE_RESULTS" "${all_merge_arguments[@]}"
    require_file "$ALL_RESULTS_TEMP"
    mv "$ALL_RESULTS_TEMP" "$ALL_RESULTS"
    write_signature "$ALL_RESULTS_SIGNATURE_FILE" "$all_results_signature"
fi

if [[ "$BUILD_SUPPORT" -eq 1 ]]; then
    motif_inputs=("${EXPECTED_MOTIF_INPUTS[@]}")
    mapfile -t family_term_inputs < <(
        find "$FAMILY_TERM2GENE_DIR" -maxdepth 1 -type f -name 'term2gene_Gvp*.csv' -print | sort
    )
    if [[ "${#motif_inputs[@]}" -eq 0 ]]; then
        echo "No motif-set inputs were found for support analysis" >&2
        exit 1
    fi
    SUPPORT_SIGNATURE_FILE="$SUPPORT_OUTPUT_DIR/support_analysis.signature"
    support_signature="$(content_signature \
        support-analysis-v2 \
        "$ALL_RESULTS" \
        "$SUPPORT_SCRIPT" \
        "$PACKAGE_DIR/support_analysis.py" \
        "$PACKAGE_DIR/reproducibility.py" \
        "random_seed=$RANDOM_SEED" \
        "random_iterations=$RANDOM_ITERATIONS" \
        "${motif_inputs[@]}" \
        "${family_term_inputs[@]}" \
        "${CATALOG_PATHS[@]}")"
    support_outputs=(
        "$SUPPORT_OUTPUT_DIR/motif_level_fairness.csv"
        "$SUPPORT_OUTPUT_DIR/topk_family_method_summary.csv"
        "$SUPPORT_OUTPUT_DIR/topk_method_summary.csv"
        "$SUPPORT_OUTPUT_DIR/random_background_summary.csv"
        "$SUPPORT_OUTPUT_DIR/support_analysis_manifest.json"
    )
    support_complete=1
    for output in "${support_outputs[@]}"; do
        if [[ ! -s "$output" ]]; then
            support_complete=0
        fi
    done
    if [[ "$support_complete" -eq 1 ]] && \
       signature_matches "$SUPPORT_SIGNATURE_FILE" "$support_signature"; then
        echo "SKIP support analysis: matching content signature"
    else
        echo "Building corrected motif support and matched-background tables"
        support_arguments=(
            --enrichment "$ALL_RESULTS"
            --family-term2gene-dir "$FAMILY_TERM2GENE_DIR"
            --output-dir "$SUPPORT_OUTPUT_DIR"
            --random-seed "$RANDOM_SEED"
            --random-iterations "$RANDOM_ITERATIONS"
        )
        for motif_input in "${motif_inputs[@]}"; do
            support_arguments+=(--motif-sets "$motif_input")
        done
        for catalog in "${CATALOG_PATHS[@]}"; do
            support_arguments+=(--catalog "$catalog")
        done
        "$PYTHON_BIN" "$SUPPORT_SCRIPT" "${support_arguments[@]}"
        for output in "${support_outputs[@]}"; do
            require_file "$output"
        done
        write_signature "$SUPPORT_SIGNATURE_FILE" "$support_signature"
    fi
fi

echo "Corrected Exp5 recomputation completed: $OUTPUT_ROOT"
