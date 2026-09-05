#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RESULTS="${RESULTS:-${GVPMINER_OUTPUT_DIR:-$PROJECT/work/results}}"
PREPROCESSED="${PREPROCESSED:-$RESULTS/preprocessed}"
RESULT_BASELINE="${RESULT_BASELINE:-$RESULTS/result_baseline}"
EXP4_OUTPUT="${EXP4_OUTPUT:-$RESULTS/exp4_motif_structure_analysis/baseline}"
EXP5_OUTPUT="${EXP5_OUTPUT:-$RESULTS/exp5_external_functional_enrichment}"
CLEAN_NCBI="${CLEAN_NCBI:-$RESULT_BASELINE/baseline_inputs/ncbi_cleaned}"
PYTHON="${PYTHON:-python}"
export PROJECT RESULTS PREPROCESSED RESULT_BASELINE EXP4_OUTPUT EXP5_OUTPUT CLEAN_NCBI PYTHON

DEFAULT_METHODS=(MEME GLAM2 STREME SLiMFinder DiMotif MotifAE)
STAGE="all"
FORCE=0
PREDICT_MISSING_STRUCTURES=0
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
ALLOW_STANDARD_CSV_ADAPTERS="${ALLOW_STANDARD_CSV_ADAPTERS:-0}"
RUN_CLUSTERPROFILER="${RUN_CLUSTERPROFILER:-1}"
MIN_FAMILY_SIZE="${MIN_FAMILY_SIZE:-100}"
MEME_NMOTIFS="${MEME_NMOTIFS:-20}"
MEME_MINW="${MEME_MINW:-6}"
MEME_MAXW="${MEME_MAXW:-50}"
STREME_MINW="${STREME_MINW:-6}"
STREME_MAXW="${STREME_MAXW:-30}"
SLIMFINDER_SCRIPT="${SLIMFINDER_SCRIPT:-}"
SLIMFINDER_MAXSEQ="${SLIMFINDER_MAXSEQ:-12000}"
SLIMFINDER_DISMASK="${SLIMFINDER_DISMASK:-F}"
SLIMFINDER_INTERACTIVE="${SLIMFINDER_INTERACTIVE:--1}"

FAMILIES=()
METHODS=()

usage() {
  cat <<'USAGE'
Usage:
  src/baselines/run_result_baseline_serial.sh [options]

Options:
  --stage all|preflight|baselines|adapters|exp4|exp5
  --families GvpA GvpK
  --methods MEME GLAM2 STREME SLiMFinder DiMotif MotifAE
  --force
  --predict-missing-structures
  -h|--help

Environment overrides:
  RESULTS, PREPROCESSED, RESULT_BASELINE, EXP4_OUTPUT, EXP5_OUTPUT, CLEAN_NCBI
  PYTHON, SKIP_PREFLIGHT, ALLOW_STANDARD_CSV_ADAPTERS, RUN_CLUSTERPROFILER
  MEME_CMD, GLAM2_CMD, STREME_CMD, SLIMFINDER_CMD, SLIMFINDER_SCRIPT
  SLIMFINDER_MAXSEQ, SLIMFINDER_DISMASK, SLIMFINDER_INTERACTIVE
  DIMOTIF_CMD, MOTIFAE_CMD
  MEME_ADAPTER_CMD, GLAM2_ADAPTER_CMD, STREME_ADAPTER_CMD
  SLIMFINDER_ADAPTER_CMD, DIMOTIF_ADAPTER_CMD, MOTIFAE_ADAPTER_CMD

For DiMotif, MotifAE, and custom adapters, the command runs with these exported variables:
  PROJECT, RESULTS, RESULT_BASELINE, FAMILY, METHOD, FAMILY_FASTA, MODEL_RAW_DIR, MODEL_OUT_DIR
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

has_stage() {
  [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

parse_multi_arg() {
  local -n target="$1"
  shift
  while [[ $# -gt 0 && "$1" != --* ]]; do
    target+=("$1")
    shift
  done
  PARSE_REMAINING=("$@")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || die "--stage requires a value"
      STAGE="$2"
      shift 2
      ;;
    --families)
      shift
      PARSE_REMAINING=()
      parse_multi_arg FAMILIES "$@"
      set -- "${PARSE_REMAINING[@]}"
      ;;
    --methods)
      shift
      PARSE_REMAINING=()
      parse_multi_arg METHODS "$@"
      set -- "${PARSE_REMAINING[@]}"
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --predict-missing-structures)
      PREDICT_MISSING_STRUCTURES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

case "$STAGE" in
  all|preflight|baselines|adapters|exp4|exp5) ;;
  *) die "Invalid --stage: $STAGE" ;;
esac

if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=("${DEFAULT_METHODS[@]}")
fi

mkdir -p "$RESULT_BASELINE/logs" "$CLEAN_NCBI"

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

family_fasta() {
  local family="$1"
  local fasta="$PREPROCESSED/by_family/${family}_cleaned.fasta"
  require_file "$fasta"
  printf '%s\n' "$fasta"
}

load_default_families() {
  local stats="$PREPROCESSED/family_statistics.csv"
  require_file "$stats"
  "$PYTHON" - "$stats" "$MIN_FAMILY_SIZE" <<'PY'
import csv
import sys
from pathlib import Path

stats = Path(sys.argv[1])
min_size = int(sys.argv[2])
with stats.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        family = (row.get("family") or "").strip()
        count = int(float(row.get("count") or row.get("cleaned_count") or 0))
        is_small = (row.get("is_small") or "").strip().upper() in {"TRUE", "1", "YES"}
        if family and count >= min_size and not is_small:
            print(family)
PY
}

if [[ ${#FAMILIES[@]} -eq 0 ]]; then
  mapfile -t FAMILIES < <(load_default_families)
fi

[[ ${#FAMILIES[@]} -gt 0 ]] || die "No families selected."

copy_cleaned_fastas_for_exp5() {
  local family fasta out_dir
  for family in "${FAMILIES[@]}"; do
    fasta="$(family_fasta "$family")"
    out_dir="$CLEAN_NCBI/$family"
    mkdir -p "$out_dir"
    cp "$fasta" "$out_dir/${family}_proteins.fasta"
  done
}

run_logged() {
  local label="$1"
  shift
  local logfile="$RESULT_BASELINE/logs/${label}.log"
  log "RUN $label"
  "$@" >"$logfile" 2>&1
  log "DONE $label -> $logfile"
}

run_shell_logged() {
  local label="$1"
  local command_text="$2"
  local logfile="$RESULT_BASELINE/logs/${label}.log"
  log "RUN $label"
  bash -lc "$command_text" >"$logfile" 2>&1
  log "DONE $label -> $logfile"
}

run_glam2_logged() {
  local label="$1"
  local cmd="$2"
  local fasta="$3"
  local raw_dir="$4"
  local logfile="$RESULT_BASELINE/logs/${label}.log"
  local output_txt="$raw_dir/glam2/glam2.txt"
  local status

  log "RUN $label"
  set +e
  "$cmd" p "$fasta" -O "$raw_dir/glam2" >"$logfile" 2>&1
  status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    if [[ -s "$output_txt" ]]; then
      printf '[WARN] GLAM2 returned %s but %s exists; treating raw motif discovery as complete.\n' "$status" "$output_txt" >>"$logfile"
      log "WARN $label returned $status after writing GLAM2 raw output; continuing"
    else
      return "$status"
    fi
  fi

  log "DONE $label -> $logfile"
}

model_lower() {
  printf '%s\n' "$1" | tr '[:upper:]' '[:lower:]'
}

adapter_env_name() {
  local method="$1"
  case "$method" in
    MEME) printf '%s\n' MEME_ADAPTER_CMD ;;
    GLAM2) printf '%s\n' GLAM2_ADAPTER_CMD ;;
    STREME) printf '%s\n' STREME_ADAPTER_CMD ;;
    SLiMFinder) printf '%s\n' SLIMFINDER_ADAPTER_CMD ;;
    DiMotif) printf '%s\n' DIMOTIF_ADAPTER_CMD ;;
    MotifAE) printf '%s\n' MOTIFAE_ADAPTER_CMD ;;
    *) printf '%s\n' "${method^^}_ADAPTER_CMD" ;;
  esac
}

preflight_errors=0

preflight_has_stage() {
  [[ "$STAGE" == "all" || "$STAGE" == "preflight" || "$STAGE" == "$1" ]]
}

preflight_fail() {
  echo "[PREFLIGHT] ERROR: $*" >&2
  preflight_errors=$((preflight_errors + 1))
}

preflight_ok() {
  echo "[PREFLIGHT] OK: $*"
}

is_standard_csv_adapter_method() {
  case "$1" in
    SLiMFinder|DiMotif) return 0 ;;
    *) return 1 ;;
  esac
}

check_baseline_command_available() {
  local method="$1"
  local cmd=""
  case "$method" in
    MEME) cmd="${MEME_CMD:-meme}" ;;
    GLAM2) cmd="${GLAM2_CMD:-glam2}" ;;
    STREME) cmd="${STREME_CMD:-streme}" ;;
    SLiMFinder)
      if [[ -n "$SLIMFINDER_SCRIPT" ]]; then
        if [[ -f "$SLIMFINDER_SCRIPT" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
          preflight_ok "SLiMFinder script found: $SLIMFINDER_SCRIPT"
        else
          preflight_fail "Set SLIMFINDER_SCRIPT to an existing slimfinder.py file and ensure PYTHON is available."
        fi
        return
      fi
      cmd="${SLIMFINDER_CMD:-slimfinder.py}"
      if [[ -f "$cmd" && "$cmd" == *.py ]]; then
        command -v "$PYTHON" >/dev/null 2>&1 \
          && preflight_ok "SLiMFinder Python script found: $cmd" \
          || preflight_fail "PYTHON command not found for SLiMFinder script: $PYTHON"
        return
      fi
      ;;
    DiMotif)
      [[ -n "${DIMOTIF_CMD:-}" ]] \
        && preflight_ok "DiMotif entrypoint provided by DIMOTIF_CMD" \
        || preflight_fail "Set DIMOTIF_CMD for DiMotif."
      return
      ;;
    MotifAE)
      [[ -n "${MOTIFAE_CMD:-}" ]] \
        && preflight_ok "MotifAE entrypoint provided by MOTIFAE_CMD" \
        || preflight_fail "Set MOTIFAE_CMD for MotifAE."
      return
      ;;
    *)
      preflight_fail "Unsupported baseline method for baseline stage: $method"
      return
      ;;
  esac

  if command -v "$cmd" >/dev/null 2>&1; then
    preflight_ok "$method command found: $cmd"
  else
    preflight_fail "$method command not found: $cmd"
  fi
}

check_adapter_available() {
  local method="$1"
  local lower adapter env_name adapter_command
  lower="$(model_lower "$method")"
  adapter="$PROJECT/src/baselines/adapters/${lower}_adapter.py"
  env_name="$(adapter_env_name "$method")"
  adapter_command="${!env_name:-}"

  if [[ -n "$adapter_command" ]]; then
    preflight_ok "$method adapter command provided by $env_name"
  elif [[ -f "$adapter" ]]; then
    if is_standard_csv_adapter_method "$method" && [[ "$ALLOW_STANDARD_CSV_ADAPTERS" != "1" ]]; then
      preflight_fail "$method currently uses a standard CSV adapter. Set $env_name to a real parser, replace $adapter, or set ALLOW_STANDARD_CSV_ADAPTERS=1 only if the model runner exports standard motif CSV files."
    else
      preflight_ok "$method adapter script found: $adapter"
    fi
  else
    preflight_fail "No adapter for $method. Create $adapter or set $env_name."
  fi
}

check_exp4_structure_cache() {
  if [[ "$PREDICT_MISSING_STRUCTURES" -eq 1 ]]; then
    preflight_ok "Exp4 may predict missing structures because --predict-missing-structures is set"
    return
  fi

  local family structure_dir
  for family in "${FAMILIES[@]}"; do
    structure_dir="$RESULTS/structures/esmfold/$family"
    if [[ ! -d "$structure_dir" ]]; then
      preflight_fail "Missing Exp4 structure cache directory: $structure_dir"
      continue
    fi
    if find "$structure_dir" -maxdepth 1 -type f \( -name '*.json' -o -name '*.pdb' -o -name '*.ent' \) -print -quit | grep -q .; then
      preflight_ok "Exp4 structure cache present for $family"
    else
      preflight_fail "No .json/.pdb/.ent structure files found in $structure_dir"
    fi
  done
}

check_exp5_annotation_inputs() {
  local term2gene="$EXP5_OUTPUT/annotations/term2gene.csv"
  local term2name="$EXP5_OUTPUT/annotations/term2name.csv"
  local interpro="$EXP5_OUTPUT/interproscan/interproscan.tsv"
  local eggnog="$EXP5_OUTPUT/eggnog/gvp_eggnog.emapper.annotations"
  local kofam="$EXP5_OUTPUT/kofamscan/gvp_kofam.tsv"

  if [[ -f "$term2gene" && -f "$term2name" ]]; then
    preflight_ok "Exp5 annotation tables can be reused"
  elif [[ -f "$interpro" || -f "$eggnog" || -f "$kofam" ]]; then
    preflight_ok "Exp5 raw annotation outputs exist and can be parsed"
  else
    preflight_fail "Exp5 annotation inputs missing. Need term2gene.csv/term2name.csv or raw InterProScan/eggNOG/KofamScan outputs under $EXP5_OUTPUT."
  fi

  if [[ "$RUN_CLUSTERPROFILER" == "1" ]]; then
    command -v Rscript >/dev/null 2>&1 \
      && preflight_ok "Rscript found for clusterProfiler stage" \
      || preflight_fail "Rscript not found. Install R/clusterProfiler or set RUN_CLUSTERPROFILER=0."
  else
    preflight_ok "RUN_CLUSTERPROFILER=0, Exp5 will stop after motif sets and R script generation"
  fi
}

run_preflight() {
  preflight_errors=0
  log "Preflight checks started"

  command -v "$PYTHON" >/dev/null 2>&1 \
    && preflight_ok "Python command found: $PYTHON" \
    || preflight_fail "Python command not found: $PYTHON"

  [[ -f "$PREPROCESSED/family_statistics.csv" ]] \
    && preflight_ok "Family statistics found: $PREPROCESSED/family_statistics.csv" \
    || preflight_fail "Missing family statistics: $PREPROCESSED/family_statistics.csv"

  local family method fasta
  for family in "${FAMILIES[@]}"; do
    fasta="$PREPROCESSED/by_family/${family}_cleaned.fasta"
    [[ -f "$fasta" ]] \
      && preflight_ok "Family FASTA found: $fasta" \
      || preflight_fail "Missing family FASTA: $fasta"
  done

  if preflight_has_stage baselines; then
    for method in "${METHODS[@]}"; do
      check_baseline_command_available "$method"
    done
  fi

  if preflight_has_stage adapters; then
    for method in "${METHODS[@]}"; do
      check_adapter_available "$method"
    done
  fi

  if preflight_has_stage exp4; then
    check_exp4_structure_cache
  fi

  if preflight_has_stage exp5; then
    check_exp5_annotation_inputs
  fi

  if [[ "$preflight_errors" -gt 0 ]]; then
    die "Preflight failed with $preflight_errors error(s). Fix the items above, or set SKIP_PREFLIGHT=1 if you intentionally want to run anyway."
  fi

  log "Preflight checks passed"
}

run_builtin_baseline() {
  local family="$1"
  local method="$2"
  local fasta="$3"
  local raw_dir="$RESULT_BASELINE/$family/$method/raw"
  mkdir -p "$raw_dir"

  case "$method" in
    MEME)
      local cmd="${MEME_CMD:-meme}"
      command -v "$cmd" >/dev/null 2>&1 || die "MEME command not found. Set MEME_CMD or load MEME Suite."
      run_logged "${family}_${method}" "$cmd" "$fasta" -protein -oc "$raw_dir/meme" -mod zoops -nmotifs "$MEME_NMOTIFS" -minw "$MEME_MINW" -maxw "$MEME_MAXW"
      ;;
    GLAM2)
      local cmd="${GLAM2_CMD:-glam2}"
      command -v "$cmd" >/dev/null 2>&1 || die "GLAM2 command not found. Set GLAM2_CMD or load MEME Suite."
      run_glam2_logged "${family}_${method}" "$cmd" "$fasta" "$raw_dir"
      ;;
    STREME)
      local cmd="${STREME_CMD:-streme}"
      command -v "$cmd" >/dev/null 2>&1 || die "STREME command not found. Set STREME_CMD or load MEME Suite."
      run_logged "${family}_${method}" "$cmd" --protein --p "$fasta" --oc "$raw_dir/streme" --minw "$STREME_MINW" --maxw "$STREME_MAXW"
      ;;
    SLiMFinder)
      local cmd="${SLIMFINDER_CMD:-slimfinder.py}"
      local -a slimfinder_cmd_args=()
      if [[ -n "$SLIMFINDER_SCRIPT" ]]; then
        [[ -f "$SLIMFINDER_SCRIPT" ]] || die "SLiMFinder script not found: $SLIMFINDER_SCRIPT"
        command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON command not found for SLiMFinder script: $PYTHON"
        slimfinder_cmd_args=("$PYTHON" "$SLIMFINDER_SCRIPT")
      elif [[ -f "$cmd" && "$cmd" == *.py ]]; then
        command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON command not found for SLiMFinder script: $PYTHON"
        slimfinder_cmd_args=("$PYTHON" "$cmd")
      else
        command -v "$cmd" >/dev/null 2>&1 || die "SLiMFinder command not found. Set SLIMFINDER_CMD to an executable command or SLIMFINDER_SCRIPT to the SLiMSuite slimfinder.py path."
        slimfinder_cmd_args=("$cmd")
      fi
      run_logged "${family}_${method}" "${slimfinder_cmd_args[@]}" \
        seqin="$fasta" \
        resdir="$raw_dir/slimfinder" \
        runid="${family}_${method}" \
        i="$SLIMFINDER_INTERACTIVE" \
        dismask="$SLIMFINDER_DISMASK" \
        maxseq="$SLIMFINDER_MAXSEQ"
      ;;
    DiMotif)
      [[ -n "${DIMOTIF_CMD:-}" ]] || die "Set DIMOTIF_CMD for the downloaded DiMotif repository entrypoint."
      export PROJECT RESULTS RESULT_BASELINE FAMILY="$family" METHOD="$method" FAMILY_FASTA="$fasta" MODEL_RAW_DIR="$raw_dir" MODEL_OUT_DIR="$RESULT_BASELINE/$family/$method"
      run_shell_logged "${family}_${method}" "$DIMOTIF_CMD"
      ;;
    MotifAE)
      [[ -n "${MOTIFAE_CMD:-}" ]] || die "Set MOTIFAE_CMD for the downloaded MotifAE repository entrypoint."
      export PROJECT RESULTS RESULT_BASELINE FAMILY="$family" METHOD="$method" FAMILY_FASTA="$fasta" MODEL_RAW_DIR="$raw_dir" MODEL_OUT_DIR="$RESULT_BASELINE/$family/$method"
      run_shell_logged "${family}_${method}" "$MOTIFAE_CMD"
      ;;
    *)
      die "Unsupported baseline method: $method"
      ;;
  esac
}

run_baselines() {
  local family method fasta out_dir sentinel
  for family in "${FAMILIES[@]}"; do
    fasta="$(family_fasta "$family")"
    for method in "${METHODS[@]}"; do
      out_dir="$RESULT_BASELINE/$family/$method"
      sentinel="$out_dir/raw/.baseline_done"
      if [[ "$FORCE" -eq 0 && -f "$sentinel" ]]; then
        log "SKIP baseline $family $method (already done)"
        continue
      fi
      mkdir -p "$out_dir/raw"
      run_builtin_baseline "$family" "$method" "$fasta"
      date '+%F %T' >"$sentinel"
    done
  done
}

run_adapters() {
  local family method out_dir occurrences catalog lower adapter env_name adapter_command
  for family in "${FAMILIES[@]}"; do
    for method in "${METHODS[@]}"; do
      out_dir="$RESULT_BASELINE/$family/$method"
      occurrences="$out_dir/motif_occurrences_${method}.csv"
      catalog="$out_dir/motif_catalog_${method}.csv"
      if [[ "$FORCE" -eq 0 && -f "$occurrences" && -f "$catalog" ]]; then
        log "SKIP adapter $family $method (standard outputs exist)"
        continue
      fi

      lower="$(model_lower "$method")"
      adapter="$PROJECT/src/baselines/adapters/${lower}_adapter.py"
      env_name="$(adapter_env_name "$method")"
      adapter_command="${!env_name:-}"

      export PROJECT RESULTS RESULT_BASELINE FAMILY="$family" METHOD="$method" MODEL_OUT_DIR="$out_dir" MODEL_RAW_DIR="$out_dir/raw"
      if [[ -n "$adapter_command" ]]; then
        run_shell_logged "${family}_${method}_adapter" "$adapter_command"
      elif [[ -f "$adapter" ]]; then
        run_logged "${family}_${method}_adapter" "$PYTHON" "$adapter" --family "$family" --method "$method" --input-dir "$out_dir/raw" --output-dir "$out_dir"
      else
        die "No adapter found for $method. Create $adapter or set $env_name. Required outputs: $occurrences and $catalog"
      fi

      require_file "$occurrences"
      require_file "$catalog"
    done
  done
}

run_exp4() {
  local structure_flag=(--no-predict-missing-structures)
  if [[ "$PREDICT_MISSING_STRUCTURES" -eq 1 ]]; then
    structure_flag=(--predict-missing-structures)
  fi
  run_logged "exp4_baseline" \
    "$PYTHON" "$PROJECT/src/exp4_motif_structure_analysis.py" \
      --from-exp3 \
      --exp3-dir "$RESULT_BASELINE" \
      --families "${FAMILIES[@]}" \
      --methods "${METHODS[@]}" \
      --output-dir "$EXP4_OUTPUT" \
      --prediction-backend esmfold \
      "${structure_flag[@]}" \
      --no-generate-figures \
      --no-progress
}

parse_exp5_annotations_if_needed() {
  local annotations_dir="$EXP5_OUTPUT/annotations"
  local term2gene="$annotations_dir/term2gene.csv"
  local term2name="$annotations_dir/term2name.csv"
  if [[ -f "$term2gene" && -f "$term2name" ]]; then
    log "REUSE Exp5 annotation tables: $annotations_dir"
    return
  fi

  local interpro="$EXP5_OUTPUT/interproscan/interproscan.tsv"
  local eggnog="$EXP5_OUTPUT/eggnog/gvp_eggnog.emapper.annotations"
  local kofam="$EXP5_OUTPUT/kofamscan/gvp_kofam.tsv"
  local args=(parse-annotations --output-dir "$annotations_dir")

  [[ -f "$interpro" ]] && args+=(--interproscan-tsv "$interpro")
  [[ -f "$eggnog" ]] && args+=(--eggnog-annotations "$eggnog")
  [[ -f "$kofam" ]] && args+=(--kofamscan-tsv "$kofam")

  if [[ ${#args[@]} -eq 3 ]]; then
    "$PYTHON" "$PROJECT/src/exp5_external_functional_enrichment/run.py" tool-commands --input-fasta "$EXP5_OUTPUT/input/all_gvp_proteins.fasta" --output-dir "$EXP5_OUTPUT" >&2 || true
    die "Exp5 annotation tables are missing and no raw InterProScan/eggNOG/KofamScan outputs were found under $EXP5_OUTPUT."
  fi

  run_logged "exp5_parse_annotations" "$PYTHON" "$PROJECT/src/exp5_external_functional_enrichment/run.py" "${args[@]}"
}

run_exp5() {
  copy_cleaned_fastas_for_exp5

  if [[ -f "$EXP5_OUTPUT/input/sequence_index.csv" ]]; then
    log "REUSE Exp5 sequence index: $EXP5_OUTPUT/input/sequence_index.csv"
  else
    run_logged "exp5_prepare" \
      "$PYTHON" "$PROJECT/src/exp5_external_functional_enrichment/run.py" \
        prepare \
        --ncbi-root "$CLEAN_NCBI" \
        --output-dir "$EXP5_OUTPUT" \
        --families "${FAMILIES[@]}"
  fi

  parse_exp5_annotations_if_needed

  run_logged "exp5_make_motif_sets" \
    "$PYTHON" "$PROJECT/src/exp5_external_functional_enrichment/run.py" \
      make-motif-sets \
      --exp3-dir "$RESULT_BASELINE" \
      --sequence-index "$EXP5_OUTPUT/input/sequence_index.csv" \
      --output-dir "$EXP5_OUTPUT/clusterprofiler_baseline" \
      --families "${FAMILIES[@]}" \
      --methods "${METHODS[@]}"

  run_logged "exp5_write_r_script" \
    "$PYTHON" "$PROJECT/src/exp5_external_functional_enrichment/run.py" \
      write-r-script \
      --output "$EXP5_OUTPUT/clusterprofiler_baseline/run_clusterprofiler_enricher.R"

  if [[ "$RUN_CLUSTERPROFILER" == "1" ]]; then
    command -v Rscript >/dev/null 2>&1 || die "Rscript not found. Install R/clusterProfiler or set RUN_CLUSTERPROFILER=0 to stop after writing motif sets and the R script."
    run_logged "exp5_clusterprofiler" \
      Rscript "$EXP5_OUTPUT/clusterprofiler_baseline/run_clusterprofiler_enricher.R" \
        --term2gene="$EXP5_OUTPUT/annotations/term2gene.csv" \
        --term2name="$EXP5_OUTPUT/annotations/term2name.csv" \
        --motif-sets="$EXP5_OUTPUT/clusterprofiler_baseline/motif_sets_long.csv" \
        --output="$EXP5_OUTPUT/clusterprofiler_baseline/clusterprofiler_enrichment.csv"
  fi
}

log "Project: $PROJECT"
log "Families: ${FAMILIES[*]}"
log "Methods: ${METHODS[*]}"
log "Stage: $STAGE"
log "Result baseline: $RESULT_BASELINE"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  run_preflight
fi

if [[ "$STAGE" == "preflight" ]]; then
  log "Preflight-only stage completed."
  exit 0
fi

if has_stage baselines; then
  run_baselines
fi

if has_stage adapters; then
  run_adapters
fi

if has_stage exp4; then
  run_exp4
fi

if has_stage exp5; then
  run_exp5
fi

log "All requested serial stages completed."
