from __future__ import annotations

from pathlib import Path



def build_clusterprofiler_r_script() -> str:
    return r'''#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0) {
    return(default)
  }
  sub(prefix, "", hit[1], fixed = TRUE)
}

term2gene_path <- get_arg("term2gene")
term2name_path <- get_arg("term2name")
motif_sets_path <- get_arg("motif-sets")
output_path <- get_arg("output", "clusterprofiler_enrichment.csv")
pvalue_cutoff <- as.numeric(get_arg("pvalue-cutoff", "1"))
qvalue_cutoff <- as.numeric(get_arg("qvalue-cutoff", "1"))

if (is.null(term2gene_path) || is.null(term2name_path) || is.null(motif_sets_path)) {
  stop("Required arguments: --term2gene=PATH --term2name=PATH --motif-sets=PATH")
}

library(clusterProfiler)

term2gene <- read.csv(term2gene_path, stringsAsFactors = FALSE)
term2name <- read.csv(term2name_path, stringsAsFactors = FALSE)
motif_sets <- read.csv(motif_sets_path, stringsAsFactors = FALSE)

names(term2gene)[1:2] <- c("term_id", "protein_id")
names(term2name)[1:2] <- c("term_id", "term_name")

all_results <- list()
keys <- unique(motif_sets[, c("family", "method", "motif_cluster")])

for (i in seq_len(nrow(keys))) {
  key <- keys[i, ]
  subset_rows <- motif_sets[
    motif_sets$family == key$family &
      motif_sets$method == key$method &
      motif_sets$motif_cluster == key$motif_cluster,
  ]

  foreground_ids <- unique(subset_rows$protein_id[subset_rows$set_type == "foreground"])
  universe_ids <- unique(subset_rows$protein_id[subset_rows$set_type == "universe"])

  if (length(foreground_ids) == 0 || length(universe_ids) == 0) {
    next
  }

  enriched <- tryCatch(
    enricher(
      gene = foreground_ids,
      universe = universe_ids,
      TERM2GENE = term2gene,
      TERM2NAME = term2name,
      pAdjustMethod = "BH",
      pvalueCutoff = pvalue_cutoff,
      qvalueCutoff = qvalue_cutoff
    ),
    error = function(e) {
      stop(
        "Enrichment failed for ",
        key$family, "/", key$method, "/", key$motif_cluster,
        ": ", conditionMessage(e),
        call. = FALSE
      )
    }
  )

  if (is.null(enriched)) {
    next
  }
  result <- as.data.frame(enriched)
  if (nrow(result) == 0) {
    next
  }
  result$family <- key$family
  result$method <- key$method
  result$motif_cluster <- key$motif_cluster
  result$foreground_total <- length(foreground_ids)
  result$universe_total <- length(universe_ids)
  all_results[[length(all_results) + 1]] <- result
}

if (length(all_results) == 0) {
  empty <- data.frame(
    family = character(),
    method = character(),
    motif_cluster = character(),
    ID = character(),
    Description = character(),
    GeneRatio = character(),
    BgRatio = character(),
    pvalue = numeric(),
    p.adjust = numeric(),
    qvalue = numeric(),
    geneID = character(),
    Count = integer(),
    foreground_total = integer(),
    universe_total = integer()
  )
  write.csv(empty, output_path, row.names = FALSE)
} else {
  write.csv(do.call(rbind, all_results), output_path, row.names = FALSE)
}
'''


def write_clusterprofiler_r_script(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_clusterprofiler_r_script(), encoding="utf-8", newline="\n")
    return path




