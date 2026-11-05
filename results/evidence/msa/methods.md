# Descriptive absolute MSA conservation

The frozen Stage C 20-aa/15-aa-stride/M2 catalogue is unchanged. All cleaned full-length sequences are aligned separately within each candidate-bearing family using MAFFT7.526 FFT-NS-1 (--amino --retree 1 --maxiterate 0 --thread 4). Saved alignments are reused only after input/alignment SHA256 checks and exact ungapped-sequence validation. Cache paths, historical alignment commands and scoring invocation provenance are retained.

Only unweighted canonical-amino-acid frequencies are used. C=1-H/ln(20), with gaps/noncanonical residues excluded. A column requires >=50% canonical occupancy across all family input sequences. This occupancy QC is an explicit adaptation beyond excluding entirely invalid columns in the reference description; scores describe the evaluable portion, and sampling/phylogenetic dependence remain limitations.

Actual zero-based half-open occurrence coordinates map through each protein's nongap columns. Each occurrence score is the arithmetic mean of valid column scores, requiring >=50% of its residues to be valid. Missing sequences, invalid coordinates and inadequate valid-position coverage remain explicit rows. The same candidate's repeated occurrences within one protein are reduced to one median score. Candidate mean, sample SD(ddof=1), median, minimum and maximum describe those equally weighted protein scores. SD is missing for one valid protein; all score summaries are missing with no valid proteins. Occurrence coverage averages include unscorable occurrences. All522 candidate rows are retained without rankings, percentiles, background windows, hypothesis tests, p/q values or high/medium/low categories.

Family/global distributions summarize the candidate means with equal candidate weight; family tables are alphabetical and candidate rows retain catalogue order. High absolute conservation does not establish relative enrichment, specificity, experimental function or independent evolutionary replication. Manuscript documents are unchanged by this run.

Run from the isolated worktree with bundled Python and PYTHONPATH=.deps:

    python -m scripts.candidate_evidence.conservation --source-root D:/研究内容/Task1 --output-root results/candidate_evidence/msa
    python -m scripts.candidate_evidence.verify_conservation --source-root D:/研究内容/Task1 --output-root results/candidate_evidence/msa

All scientific score outputs are independently recomputed from retained full alignments by verify_conservation.py. Version, input hashes, configuration and command arguments are recorded in summary.json.
