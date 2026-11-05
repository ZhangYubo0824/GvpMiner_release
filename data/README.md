# Experimental inputs

`ncbi/` contains the local NCBI input snapshot for 23 non-hybrid Gvp family directories, including families outside the sixteen-family final motif comparison. Accession/provenance metadata are retained. Large `.gp` records use `.gp.gz`; decompression preserves original bytes.

`reference/preprocessed/` contains the completed-experiment sequence table and family FASTA/CSV inputs. `reference/alignments/` contains fifteen verified family MSAs with input FASTA, identifier maps and content hashes; the remaining main-comparison family has no primary GvpMiner candidate requiring a profile. These are reference inputs for current sequence/coordinate checks. New alignment caches belong under `work/`.

`literature/` contains curated sequence-verified reference events and the candidate support-window inventory consumed by current site mapping. It does not contain writing or literature-discovery automation. Site compatibility alone does not assign a final reviewed biological interpretation.

Records are retained with source identity. See the project `LICENSES.md` for the unresolved original-project license and third-party terms.
