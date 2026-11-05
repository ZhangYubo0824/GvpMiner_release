import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package import annotations
from package import cli
from package import clusterprofiler
from package import motif_sets
from package import motif_validation
from package import prepare


class TestExp5ExternalFunctionalEnrichment(unittest.TestCase):








    def test_discovers_occurrence_files_from_exp3_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp3_dir = Path(tmp) / "exp3_domain_discovery"
            family_dir = exp3_dir / "GvpA"
            family_dir.mkdir(parents=True)
            occurrence_path = family_dir / "motif_occurrences_M2.csv"
            occurrence_path.write_text(
                "motif_uid,family,method,parent_id\n"
                "GvpA_M2_motif_001,GvpA,M2,P1\n",
                encoding="utf-8",
            )

            paths = motif_sets.discover_occurrence_files(
                exp3_dir=exp3_dir,
                families=["GvpA"],
                methods=["M2"],
            )

        self.assertEqual(paths, [occurrence_path])

    def test_build_sequence_index_merges_family_fastas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ncbi = root / "ncbi"
            (ncbi / "GvpA").mkdir(parents=True)
            (ncbi / "GvpC").mkdir(parents=True)
            (ncbi / "GvpA" / "GvpA_proteins.fasta").write_text(
                ">P1.1 gas vesicle protein\nACDE\nFG\n>P2.1 another protein\nMMMM\n",
                encoding="utf-8",
            )
            (ncbi / "GvpC" / "GvpC_proteins.fasta").write_text(
                ">C1.1 gvp c\nTTTT\n",
                encoding="utf-8",
            )

            rows = prepare.build_sequence_index(
                ncbi_root=ncbi,
                output_dir=root / "external",
                families=["GvpA", "GvpC"],
            )

            self.assertEqual([row.protein_id for row in rows], ["P1.1", "P2.1", "C1.1"])
            self.assertEqual([row.family for row in rows], ["GvpA", "GvpA", "GvpC"])
            fasta_text = (root / "external" / "input" / "all_gvp_proteins.fasta").read_text(encoding="utf-8")
            index_text = (root / "external" / "input" / "sequence_index.csv").read_text(encoding="utf-8")
            self.assertIn(">P1.1 family=GvpA", fasta_text)
            self.assertIn("P2.1,GvpA", index_text)

    def test_parse_external_annotation_sources_to_term2gene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interpro = root / "interpro.tsv"
            interpro.write_text(
                "P1\tmd5\t80\tPfam\tPF00741\tGas vesicle protein\t1\t30\t1e-9\tT\t2026-01-01\t"
                "IPR000638\tGas vesicle protein\tGO:0031411|GO:0005198\tReactome:R-TEST\n",
                encoding="utf-8",
            )
            eggnog = root / "eggnog.emapper.annotations"
            eggnog.write_text(
                "#query\tseed_ortholog\tevalue\tscore\teggNOG_OGs\tmax_annot_lvl\tCOG_category\tDescription\t"
                "Preferred_name\tGOs\tEC\tKEGG_ko\tKEGG_Pathway\tKEGG_Module\tKEGG_Reaction\tKEGG_rclass\t"
                "BRITE\tKEGG_TC\tCAZy\tBiGG_Reaction\tPFAMs\n"
                "P2\tseed\t1e-30\t200\tOG\tBacteria\tM\tmembrane\tgvp\tGO:0012506\t3.6.4.-\t"
                "ko:K02446\tko02040\t-\t-\t-\t-\t-\t-\t-\tPF00004,PF00741\n",
                encoding="utf-8",
            )
            kofam = root / "kofam.tsv"
            kofam.write_text(
                "# gene name KO threshold score E-value KO definition\n"
                "* P3 K02446 100.0 120.0 1e-40 gas vesicle protein\n"
                "  P4 K06137 102.63 12.4 0.12 subthreshold candidate\n",
                encoding="utf-8",
            )

            terms = []
            terms.extend(annotations.parse_interproscan_tsv(interpro))
            terms.extend(annotations.parse_eggnog_annotations(eggnog))
            terms.extend(annotations.parse_kofamscan_tsv(kofam))
            annotations.write_term_tables(terms, root / "tables")

            term2gene = (root / "tables" / "term2gene.csv").read_text(encoding="utf-8")
            term2name = (root / "tables" / "term2name.csv").read_text(encoding="utf-8")
            long_table = (root / "tables" / "functional_annotations_long.csv").read_text(encoding="utf-8")
            self.assertIn("GO:0031411,P1", term2gene)
            self.assertIn("PF00741,P2", term2gene)
            self.assertIn("K02446,P3", term2gene)
            self.assertNotIn("K06137,P4", term2gene)
            self.assertIn("K02446,gas vesicle protein", term2name)
            self.assertIn("IPR000638,Gas vesicle protein", term2name)
            self.assertIn("eggNOG,KEGG_KO,K02446", long_table)

    def test_interproscan_pathways_are_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpro = Path(tmp) / "interpro.tsv"
            interpro.write_text(
                "P1\tmd5\t80\tPfam\tPF00741\tGas vesicle protein\t1\t30\t1e-9\tT\t2026-01-01\t"
                "IPR000638\tGas vesicle protein\tGO:0031411\tReactome:R-TEST\n",
                encoding="utf-8",
            )

            terms = annotations.parse_interproscan_tsv(interpro)

        self.assertFalse(
            any(term.namespace == "Pathway" for term in terms),
            "InterProScan Pathway records must be opt-in, not part of the default TERM2GENE input",
        )

    def test_interproscan_pathways_can_be_explicitly_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpro = Path(tmp) / "interpro.tsv"
            interpro.write_text(
                "P1\tmd5\t80\tPfam\tPF00741\tGas vesicle protein\t1\t30\t1e-9\tT\t2026-01-01\t"
                "IPR000638\tGas vesicle protein\tGO:0031411\tReactome:R-TEST\n",
                encoding="utf-8",
            )

            terms = annotations.parse_interproscan_tsv(
                interpro,
                include_pathways=True,
            )

        self.assertIn(
            ("InterProScan", "Pathway", "Reactome:R-TEST", "P1"),
            {
                (term.source, term.namespace, term.term_id, term.protein_id)
                for term in terms
            },
        )

    def test_annotation_writer_emits_source_namespace_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "annotations"
            annotations.write_term_tables(
                [
                    annotations.AnnotationTerm(
                        "InterProScan", "Pfam", "PF00741", "P1", "Gas vesicle protein"
                    ),
                    annotations.AnnotationTerm(
                        "InterProScan", "Pfam", "PF00741", "P2", "Gas vesicle protein"
                    ),
                    annotations.AnnotationTerm(
                        "eggNOG", "KEGG_KO", "K02446", "P2", "Gvp"
                    ),
                ],
                output_dir,
            )

            qc_path = output_dir / "annotation_namespace_counts.csv"
            self.assertTrue(qc_path.exists())
            with qc_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            rows,
            [
                {
                    "source": "InterProScan",
                    "namespace": "Pfam",
                    "annotation_links": "2",
                    "unique_terms": "1",
                    "unique_proteins": "2",
                },
                {
                    "source": "eggNOG",
                    "namespace": "KEGG_KO",
                    "annotation_links": "1",
                    "unique_terms": "1",
                    "unique_proteins": "1",
                },
            ],
        )

    def test_parse_annotations_writes_reproducibility_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interpro = root / "interpro.tsv"
            output_dir = root / "annotations"
            fields = [
                "P1",
                "md5",
                "100",
                "Pfam",
                "PF00741",
                "Gas vesicle protein",
                "1",
                "40",
                "1e-10",
                "T",
                "2026-01-01",
                "IPR000001",
                "Example",
                "GO:0008150",
                "Reactome:R-TEST",
            ]
            interpro.write_text("\t".join(fields) + "\n", encoding="utf-8")

            cli.main(
                [
                    "parse-annotations",
                    "--interproscan-tsv",
                    str(interpro),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            manifest_path = output_dir / "annotation_run_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with (output_dir / "annotation_filter_counts.csv").open(
                encoding="utf-8"
            ) as handle:
                filter_rows = list(csv.DictReader(handle))

        self.assertEqual(manifest["operation"], "parse-annotations")
        self.assertFalse(manifest["parameters"]["include_interpro_pathways"])
        self.assertEqual(manifest["inputs"][0]["path"], str(interpro.resolve()))
        self.assertEqual(len(manifest["inputs"][0]["sha256"]), 64)
        self.assertEqual(
            manifest["statistical_boundary"],
            "existing clusterProfiler p.adjust <= alpha; no additional multiplicity correction",
        )
        pathway_audit = next(
            row
            for row in filter_rows
            if row["source"] == "InterProScan" and row["namespace"] == "Pathway"
        )
        self.assertEqual(pathway_audit["parsed_links"], "1")
        self.assertEqual(pathway_audit["retained_links"], "0")
        self.assertEqual(pathway_audit["filtered_links"], "1")

    def test_build_motif_sets_uses_family_matched_universe(self):
        occurrence_rows = [
            {
                "motif_uid": "GvpA_M2_motif_001",
                "family": "GvpA",
                "method": "M2",
                "parent_id": "P1",
                "occ_start": "2",
                "occ_end": "8",
            },
            {
                "motif_uid": "GvpA_M2_motif_001",
                "family": "GvpA",
                "method": "M2",
                "parent_id": "P2",
                "occ_start": "1",
                "occ_end": "7",
            },
        ]
        protein_rows = [
            prepare.SequenceRecord("P1", "GvpA", "AAAA", "p1.fasta", "desc"),
            prepare.SequenceRecord("P2", "GvpA", "CCCC", "p2.fasta", "desc"),
            prepare.SequenceRecord("P3", "GvpA", "GGGG", "p3.fasta", "desc"),
            prepare.SequenceRecord("C1", "GvpC", "TTTT", "c1.fasta", "desc"),
        ]

        sets = motif_sets.build_motif_sets(occurrence_rows, protein_rows)

        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0].motif_cluster, "GvpA_M2_motif_001")
        self.assertEqual(sets[0].foreground_ids, ("P1", "P2"))
        self.assertEqual(sets[0].background_ids, ("P3",))
        self.assertEqual(sets[0].universe_ids, ("P1", "P2", "P3"))

    def test_clusterprofiler_script_uses_enricher_with_custom_universe(self):
        script = clusterprofiler.build_clusterprofiler_r_script()

        self.assertIn("library(clusterProfiler)", script)
        self.assertIn("enricher(", script)
        self.assertIn("TERM2GENE = term2gene", script)
        self.assertIn("TERM2NAME = term2name", script)
        self.assertIn("universe = universe_ids", script)

    def test_clusterprofiler_script_does_not_hide_enrichment_errors(self):
        script = clusterprofiler.build_clusterprofiler_r_script()

        self.assertIn("Enrichment failed for ", script)
        self.assertIn("stop(", script)
        self.assertNotIn("message(\"Skipping \", key$motif_cluster", script)

    def test_extract_motif_windows(self):
        sequences = {
            "P1": prepare.SequenceRecord("P1", "GvpA", "MABCDEFGHIK", "p1.fasta", "desc"),
            "P2": prepare.SequenceRecord("P2", "GvpA", "QRSTABCDEYY", "p2.fasta", "desc"),
        }
        occurrence_rows = [
            {
                "motif_uid": "GvpA_M2_motif_001",
                "family": "GvpA",
                "method": "M2",
                "parent_id": "P1",
                "occ_start": "1",
                "occ_end": "6",
            },
            {
                "motif_uid": "GvpA_M2_motif_001",
                "family": "GvpA",
                "method": "M2",
                "parent_id": "P2",
                "occ_start": "4",
                "occ_end": "9",
            },
        ]

        windows = motif_validation.extract_motif_windows(occurrence_rows, sequences)
        self.assertEqual([window.sequence for window in windows["GvpA_M2_motif_001"]], ["ABCDE", "ABCDE"])


if __name__ == "__main__":
    unittest.main()
