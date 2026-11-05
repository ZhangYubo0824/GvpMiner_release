import unittest
import csv
from pathlib import Path
import sys
import tempfile
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp3_domain_discovery as exp3


class TestExp3CoreMotifDiscovery(unittest.TestCase):
    def test_process_family_passes_explicit_window_size_and_step(self):
        records = [
            {
                "sequence": "ABCDEFGHIJKLMNOPQRST",
                "protein_id": "p1",
                "family": "GvpA",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            exp3, "EXP3_MIN_WINDOWS", 3
        ), mock.patch.object(
            exp3,
            "embed_windows",
            return_value=np.ones((4, 2), dtype=np.float32),
        ) as embed, mock.patch.object(
            exp3,
            "run_clustering",
            return_value=(np.full(4, -1), {"n_motifs": 0}),
        ):
            output_dir = Path(tmp)
            metrics = exp3.process_family(
                "GvpA",
                records,
                ["M2"],
                output_dir,
                window_size=5,
                window_step=5,
                residue_cache_dirs=[],
                residue_cache_fallback_dir=output_dir / "cache",
                dataset_fingerprint="fixture",
            )

            with (output_dir / "GvpA" / "windows_metadata.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [(int(row["start"]), int(row["end"])) for row in rows],
                [(0, 5), (5, 10), (10, 15), (15, 20)],
            )
            self.assertEqual(embed.call_args.kwargs["window_size"], 5)
            self.assertEqual(embed.call_args.kwargs["window_step"], 5)
            self.assertEqual(metrics[0]["window_size"], 5)
            self.assertEqual(metrics[0]["window_step"], 5)
            self.assertEqual(metrics[0]["n_windows_raw"], 4)
            self.assertEqual(metrics[0]["n_windows_deduplicated"], 4)

    def test_window_cache_manifest_rejects_different_window_size(self):
        windows = [
            {
                "window_id": "p1_w0_30",
                "window_seq_full": "A" * 40,
                "start": 0,
                "end": 30,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            exp3.save_window_embedding_cache(
                cache_dir,
                "M2",
                np.ones((1, 1280), dtype=np.float32),
                windows,
                window_size=30,
                window_step=5,
                dataset_fingerprint="fixture",
            )

            with self.assertRaisesRegex(
                ValueError,
                "window cache manifest mismatch: window_size",
            ):
                exp3.load_window_embedding_cache(
                    cache_dir,
                    "M2",
                    windows,
                    window_size=25,
                    window_step=5,
                    dataset_fingerprint="fixture",
                )

    def test_residue_cache_hit_does_not_load_embedding_model(self):
        sequence = "ACDEF"
        windows = [{
            "window_id": "p1_w0_5",
            "window_seq_full": sequence,
            "start": 0,
            "end": 5,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            primary.mkdir()
            np.save(
                primary / exp3.residue_cache.cache_filename("M2", sequence),
                np.ones((5, 1280), dtype=np.float32),
            )
            with mock.patch.object(
                exp3,
                "get_window_embedder",
                side_effect=AssertionError("model must stay unloaded on cache hit"),
            ):
                result = exp3.embed_windows(
                    "M2",
                    windows,
                    root / "window_cache",
                    window_size=5,
                    window_step=5,
                    dataset_fingerprint="fixture",
                    residue_cache_dirs=[primary],
                    residue_cache_fallback_dir=root / "supplement",
                )

            self.assertEqual(result.shape, (1, 1280))

    def test_force_recompute_bypasses_window_embedding_cache(self):
        windows = [{
            "window_id": "p1_w0_5",
            "window_seq_full": "ACDEF",
            "start": 0,
            "end": 5,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp3.save_window_embedding_cache(
                root,
                "M2",
                np.zeros((1, 1280), dtype=np.float32),
                windows,
                window_size=5,
                window_step=5,
                dataset_fingerprint="fixture",
            )
            primary = root / "primary"
            primary.mkdir()
            np.save(
                primary / exp3.residue_cache.cache_filename("M2", "ACDEF"),
                np.ones((5, 1280), dtype=np.float32),
            )

            result = exp3.embed_windows(
                "M2",
                windows,
                root,
                window_size=5,
                window_step=5,
                dataset_fingerprint="fixture",
                residue_cache_dirs=[primary],
                residue_cache_fallback_dir=root / "supplement",
                force_recompute=True,
            )

            self.assertTrue(np.all(result == 1.0))

    def test_zero_motifs_still_writes_empty_contract_files(self):
        records = [{"sequence": "A" * 20, "protein_id": "p1", "family": "GvpA"}]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            exp3, "EXP3_MIN_WINDOWS", 3
        ), mock.patch.object(
            exp3, "embed_windows", return_value=np.ones((4, 2), dtype=np.float32)
        ), mock.patch.object(
            exp3, "run_clustering", return_value=(np.full(1, -1), {"n_motifs": 0})
        ):
            exp3.process_family(
                "GvpA", records, ["M2"], Path(tmp), window_size=5, window_step=5,
                residue_cache_dirs=[], residue_cache_fallback_dir=Path(tmp) / "cache",
                dataset_fingerprint="fixture",
            )

            family = Path(tmp) / "GvpA"
            with (family / "motif_catalog_M2.csv").open(encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with (family / "motif_occurrences_M2.csv").open(encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_summarize_motif_clusters_emits_no_core_derived_fields(self):
        windows = [
            {
                "window_id": "seq1_w10_40",
                "parent_id": "seq1",
                "family": "famA",
                "start": 10,
                "end": 40,
                "window_seq": "A" * 30,
            },
            {
                "window_id": "seq2_w15_45",
                "parent_id": "seq2",
                "family": "famA",
                "start": 15,
                "end": 45,
                "window_seq": "C" * 30,
            },
        ]
        labels = np.array([7, 7])

        motifs = exp3.summarize_motif_clusters(windows, labels, "M2")

        self.assertEqual(len(motifs), 1)
        motif = motifs[0]
        self.assertEqual(motif["cluster_label"], 7)
        self.assertEqual(motif["n_windows"], 2)
        self.assertEqual(motif["n_unique_sequences"], 2)
        forbidden = {
            "core_start", "core_end", "core_length", "coverage_peak", "coverage_mass",
            "motif_consensus", "conservation_score", "position_frequency_matrix",
            "support_fragments",
        }
        self.assertTrue(forbidden.isdisjoint(motif))

    def test_finalize_motif_catalog_assigns_stable_motif_uids(self):
        motifs = [
            {
                "cluster_label": 8,
                "n_unique_sequences": 4,
                "n_windows": 10,
            },
            {
                "cluster_label": 3,
                "n_unique_sequences": 7,
                "n_windows": 8,
            },
            {
                "cluster_label": 5,
                "n_unique_sequences": 7,
                "n_windows": 9,
            },
        ]

        catalog = exp3.finalize_motif_catalog(family_name="famA", method_name="M2", motifs=motifs)

        self.assertEqual([row["motif_uid"] for row in catalog], [
            "famA_M2_motif_001",
            "famA_M2_motif_002",
            "famA_M2_motif_003",
        ])
        self.assertEqual([row["cluster_label"] for row in catalog], [5, 3, 8])
        self.assertEqual([row["rank_within_method"] for row in catalog], [1, 2, 3])

    def test_exact_window_deduplication_keeps_distinct_overlapping_windows(self):
        windows = [
            {"parent_id": "seq1", "window_seq": "AAAA", "start": 0, "end": 30},
            {"parent_id": "seq1", "window_seq": "AAAB", "start": 5, "end": 35},
            {"parent_id": "seq1", "window_seq": "AABB", "start": 10, "end": 40},
        ]

        keep_idx = exp3.deduplicate_exact_windows_within_parent(windows)

        self.assertEqual(keep_idx.tolist(), [0, 1, 2])

    def test_exact_window_deduplication_removes_only_same_parent_exact_duplicates(self):
        windows = [
            {"parent_id": "seq1", "window_seq": "AAAA", "start": 0, "end": 4},
            {"parent_id": "seq1", "window_seq": "AAAA", "start": 5, "end": 9},
            {"parent_id": "seq2", "window_seq": "AAAA", "start": 0, "end": 4},
        ]

        keep_idx = exp3.deduplicate_exact_windows_within_parent(windows)

        self.assertEqual(keep_idx.tolist(), [0, 2])

    def test_build_sequence_motif_occurrences_keeps_different_motifs_separate_on_same_sequence(self):
        windows = [
            {"window_id": "seq1_w0", "parent_id": "seq1", "family": "famA", "start": 10, "end": 40},
            {"window_id": "seq1_w1", "parent_id": "seq1", "family": "famA", "start": 15, "end": 45},
            {"window_id": "seq1_w2", "parent_id": "seq1", "family": "famA", "start": 42, "end": 72},
            {"window_id": "seq1_w3", "parent_id": "seq1", "family": "famA", "start": 47, "end": 77},
        ]
        labels = np.array([0, 0, 1, 1])
        motifs = [
            {"cluster_label": 0, "motif_uid": "famA_M2_motif_001"},
            {"cluster_label": 1, "motif_uid": "famA_M2_motif_002"},
        ]

        occurrences = exp3.build_sequence_motif_occurrences(windows, labels, motifs, method_name="M2")

        self.assertEqual(len(occurrences), 2)
        self.assertEqual([row["motif_uid"] for row in occurrences], ["famA_M2_motif_001", "famA_M2_motif_002"])
        self.assertEqual([(row["occ_start"], row["occ_end"]) for row in occurrences], [(10, 45), (42, 77)])
        forbidden = {"occ_core_start", "occ_core_end", "occ_core_length"}
        self.assertTrue(all(forbidden.isdisjoint(row) for row in occurrences))

    def test_build_sequence_motif_occurrences_splits_disjoint_occurrences_of_same_motif(self):
        windows = [
            {"window_id": "seq1_w0", "parent_id": "seq1", "family": "famA", "start": 10, "end": 40},
            {"window_id": "seq1_w1", "parent_id": "seq1", "family": "famA", "start": 15, "end": 45},
            {"window_id": "seq1_w2", "parent_id": "seq1", "family": "famA", "start": 90, "end": 120},
            {"window_id": "seq1_w3", "parent_id": "seq1", "family": "famA", "start": 95, "end": 125},
        ]
        labels = np.array([0, 0, 0, 0])
        motifs = [{"cluster_label": 0, "motif_uid": "famA_M2_motif_001"}]

        occurrences = exp3.build_sequence_motif_occurrences(windows, labels, motifs, method_name="M2")

        self.assertEqual(len(occurrences), 2)
        self.assertEqual([row["occurrence_id"] for row in occurrences], [
            "famA_M2_motif_001_occ_001",
            "famA_M2_motif_001_occ_002",
        ])
        self.assertEqual([(row["occ_start"], row["occ_end"]) for row in occurrences], [(10, 45), (90, 125)])


if __name__ == "__main__":
    unittest.main()
