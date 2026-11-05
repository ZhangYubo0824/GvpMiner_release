"""Regression checks for Exp5 protein ID mapping; no input cleaning is performed."""
from contextlib import redirect_stdout
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(os.environ.get(
    "EXP5_MAPPING_PACKAGE_ROOT", str(Path(__file__).resolve().parents[1])
))
sys.path.insert(0, str(PACKAGE_ROOT))
from package import cli, motif_sets, prepare


class TestExp5IdMappingRegression(unittest.TestCase):
    def record(self, protein_id, family="GvpA"):
        return prepare.SequenceRecord(protein_id, family, "ACDEFG", "original.fasta")

    def occurrence(self, protein_id, family="GvpA"):
        return {"family": family, "method": "M2", "motif_uid": family + "_M2_001", "parent_id": protein_id}

    def test_alias_uses_existing_background_id_and_deduplicates(self):
        indexed = "sp|A0A1Q5LR04.1|GVPA_STRX0"
        records = [self.record(indexed), self.record("BACKGROUND_ONLY")]
        occurrences = [self.occurrence("A0A1Q5LR04.1"), self.occurrence(indexed)]
        sets = motif_sets.build_motif_sets(occurrences, records)
        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0].foreground_ids, (indexed,))
        self.assertEqual(sets[0].background_ids, ("BACKGROUND_ONLY",))
        self.assertEqual(set(sets[0].universe_ids), {indexed, "BACKGROUND_ONLY"})

    def test_unique_accession_mapping_is_scoped_to_family(self):
        a, c = "sp|P1|GVPA", "tr|P1|GVPC"
        sets = motif_sets.build_motif_sets(
            [self.occurrence("P1")], [self.record(a), self.record(c, "GvpC")]
        )
        self.assertEqual(sets[0].foreground_ids, (a,))
        self.assertNotIn(c, sets[0].universe_ids)

    def test_id_only_present_in_another_family_is_excluded(self):
        sets = motif_sets.build_motif_sets(
            [self.occurrence("P1")], [self.record("A_OTHER"), self.record("sp|P1|GVPC", "GvpC")]
        )
        self.assertEqual(sets, [])

    def test_ambiguous_accession_is_not_assigned_arbitrarily(self):
        records = [self.record("sp|P1|FIRST"), self.record("tr|P1|SECOND")]
        self.assertEqual(motif_sets.build_motif_sets([self.occurrence("P1")], records), [])
        exact = motif_sets.build_motif_sets([self.occurrence("sp|P1|FIRST")], records)
        self.assertEqual(exact[0].foreground_ids, ("sp|P1|FIRST",))

    def test_missing_family_does_not_invent_a_universe(self):
        sets = motif_sets.build_motif_sets(
            [self.occurrence("MISSING", "GvpJ")], [self.record("A_ONLY")]
        )
        self.assertEqual(sets, [])

    def test_cli_reports_unique_excluded_proteins_and_keeps_original_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "sequence_index.csv"
            indexed = "sp|A0A1Q5LR04.1|GVPA_STRX0"
            prepare.write_sequence_index([self.record(indexed), self.record("BACKGROUND_ONLY")], index)
            family = root / "exp3/GvpA"
            family.mkdir(parents=True)
            source = family / "motif_occurrences_M2.csv"
            occurrences = [self.occurrence("A0A1Q5LR04.1")]
            occurrences += [self.occurrence("MISSING_" + str(i)) for i in range(3)]
            occurrences.append(self.occurrence("MISSING_0"))
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(occurrences[0]))
                writer.writeheader()
                writer.writerows(occurrences)
            output = root / "new_output"
            capture = io.StringIO()
            before = index.read_bytes()
            with redirect_stdout(capture):
                cli.main(["make-motif-sets", "--exp3-dir", str(root / "exp3"),
                          "--sequence-index", str(index), "--methods", "M2", "--output-dir", str(output)])
            self.assertIn("Excluded 3 family/method/protein IDs", capture.getvalue())
            with (output / "motif_sets_long.csv").open(encoding="utf-8-sig", newline="") as handle:
                result = list(csv.DictReader(handle))
            foreground = {r["protein_id"] for r in result if r["set_type"] == "foreground"}
            universe = {r["protein_id"] for r in result if r["set_type"] == "universe"}
            self.assertEqual(foreground, {indexed})
            self.assertEqual(universe, {indexed, "BACKGROUND_ONLY"})
            self.assertLessEqual(foreground, universe)
            self.assertEqual(index.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
