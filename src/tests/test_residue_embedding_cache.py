import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings import residue_cache


class TestResidueEmbeddingCache(unittest.TestCase):
    def test_primary_cache_hit_is_read_only_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            supplement = root / "supplement"
            primary.mkdir()
            sequence = "ACDEF"
            name = residue_cache.cache_filename("M2", sequence)
            np.save(primary / name, np.ones((5, 1280), dtype=np.float32))

            result = residue_cache.load_or_compute(
                method="M2",
                sequence=sequence,
                primary_dirs=[primary],
                supplement_dir=supplement,
                compute=lambda: self.fail("cache hit must not compute"),
            )

            self.assertEqual(result.source, "primary")
            self.assertEqual(result.array.shape, (5, 1280))
            self.assertFalse(supplement.exists())

    def test_wrong_embedding_dimension_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = "ACD"
            np.save(
                root / residue_cache.cache_filename("M2", sequence),
                np.ones((3, 7), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "expected shape"):
                residue_cache.load_validated(root, "M2", sequence)

    def test_non_finite_embedding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = "ACD"
            array = np.ones((3, 1024), dtype=np.float32)
            array[1, 2] = np.nan
            np.save(root / residue_cache.cache_filename("M3", sequence), array)

            with self.assertRaisesRegex(ValueError, "invalid values"):
                residue_cache.load_validated(root, "M3", sequence)

    def test_supplement_cache_is_used_after_primary_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            supplement = root / "supplement"
            primary.mkdir()
            supplement.mkdir()
            sequence = "ACD"
            target = supplement / residue_cache.cache_filename("M3", sequence)
            np.save(target, np.ones((3, 1024), dtype=np.float32))

            result = residue_cache.load_or_compute(
                method="M3",
                sequence=sequence,
                primary_dirs=[primary],
                supplement_dir=supplement,
                compute=lambda: self.fail("supplement hit must not compute"),
            )

            self.assertEqual(result.source, "supplement")
            self.assertEqual(result.path, target)

    def test_cache_miss_writes_only_supplement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            supplement = root / "supplement"
            primary.mkdir()
            sequence = "ACD"

            result = residue_cache.load_or_compute(
                method="M3",
                sequence=sequence,
                primary_dirs=[primary],
                supplement_dir=supplement,
                compute=lambda: np.full((3, 1024), 2.0, dtype=np.float32),
            )

            self.assertEqual(result.source, "computed")
            self.assertFalse(
                (primary / residue_cache.cache_filename("M3", sequence)).exists()
            )
            self.assertTrue(
                (supplement / residue_cache.cache_filename("M3", sequence)).exists()
            )

    def test_audit_record_contains_hash_but_not_raw_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            sequence = "ACD"

            residue_cache.load_or_compute(
                method="M3",
                sequence=sequence,
                primary_dirs=[],
                supplement_dir=root / "supplement",
                compute=lambda: np.ones((3, 1024), dtype=np.float32),
                audit_callback=records.append,
            )

            self.assertEqual(len(records), 1)
            self.assertNotIn("sequence", records[0])
            self.assertNotIn(sequence, records[0].values())
            self.assertEqual(records[0]["source"], "computed")


if __name__ == "__main__":
    unittest.main()
