import unittest
import numpy as np
from scripts.motif_representation.core import project, profile, consensus


class RepresentationTests(unittest.TestCase):
    def test_projection_preserves_boundary_and_internal_gap(self):
        self.assertEqual(project("A-CD-E", 1, 3, "CD"), (2, 4, "CD"))
        self.assertEqual(project("A-CD-E", 0, 3, "ACD"), (0, 4, "A-CD"))
        with self.assertRaises(ValueError):
            project("A-CD-E", 0, 3, "AAA")
        with self.assertRaises(ValueError):
            project("A-CD-E", -1, 3, "ACD")

    def test_gap_absence_and_ambiguity_are_distinct(self):
        p = profile([("p1", 0, "A-X"), ("p2", 1, "C-")])
        np.testing.assert_allclose(p["coverage"], [.5, .5, 0])
        np.testing.assert_allclose(p["gap"], [0, .5, .5])
        np.testing.assert_allclose(p["outside"], [.5, 0, 0])
        np.testing.assert_allclose(p["ambiguous"], [0, 0, .5])
        self.assertEqual(consensus(p), "AC.")

    def test_each_protein_has_total_weight_one(self):
        p = profile([("p1", 0, "A"), ("p1", 0, "A"), ("p2", 0, "C")])
        self.assertAlmostEqual(p["probabilities"][0, 0], .5)
        self.assertAlmostEqual(p["occurrence_probabilities"][0, 0], 2/3)
        self.assertAlmostEqual(p["tv"][0], 1/6)
        self.assertEqual(consensus(p), "X")

    def test_disconnected_columns_not_joined_as_contiguous(self):
        p = profile([("p1", 0, "AA"), ("p2", 5, "CC")])
        self.assertEqual(consensus(p), "AA...CC")
        self.assertEqual(p["columns"].tolist(), list(range(7)))

    def test_all_ambiguous_no_fabricated_probabilities(self):
        p = profile([("p1", 5, "XX")])
        self.assertTrue(np.isnan(p["probabilities"]).all())
        self.assertEqual(consensus(p), "..")

    def test_frequency_majority_distinct_from_coverage(self):
        p = profile([("p1", 0, "AC"), ("p2", 0, "AC"), ("p3", 1, "D")])
        self.assertEqual(consensus(p), "AC")
        self.assertAlmostEqual(p["coverage"][0], 2/3)
        self.assertAlmostEqual(p["probabilities"][1, 1], 2/3)


if __name__ == "__main__":
    unittest.main()
