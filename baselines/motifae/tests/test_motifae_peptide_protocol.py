"""Preprint-v1 segmentation boundaries retained across the adapter removal."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/baseline_adapters'))
from motifae_peptide_protocol import _paper_peptide_points
from motifae_candidate_filter import process_candidates


class PeptideProtocolTests(unittest.TestCase):
    def test_gap_bridging_and_strict_mean_boundary(self):
        self.assertEqual(_paper_peptide_points([(0,1,.7),(3,1,.7),(4,1,.7)])[0][:3], (1,0,5))
        self.assertEqual(_paper_peptide_points([(0,1,.2),(3,1,.2),(4,1,.2)]), [])
        self.assertEqual(_paper_peptide_points([(i,1,.6) for i in range(31)]), [])

    def test_separated_segments_retain_original_residue_coordinates(self):
        result = _paper_peptide_points([(i,1,.5) for i in [*range(5),*range(9,14)]])
        self.assertEqual(result, [(1,0,5,.5),(1,9,14,.5)])

    def test_feature_total_activation_boundary_is_strict(self):
        self.assertEqual(_paper_peptide_points([(0,1,.5),(4,1,.5)]), [])

    def test_near_threshold_mean_survives_candidate_export_accounting(self):
        feature,start,end,score = _paper_peptide_points([(i,1,float(np.nextafter(.2,1.))) for i in range(5)])[0]
        row = dict(family='GvpA',method='MotifAE',motif_uid=str(feature),parent_id='P1',sequence_id='P1',
                   occ_start=start,occ_end=end,motif_sequence='ACDEF',score=score)
        self.assertGreater(score,.2)
        result = process_candidates([row])
        self.assertGreater(float(result['occurrences'][0]['score']),.2)


if __name__ == '__main__':
    unittest.main()
