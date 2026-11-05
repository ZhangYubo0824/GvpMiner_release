import csv
import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/baseline_adapters'))


def occurrence(feature, protein, start=0, end=10, score=0.6, family='GvpA'):
    return dict(family=family, method='MotifAE', motif_uid=feature,
                sequence_id=protein, parent_id=protein, occ_start=start,
                occ_end=end, motif_sequence='A' * (end-start), score=score)


class CandidateFilterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_candidate_filter'),
                             'candidate support and redundancy processor is missing')
        self.module = importlib.import_module('motifae_candidate_filter')

    def process(self, rows):
        return self.module.process_candidates(rows)

    def test_canonical_aliases_do_not_count_as_independent_proteins(self):
        rows = [occurrence('f1', f'P1|alias{i}') for i in range(5)]
        result = self.process(rows)
        self.assertEqual(result['decisions'][0]['n_proteins'], 1)
        self.assertEqual(result['counts']['duplicate_occurrences_removed'], 4)

    def test_uniprot_headers_remain_different_proteins(self):
        rows = [occurrence('a', f'sp|P{i}|NAME{i}') for i in range(5)]
        result = self.process(rows)
        self.assertEqual(result['counts']['retained_features'], 1)
        self.assertEqual(result['catalog'][0]['n_proteins'], 5)
        self.assertEqual({r['parent_id'] for r in result['occurrences']}, {f'P{i}' for i in range(5)})

    def test_uniprot_sp_tr_aliases_of_same_accession_count_once(self):
        result = self.process([occurrence('a','sp|P1|NAME'), occurrence('a','tr|P1|OTHER')])
        self.assertEqual(result['catalog'][0]['n_proteins'],1)
        self.assertEqual(result['counts']['duplicate_occurrences_removed'],1)

    def test_duplicates_removed_before_counts_and_keep_max_activation(self):
        rows = [occurrence('f1', f'P{i}') for i in range(5)]
        result = self.process(rows + [occurrence('f1', 'P0', score=0.8)])
        self.assertEqual(len(result['occurrences']), 5)
        self.assertEqual(result['counts']['duplicate_occurrences_removed'], 1)
        self.assertEqual(float(result['occurrences'][0]['score']), 0.8)


    def test_same_proteins_different_regions_remain_distinct(self):
        rows = [occurrence(f, f'P{i}', s, s+10) for f, s in [('f1', 0), ('f2', 20)] for i in range(5)]
        self.assertEqual(self.process(rows)['counts']['retained_features'], 2)


    def test_extra_unmatched_occurrences_prevent_collapse(self):
        rows = [occurrence(f, f'P{i}') for f in ['a', 'b'] for i in range(5)]
        rows += [occurrence('b', f'P{i}', 20, 30) for i in range(5)]
        self.assertEqual(self.process(rows)['counts']['retained_features'], 2)

    def test_rank_uses_proteins_and_not_number_of_segments(self):
        rows = [occurrence('few', 'P1', i*12, i*12+10, .9) for i in range(20)]
        rows += [occurrence('many', f'P{i}', 0, 10, .4) for i in range(5)]
        result = self.process(rows)
        self.assertEqual(result['catalog'][0]['motif_uid'], 'many')

    def test_no_implicit_feature_cap(self):
        rows = [occurrence(f'f{i:03}', 'P1', i*12, i*12+10) for i in range(205)]
        self.assertEqual(self.process(rows)['counts']['retained_features'], 205)

    def test_no_merge_between_families(self):
        rows = [occurrence('f1', f'P{i}', family=f) for f in ['GvpA', 'GvpC'] for i in range(5)]
        self.assertEqual(self.process(rows)['counts']['retained_features'], 2)

    def test_reject_invalid_scores_sequences_and_coordinates(self):
        for change in [dict(score=math.nan), dict(score=math.inf), dict(score=0),
                       dict(occ_start=-1), dict(occ_end=10.5),
                       dict(motif_sequence='feature_1'), dict(motif_sequence='AAAA')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.process([{**occurrence('f1', 'P1'), **change}])


    def test_conflicting_sequence_for_same_parent_interval_rejected(self):
        rows = [occurrence('a', 'P1'), {**occurrence('b', 'P1'), 'motif_sequence': 'C'*10}]
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.process(rows)

    def test_counts_reconcile_and_order_is_deterministic(self):
        rows = [occurrence(f, f'P{i}', score=s) for f, s in [('a', .9), ('b', .8)] for i in range(5)] + [occurrence('low', 'P1')]
        a = self.process(rows)
        self.assertEqual(a, self.process(list(reversed(rows))))
        c = a['counts']
        self.assertEqual(c['raw_features'], c['below_min_proteins']+c['redundant_features']+c['capped_features']+c['retained_features'])

    def test_empty_catalogue_exports_headers_and_zero_count_manifest(self):
        result = self.process([])
        with tempfile.TemporaryDirectory() as folder:
            self.module.export_results(Path(folder), 'MotifAE', result, {'mode': 'test'})
            with (Path(folder)/'motif_catalog_MotifAE.csv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            manifest = json.loads((Path(folder)/'motifae_manifest.json').read_text())
            self.assertEqual(manifest['counts']['retained_features'], 0)


if __name__ == '__main__':
    unittest.main()
