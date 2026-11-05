import csv
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))
from motifae_candidate_filter import process_candidates


def row(feature='f1',protein='P1',score=.6,start=0,length=5,sequence=None):
    return dict(family='GvpA',method='MotifAE',motif_uid=feature,parent_id=protein,
                sequence_id=protein,occ_start=start,occ_end=start+length,
                motif_sequence=sequence or 'A'*length,score=score)


class PreparationTests(unittest.TestCase):
    def test_default_preserves_single_protein_features_and_similar_features(self):
        result=process_candidates([row('a'),row('b')])
        self.assertEqual(result['counts']['retained_features'],2)
        self.assertEqual(result['counts']['below_min_proteins'],0)
        self.assertEqual(result['counts']['redundant_features'],0)

    def test_inventory_does_not_apply_preprint_mean_threshold_to_published_input(self):
        result=process_candidates([row(score=.1)])
        self.assertEqual(result['counts']['retained_features'],1)

    def test_gibbs_preparation_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_gibbs_inputs'),
                             'published feature-wise GibbsCluster preparation is missing')

    def test_gibbs_top1000_is_per_feature_and_keeps_repeated_sequences(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_gibbs_inputs'))
        module=importlib.import_module('motifae_gibbs_inputs')
        rows=[row('a',f'P{i:04}',score=.3+i/10000) for i in range(1005)]
        rows += [row('b','P1')]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            manifest=module.prepare_inputs(rows,out,source_protocol='legacy_preprint_v1')
            self.assertEqual(manifest['counts']['features'],2)
            self.assertEqual(manifest['counts']['selected_peptides'],1001)
            self.assertEqual(manifest['status'],'awaiting_gibbscluster_2_0')
            self.assertIsNone(manifest['completed_motif_count'])
            with (out/'feature_index.csv').open() as handle:index=list(csv.DictReader(handle))
            first=next(r for r in index if r['motif_uid']=='a')
            peptides=(out/first['peptide_file']).read_text().splitlines()
            self.assertEqual(len(peptides),1000)
            self.assertEqual(len(set(peptides)),1)
            with (out/first['mapping_file']).open() as handle:mapping=list(csv.DictReader(handle))
            self.assertEqual(mapping[0]['parent_id'],'P1004')
            self.assertEqual(mapping[-1]['parent_id'],'P0005')

    def test_gibbs_core6_excludes_length5_without_dropping_feature_silently(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_gibbs_inputs'))
        module=importlib.import_module('motifae_gibbs_inputs')
        with tempfile.TemporaryDirectory() as tmp:
            result=module.prepare_inputs([row('a'),row('b',length=6)],Path(tmp),
                                         source_protocol='legacy_preprint_v1',core_length=6)
            self.assertEqual(result['counts']['features'],2)
            self.assertEqual(result['counts']['features_with_input'],1)
            self.assertEqual(result['counts']['too_short_for_core'],1)


    def test_preparation_overwrite_removes_only_stale_managed_inputs(self):
        module=importlib.import_module('motifae_gibbs_inputs')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            module.prepare_inputs([row('a'),row('b')],out,source_protocol='legacy_preprint_v1')
            (out/'notes.txt').write_text('user notes')
            module.prepare_inputs([row('a')],out,source_protocol='legacy_preprint_v1',overwrite=True)
            self.assertEqual(len(list((out/'peptides').glob('*.txt'))),1)
            self.assertEqual(len(list((out/'mappings').glob('*.csv'))),1)
            self.assertEqual((out/'notes.txt').read_text(),'user notes')

    def test_preparation_ready_to_blocked_leaves_no_old_peptide_file(self):
        module=importlib.import_module('motifae_gibbs_inputs')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            module.prepare_inputs([row('a')],out,source_protocol='legacy_preprint_v1')
            manifest=module.prepare_inputs([row('a',sequence='AAXAA')],out,
                                           source_protocol='legacy_preprint_v1',overwrite=True)
            self.assertEqual(manifest['counts']['features_with_input'],0)
            self.assertEqual(list((out/'peptides').glob('*.txt')),[])


if __name__=='__main__':
    unittest.main()
