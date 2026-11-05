import csv
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))
from motifae_gibbs_inputs import prepare_inputs


class GibbsResultTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_gibbs_results'))
        self.api=importlib.import_module('motifae_gibbs_results')

    def native(self,folder,records,trash=0,params=None):
        (folder/'res').mkdir(parents=True,exist_ok=True)
        (folder/'cores').mkdir(exist_ok=True)
        settings=params or {'l':5,'g':1,'trash':1,'base':1,'ilen':1,'dlen':1}
        header=''.join(f'# -{k} {v}\n' for k,v in settings.items())
        lines=[f'G 0 {i} {seq} {core} o {off} ip {ip} il {il} dp {dp} dl {dl} Peplist sS 4.0 bgG -1 bgS 1.0 cS 3.2'
               for i,(seq,core,off,ip,il,dp,dl) in enumerate(records)]
        summary=f'\n# Trash cluster: removed {trash} outliers\n### Final KLD sum after 10 iterations: 12.0\n'
        (folder/'res/gibbs.1g.out').write_text(header+'\n'.join(lines)+summary)
        (folder/'res/gibbs.1g.ds.out').write_text('G Gn Num Sequence Core o of ip IP il IL dp DP dl DL Annotation sS Self bgG bgG bgS bgScor cS cScore\n'+'\n'.join(lines)+'\n')
        (folder/'cores/gibbs.1g.cores').write_text('## Alignment cores for group 1\n'+''.join(r[1]+'\n' for r in records))

    def prepare(self,root,extra=False):
        rows=[dict(family='GvpA',method='MotifAE',motif_uid='GvpA_MotifAE_feature_42',
                   parent_id=p,sequence_id=p,occ_start=2,occ_end=7,motif_sequence='ACDEF',score=.5)
              for p in ['P1','P2']]
        if extra:
            rows.append(dict(rows[0],motif_uid='GvpA_MotifAE_feature_43',parent_id='P3',sequence_id='P3',motif_sequence='GHIKL'))
        prepare_inputs(rows,root/'prep',source_protocol='legacy_preprint_v1')
        (root/'input.fasta').write_text('>P1\nMMACDEFGG\n>P2\nMMACDEFGG\n>P3\nMMGHIKLGG\n')
        return root/'prep',root/'input.fasta'

    def test_contiguous_core_positions(self):
        self.assertEqual(self.api.core_positions('MMACDEFGG','ACDEF',2),[2,3,4,5,6])

    def test_actual_pilot_deletion_core_is_not_contiguous(self):
        self.assertEqual(self.api.core_positions('AGAKRAASSVGGKVKG','GKRAA',1,dp=1,dl=1),[1,3,4,5,6])

    def test_official_example_insertion_maps_gap_to_none(self):
        self.assertEqual(self.api.core_positions('FTDKAASY','FTDKA-ASY',0,ip=5,il=1),[0,1,2,3,4,None,5,6,7])

    def test_invalid_core_or_indel_does_not_guess_a_position(self):
        with self.assertRaises(ValueError):self.api.core_positions('ACDEF','AAAAA',0)
        with self.assertRaises(ValueError):self.api.core_positions('ACDEF','ACDEF',0,ip=2,il=1)
        with self.assertRaises(ValueError):self.api.core_positions('ACDEF','ACDEF',0,ip=2,il=1,dp=3,dl=1)

    def test_inconsistent_indel_fields_are_rejected(self):
        with self.assertRaises(ValueError):self.api.core_positions('ACDEF','ACDEF',0,ip=2,il=-99)
        with self.assertRaises(ValueError):self.api.core_positions('ACDEF','ACDEF',0,dp=2,dl=-99)

    def test_parse_checks_trash_accounting_and_input_identity(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.native(root,[('ACDEF','ACDEF',0,-99,-99,-99,-99)],trash=1)
            result=self.api.parse_native_run(root,['ACDEF','ACDEF','GHIKL'])
            self.assertEqual(result['trash_sequences'],['GHIKL'])
            self.assertEqual(len(result['records']),1)

    def test_missing_retained_rows_not_silently_called_trash(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.native(root,[('ACDEF','ACDEF',0,-99,-99,-99,-99)],trash=0)
            with self.assertRaises(ValueError):self.api.parse_native_run(root,['ACDEF','GHIKL'])

    def test_published_algorithm_parameters_are_required(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.native(root,[('ACDEF','ACDEF',0,-99,-99,-99,-99)],params={'l':5,'g':1,'trash':0,'base':1,'ilen':1,'dlen':1})
            with self.assertRaises(ValueError):self.api.parse_native_run(root,['ACDEF'])

    def test_ds_cores_must_agree(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.native(root,[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            (root/'cores/gibbs.1g.cores').write_text('AAAAA\n')
            with self.assertRaises(ValueError):self.api.parse_native_run(root,['ACDEF'])

    def test_import_restores_duplicate_instances_and_full_protein_positions(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.prepare(root)
            self.native(root/'native',[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            report=self.api.import_alignment(prep,'feature_000001',root/'native',fasta,root/'parsed')
            self.assertEqual(report['retained_unique_sequences'],1)
            self.assertEqual(report['mapped_instances'],2)
            with (root/'parsed/aligned_occurrences.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual({r['parent_id'] for r in rows},{'P1','P2'})
            self.assertTrue(all((int(r['occ_start']),int(r['occ_end']))==(2,7) for r in rows))
            self.assertTrue(all(json.loads(r['core_residue_positions_0based'])==[2,3,4,5,6] for r in rows))
            self.assertFalse((root/'parsed/motif_occurrences_MotifAE.csv').exists())
            self.assertEqual(report['localization_scope'],'selected_top1000_alignment_instances')
            with (root/'parsed/frequency_matrix.csv').open(newline='') as handle:matrix=list(csv.DictReader(handle))
            self.assertEqual(len(matrix),5)
            self.assertEqual(float(matrix[0]['A']),1.)
            self.assertEqual(int(matrix[0]['non_gap_sequences']),1)

    def test_changed_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.prepare(root)
            (prep/'peptides/feature_000001.txt').write_text('AAAAA\n')
            self.native(root/'native',[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            with self.assertRaises(ValueError):self.api.import_alignment(prep,'feature_000001',root/'native',fasta,root/'parsed')

    def test_wrong_full_protein_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.prepare(root)
            fasta.write_text('>P1\nMMAAAAAAA\n>P2\nMMACDEFGG\n')
            self.native(root/'native',[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            with self.assertRaises(ValueError):self.api.import_alignment(prep,'feature_000001',root/'native',fasta,root/'parsed')

    def test_all_trash_is_valid_but_does_not_build_a_pattern(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.prepare(root)
            self.native(root/'native',[],trash=1)
            raw=root/'native/res/gibbs.1g.out'
            raw.write_text(raw.read_text().replace('### Final KLD',
                '#Trash1 ACDEF Peplist\n# Final number of non-empty clusters: 0\n### Final KLD'))
            report=self.api.import_alignment(prep,'feature_000001',root/'native',fasta,root/'parsed')
            self.assertEqual(report['status'],'all_sequences_trash')
            self.assertEqual(report['mapped_instances'],0)
            self.assertFalse(report['pattern_built'])

    def test_empty_frequency_column_is_not_claimed_as_built_pattern(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.prepare(root)
            self.native(root/'native',[('ACDEF','ACDE-',0,4,1,-99,-99)])
            report=self.api.import_alignment(prep,'feature_000001',root/'native',fasta,root/'parsed')
            self.assertFalse(report['pattern_built'])
            self.assertEqual(report['status'],'alignment_has_empty_frequency_columns')



















if __name__=='__main__':unittest.main()
