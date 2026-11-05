"""Portable qualification tests; mocked execution is not Linux sampler evidence."""
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/baseline_adapters'),str(ROOT/'tests')]
import motifae_gibbs_qualification as qualification
import motifae_native_guard as guard
import motifae_gibbs_results as results
import test_motifae_gibbs_results as fixtures
import test_motifae_gibbs_all_trash as zero


class GenericQualificationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(qualification,'qualify_installation'), 'portable qualification API is missing')
        self.qualification=importlib.import_module('motifae_gibbs_qualification')
        self.fixture=fixtures.GibbsResultTests()

    def runner(self,peptides,root,output,**kwargs):
        output=Path(output)
        if set(peptides)=={'AIGSE','SSAPQ'}:
            zero.zero_native(output,tuple(sorted(set(peptides))));count=3
        elif peptides==self.qualification.SINGLETON_PEPTIDES:
            zero.zero_native(output,tuple(peptides));count=0
        else:
            shutil.copytree(self.qualification.FIXTURES/'normal_reference',output)
            count=0
        return dict(status='completed_guarded_sampler',indel_moves_skipped=count,phase_moves_skipped=0)

    def qualify(self,root):
        installed=(root/'installed').resolve();installed.mkdir()
        with mock.patch.object(self.qualification,'verified_installation',return_value=installed), \
             mock.patch.object(guard,'verified_installation',return_value=installed), \
             mock.patch.object(self.qualification,'run_guarded_native',side_effect=self.runner):
            state=qualification.qualify_installation(installed,root/'qualified')
        return installed,state



    def test_qualification_is_independent_of_family_and_has_fixed_real_fixture(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);installed,state=self.qualify(root)
            self.assertEqual(state['status'],'installation_qualification_passed')
            self.assertNotIn('fasta_sha256',state['signature'])
            self.assertNotIn('preparation_sha256',state['signature'])
            self.assertGreater(state['normal_comparison']['retained_unique_sequences'],2)
            self.qualification.verify_qualification(root/'qualified')


    def test_missing_hash_entry_cannot_bypass_artifact_recheck(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.qualify(root)
            path=root/'qualified/qualification_manifest.json';state=json.loads(path.read_text())
            name=next(n for n in state['files_sha256'] if n.endswith('gibbs.1g.ds.out'))
            del state['files_sha256'][name];results.write_json(path,state)
            with self.assertRaisesRegex(ValueError,'inventory|hash'):
                self.qualification.verify_qualification(root/'qualified')

    def test_changed_qualification_input_and_code_signature_are_rejected(self):
        for damage in ('input','version'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as t:
                root=Path(t);self.qualify(root)
                if damage=='input':
                    (root/'qualified/normal_case/input_peptides.txt').write_text('KLMNP\nQRSTV\n')
                else:
                    path=root/'qualified/qualification_manifest.json';state=json.loads(path.read_text())
                    state['signature']['code_sha256']={};results.write_json(path,state)
                with self.assertRaisesRegex(ValueError,'hash|stale|signature'):
                    self.qualification.verify_qualification(root/'qualified')





class NativeComparisonPortabilityTests(unittest.TestCase):
    def native_pair(self,root,params=None):
        fixture=fixtures.GibbsResultTests()
        for name in ('reference','trial'):
            fixture.native(root/name,[('ACDEF','ACDEF',0,-99,-99,-99,-99),
                ('GHIKL','GHIKL',0,-99,-99,-99,-99)],params=params)

    def test_display_background_fields_and_core_order_must_match_exactly(self):
        for damage in ('display','core_order'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as t:
                root=Path(t);self.native_pair(root)
                if damage=='display':
                    path=root/'trial/res/gibbs.1g.ds.out';path.write_text(path.read_text().replace('bgG -1','bgG 0'))
                else:
                    path=root/'trial/cores/gibbs.1g.cores';path.write_text('## Alignment cores for group 1\nGHIKL\nACDEF\n')
                with self.assertRaises(ValueError):
                    guard.compare_native(root/'reference',root/'trial',['ACDEF','GHIKL'])

    def test_only_verified_blosum_installation_root_may_be_normalized(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.native_pair(root,params=dict(l=5,g=1,trash=1,base=1,ilen=1,dlen=1,
                blf='/original/data/blosum%i.freq_rownorm'))
            path=root/'trial/res/gibbs.1g.out';path.write_text(path.read_text().replace('/original/','/new/'))
            with self.assertRaisesRegex(ValueError,'parameter'):
                guard.compare_native(root/'reference',root/'trial',['ACDEF','GHIKL'])
            with mock.patch.object(guard,'verified_installation',return_value=root/'new') as verify:
                guard.compare_native(root/'reference',root/'trial',['ACDEF','GHIKL'],blosum_root=root/'new')
            verify.assert_called_once()
            with self.assertRaisesRegex(ValueError,'installation'):
                guard.compare_native(root/'reference',root/'trial',['ACDEF','GHIKL'],blosum_root=root/'missing')

    def test_kld_is_exact_not_tolerated_to_six_decimal_places(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture=fixtures.GibbsResultTests()
            for name in ('reference','trial'):
                fixture.native(root/name,[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            path=root/'trial/res/gibbs.1g.out';path.write_text(path.read_text().replace('12.0','12.0000001'))
            with self.assertRaisesRegex(ValueError,'KLD'):
                guard.compare_native(root/'reference',root/'trial',['ACDEF'])


if __name__=='__main__':unittest.main()
