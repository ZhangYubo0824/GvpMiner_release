"""Regression for the user-reported completed sampler / Perl plot failure."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))
sys.path.insert(0,str(ROOT/'tests'))
fixtures=importlib.import_module('test_motifae_gibbs_results')
api=importlib.import_module('motifae_gibbs_results')


def zero_native(folder,sequences=('ACDEF',),kld='-nan'):
    (folder/'res').mkdir(parents=True,exist_ok=True)
    header=''.join(f'# -{k} {v}\n' for k,v in dict(l=5,g=1,trash=1,base=1,ilen=1,dlen=1).items())
    text=header+'# G 0 1\n'
    text+=''.join(f'# Doing temprature step {i} {1.5*(19-i)/19:.6f}\n' for i in range(20))
    text+=f'## Trash cluster ({len(sequences)} outliers)\n'
    text+=''.join(f'#Trash{i} {seq} Peplist\n' for i,seq in enumerate(sequences,1))
    text+=f'# Trash cluster: removed {len(sequences)} outliers\n#\n# Final number of non-empty clusters: 0\n\n'
    text+=f'### Final KLD sum after {200*len(sequences)} iterations: {kld}\n'
    (folder/'res/gibbs.1g.out').write_text(text)
    (folder/'res/gibbs.1g.ds.out').write_text('G Gn Num Sequence Core o of ip IP il IL dp DP dl DL Annotation sS Self bgG bgG bgS bgScor cS cScore\n')


class AllTrashTests(unittest.TestCase):
    def setUp(self):self.fixture=fixtures.GibbsResultTests()

    def test_reported_native_zero_retained_nan_and_missing_cores_is_assessed(self):
        for peptide in ('QPSHRLPSEDE','ELIAIEA'):
            with self.subTest(peptide=peptide),tempfile.TemporaryDirectory() as t:
                folder=Path(t);zero_native(folder,(peptide,))
                raw=(folder/'res/gibbs.1g.out').read_bytes()
                result=api.parse_native_run(folder,[peptide])
                self.assertEqual(result['trash_sequences'],[peptide])
                self.assertEqual(result['records'],[])
                self.assertIsNone(result['kld'])
                self.assertEqual(result['kld_raw'],'-nan')
                self.assertEqual(result['completion_status'],'all_sequences_trash')
                self.assertEqual((folder/'res/gibbs.1g.out').read_bytes(),raw)
                self.assertFalse((folder/'cores/gibbs.1g.cores').exists())

    def test_zero_retained_import_accounts_for_all_duplicate_instances_without_pattern(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.fixture.prepare(root);zero_native(root/'native')
            report=api.import_alignment(prep,'feature_000001',root/'native',fasta,root/'parsed')
            self.assertEqual((report['mapped_instances'],report['trash_instances'],report['unassessed_instances']),(0,2,0))
            self.assertFalse(report['pattern_built']);self.assertIsNone(report['kld'])
            encoded=(root/'parsed/parse_manifest.json').read_text()
            self.assertNotIn('NaN',encoded);self.assertIn('"kld": null',encoded)

    def test_incomplete_or_inconsistent_all_trash_is_rejected(self):
        mutations={
            'no_final_marker':lambda s:s[:s.index('### Final KLD')],
            'no_cluster_marker':lambda s:s.replace('# Final number of non-empty clusters: 0',''),
            'wrong_cluster_count':lambda s:s.replace('non-empty clusters: 0','non-empty clusters: 1'),
            'foreign_sequence':lambda s:s.replace('#Trash1 ACDEF','#Trash1 GHIKL'),
            'missing_trash_identity':lambda s:s.replace('#Trash1 ACDEF Peplist\n',''),
            'duplicate_trash_identity':lambda s:s.replace('#Trash1 ACDEF Peplist','#Trash1 ACDEF Peplist\n#Trash2 ACDEF Peplist'),
            'wrong_trash_count':lambda s:s.replace('removed 1 outliers','removed 2 outliers'),
            'infinite_kld':lambda s:s.replace('iterations: -nan','iterations: inf'),
            'changed_parameters':lambda s:s.replace('# -trash 1','# -trash 0'),
            'nonterminal_final_marker':lambda s:s+'# another incomplete run starts here\n',
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as t:
                folder=Path(t);zero_native(folder);p=folder/'res/gibbs.1g.out';p.write_text(mutate(p.read_text()))
                with self.assertRaises((ValueError,FileNotFoundError)):api.parse_native_run(folder,['ACDEF'])

    def test_missing_optional_display_for_verified_all_trash_is_allowed(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);zero_native(folder);(folder/'res/gibbs.1g.ds.out').unlink()
            result=api.parse_native_run(folder,['ACDEF'])
            self.assertEqual(result['trash_sequences'],['ACDEF'])

    def test_contradictory_display_or_core_cannot_be_ignored(self):
        for name,text in [('res/gibbs.1g.ds.out','G 0 0 ACDEF ACDEF o 0 ip -99 il -99 dp -99 dl -99 Peplist sS 4.0 bgG -1 bgS 1.0 cS 3.2\n'),
                          ('cores/gibbs.1g.cores','ACDEF\n')]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as t:
                folder=Path(t);zero_native(folder);p=folder/name;p.parent.mkdir(exist_ok=True);p.write_text(text)
                with self.assertRaises(ValueError):api.parse_native_run(folder,['ACDEF'])

    def test_nonempty_alignment_still_requires_core_file_and_finite_kld(self):
        for damage in ('missing_core','nan'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as t:
                folder=Path(t);self.fixture.native(folder,[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
                if damage=='missing_core':(folder/'cores/gibbs.1g.cores').unlink()
                else:
                    p=folder/'res/gibbs.1g.out';p.write_text(p.read_text().replace('iterations: 12.0','iterations: -nan'))
                with self.assertRaises((ValueError,FileNotFoundError)):api.parse_native_run(folder,['ACDEF'])



if __name__=='__main__':unittest.main()
