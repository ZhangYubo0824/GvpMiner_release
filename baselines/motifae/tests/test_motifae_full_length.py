from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))
import run_motifae_pipeline as api


class FullLengthTests(unittest.TestCase):
    def test_full_policy_reaches_model_for_1026_residues_without_truncation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);f=p/'C.fasta';f.write_text('>P1\n'+'A'*1026+'\n')
            with mock.patch.object(api,'load_autoencoder',side_effect=RuntimeError('reached model load')):
                with self.assertRaisesRegex(RuntimeError,'reached model load'):
                    api.run_pipeline(family='GvpC',fasta=f,checkpoint=p/'model.pt',output_dir=p/'out',long_sequence_policy='full')

    def test_explicit_error_still_rejects_long_input_before_model(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);f=p/'C.fasta';f.write_text('>P1\n'+'A'*1026+'\n')
            with self.assertRaisesRegex(ValueError,'1022'):
                api.run_pipeline(family='GvpC',fasta=f,checkpoint=p/'model.pt',output_dir=p/'out',long_sequence_policy='error')

    def test_local_esm_rejects_unknown_file_before_loading_or_download(self):
        self.assertTrue(hasattr(api,'load_local_esm'),'explicit local ESM loader missing')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.pt';p.write_bytes(b'wrong file')
            with self.assertRaisesRegex(ValueError,'hash|official|checkpoint'):api.load_local_esm(p)


if __name__=='__main__':unittest.main()
