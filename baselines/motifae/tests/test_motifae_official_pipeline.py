import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))


class OfficialPipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_official_inference'),
                             'official-model inference module missing')
        self.api=importlib.import_module('motifae_official_inference')

    def torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Install test requirements to exercise actual official Torch model')
        return torch

    def checkpoint(self,root):
        torch=self.torch()
        cls,_=self.api.official_model_class()
        model=cls(2,3,tied=False)
        with torch.no_grad():
            model.bias.copy_(torch.tensor([1.,0.]))
            model.encoder.weight.copy_(torch.tensor([[1.,0.],[1.,0.],[0.,1.]]))
            model.encoder.bias.zero_()
        path=root/'synthetic.pt'
        torch.save(model.state_dict(),path)
        return path

    def cached(self,root,header='sp|P1|NAME'):
        raw=root/'raw';(raw/'input').mkdir(parents=True);(raw/'embeddings').mkdir()
        (raw/'input/motifae_input.csv').write_text('uniprot,sequence\nS1,ACDEFGH\n')
        (raw/'input/id_map.csv').write_text('safe_id,original_id\nS1,'+header+'\n')
        values=np.array([0,.7,.8,0,.7,.8,0],dtype=np.float32)
        array=np.zeros((1,9,2),dtype=np.float32);array[0,1:-1,0]=values+1
        np.savez_compressed(raw/'embeddings/S1.npz',repr=array)
        return raw

    def test_vendored_class_is_unmodified_pinned_author_source(self):
        self.torch()
        cls,metadata=self.api.official_model_class()
        self.assertEqual(cls.__name__,'AutoEncoder')
        self.assertEqual(metadata['source_sha256'],metadata['pinned_sha256'])
        self.assertEqual(metadata['upstream_commit'],'8dd6a7588d30726fef98c789dac3042370e829c5')


    def test_default_rejects_unverified_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=self.checkpoint(Path(tmp))
            with self.assertRaisesRegex(ValueError,'published checkpoint'):
                self.api.load_autoencoder(checkpoint)


    def test_supports_cached_token_and_author_residue_layouts(self):
        x=np.ones((5,2),dtype=np.float32)
        np.testing.assert_array_equal(self.api.residue_view(x,'ACDEF',2),x)
        np.testing.assert_array_equal(self.api.residue_view(np.ones((1,7,2)),'ACDEF',2),x)
        with self.assertRaises(ValueError):
            self.api.residue_view(np.ones((1024,2)),'A'*1100,2)
        with self.assertRaises(ValueError):
            self.api.residue_view(np.full((5,2),np.nan),'ACDEF',2)

    def test_direct_encode_chunking_preserves_all_features_and_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=self.checkpoint(root)
            cls,_=self.api.official_model_class()
            model=cls(2,3,tied=False)
            model.load_state_dict(self.torch().load(checkpoint,weights_only=True));model.eval()
            array=np.zeros((7,2),dtype=np.float32);array[:,0]=np.array([0,.7,.8,0,.7,.8,0])+1
            rows=self.api.extract_peptides(model,array,'ACDEFGH','P1','GvpA',chunk_size=2)
            self.assertEqual(len(rows),2)
            self.assertEqual({r['motif_uid'] for r in rows},{'GvpA_MotifAE_feature_0','GvpA_MotifAE_feature_1'})
            self.assertTrue(all((r['occ_start'],r['occ_end'],r['motif_sequence'])==(1,6,'CDEFG') for r in rows))
            self.assertEqual(rows,self.api.extract_peptides(model,array,'ACDEFGH','P1','GvpA',chunk_size=128))

    def test_cache_index_drives_processing_not_orphan_npz_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=self.cached(root)
            np.savez(raw/'embeddings/orphan.npz',repr=np.ones((1,7,2)))
            records,meta=self.api.load_cached_records(raw)
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['parent_id'],'P1')
            self.assertEqual(meta['orphan_embedding_files'],['orphan.npz'])

    def test_reject_missing_or_unsafe_embedding_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=self.cached(root)
            (raw/'input/motifae_input.csv').write_text('uniprot,sequence\n../outside,ACDEF\n')
            with self.assertRaises(ValueError):
                self.api.load_cached_records(raw)



    def test_fasta_all_long_skip_does_not_load_models(self):
        from run_motifae_pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fasta=root/'long.fasta';fasta.write_text('>P1\n'+'A'*1100+'\n')
            with self.assertRaisesRegex(ValueError,'No proteins'):
                run_pipeline(family='GvpA',fasta=fasta,checkpoint=root/'not_loaded.pt',
                             output_dir=root/'out',long_sequence_policy='skip')

    def test_esm_boundary_removes_special_tokens(self):
        torch=self.torch()
        from run_motifae_pipeline import esm_embeddings
        class Model:
            def __call__(self,tokens,repr_layers):
                self.layers=repr_layers
                n=tokens.shape[1]
                return {'representations':{33:torch.arange(n,dtype=torch.float32).view(1,n,1).repeat(1,1,1280)}}
        def converter(records):
            return None,None,torch.zeros((1,len(records[0][1])+2),dtype=torch.long)
        model=Model();array=esm_embeddings('ACDEF','P1',model,converter,'cpu')
        self.assertEqual(array.shape,(5,1280))
        np.testing.assert_array_equal(array[:,0],[1,2,3,4,5])
        self.assertEqual(model.layers,[33])







if __name__=='__main__':
    unittest.main()
