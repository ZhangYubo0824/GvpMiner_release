import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))
from motifae_candidate_filter import sha256
from motifae_gibbs_inputs import prepare_inputs


def discovery(root,family='GvpA',run='run_001',sequence='ACDEF',rows=True):
    fasta_dir=root/'fasta';fasta_dir.mkdir(exist_ok=True)
    fasta=fasta_dir/(family+'_cleaned.fasta');fasta.write_text('>P_'+family+'\n'+sequence+'\n')
    folder=root/'discovery'/family/run;folder.mkdir(parents=True)
    candidates=[dict(family=family,method='MotifAE',motif_uid=family+'_MotifAE_feature_42',parent_id='P_'+family,sequence_id='P_'+family,occ_start=0,occ_end=5,motif_sequence=sequence[:5],score=.5)] if rows else []
    source=folder/'features/motif_occurrences_MotifAE_raw.csv';source.parent.mkdir()
    fields=['family','method','motif_uid','parent_id','sequence_id','occ_start','occ_end','motif_sequence','score']
    with source.open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(candidates)
    prepare_inputs(candidates,folder/'gibbs_inputs',source_protocol='legacy_preprint_v1',source_files=[dict(path=str(source),sha256=sha256(source))])
    manifest=dict(status='awaiting_gibbscluster_2_0',family=family,protocol='preprint-v1',processed_proteins=1,skipped_proteins=[],feature_count=len(candidates),peptide_count=len(candidates),model=dict(published_checkpoint_verified=True,checkpoint_sha256='e322c859a593c6c3dba461a2fae6c648ccd064a46422221f05f25c6eac03f28d'),input=dict(source_files=[dict(path=str(fasta),sha256=sha256(fasta))]))
    (folder/'pipeline_manifest.json').write_text(json.dumps(manifest))
    return folder,fasta


class FamilyAlignmentTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_family_alignment'),'family alignment orchestrator missing')
        import motifae_family_alignment
        return motifae_family_alignment

    def test_discovers_two_families_and_preserves_long_input_without_models(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);discovery(root);discovery(root,'GvpC',sequence='A'*1026)
            cases=self.api().inspect_discovery(root/'discovery',root/'fasta')
            self.assertEqual([x['family'] for x in cases],['GvpA','GvpC'])
            self.assertEqual(cases[1]['max_protein_length'],1026)

    def test_multiple_runs_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);discovery(root);discovery(root,run='run_002')
            api=self.api()
            with self.assertRaisesRegex(ValueError,'multiple|ambiguous'):api.inspect_discovery(root/'discovery',root/'fasta')
            self.assertEqual(Path(api.inspect_discovery(root/'discovery',root/'fasta',select_run='latest')[0]['discovery_dir']).name,'run_002')

    def test_rejects_changed_fasta_before_any_alignment(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder,fasta=discovery(root);fasta.write_text('>P_GvpA\nAAAAA\n')
            with self.assertRaisesRegex(ValueError,'FASTA|hash'):self.api().inspect_discovery(root/'discovery',root/'fasta')

    def test_rejects_previously_skipped_proteins(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder,fasta=discovery(root);p=folder/'pipeline_manifest.json';m=json.loads(p.read_text());m['skipped_proteins']=[dict(parent_id='P_GvpA')];p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'skip|processed'):self.api().inspect_discovery(root/'discovery',root/'fasta')

    def test_rejects_tampered_mapping_even_if_hash_rewritten(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder,fasta=discovery(root);p=folder/'gibbs_inputs/mappings/feature_000001.csv';p.write_text(p.read_text().replace('P_GvpA','UNKNOWN'))
            mp=folder/'gibbs_inputs/gibbs_preparation_manifest.json';m=json.loads(mp.read_text());m['files_sha256']['mappings/feature_000001.csv']=sha256(p);mp.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'protein|mapping|sequence|Top'):self.api().inspect_discovery(root/'discovery',root/'fasta')

    def test_rejects_nonofficial_checkpoint_and_wrong_family(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder,fasta=discovery(root);p=folder/'pipeline_manifest.json';m=json.loads(p.read_text());m['model']['published_checkpoint_verified']=False;p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'official|published|checkpoint'):self.api().inspect_discovery(root/'discovery',root/'fasta')
            m['model']['published_checkpoint_verified']=True;m['family']='GvpF';p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'family'):self.api().inspect_discovery(root/'discovery',root/'fasta')

    def test_zero_candidate_family_is_a_valid_explicit_input(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);discovery(root,rows=False)
            case=self.api().inspect_discovery(root/'discovery',root/'fasta')[0]
            self.assertEqual(case['groups'],0)
            self.assertEqual(case['selected_instances'],0)

    def test_unsafe_family_and_nested_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);discovery(root);api=self.api()
            with self.assertRaises(ValueError):api.inspect_discovery(root/'discovery',root/'fasta',families=['../GvpA'])
            with self.assertRaisesRegex(ValueError,'separate|overlap|inside'):
                api.align_families(root/'discovery',root/'fasta',root/'discovery'/'out',root/'gibbs','r1',root/'q',check_runtime=False)

    def test_cli_preflight_needs_no_weights_or_torch_cuda(self):
        import subprocess
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);discovery(root)
            self.api()
            result=subprocess.run([sys.executable,str(ROOT/'scripts/baseline_adapters/motifae_family_alignment.py'),'preflight','--discovery-root',str(root/'discovery'),'--fasta-dir',str(root/'fasta'),'--input-only'],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['families'][0]['family'],'GvpA')




if __name__=='__main__':unittest.main()
