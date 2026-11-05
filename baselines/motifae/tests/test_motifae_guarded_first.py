"""First-pass execution tests; native execution is mocked on Windows."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/baseline_adapters'),str(ROOT/'tests')]
import motifae_native_guard as guard
import motifae_gibbs_results as results
import motifae_family_alignment as families
from motifae_process_control import JobTimedOut
from motifae_gibbs_inputs import prepare_inputs
import test_motifae_gibbs_results as fixture_module
import test_motifae_gibbs_all_trash as zero_module
import test_motifae_family_alignment as family_fixtures


class FirstPassTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_guarded_batch'),
                             'first-pass guarded batch is missing')
        return importlib.import_module('motifae_guarded_batch')

    def inputs(self,root):
        rows=[]
        for uid,peptides in [('normal',['ACDEF','GHIKL']),('fault',['AIGSE','SSAPQ']),('single',['QPSHRLPSEDE'])]:
            for i,seq in enumerate(peptides):
                protein=f'{uid}_{i}'
                rows.append(dict(family='GvpA',method='MotifAE',motif_uid=uid,parent_id=protein,
                    sequence_id=protein,occ_start=0,occ_end=len(seq),motif_sequence=seq,score=.5))
        prep=root/'prep';prepare_inputs(rows,prep,source_protocol='legacy_preprint_v1')
        fasta=root/'input.fasta';fasta.write_text(''.join(f">{r['parent_id']}\n{r['motif_sequence']}\n" for r in rows))
        return prep,fasta

    def runner(self,peptides,installed,output,**kwargs):
        if 'ACDEF' in peptides:
            fixture_module.GibbsResultTests().native(Path(output),[(p,p,0,-99,-99,-99,-99) for p in sorted(set(peptides))]);count=0
        else:
            zero_module.zero_native(Path(output),tuple(sorted(set(peptides))));count=3
        return dict(status='completed_guarded_sampler',indel_moves_skipped=count,phase_moves_skipped=0)

    def context(self,root):
        installed=root/'installed';installed.mkdir();q=root/'q';q.mkdir()
        (q/'qualification_manifest.json').write_text('{}')
        return installed,q

    def test_each_group_is_sampled_once_under_guard_and_exportable_including_singleton(self):
        api=self.api()
        from motifae_handoff import export_handoff
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.inputs(root);installed,q=self.context(root)
            with mock.patch.object(api,'verified_installation',return_value=installed), \
                 mock.patch.object(api,'verify_qualification',return_value={}), \
                 mock.patch.object(api,'run_guarded_native',side_effect=self.runner) as sampler:
                state=api.run_guarded_batch(prep,installed,fasta,root/'batch',q)
            self.assertEqual(sampler.call_count,3)
            self.assertEqual(state['status'],'alignment_samples_parsed',state)
            self.assertEqual(state['pattern_built_groups'],1)
            self.assertEqual(state['all_trash_groups'],2)
            self.assertEqual(state['timed_out_groups'],0)
            self.assertEqual(state['execution_mode'],'guarded-first')
            self.assertTrue(all(j['sampling_executed'] for j in state['jobs']))
            export_handoff([dict(family='GvpA',batch_dir=str(root/'batch'),preparation_dir=str(prep),fasta=str(fasta))],root/'handoff')
            self.assertTrue((root/'handoff/handoff_manifest.json').is_file())

    def test_resume_reuses_verified_jobs_without_new_sampling_and_detects_changed_files(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.inputs(root);installed,q=self.context(root)
            with mock.patch.object(api,'verified_installation',return_value=installed), \
                 mock.patch.object(api,'verify_qualification',return_value={}), \
                 mock.patch.object(api,'run_guarded_native',side_effect=self.runner) as sampler:
                state=api.run_guarded_batch(prep,installed,fasta,root/'batch',q)
                sampler.reset_mock()
                api.run_guarded_batch(prep,installed,fasta,root/'batch',q,resume=True)
                sampler.assert_not_called()
                job=state['jobs'][0]
                (root/'batch'/job['parsed_dir']/'aligned_occurrences.csv').write_text('damaged')
                with self.assertRaisesRegex(ValueError,'hash|changed|verified'):
                    api.run_guarded_batch(prep,installed,fasta,root/'batch',q,resume=True)
                sampler.assert_not_called()

    def test_real_timeout_is_incomplete_and_is_not_retried_automatically(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.inputs(root);installed,q=self.context(root)
            with mock.patch.object(api,'verified_installation',return_value=installed), \
                 mock.patch.object(api,'verify_qualification',return_value={}), \
                 mock.patch.object(api,'run_guarded_native',side_effect=JobTimedOut('genuine slow task')) as sampler:
                state=api.run_guarded_batch(prep,installed,fasta,root/'batch',q)
                self.assertEqual(sampler.call_count,3)
                self.assertEqual(state['timed_out_groups'],3)
                sampler.reset_mock()
                api.run_guarded_batch(prep,installed,fasta,root/'batch',q,resume=True)
                sampler.assert_not_called()

    def test_resume_derives_guard_provenance_from_hashed_attempt_record(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prep,fasta=self.inputs(root);installed,q=self.context(root)
            with mock.patch.object(api,'verified_installation',return_value=installed), \
                 mock.patch.object(api,'verify_qualification',return_value={}), \
                 mock.patch.object(api,'run_guarded_native',side_effect=self.runner):
                api.run_guarded_batch(prep,installed,fasta,root/'batch',q)
                job_path=root/'batch/jobs/feature_000001/job_manifest.json'
                job=json.loads(job_path.read_text());job['guard_event']['indel_moves_skipped']=999999
                job_path.write_text(json.dumps(job))
                resumed=api.run_guarded_batch(prep,installed,fasta,root/'batch',q,resume=True)
                self.assertEqual(resumed['jobs'][0]['guard_event']['indel_moves_skipped'],3)

    def test_guard_accepts_singleton_but_never_empty_or_unaccounted_sequences(self):
        self.assertTrue(guard.guard_decision(1,[0,1],1))
        self.assertFalse(guard.guard_decision(1,[1,0],1))
        script=guard.guard_commands(['-trash'],1,Path('/tmp/out'),Path('/tmp/err'))
        self.assertIn('!= 1',script)
        for sizes,n in [([0,0],0),([0,2],1)]:
            with self.assertRaises(ValueError):guard.guard_decision(1,sizes,n)

    def test_default_family_path_never_calls_legacy_batch_or_recovery(self):
        self.api()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);family_fixtures.discovery(root)
            installed=root/'installed';q=root/'q';q.mkdir()
            (q/'qualification_manifest.json').write_text('{}')
            for name in ['gibbscluster',*guard.INSTALL_SHA]:
                p=installed/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test '+name)
            digests={name:results.sha256(installed/name) for name in guard.INSTALL_SHA}
            api=importlib.import_module('motifae_guarded_batch')
            with mock.patch.dict(guard.INSTALL_SHA,digests,clear=True), \
                 mock.patch.object(families,'verify_qualification',return_value={}), \
                 mock.patch.object(api,'verify_qualification',return_value={}), \
                 mock.patch.object(api,'run_guarded_native',side_effect=self.runner):
                state=families.align_families(root/'discovery',root/'fasta',root/'alignment',installed,'first',q,check_runtime=False)
            self.assertFalse(hasattr(families,'run_batch'))
            self.assertFalse(hasattr(families,'recover_batch'))
            self.assertEqual(state['status'],'all_families_aligned',state)
            self.assertTrue(Path(state['families'][0]['final_result_dir']).name=='guarded')


if __name__=='__main__':unittest.main()
