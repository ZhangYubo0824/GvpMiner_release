import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))
sys.path.insert(0,str(ROOT/'tests'))
fixtures=importlib.import_module('test_motifae_gibbs_results')
zero=importlib.import_module('test_motifae_gibbs_all_trash')


class NativeGuardTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_native_guard'))
        self.api=importlib.import_module('motifae_native_guard')

    def test_guard_skips_only_empty_single_ordinary_cluster(self):
        self.assertTrue(self.api.guard_decision(1,[0,2],2))
        self.assertFalse(self.api.guard_decision(1,[1,1],2))
        for groups,sizes,n in [(2,[0,0,2],2),(1,[0,1],2),(1,[-1,3],2),(0,[2],2)]:
            with self.assertRaises(ValueError):self.api.guard_decision(groups,sizes,n)

    def test_generated_gdb_guard_precedes_selection_and_redirects_native_output(self):
        text=self.api.guard_commands(['-trash','-l','5','/tmp/input with spaces.txt'],2,
                                     Path('/tmp/native output.txt'),Path('/tmp/native.err'))
        for token in ('0x402b7a','0x402fb4','0x402fdb','0x4031ad','MOTIFAE_GUARD_COUNTS'):
            self.assertIn(token,text)
        self.assertIn('set startup-with-shell on',text)
        self.assertIn("'/tmp/input with spaces.txt'",text)
        self.assertIn("> '/tmp/native output.txt' 2> /tmp/native.err",text)
        self.assertNotIn('set {',text)

    def test_native_parameters_match_official_driver_defaults(self):
        command=self.api.native_command(Path('/gibbs'),Path('/input'))
        for flag,value in [('-g','1'),('-l','5'),('-base','1'),('-ilen','1'),('-dlen','1'),
                           ('-i','10'),('-nt','20'),('-s','1'),('-fr','20'),('-fs','100')]:
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('-trash',command)

    def test_auxiliary_files_derive_from_real_raw_records_without_changing_raw(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture=fixtures.GibbsResultTests()
            fixture.native(root,[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            raw=root/'res/gibbs.1g.out';before=raw.read_bytes()
            (root/'cores/gibbs.1g.cores').unlink();(root/'res/gibbs.1g.ds.out').unlink()
            self.api.materialize_auxiliary(root,['ACDEF'])
            parsed=self.api.parse_native_run(root,['ACDEF'])
            self.assertEqual(parsed['records'][0]['core'],'ACDEF')
            self.assertEqual(raw.read_bytes(),before)

    def test_completed_all_trash_keeps_nan_undefined_and_does_not_invent_cores(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);zero.zero_native(root,('AIGSE','SSAPQ'))
            self.api.materialize_auxiliary(root,['AIGSE','SSAPQ'])
            result=self.api.parse_native_run(root,['AIGSE','SSAPQ'])
            self.assertIsNone(result['kld']);self.assertFalse((root/'cores/gibbs.1g.cores').exists())

    def test_partial_raw_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);zero.zero_native(root,('AIGSE','SSAPQ'))
            p=root/'res/gibbs.1g.out';p.write_text(p.read_text().split('### Final KLD')[0])
            with self.assertRaises(ValueError):self.api.materialize_auxiliary(root,['AIGSE','SSAPQ'])

    def test_reference_comparison_checks_entire_native_assignments(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture=fixtures.GibbsResultTests()
            fixture.native(root/'reference',[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            fixture.native(root/'trial',[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
            self.api.compare_native(root/'reference',root/'trial',['ACDEF'])
            p=root/'trial/res/gibbs.1g.out';p.write_text(p.read_text().replace('12.0','13.0'))
            with self.assertRaises(ValueError):self.api.compare_native(root/'reference',root/'trial',['ACDEF'])

    def test_reference_comparison_checks_background_and_corrected_scores(self):
        for field,value in [('bgG -1','bgG 0'),('bgS 1.0','bgS 2.0'),('cS 3.2','cS 2.2')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
                root=Path(t);fixture=fixtures.GibbsResultTests()
                for folder in (root/'reference',root/'trial'):
                    fixture.native(folder,[('ACDEF','ACDEF',0,-99,-99,-99,-99)])
                for name in ('gibbs.1g.out','gibbs.1g.ds.out'):
                    p=root/'trial/res'/name;p.write_text(p.read_text().replace(field,value))
                with self.assertRaises(ValueError):self.api.compare_native(root/'reference',root/'trial',['ACDEF'])

    def runner_context(self,stack,installed,runner,cleanup_error=None):
        stack.enter_context(mock.patch.object(self.api.platform,'system',return_value='Linux'))
        stack.enter_context(mock.patch.object(self.api.platform,'machine',return_value='x86_64'))
        stack.enter_context(mock.patch.object(self.api,'verified_installation',return_value=installed))
        stack.enter_context(mock.patch.object(self.api.shutil,'which',return_value='mock-gdb'))
        stack.enter_context(mock.patch.object(self.api,'PidfdBackend'))
        stack.enter_context(mock.patch.object(self.api,'check_backend'))
        stack.enter_context(mock.patch.object(self.api,'run_external',side_effect=runner))
        return stack.enter_context(mock.patch.object(self.api,'cleanup_probe',return_value=0,side_effect=cleanup_error))

    def test_guarded_runner_records_actual_counters_and_completed_native_result(self):
        with tempfile.TemporaryDirectory() as t,ExitStack() as stack:
            root=Path(t)
            def run(command,**kwargs):
                zero.zero_native(Path(kwargs['cwd']),('AIGSE','SSAPQ'))
                kwargs['stdout'].write(b'MOTIFAE_GUARD_COUNTS 3 1\n')
                return subprocess.CompletedProcess(command,0)
            cleanup=self.runner_context(stack,root/'installed',run)
            event=self.api.run_guarded_native(['AIGSE','SSAPQ'],root/'installed',root/'out')
            self.assertEqual((event['indel_moves_skipped'],event['phase_moves_skipped']),(3,1))
            self.assertEqual(event['result_status'],'all_sequences_trash')
            cleanup.assert_called_once()

    def test_singleton_runner_completes_and_preserves_one_native_sequence(self):
        with tempfile.TemporaryDirectory() as t,ExitStack() as stack:
            root=Path(t)
            def run(command,**kwargs):
                zero.zero_native(Path(kwargs['cwd']),('QPSHRLPSEDE',))
                kwargs['stdout'].write(b'MOTIFAE_GUARD_COUNTS 0 0\n')
                return subprocess.CompletedProcess(command,0)
            self.runner_context(stack,root/'installed',run)
            event=self.api.run_guarded_native(['QPSHRLPSEDE']*3,root/'installed',root/'out')
            self.assertEqual(event['unique_inputs'],1)
            self.assertEqual(event['original_instances'],3)
            self.assertEqual(event['result_status'],'all_sequences_trash')

    def test_timeout_and_interrupt_always_clean_native_and_remain_unresolved(self):
        from motifae_process_control import JobTimedOut
        for error in (JobTimedOut('test'),KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__),tempfile.TemporaryDirectory() as t,ExitStack() as stack:
                root=Path(t)
                def run(*args,**kwargs):raise error
                cleanup=self.runner_context(stack,root/'installed',run)
                with self.assertRaises(type(error)):
                    self.api.run_guarded_native(['AIGSE','SSAPQ'],root/'installed',root/'out')
                cleanup.assert_called_once()
                event=json.loads((root/'out/native_guard_manifest.json').read_text())
                self.assertEqual(event['status'],'unresolved_guarded_sampler')

    def test_inferior_cleanup_failure_has_fatal_cleanup_type(self):
        from motifae_process_control import ProcessCleanupError
        with tempfile.TemporaryDirectory() as t,ExitStack() as stack:
            root=Path(t)
            self.runner_context(stack,root/'installed',lambda *a,**k:subprocess.CompletedProcess([],0),RuntimeError('cleanup failed'))
            with self.assertRaises(ProcessCleanupError):
                self.api.run_guarded_native(['AIGSE','SSAPQ'],root/'installed',root/'out')


if __name__=='__main__':unittest.main()
