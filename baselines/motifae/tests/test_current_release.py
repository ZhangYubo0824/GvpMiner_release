"""Current-release contracts; these do not execute the licensed native sampler."""
import inspect
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'baseline_adapters'))


class CurrentReleaseTests(unittest.TestCase):
    def test_alignment_exposes_only_guarded_first_sampling(self):
        import motifae_family_alignment as alignment
        self.assertNotIn('execution_mode', inspect.signature(alignment.align_families).parameters)
        self.assertNotIn('recovery_timeout', inspect.signature(alignment.align_families).parameters)
        result = subprocess.run([sys.executable, str(Path(alignment.__file__)), 'run', '--help'],
                                capture_output=True, text=True, check=True)
        self.assertNotIn('legacy-recovery', result.stdout)
        self.assertNotIn('--execution-mode', result.stdout)

    def test_inference_has_no_local_training_or_custom_checkpoint_override(self):
        import run_motifae_pipeline as pipeline
        import motifae_official_inference as inference
        self.assertEqual(inspect.signature(pipeline.run_pipeline).parameters['long_sequence_policy'].default, 'full')
        for name in ('allow_custom_checkpoint', 'allow_incomplete_training', 'gibbs_command_file', 'author_repo'):
            self.assertNotIn(name, inspect.signature(pipeline.run_pipeline).parameters)
        self.assertNotIn('require_published', inspect.signature(inference.load_autoencoder).parameters)
        self.assertNotIn('source_path', inspect.signature(inference.load_autoencoder).parameters)
        self.assertEqual(set(inference.RELEASED_MD5.values()), {'k3'})

    def test_unverified_checkpoint_is_rejected_before_torch_is_loaded(self):
        import motifae_official_inference as inference
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'custom.pt'
            path.write_bytes(b'not published weights')
            with self.assertRaisesRegex(ValueError, 'k3/80000'):
                inference.load_autoencoder(path)

    def test_excluded_modules_are_not_shipped(self):
        for name in ('motifae_adapter_paper_peptides.py', 'motifae_gibbs_guard_recovery.py', 'motifae_gibbs_reparse.py'):
            self.assertFalse((ROOT / 'scripts/baseline_adapters' / name).exists(), name)
        self.assertTrue((ROOT / 'src/exp4_motif_structure_analysis/cli.py').is_file())
        self.assertFalse((ROOT / 'src3').exists())

    def test_result_parser_cannot_launch_an_unguarded_batch(self):
        import motifae_gibbs_results as results
        self.assertFalse(hasattr(results, 'run_batch'))
        self.assertFalse(hasattr(results, 'validate_postprocessing_error'))

    def test_candidate_inventory_has_no_unused_local_filter_modes(self):
        import motifae_candidate_filter as candidates
        self.assertEqual(list(inspect.signature(candidates.process_candidates).parameters), ['rows'])
        self.assertFalse(hasattr(candidates, '_matched_intervals'))

    def test_unused_exp5_exploratory_modules_and_scripts_are_excluded(self):
        base = ROOT / 'src/exp5_external_functional_enrichment'
        for name in ('exploratory_analysis.py', 'support_analysis.py'):
            self.assertFalse((base / 'package' / name).exists())
        self.assertFalse((base / 'scripts').exists())

    def test_exp4_and_exp5_defaults_resolve_current_reference_inputs_and_work_outputs(self):
        sys.path.insert(0, str(ROOT / 'src'))
        from exp4_motif_structure_analysis import cli, family_structures
        from exp4_motif_structure_analysis.structure_backends import ESMFoldBackend
        from exp5_external_functional_enrichment.package import cli as exp5
        self.assertEqual(cli._default_sequence_table().resolve(), (PACKAGE / 'data/reference/preprocessed/cleaned_gvp_dataset.csv').resolve())
        self.assertTrue(cli._default_sequence_table().is_file())
        self.assertEqual(cli._default_output_root().resolve(), (PACKAGE / 'work/results/exp4_motif_structure_analysis').resolve())
        self.assertEqual(cli._default_exp3_dir().resolve(), (PACKAGE / 'work/results/exp3_domain_discovery').resolve())
        self.assertEqual(cli._default_structure_dir('GvpA','MotifAE','esmfold').resolve(), (PACKAGE / 'work/results/structures/esmfold/GvpA').resolve())
        self.assertEqual(family_structures._default_sequence_table().resolve(), cli._default_sequence_table().resolve())
        self.assertEqual(family_structures.STRUCTURES_DIR.resolve(), (PACKAGE / 'work/results/structures').resolve())
        self.assertEqual(ESMFoldBackend._project_root().resolve(), PACKAGE.resolve())
        self.assertEqual(ESMFoldBackend._models_dir().resolve(), (PACKAGE / 'work/models').resolve())
        self.assertEqual(exp5._default_project_root().resolve(), PACKAGE.resolve())
        self.assertEqual(exp5._default_results_dir().resolve(), (PACKAGE / 'work/results').resolve())
        self.assertEqual(exp5._default_ncbi_root().resolve(), (PACKAGE / 'data/ncbi').resolve())
        self.assertTrue(exp5._default_ncbi_root().is_dir())

    def test_removed_methods_are_rejected_by_current_exp4_exp5_parsers(self):
        sys.path.insert(0, str(ROOT / 'src'))
        from exp4_motif_structure_analysis import cli
        from exp5_external_functional_enrichment.package import cli as exp5, motif_sets
        for removed in ('SLiMFinder', 'DiMotif'):
            self.assertNotIn(removed, cli.SUPPORTED_MOTIF_METHODS)
            self.assertNotIn(removed, motif_sets.SUPPORTED_MOTIF_METHODS)
            with self.assertRaises(SystemExit):
                cli.build_arg_parser().parse_args(['--methods', removed])
            with self.assertRaises(SystemExit):
                exp5.build_arg_parser().parse_args(['make-motif-sets', '--methods', removed])

    def test_structure_backend_torch_hub_uses_the_model_work_cache(self):
        sys.path.insert(0, str(ROOT / 'src'))
        from exp4_motif_structure_analysis.structure_backends import ESMFoldBackend
        hub = SimpleNamespace(set_dir=mock.Mock())
        with tempfile.TemporaryDirectory() as folder:
            models = Path(folder) / 'models'
            with mock.patch.object(ESMFoldBackend, '_models_dir', return_value=models):
                ESMFoldBackend._prepare_model_cache(SimpleNamespace(hub=hub))
            self.assertTrue(models.is_dir())
            hub.set_dir.assert_called_once_with(str(models / 'torch_hub'))

    def test_shell_wrapper_actual_defaults_resolve_the_release_root(self):
        bash = shutil.which('bash')
        git = shutil.which('git')
        if not bash and git:
            candidate = Path(git).resolve().parents[1] / 'bin/bash.exe'
            if candidate.is_file():
                bash = str(candidate)
        if not bash:
            self.skipTest('Bash is required to evaluate wrapper defaults')
        # Source the actual wrapper with command doubles; native tools and writes
        # are never invoked. The shell still evaluates all real path defaults.
        script = '''
path_line() { if command -v cygpath >/dev/null; then cygpath -m "$1"; else printf '%s\\n' "$1"; fi; }
python() { path_line "$PACKAGE_ROOT"; path_line "$MOTIFAE_RESULTS"; path_line "$MOTIFAE_FASTA"; }
mkdir() { :; }
tee() { cat; }
wrapper="$1"; shift; source "$wrapper"
'''
        env = {key: value for key, value in os.environ.items() if key not in ('MOTIFAE_RESULTS','MOTIFAE_FASTA','PYTHON')}
        env['MOTIFAE_GIBBS'] = '/unused/qualification-test-installation'
        result = subprocess.run([bash, '--noprofile', '--norc', '-c', script, '--', (ROOT / 'run_all_families.sh').as_posix()],
                                env=env, capture_output=True, text=True, encoding='utf-8', check=True)
        paths = result.stdout.splitlines()
        self.assertEqual(Path(paths[0]).resolve(), PACKAGE.resolve())
        self.assertEqual(Path(paths[1]).resolve(), (PACKAGE / 'work/results/baselines/motifae').resolve())
        self.assertEqual(Path(paths[2]).resolve(), (PACKAGE / 'data/reference/preprocessed/by_family').resolve())
        self.assertTrue(Path(paths[2]).is_dir())

    def test_qualification_signature_and_fixed_fixture_inventory_are_complete(self):
        import motifae_gibbs_qualification as qualification
        normal, fault = qualification._fixtures()
        self.assertGreater(len(normal), 100)
        self.assertEqual(len(fault), 24)
        code = qualification._signature()['code_sha256']
        self.assertNotIn('motifae_gibbs_guard_recovery.py', code)
        self.assertNotIn('motifae_gibbs_reparse.py', code)
        self.assertIn('motifae_guarded_batch.py', code)

    def test_active_python_and_shell_sources_have_no_historical_root_defaults(self):
        for path in ROOT.rglob('*'):
            if path.suffix not in ('.py', '.sh') or 'tests' in path.parts:
                continue
            text = path.read_text(encoding='utf-8')
            for forbidden in ('/data/coding', 'results6', 'oldsrc', 'src3'):
                self.assertNotIn(forbidden, text, str(path))


if __name__ == '__main__':
    unittest.main()
