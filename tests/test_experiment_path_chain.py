"""Default producer/consumer paths must agree without loading a model."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import config
from exp4_motif_structure_analysis import cli, family_structures
from exp4_motif_structure_analysis.structure_backends import ESMFoldBackend
from motif_stability.cli import default_source_exp3_dir


def test_structure_prediction_and_analysis_share_new_output_root():
    assert config.EXP4_STRUCTURE_INPUT_DIR==config.STRUCTURES_DIR
    assert cli._default_sequence_table()==config.SEQUENCE_TABLE
    assert cli._default_structure_dir('GvpA','esmfold')==config.STRUCTURES_DIR/'esmfold/GvpA'


def test_structure_prediction_uses_the_bundled_sequence_fallback():
    assert family_structures._default_sequence_table()==config.SEQUENCE_TABLE
    assert family_structures._default_sequence_table().is_file()


def test_stability_consumes_the_normal_discovery_output():
    assert default_source_exp3_dir()==config.EXP3_DIR


def test_structure_models_use_the_shared_work_cache():
    assert ESMFoldBackend._models_dir()==config.MODELS_DIR


def test_motifae_structure_defaults_share_the_release_paths():
    import subprocess
    code = """
from pathlib import Path
import sys
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'baselines/motifae/src'))
from exp4_motif_structure_analysis import cli, family_structures
assert cli._default_structure_dir('GvpA','MotifAE','esmfold') == family_structures.STRUCTURES_DIR/'esmfold/GvpA'
assert cli._default_sequence_table() == family_structures._default_sequence_table()
assert cli._default_sequence_table().is_file()
assert family_structures.STRUCTURES_DIR == root/'work/results/structures'
"""
    subprocess.run([sys.executable,'-c',code,str(ROOT)],check=True)
