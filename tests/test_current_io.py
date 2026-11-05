import gzip
import pytest
from scripts.candidate_evidence.msa_alignment import read_fasta
from scripts.paths import output_directory, table_path


def test_compressed_alignment_preserves_sequence_identifiers(tmp_path):
    path=tmp_path/'alignment.fasta.gz'
    path.write_bytes(gzip.compress(b'>one\nAC-D\n>two\nA-CD\n',mtime=0))
    assert read_fasta(path)=={'one':'AC-D','two':'A-CD'}
    assert table_path(tmp_path/'alignment.fasta')==path


def test_current_outputs_can_use_work_but_never_frozen_inputs(tmp_path):
    root=tmp_path/'project';root.mkdir()
    assert output_directory(root,root/'work/derived')==root/'work/derived'
    for target in [root,root/'data/derived',root/'results/derived',root/'src/derived']:
        with pytest.raises(ValueError):output_directory(root,target)
