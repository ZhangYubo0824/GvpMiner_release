"""Scientific behavior tests for descriptive-only absolute MSA conservation."""
import math
import numpy as np
import pandas as pd
import pytest
from scripts.candidate_evidence import conservation as c
from scripts.candidate_evidence.msa_alignment import validate_alignment


def test_msa_output_paths_allow_external_outputs_after_merge(tmp_path):
    source=tmp_path/'main'
    output=tmp_path/'derived/msa'
    cache=tmp_path/'derived/alignments'
    assert c.output_paths(source,output,cache)==tuple(p.resolve() for p in (source,output,cache))
    for bad_output,bad_cache in [(source,cache),(source/'results',cache),
                                 (output,source),(output,source/'cache'),
                                 (output,output/'cache'),(cache/'results',cache)]:
        with pytest.raises(ValueError):
            c.output_paths(source,bad_output,bad_cache)


def test_entropy_is_unweighted_and_normalized():
    scores, occupancy = c.column_conservation(['AA', 'AC', 'CC'])
    expected = 1 - (-(2/3)*math.log(2/3)-(1/3)*math.log(1/3))/math.log(20)
    assert np.allclose(scores, expected)
    assert occupancy.tolist() == [1, 1]
    scores, _ = c.column_conservation(list(c.AA))
    assert scores[0] == pytest.approx(0, abs=1e-12)


def test_invalid_or_low_occupancy_columns_are_missing_not_perfectly_conserved():
    scores, occupancy = c.column_conservation(['AX-', '-X-', '-A-'])
    assert np.isnan(scores).all()
    assert np.allclose(occupancy, [1/3, 1/3, 0])


def test_mapping_and_half_valid_residue_rule():
    assert c.residue_to_column('--AC-D-E') == [2, 3, 5, 7]
    assert c.window_score(np.array([.2, np.nan, np.nan]), 0, 3) == (None, 1/3)
    score, coverage = c.window_score(np.array([.2, .4, np.nan]), 0, 3)
    assert score == pytest.approx(.3)
    assert coverage == pytest.approx(2/3)


def test_one_protein_is_one_descriptive_unit_despite_repeated_occurrences():
    rows = [dict(parent_id='a', candidate_score=x, status='scored', valid_position_coverage=1)
            for x in [.1, .5, .9]]
    rows.append(dict(parent_id='b', candidate_score=.9, status='scored', valid_position_coverage=1))
    result = c.candidate_statistics(rows)
    assert result['n_valid_proteins'] == 2
    assert result['mean_conservation'] == pytest.approx(.7)
    assert result['median_conservation'] == pytest.approx(.7)
    assert result['sd_conservation'] == pytest.approx(np.std([.5, .9], ddof=1))
    assert result['status'] == 'scored'


def test_missing_candidate_retained_and_singleton_sd_is_missing():
    row = dict(parent_id='a', candidate_score=None, status='inadequate_valid_positions', valid_position_coverage=0)
    result = c.candidate_statistics([row])
    assert result['n_total_proteins'] == 1 and result['n_valid_proteins'] == 0
    assert result['mean_conservation'] is None
    assert result['status'] == 'no_valid_occurrences'
    result = c.candidate_statistics([{**row, 'candidate_score': 1, 'status': 'scored', 'valid_position_coverage': 1}])
    assert result['mean_conservation'] == 1 and result['sd_conservation'] is None
    assert c.candidate_statistics([])['status'] == 'no_occurrences'


def test_full_length_candidate_needs_no_control_and_invalid_coordinates_stay_explicit(tmp_path, monkeypatch):
    metadata = pd.DataFrame({'protein_id': ['a'], 'sequence': ['ACDE']})
    catalog = pd.DataFrame({'motif_uid': ['full', 'invalid', 'absent']})
    occ = pd.DataFrame({'motif_uid': ['full', 'invalid'], 'parent_id': ['a', 'a'],
                        'occ_start': [0, -1], 'occ_end': [4, 2], 'occurrence_id': ['o1', 'o2']})
    monkeypatch.setattr(c, 'align_family', lambda *args: ({'a': 'ACDE'}, {'n_sequences': 1}))
    rows, _ = c.evaluate_family('test', metadata, catalog, occ, tmp_path,
                               tmp_path/'cache', tmp_path/'unused.bat', 4)
    assert rows[0]['mean_conservation'] == 1
    assert rows[1]['status'] == 'no_valid_occurrences'
    assert rows[2]['status'] == 'no_occurrences'
    saved = pd.read_csv(tmp_path/'occurrence_conservation.csv.gz')
    assert saved.status.tolist() == ['scored', 'invalid_coordinates']
    assert not any(x in ' '.join(saved.columns) for x in ['background', 'difference', 'p_value', 'q_value', 'rank', 'percentile'])


def test_alignment_validation_and_cache_integrity(tmp_path):
    validate_alignment({'a': 'AC'}, {'a': 'A-C'})
    with pytest.raises(ValueError):
        validate_alignment({'a': 'AC'}, {'a': 'A-D'})
    from scripts.candidate_evidence.msa_alignment import alignment_reusable, sha256
    import json
    source, aligned, manifest = [tmp_path/x for x in ['input.fasta', 'alignment.fasta', 'alignment_manifest.json']]
    source.write_bytes(b'>a\r\nAC\r\n'); aligned.write_bytes(b'>a\nA-C\n')
    manifest.write_text(json.dumps({'input_sha256': sha256(source), 'alignment_sha256': sha256(aligned), 'threads': 4}))
    assert alignment_reusable(source, aligned, manifest, '>a\nAC\n', 4)
    assert not alignment_reusable(source, aligned, manifest, '>a\nAD\n', 4)


def test_compressed_result_replacement_preserves_previous_file_on_publish_failure(tmp_path, monkeypatch):
    import os
    path = tmp_path/'scores.csv.gz'
    old = b'previous complete artifact'
    path.write_bytes(old)
    def denied(*args):
        raise OSError('simulated interrupted publish')
    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', denied)
        with pytest.raises(OSError, match='interrupted publish'):
            c.write_compressed_csv(pd.DataFrame({'score': [.2, .4]}), path)
    assert path.read_bytes() == old
    c.write_compressed_csv(pd.DataFrame({'score': [.2, .4]}), path)
    assert pd.read_csv(path).score.tolist() == [.2, .4]
