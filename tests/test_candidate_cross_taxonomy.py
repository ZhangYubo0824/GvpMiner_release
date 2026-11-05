import importlib

import pandas as pd
import pytest


def module(name):
    try:
        return importlib.import_module('scripts.candidate_evidence.' + name)
    except ModuleNotFoundError:
        pytest.fail(f'{name} analysis is not implemented')


def test_residue_unions_do_not_double_count_or_cross_proteins():
    cm = module('cross_model')
    left = cm.residue_unions([('p', 0, 20), ('p', 15, 35), ('q', 0, 10)])
    right = cm.residue_unions([('p', 10, 30), ('r', 0, 10)])
    scores = cm.compare_unions(left, right)
    assert scores['left_residues'] == 45
    assert scores['right_residues'] == 30
    assert scores['intersection_residues'] == 20
    assert scores['residue_jaccard'] == pytest.approx(20 / 55)
    assert scores['left_coverage'] == pytest.approx(20 / 45)
    assert scores['right_coverage'] == pytest.approx(20 / 30)
    assert scores['protein_jaccard'] == pytest.approx(1 / 3)


def test_half_open_intervals_and_invalid_input():
    cm = module('cross_model')
    assert cm.compare_unions(cm.residue_unions([('p', 0, 20)]),
                             cm.residue_unions([('p', 20, 40)]))['intersection_residues'] == 0
    with pytest.raises(ValueError):
        cm.residue_unions([('p', -1, 20)])
    with pytest.raises(ValueError):
        cm.residue_unions([('p', 20, 20)])


def test_matching_retains_split_and_no_match_candidates():
    cm = module('cross_model')
    left = {'a': cm.residue_unions([('p', 0, 40)]),
            'b': cm.residue_unions([('z', 0, 20)])}
    right = {'x': cm.residue_unions([('p', 0, 20)]),
             'y': cm.residue_unions([('p', 20, 40)])}
    candidates, pairs = cm.match_family('GvpA', left, right)
    a = next(r for r in candidates if r['motif_uid'] == 'a')
    b = next(r for r in candidates if r['motif_uid'] == 'b')
    assert len(candidates) == 4
    assert a['partner_count_j050'] == 2
    assert a['best_match_tie_count'] == 2
    assert a['best_match_uid'] == 'x'
    assert a['reciprocal_best']
    assert a['hungarian_match_j050']
    assert not b['hungarian_match_j030']
    assert b['status'] == 'no_residue_overlap'
    assert sum(r['hungarian_match_j050'] for r in pairs) == 1


def test_thresholded_assignment_does_not_let_subthreshold_edges_steal_matches():
    cm = module('cross_model')
    # Unconstrained assignment prefers .49+.49; thresholded matching must retain .6.
    assert cm.hungarian_matches([[.6, .49], [.49, 0]], .5) == {(0, 0)}


def test_taxonomy_uses_binomial_and_preserves_uncertainty():
    tx = module('taxonomy')
    known = tx.parse_organism('Haloferax mediterranei ATCC 33500', 'Archaea; Halobacteria; Haloferax')
    assert known['recognized_species'] == 'Haloferax mediterranei'
    assert known['recognized_genus'] == 'Haloferax'
    unresolved = tx.parse_organism('Bacillus sp. ABC', 'Bacteria; Bacillota; Bacillus')
    assert unresolved['recognized_species'] is None
    assert unresolved['recognized_genus'] == 'Bacillus'
    assert unresolved['organism_status'] == 'unresolved_species'
    unknown = tx.parse_organism('uncultured bacterium', 'Bacteria; Bacillota')
    assert unknown['recognized_species'] is None
    assert unknown['recognized_genus'] is None  # No guessed genus from last lineage token.
    assert tx.parse_organism('', '')['organism_status'] == 'missing_organism'
    assert tx.parse_organism('Candidatus Microcystis aeruginosa', '')['recognized_species'] == 'Candidatus Microcystis aeruginosa'
    assert tx.parse_organism('Thermoleophilia bacterium SCSIO 60948', '')['recognized_genus'] is None
    assert tx.parse_organism('Gammaproteobacteria', '')['recognized_genus'] is None
    assert tx.parse_organism('Oscillatoriophycideae', '')['recognized_genus'] is None
    assert tx.parse_organism('Bacillus group bacterium X', '')['recognized_species'] is None


def test_taxonomic_coverage_has_species_and_protein_denominators():
    tx = module('taxonomy')
    frame = pd.DataFrame([
        {'protein_id': 'a', 'family': 'GvpA', 'organism': 'Bacillus subtilis strain 1', 'taxonomy': 'Bacteria; Bacillus'},
        {'protein_id': 'b', 'family': 'GvpA', 'organism': 'Bacillus subtilis strain 2', 'taxonomy': 'Bacteria; Bacillus'},
        {'protein_id': 'c', 'family': 'GvpA', 'organism': 'Bacillus cereus', 'taxonomy': 'Bacteria; Bacillus'},
        {'protein_id': 'd', 'family': 'GvpA', 'organism': 'Bacillus sp. X', 'taxonomy': ''},
    ])
    prepared = tx.prepare_metadata(frame)
    result = tx.candidate_coverage('GvpA', 'm', ['a', 'b', 'd', 'absent'], prepared)
    assert result['n_proteins'] == 4
    assert result['n_recognized_species'] == 1
    assert result['n_recognized_genera'] == 1
    assert result['species_background_coverage'] == .5
    assert result['dominant_species_share_all_proteins'] == .5
    assert result['dominant_species_share_resolved_proteins'] == 1
    assert result['n_missing_metadata'] == 1
    assert result['n_unresolved_species'] == 1
    assert result['protein_background_coverage'] == .75


def test_outputs_cannot_be_written_inside_source(tmp_path):
    cm = module('cross_model')
    with pytest.raises(ValueError):
        cm.safe_output_root(tmp_path, tmp_path / 'results')


def test_duplicate_metadata_is_rejected():
    tx = module('taxonomy')
    row = {'protein_id': 'a', 'family': 'GvpA', 'organism': '', 'taxonomy': ''}
    with pytest.raises(ValueError):
        tx.prepare_metadata(pd.DataFrame([row, row]))


def test_frozen_loader_accepts_occurrence_ids_reused_on_different_proteins(tmp_path, monkeypatch):
    cm = module('cross_model')
    monkeypatch.setattr(cm, 'FROZEN', 'frozen')
    folder = tmp_path / cm.FROZEN / 'GvpA'
    folder.mkdir(parents=True)
    pd.DataFrame([{'family': 'GvpA', 'motif_uid': 'm'}]).to_csv(folder / 'motif_catalog_M2.csv', index=False)
    rows = [{'family': 'GvpA', 'motif_uid': 'm', 'occurrence_id': 'm_occ_001',
             'parent_id': p, 'occ_start': 0, 'occ_end': 20} for p in ['p', 'q']]
    pd.DataFrame(rows).to_csv(folder / 'motif_occurrences_M2.csv', index=False)
    _, data, _ = cm.load_frozen(tmp_path, methods=('M2',))
    assert set(data['GvpA', 'M2'][2]['m']) == {'p', 'q'}


def test_family_summaries_retain_empty_evaluated_families():
    cm = module('cross_model')
    rows, _ = cm.match_family('GvpA', {'m': cm.residue_unions([('p', 0, 20)])}, {})
    summary = cm.summarize_candidates(pd.DataFrame(rows), families=['GvpA', 'GvpH'])
    assert len(summary) == 4
    assert summary.loc[(summary.family == 'GvpH') & (summary.method == 'M2'), 'n_candidates'].iloc[0] == 0
    tx = module('taxonomy')
    row = tx.candidate_coverage('GvpA', 'm', ['p'], tx.prepare_metadata(pd.DataFrame([
        {'family': 'GvpA', 'protein_id': 'p', 'organism': 'Bacillus subtilis', 'taxonomy': ''}])))
    assert tx._candidate_summary(pd.DataFrame([row]), families=['GvpA', 'GvpH']).set_index('family').loc['GvpH', 'n_candidates'] == 0
