import csv, json, sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from exp4_motif_structure_analysis.exp3_adapter import build_exp3_motif_rows
from exp4_motif_structure_analysis.features import compute_interval_features
from exp4_motif_structure_analysis.parsing import ParsedStructure
from exp4_motif_structure_analysis.controls import build_composition_window_index, build_composition_feature_index, sample_composition_matched_windows_from_feature_index
from exp4_motif_structure_analysis.pipeline import run_exp4_motif_structure_analysis
from exp4_motif_structure_analysis.rmsd import backfill_local_rmsd_resumable


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def parsed(sequence='GWKRAA'):
    return ParsedStructure(sequence, np.array([90,0,90,90,90,90]), np.array([[0,0,0],[999,999,999],[1,0,0],[2,0,0],[3,0,0],[4,0,0]], dtype=float), ['H','E','H','H','H','H'], Path('test.json'), np.array([.2,99,.2,.2,.2,.2]))


def test_adapter_preserves_actual_positions_and_null(tmp_path):
    path=tmp_path/'F'/'motif_occurrences_MotifAE.csv'
    write_csv(path,[dict(parent_id='p',motif_uid='m',occ_start=0,occ_end=6,core_residue_positions_0based='[0, 2, null, 4, 5]',aligned_core='GK-AA')])
    rows=build_exp3_motif_rows(exp3_dir=tmp_path, sequence_rows=[dict(sequence_id='p',sequence='GWKRAA')],families=['F'],methods=['MotifAE'])
    assert rows[0]['motif_sequence']=='GKAA'
    assert json.loads(rows[0]['core_residue_positions_0based'])==[0,2,None,4,5]
    assert rows[0]['aligned_core']=='GK-AA'
    assert rows[0]['motif_observed_length']==4
    assert rows[0]['motif_span_length']==6


def test_features_exclude_extreme_skipped_residue_and_use_original_separation():
    f=compute_interval_features(parsed(),0,6,residue_positions=[0,2,3,4,5],long_range_min_separation=5)
    assert f['motif_sequence']=='GKRAA'
    assert f['motif_length']==5
    assert f['motif_mean_pLDDT']==90
    assert f['relative_SASA']==pytest.approx(.2)
    assert f['helix_fraction']==1
    assert f['contact_density']==1
    assert f['long_range_contact_count']==1
    assert f['radius_of_gyration']==pytest.approx(np.sqrt(2))


def test_null_is_missing_alignment_slot_not_a_residue():
    f=compute_interval_features(parsed(),0,6,residue_positions=[0,2,None,4,5])
    assert f['motif_sequence']=='GKAA'
    assert f['motif_length']==4
    assert f['motif_observed_length']==4
    assert json.loads(f['core_residue_positions_0based'])==[0,2,None,4,5]


def test_contiguous_compatibility():
    f=compute_interval_features(parsed(),2,6)
    assert f['motif_sequence']=='KRAA'
    assert f['motif_mean_pLDDT']==90
    assert f['motif_length']==4


def test_composition_index_compares_layout_core_not_span():
    sequences={'a':dict(family='F',sequence='GWKRAA'), 'b':dict(family='F',sequence='GKRAAW')}
    idx=build_composition_window_index(sequences=sequences,excluded_by_sequence={},window_lengths=[6],residue_offset_layouts=[(0,2,3,4,5)])
    out=sample_composition_matched_windows_from_feature_index(motif_sequence='GKRAA',motif_family='F',window_length=6,residue_offsets=(0,2,3,4,5),index=build_composition_feature_index(idx),num_controls=1)
    assert out==[('a',(0,6))]


def fixture_pipeline(tmp_path):
    sequences=[dict(sequence_id='p',family='F',sequence='GWKRAAGWKRAAGWKRAA'),dict(sequence_id='q',family='F',sequence='GWKRAAGWKRAAGWKRAA')]
    table=tmp_path/'sequence.csv'; write_csv(table,sequences)
    structures=tmp_path/'structures'; structures.mkdir()
    for row in sequences:
        n=len(row['sequence']); (structures/(row['sequence_id']+'.json')).write_text(json.dumps(dict(sequence=row['sequence'],plddt=[90,0,90,90,90,90]*3,ca_coords=[[i,0,0] for i in range(n)],secondary_structure=['H','E','H','H','H','H']*3,relative_sasa=[.2,99,.2,.2,.2,.2]*3)))
    motifs=[dict(sequence_id='p',family='F',method='MotifAE',motif_start=0,motif_end=6,motif_id='m',motif_instance_id='m:p',core_residue_positions_0based='[0,2,3,4,5]',aligned_core='GKRAA')]
    return dict(sequence_table=table,structure_dir=structures,output_dir=tmp_path/'out',motif_rows_override=motifs,num_random_controls=1,show_progress=False,generate_figures=False)


def test_minimal_end_to_end_outputs_and_controls_use_same_layout(tmp_path):
    kw=fixture_pipeline(tmp_path); summary=run_exp4_motif_structure_analysis(**kw)
    with (kw['output_dir']/'motif_structure_features.csv').open() as f: rows=list(csv.DictReader(f))
    assert summary['motif_instances_analyzed']==1
    assert rows[0]['motif_sequence']=='GKRAA'
    assert rows[0]['motif_length']=='5'
    assert rows[0]['motif_span_length']=='6'
    assert rows[0]['aligned_core']=='GKRAA'
    with (kw['output_dir']/'matched_control_features.csv').open() as f: controls=list(csv.DictReader(f))
    assert {c['control_type'] for c in controls}=={'within_random','position_matched','composition_matched'}
    for row in controls:
        start=int(row['control_start']); pos=json.loads(row['core_residue_positions_0based'])
        assert [p-start for p in pos]==[0,2,3,4,5]
        assert int(row['control_end'])-start==6
        assert len(row['motif_sequence'])==5


def test_resume_rejects_changed_layout_with_same_ids(tmp_path):
    kw=fixture_pipeline(tmp_path); run_exp4_motif_structure_analysis(**kw)
    progress=kw['output_dir']/'analysis_progress.json'; payload=json.loads(progress.read_text()); payload['stage']='local_rmsd_complete'; progress.write_text(json.dumps(payload))
    kw['motif_rows_override'][0]['core_residue_positions_0based']='[0,1,3,4,5]'
    with pytest.raises(ValueError,match='signature|semantics|layout'):
        run_exp4_motif_structure_analysis(**kw,resume=True)


def test_rmsd_uses_actual_core_positions_even_with_default_flank(tmp_path):
    p=parsed(); q=ParsedStructure(p.sequence,p.plddt,p.ca_coords.copy(),p.secondary_structure,Path('q.json'),p.relative_sasa)
    q.ca_coords[1]=[-999,42,0]
    rows=[dict(sequence_id=s,motif_start=0,motif_end=6,motif_cluster='m',motif_instance_id=s,core_residue_positions_0based='[0,2,3,4,5]') for s in ['p','q']]
    backfill_local_rmsd_resumable(rows,dict(p=p,q=q),tmp_path,show_progress=False,reuse_checkpoints=False)
    assert rows[0]['local_RMSD_or_structural_similarity']==0

def test_resume_accepts_unchanged_core_layout_and_rmsd_csv_roundtrip(tmp_path):
    kw=fixture_pipeline(tmp_path); run_exp4_motif_structure_analysis(**kw)
    progress=kw['output_dir']/'analysis_progress.json'; payload=json.loads(progress.read_text()); payload['stage']='features_and_controls_complete'; progress.write_text(json.dumps(payload))
    result=run_exp4_motif_structure_analysis(**kw,resume=True)
    assert result['motif_instances_analyzed']==1


def test_same_span_different_offsets_do_not_share_feature_cache(tmp_path):
    kw=fixture_pipeline(tmp_path)
    second=dict(kw['motif_rows_override'][0]); second.update(motif_instance_id='other',core_residue_positions_0based='[0,1,3,4,5]',aligned_core='GWRAA')
    kw['motif_rows_override'].append(second); run_exp4_motif_structure_analysis(**kw)
    with (kw['output_dir']/'motif_structure_features.csv').open() as f: a,b=list(csv.DictReader(f))
    assert a['motif_sequence']=='GKRAA'
    assert b['motif_sequence']=='GWRAA'
    assert float(a['motif_mean_pLDDT'])==90
    assert float(b['motif_mean_pLDDT'])==72


def test_rmsd_null_slots_compare_shared_alignment_columns(tmp_path):
    p=parsed(); q=ParsedStructure(p.sequence,p.plddt,p.ca_coords.copy(),p.secondary_structure,Path('q.json'),p.relative_sasa)
    rows=[dict(sequence_id='p',motif_start=0,motif_end=6,motif_cluster='m',motif_instance_id='p',core_residue_positions_0based='[0,2,null,4,5]'), dict(sequence_id='q',motif_start=0,motif_end=6,motif_cluster='m',motif_instance_id='q',core_residue_positions_0based='[0,null,3,4,5]')]
    backfill_local_rmsd_resumable(rows,dict(p=p,q=q),tmp_path,show_progress=False,reuse_checkpoints=False)
    assert rows[0]['local_RMSD_or_structural_similarity']==0


def test_missing_structure_remains_explicitly_skipped(tmp_path):
    kw=fixture_pipeline(tmp_path); (kw['structure_dir']/'p.json').unlink()
    result=run_exp4_motif_structure_analysis(**kw)
    assert result['motif_instances_analyzed']==0
    assert result['skipped_motif_instances']==1
    with (kw['output_dir']/'skipped_motif_instances.csv').open() as f: rows=list(csv.DictReader(f))
    assert 'missing' in rows[0]['reason']


def test_cli_completed_resume_cannot_bypass_exact_input_signature(tmp_path,monkeypatch):
    from src.exp4_motif_structure_analysis import cli
    import argparse
    kw=fixture_pipeline(tmp_path)
    out=tmp_path/'cliout'/'F'/'MotifAE';out.mkdir(parents=True)
    (out/'analysis_progress.json').write_text(json.dumps(dict(complete=True,stage='complete')))
    (out/'analysis_summary.json').write_text(json.dumps(dict(motif_instances_analyzed=1)))
    monkeypatch.setattr(cli,'build_exp3_motif_rows',lambda **kwargs:kw['motif_rows_override'])
    def verify_signature(**kwargs):raise ValueError('exact input signature changed')
    monkeypatch.setattr(cli,'run_exp4_motif_structure_analysis',verify_signature)
    args=argparse.Namespace(structure_dir=kw['structure_dir'],structure_root=None,prediction_backend='esmfold',predict_missing_structures=False,resume=True,num_random_controls=1,seed=42,contact_threshold=8.,long_range_min_separation=12,low_plddt_cutoff=70.,rmsd_workers=1,no_progress=True,generate_figures=False,figure_formats=['png'])
    with pytest.raises(ValueError,match='signature'):
        cli._run_one_exp3_unit(args,sequence_table=kw['sequence_table'],sequence_rows=[],exp3_dir=tmp_path/'handoff',output_root=tmp_path/'cliout',family='F',method='MotifAE')


def test_rejects_same_id_structure_for_a_different_sequence(tmp_path):
    kw=fixture_pipeline(tmp_path)
    path=kw['structure_dir']/'p.json';payload=json.loads(path.read_text());payload['sequence']='A'+payload['sequence'][1:];path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='sequence.*mismatch|mismatch.*sequence'):
        run_exp4_motif_structure_analysis(**kw)


def test_adapter_does_not_silently_drop_motifae_missing_sequence(tmp_path):
    path=tmp_path/'F'/'motif_occurrences_MotifAE.csv'
    write_csv(path,[dict(parent_id='missing',motif_uid='m',occ_start=0,occ_end=5,core_residue_positions_0based='[0,1,2,3,4]',aligned_core='ACDEF')])
    with pytest.raises(ValueError,match='sequence|protein'):
        build_exp3_motif_rows(exp3_dir=tmp_path,sequence_rows=[dict(sequence_id='other',sequence='ACDEF')],families=['F'],methods=['MotifAE'])


def test_all_trash_family_exp4_reports_no_instances_without_requiring_structures(tmp_path):
    from src.exp4_motif_structure_analysis import cli
    import argparse
    occurrence=tmp_path/'handoff/F/MotifAE/motif_occurrences_MotifAE.csv';occurrence.parent.mkdir(parents=True)
    occurrence.write_text('family,method,motif_uid,parent_id,occ_start,occ_end\n')
    args=argparse.Namespace(structure_dir=tmp_path/'missing_structures',structure_root=None,prediction_backend='esmfold',predict_missing_structures=False,resume=False)
    result=cli._run_one_exp3_unit(args,sequence_table=tmp_path/'not_needed.csv',sequence_rows=[dict(sequence_id='p',family='F',sequence='ACDEF')],exp3_dir=tmp_path/'handoff',output_root=tmp_path/'out',family='F',method='MotifAE')
    assert result['status']=='no_motif_instances'
    assert result['motif_instances_analyzed']==0
    assert (tmp_path/'out/F/MotifAE/motif_structure_features.csv').is_file()


def test_shared_accession_does_not_change_current_family_control_background(tmp_path):
    kw=fixture_pipeline(tmp_path)
    with kw['sequence_table'].open() as h:rows=list(csv.DictReader(h))
    rows[1]['sequence']='L'*18
    q=kw['structure_dir']/'q.json';payload=json.loads(q.read_text());payload['sequence']='L'*18;q.write_text(json.dumps(payload))
    rows.append(dict(rows[0],family='OtherFamily'))
    write_csv(kw['sequence_table'],rows)
    run_exp4_motif_structure_analysis(**kw)
    with (kw['output_dir']/'matched_control_features.csv').open() as h:controls=list(csv.DictReader(h))
    composition=[r for r in controls if r['control_type']=='composition_matched']
    assert composition[0]['control_sequence_id']=='p'


def test_exact_completed_analysis_can_resume_without_changing_results(tmp_path):
    kw=fixture_pipeline(tmp_path);first=run_exp4_motif_structure_analysis(**kw)
    second=run_exp4_motif_structure_analysis(**kw,resume=True)
    assert first['motif_instances_analyzed']==second['motif_instances_analyzed']==1


def test_same_sequence_changed_structure_invalidates_resume(tmp_path):
    kw=fixture_pipeline(tmp_path);run_exp4_motif_structure_analysis(**kw)
    p=kw['output_dir']/'analysis_progress.json';meta=json.loads(p.read_text());meta['stage']='features_and_controls_complete';p.write_text(json.dumps(meta))
    p=kw['structure_dir']/'p.json';meta=json.loads(p.read_text());meta['plddt']=[10]*18;p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='signature|structure'):
        run_exp4_motif_structure_analysis(**kw,resume=True)


def test_feature_and_summary_keep_sampling_scope_and_guard_source(tmp_path):
    kw=fixture_pipeline(tmp_path)
    kw['motif_rows_override'][0].update(localization_scope='selected_top1000_alignment_instances',sampler_kind='native_empty_g1_move_guard',guard_provenance='{"indel_moves_skipped":1}')
    summary=run_exp4_motif_structure_analysis(**kw)
    with (kw['output_dir']/'motif_structure_features.csv').open() as h:row=next(csv.DictReader(h))
    assert row['localization_scope']=='selected_top1000_alignment_instances'
    assert row['sampler_kind']=='native_empty_g1_move_guard'
    assert summary['localization_scopes']==['selected_top1000_alignment_instances']
    assert summary['contains_guarded_sampler_results'] is True


def test_rmsd_checkpoint_is_bound_to_actual_structure_coordinates(tmp_path):
    p=parsed();q=ParsedStructure(p.sequence,p.plddt,p.ca_coords.copy(),p.secondary_structure,Path('q.json'),p.relative_sasa)
    rows=[dict(sequence_id=s,motif_start=0,motif_end=6,motif_cluster='m',motif_instance_id=s,core_residue_positions_0based='[0,2,3,4,5]') for s in ['p','q']]
    backfill_local_rmsd_resumable(rows,dict(p=p,q=q),tmp_path,show_progress=False,reuse_checkpoints=False)
    q.ca_coords[2]=[999,7,0]
    with pytest.raises(ValueError,match='signature'):
        backfill_local_rmsd_resumable(rows,dict(p=p,q=q),tmp_path,show_progress=False,reuse_checkpoints=True)
