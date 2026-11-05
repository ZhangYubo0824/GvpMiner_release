#!/usr/bin/env python3
"""Reuse official-model discovery files across families; no model inference.

Each family has qualified guarded-first native sampling. A
successful result is published through family_state.json, never guessed from
the presence of a directory or the sampler's exit code.
"""
from __future__ import annotations
import argparse
import csv
import heapq
from itertools import groupby
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from motifae_candidate_filter import sha256
from motifae_gibbs_results import (_preparation,_sequences,write_json,SCOPE)
from motifae_guarded_batch import run_guarded_batch
from motifae_gibbs_qualification import qualify_installation,verify_qualification
from motifae_native_guard import verified_installation
from motifae_process_control import batch_lock,ProcessCleanupError
from motifae_handoff import _family as validate_handoff_family

CHECKPOINT_SHA='e322c859a593c6c3dba461a2fae6c648ccd064a46422221f05f25c6eac03f28d'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_name(name):
    if not isinstance(name,str) or name in ('.','..') or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name):
        raise ValueError('Unsafe family/run name: '+str(name))
    return name


def _candidate_runs(family_dir):
    result=[]
    for path in [family_dir,*sorted(p for p in family_dir.iterdir() if p.is_dir())]:
        if (path/'gibbs_inputs/gibbs_preparation_manifest.json').is_file():result.append(path)
    return result


def _rank(row):
    return (-float(row['score']),row['parent_id'],int(row['occ_start']),int(row['occ_end']),row['motif_sequence'])


def _raw_source(folder,manifest):
    path=folder/'features/motif_occurrences_MotifAE_raw.csv'
    matching=[s for s in manifest.get('source_files',[]) if Path(str(s['path']).replace('\\','/')).name==path.name]
    if len(matching)!=1 or not path.is_file() or sha256(path)!=matching[0]['sha256']:
        raise ValueError('Original candidate source hash is missing or differs: '+str(path))
    return path


def inspect_discovery(discovery_root,fasta_dir,*,families=None,discovery_run=None,select_run=None):
    root=Path(discovery_root).resolve();fasta_dir=Path(fasta_dir).resolve()
    if not root.is_dir() or not fasta_dir.is_dir():raise FileNotFoundError('Discovery root or FASTA directory missing')
    if discovery_run is not None and select_run is not None:raise ValueError('Choose an explicit discovery run or latest, not both')
    names=list(families) if families is not None else sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not names or len(set(names))!=len(names):raise ValueError('No families, or duplicate family names')
    result=[]
    for name in sorted(names):
        safe_name(name);family_dir=root/name
        if not family_dir.is_dir():raise FileNotFoundError(family_dir)
        runs=_candidate_runs(family_dir)
        if discovery_run is not None:runs=[p for p in runs if p.name==safe_name(discovery_run)]
        if select_run is not None:
            if select_run!='latest':raise ValueError('Supported run selection: latest')
            def order(p):
                m=re.fullmatch('run_([0-9]+)',p.name)
                return (1,int(m[1])) if m else (0,p.name)
            runs=sorted(runs,key=order)[-1:]
        if len(runs)!=1:raise ValueError(f'{name}: missing or multiple ambiguous discovery runs: {[p.name for p in runs]}; use --discovery-run or --select-run latest')
        folder=runs[0];pipeline_path=folder/'pipeline_manifest.json';pipeline=load(pipeline_path)
        if pipeline.get('family')!=name:raise ValueError(name+': pipeline family differs')
        if pipeline.get('status')!='awaiting_gibbscluster_2_0' or pipeline.get('protocol')!='preprint-v1':
            raise ValueError(name+': incomplete or unsupported discovery protocol/status')
        model=pipeline.get('model',{})
        if not model.get('published_checkpoint_verified') or model.get('checkpoint_sha256')!=CHECKPOINT_SHA:
            raise ValueError(name+': discovery does not verify the official published k3 checkpoint')
        fasta=next((fasta_dir/n for n in (name+'_cleaned.fasta',name+'.fasta',name+'_cleaned.fa',name+'.fa') if (fasta_dir/n).is_file()),None)
        if fasta is None:raise FileNotFoundError(name+': family FASTA missing')
        sequences=_sequences(fasta);fasta_hash=sha256(fasta)
        if fasta_hash not in {s.get('sha256') for s in pipeline.get('input',{}).get('source_files',[])}:
            raise ValueError(name+': FASTA hash does not match the discovery input; do not substitute another dataset')
        if pipeline.get('skipped_proteins') or pipeline.get('processed_proteins')!=len(sequences):
            raise ValueError(name+': discovery skipped proteins or processed only part of this FASTA; complete discovery first')
        prep,manifest,index=_preparation(folder/'gibbs_inputs');selected={};seen=set()
        for entry in index:
            uid=entry['motif_uid']
            if uid in seen or entry['family']!=name or entry['method']!='MotifAE':raise ValueError(name+': incorrect or duplicate preparation family/feature')
            seen.add(uid)
            if entry['status']!='ready_for_gibbscluster':raise ValueError(name+': blocked preparation group '+uid)
            with (prep/entry['mapping_file']).open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
            lines=(prep/entry['peptide_file']).read_text(encoding='ascii').splitlines()
            if len(rows)!=int(entry['selected_peptides']) or not 1<=len(rows)<=1000 or [r['motif_sequence'] for r in rows]!=lines:
                raise ValueError(name+': selected peptide/mapping count or order differs')
            for i,row in enumerate(rows,1):
                protein=row['parent_id'];start=int(row['occ_start']);end=int(row['occ_end']);score=float(row['score'])
                if (row['family'],row['method'],row['motif_uid'])!=(name,'MotifAE',uid) or int(row['selection_rank'])!=i:
                    raise ValueError(name+': selected mapping identity/rank differs')
                if protein not in sequences or not 0<=start<end<=len(sequences[protein]) or sequences[protein][start:end]!=row['motif_sequence']:
                    raise ValueError(name+': selected protein sequence/mapping differs')
                if not 5<=end-start<=30 or not math.isfinite(score) or score<=.2:raise ValueError(name+': peptide length/mean does not match declared extraction protocol')
            selected[uid]=rows
        raw=_raw_source(folder,manifest);raw_groups=set();raw_count=0
        # The official pipeline exports feature-grouped rows. Streaming nsmallest
        # bounds memory to Top1000, without changing selection or deduplicating.
        with raw.open(encoding='utf-8-sig',newline='') as handle:
            for uid,members in groupby(csv.DictReader(handle),lambda r:r['motif_uid']):
                if uid in raw_groups or uid not in selected:raise ValueError(name+': candidate source feature grouping/coverage differs')
                raw_groups.add(uid);count=[0]
                def checked_members():
                    for row in members:
                        count[0]+=1
                        if (row['family'],row['method'])!=(name,'MotifAE'):raise ValueError(name+': raw source family differs')
                        yield row
                expected=heapq.nsmallest(1000,checked_members(),key=_rank)
                actual=selected[uid]
                keys=('family','method','motif_uid','parent_id','sequence_id','occ_start','occ_end','motif_sequence')
                if len(actual)!=len(expected) or any(any(str(a[k])!=str(b[k]) for k in keys) or float(a['score'])!=float(b['score']) for a,b in zip(actual,expected)):
                    raise ValueError(name+': Top1000 mapping/ranking differs from original candidates')
                entry=next(e for e in index if e['motif_uid']==uid)
                if count[0]!=int(entry['total_peptides']):raise ValueError(name+': raw candidate count differs')
                raw_count+=count[0]
        if raw_groups!=seen or pipeline.get('feature_count')!=len(index) or pipeline.get('peptide_count')!=raw_count:
            raise ValueError(name+': candidate/preparation/pipeline totals differ')
        result.append(dict(family=name,discovery_dir=str(folder),preparation_dir=str(prep),fasta=str(fasta),proteins=len(sequences),max_protein_length=max(map(len,sequences.values())),groups=len(index),candidate_instances=raw_count,selected_instances=sum(len(v) for v in selected.values()),signature=dict(preparation_sha256=sha256(prep/'gibbs_preparation_manifest.json'),fasta_sha256=fasta_hash,pipeline_manifest_sha256=sha256(pipeline_path),raw_candidates_sha256=sha256(raw))))
        print(f'[INPUT] {name}: {len(sequences)} proteins, {len(index)} groups, {raw_count} candidates verified',file=sys.stderr,flush=True)
    return result


def check_runtime(gibbs_root):
    if platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64'):
        raise RuntimeError('Actual Gibbs execution requires the verified Linux x86_64 installation')
    root=verified_installation(gibbs_root)
    for command in ('gdb','tcsh','perl'):
        if shutil.which(command) is None:raise FileNotFoundError('Required runtime command missing: '+command)
    wrapper=root/'gibbscluster'
    if not wrapper.is_file() or not os.access(wrapper,os.X_OK):raise FileNotFoundError('Executable Gibbs wrapper missing: '+str(wrapper))
    return root


def _code_signature():
    return {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}


def _separate(output,inputs):
    output=Path(output).resolve()
    for source in inputs:
        source=Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Keep alignment output separate from input/program directories; overlapping paths are forbidden')


def _inventory(directory):
    root=Path(directory).resolve();result={}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Alignment results cannot contain symlinks')
        if path.is_file() and path.name!='.batch.lock':result[path.relative_to(root).as_posix()]=sha256(path)
    return result


def validate_alignment(case,batch_dir):
    family,sequences,occurrences,catalog,summary=validate_handoff_family(dict(family=case['family'],batch_dir=batch_dir,preparation_dir=case['preparation_dir'],fasta=case['fasta']))
    manifest=load(Path(batch_dir)/'batch_manifest.json')
    if manifest.get('status')!='alignment_samples_parsed' or len(manifest['jobs'])!=case['groups'] or manifest.get('blocked_groups',0):
        raise ValueError(family+': final alignment is incomplete')
    return dict(summary,final_result_dir=str(Path(batch_dir).resolve()))


def align_families(discovery_root,fasta_dir,alignment_root,gibbs_root,run_name,qualification_dir,*,families=None,discovery_run=None,select_run=None,resume=False,job_timeout=300,heartbeat=15,retry_failed=False,check_runtime=True):
    safe_name(run_name);output=Path(alignment_root).resolve();gibbs=Path(gibbs_root).resolve();qualification=Path(qualification_dir).resolve()
    _separate(output,[discovery_root,fasta_dir,gibbs])
    if any(not math.isfinite(v) or v<=0 for v in (job_timeout,heartbeat)):
        raise ValueError('Timeouts and heartbeat must be finite positive seconds')
    cases=inspect_discovery(discovery_root,fasta_dir,families=families,discovery_run=discovery_run,select_run=select_run)
    if check_runtime:globals()['check_runtime'](gibbs)
    verify_qualification(qualification,gibbs_root=gibbs)
    signature=dict(code_sha256=_code_signature(),qualification_manifest_sha256=sha256(qualification/'qualification_manifest.json'),inputs={c['family']:c['signature'] for c in cases},execution_mode='guarded-first')
    runhome=output/'_runs'/run_name;manifest_path=runhome/'orchestration_manifest.json'
    with batch_lock(runhome):
        if manifest_path.exists():
            if not resume:raise FileExistsError('Run exists; choose a new --run-name or use validated --resume')
            if load(manifest_path).get('signature')!=signature:raise ValueError('Run code, qualification or inputs changed; use a new run name')
        state=dict(status='running',signature=signature,run_name=run_name,localization_scope=SCOPE,families=[],model_inference_executed=False,execution_mode='guarded-first')
        write_json(manifest_path,state)
        for case in cases:
            name=case['family'];unit=output/name/run_name;family_state=unit/'family_state.json'
            unit.mkdir(parents=True,exist_ok=True)
            entry=dict(family=name,status='running',inputs=case,signature=signature)
            try:
                with batch_lock(unit):
                    old=load(family_state) if family_state.is_file() else None
                    if old and old.get('signature')!=signature:raise ValueError('Family input/code/qualification signature changed')
                    if old and old.get('status')=='completed':
                        final=Path(old['final_result_dir'])
                        if not final.is_relative_to(unit):raise ValueError('Family result pointer escapes its output')
                        if _inventory(final)!=old['files_sha256']:raise ValueError('Completed family output hashes changed; preserve it and use a new run name')
                        summary=validate_alignment(case,final);entry=dict(old,resumed=True)
                    else:
                        write_json(family_state,entry)
                        final=unit/'guarded'
                        run_guarded_batch(case['preparation_dir'],gibbs,case['fasta'],final,qualification,
                            resume=resume and (final/'batch_manifest.json').is_file(),job_timeout=job_timeout,
                            heartbeat=heartbeat,retry_failed=retry_failed)
                        summary=validate_alignment(case,final)
                        # Recheck input identities after external work.
                        for key,path in [('fasta_sha256',case['fasta']),('preparation_sha256',Path(case['preparation_dir'])/'gibbs_preparation_manifest.json'),('pipeline_manifest_sha256',Path(case['discovery_dir'])/'pipeline_manifest.json'),('raw_candidates_sha256',Path(case['discovery_dir'])/'features/motif_occurrences_MotifAE_raw.csv')]:
                            if sha256(path)!=case['signature'][key]:raise ValueError('Family input changed during alignment')
                        entry.update(status='completed',**{k:v for k,v in summary.items() if k!='status'},files_sha256=_inventory(final))
                        write_json(family_state,entry)
                print(f'[FAMILY] {name}: completed; {entry.get("motifs",0)} patterns; {entry.get("occurrences",0)} mapped',file=sys.stderr,flush=True)
            except (KeyboardInterrupt,ProcessCleanupError) as exc:
                entry.update(status='interrupted_or_cleanup_unconfirmed',error=f'{type(exc).__name__}: {exc}')
                write_json(family_state,entry);state['families'].append({k:v for k,v in entry.items() if k not in ('files_sha256','signature')});state['status']='interrupted_or_cleanup_unconfirmed';write_json(manifest_path,state);raise
            except Exception as exc:
                entry.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                write_json(family_state,entry)
                print(f'[FAMILY] {name}: failed: {exc}',file=sys.stderr,flush=True)
            state['families'].append({k:v for k,v in entry.items() if k not in ('files_sha256','signature')})
            write_json(manifest_path,state)
        state['successful_families']=sum(e['status']=='completed' for e in state['families'])
        state['failed_families']=len(cases)-state['successful_families']
        state['status']='all_families_aligned' if state['failed_families']==0 else 'incomplete_families'
        state['biological_function_validated']=False
        write_json(manifest_path,state)
        family_runs=[dict(family=e['family'],batch_dir=e['final_result_dir'],preparation_dir=e['inputs']['preparation_dir'],fasta=e['inputs']['fasta']) for e in state['families'] if e['status']=='completed']
        write_json(runhome/'family_runs.json',family_runs if not state['failed_families'] else
                   dict(status='incomplete_families',reason='Resolve failed selected families before exporting a complete handoff',successful_family_runs=family_runs))
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    qualify=sub.add_parser('qualify');qualify.add_argument('--gibbs-root',type=Path,required=True);qualify.add_argument('--output-dir',type=Path,required=True);qualify.add_argument('--timeout',type=float,default=300);qualify.add_argument('--reuse',action='store_true',help='Verify an already successful qualification instead of executing it again')
    for name in ('preflight','run'):
        p=sub.add_parser(name)
        p.add_argument('--discovery-root',type=Path,required=True);p.add_argument('--fasta-dir',type=Path,required=True)
        p.add_argument('--families',nargs='+');selection=p.add_mutually_exclusive_group();selection.add_argument('--discovery-run');selection.add_argument('--select-run',choices=['latest'])
        p.add_argument('--gibbs-root',type=Path,required=name=='run')
        if name=='preflight':p.add_argument('--input-only',action='store_true');p.add_argument('--report',type=Path)
        else:
            p.add_argument('--alignment-root',type=Path,required=True);p.add_argument('--run-name',required=True);p.add_argument('--qualification-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--retry-failed',action='store_true');p.add_argument('--job-timeout',type=float,default=300);p.add_argument('--heartbeat',type=float,default=15)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='qualify':
        reuse=args.pop('reuse')
        result=verify_qualification(args['output_dir'],gibbs_root=args['gibbs_root']) if reuse and (args['output_dir']/'qualification_manifest.json').is_file() else qualify_installation(**args)
    elif command=='preflight':
        report=args.pop('report');inputs_only=args.pop('input_only');gibbs=args.pop('gibbs_root')
        result=dict(status='inputs_verified',families=inspect_discovery(**args),model_inference_required=False)
        if not inputs_only:
            if gibbs is None:raise ValueError('Use --gibbs-root for runtime checks, or explicit --input-only')
            check_runtime(gibbs);result['runtime_verified']=True
        if report:write_json(report,result)
    else:result=align_families(**args)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('failed_families',0):raise SystemExit(2)


if __name__=='__main__':main()
