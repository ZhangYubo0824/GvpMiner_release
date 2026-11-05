"""Qualified native sampling on the first attempt; no unguarded timeout pass."""
from __future__ import annotations
import json
import math
from pathlib import Path
import re

from motifae_candidate_filter import sha256
from motifae_gibbs_results import (_preparation,_sequences,_import,_resume_job,
    publish_batch,write_json,SCOPE)
from motifae_native_guard import run_guarded_native,verified_installation,INSTALL_SHA,code_signature
from motifae_gibbs_qualification import verify_qualification
from motifae_process_control import (batch_lock,termination_signals,attempt_has_live_processes,
    ProcessCleanupError,JobTimedOut)


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _signature(prep,fasta,qualification):
    code=code_signature()
    code[Path(__file__).name]=sha256(Path(__file__))
    return dict(preparation_sha256=sha256(prep/'gibbs_preparation_manifest.json'),
        fasta_sha256=sha256(fasta),qualification_manifest_sha256=sha256(qualification/'qualification_manifest.json'),
        code_sha256=code,installation_sha256=INSTALL_SHA.copy(),
        localization_scope=SCOPE,execution_mode='guarded-first')


def _previous(path,signature,entry,stem,folder,output,retry_failed):
    old=_load(path)
    if (old.get('signature')!=signature or old.get('feature_stem')!=stem
            or old.get('motif_uid')!=entry['motif_uid']):
        raise ValueError('Previous job identity/signature changed; use a new run')
    attempt=(output/old.get('attempt','')).resolve()
    if attempt.parent!=folder.resolve() or not re.fullmatch(r'attempt_[0-9]{4}',attempt.name):
        raise ValueError('Previous attempt path is invalid')
    if attempt_has_live_processes(attempt):
        raise ProcessCleanupError('Previous guarded attempt still has live processes: '+str(attempt))
    if old.get('status')=='success':
        verified=_resume_job(old,signature,entry,stem,folder,output)
        if verified is None:raise ValueError('Previously successful job hashes changed; preserve evidence and use a new run')
        event_path=attempt/'guard_event.json'
        if str(event_path.relative_to(output)) not in old['files_sha256']:
            raise ValueError('Verified guard provenance is missing')
        verified['guard_event']=_load(event_path)
        verified['sampler_kind']='version_locked_gdb_empty_g1_guard'
        verified['engineering_fix']='empty_g1_move_guard'
        verified['resumed']=True
        return verified
    if old.get('status') in ('failed','timed_out') and not retry_failed:return old
    return None


def run_guarded_batch(preparation_dir,gibbs_root,fasta,output_dir,qualification_dir,*,
                      resume=False,job_timeout=300,heartbeat=15,retry_failed=False):
    if any(not math.isfinite(v) or v<=0 for v in (job_timeout,heartbeat)):
        raise ValueError('Guarded runtime limits must be finite and positive')
    prep,manifest,index=_preparation(preparation_dir)
    root=verified_installation(gibbs_root);fasta=Path(fasta).resolve()
    output=Path(output_dir).resolve();qualification=Path(qualification_dir).resolve()
    for source in (prep,root,fasta,qualification):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Keep guarded output separate from input/program/qualification paths')
    verify_qualification(qualification,gibbs_root=root)
    sequences=_sequences(fasta);signature=_signature(prep,fasta,qualification)
    state_path=output/'batch_manifest.json'
    with batch_lock(output),termination_signals():
        if any(p.name!='.batch.lock' for p in output.iterdir()):
            if not resume or not state_path.is_file():raise FileExistsError('Use a new guarded output or validated --resume')
            if _load(state_path).get('signature')!=signature:raise ValueError('Guarded batch signature changed; use a new run')
        state=dict(status='running_guarded_alignment',signature=signature,total_groups=len(index),
            jobs=[],successful_groups=0,failed_groups=0,timed_out_groups=0,blocked_groups=0,
            execution_mode='guarded-first',localization_scope=SCOPE,qualification_dir=str(qualification),
            engineering_fix='validated_empty_g1_move_guard',scientific_parameters_changed=False,
            original_binary_modified=False,contains_guarded_sampler_results=bool(index),
            guarded_groups_executed=0,unguarded_stage_executed=False,
            runtime_controls=dict(job_timeout_seconds=job_timeout,heartbeat_seconds=heartbeat,retry_failed=retry_failed))
        write_json(state_path,state)
        for number,entry in enumerate(index,1):
            stem=Path(entry['peptide_file']).stem
            if not re.fullmatch(r'feature_[0-9]{6}',stem):raise ValueError('Unsafe feature stem')
            folder=output/'jobs'/stem;job_path=folder/'job_manifest.json'
            if entry['status']!='ready_for_gibbscluster':
                state['blocked_groups']+=1
                job=dict(motif_uid=entry['motif_uid'],feature_stem=stem,status=entry['status'])
            else:
                job=_previous(job_path,signature,entry,stem,folder,output,retry_failed) if resume and job_path.is_file() else None
                if job is None:
                    folder.mkdir(parents=True,exist_ok=True);attempt_number=1
                    while (folder/f'attempt_{attempt_number:04d}').exists():attempt_number+=1
                    attempt=folder/f'attempt_{attempt_number:04d}';attempt.mkdir()
                    job=dict(motif_uid=entry['motif_uid'],feature_stem=stem,status='running',signature=signature,
                        attempt=str(attempt.relative_to(output)),execution_mode='guarded-first',
                        sampler_kind='version_locked_gdb_empty_g1_guard',sampling_executed=True)
                    write_json(job_path,job)
                    state['guarded_groups_executed']+=1
                    try:
                        peptides=(prep/entry['peptide_file']).read_text(encoding='ascii').splitlines()
                        event=run_guarded_native(peptides,root,attempt/'native',timeout=job_timeout,heartbeat=heartbeat)
                        write_json(attempt/'guard_event.json',event)
                        report=_import(prep,entry,attempt/'native',sequences,attempt/'parsed')
                        report.update(sampler_kind=job['sampler_kind'],sampler_engineering_fix='empty_g1_move_guard',
                            indel_moves_skipped=event['indel_moves_skipped'],phase_moves_skipped=event['phase_moves_skipped'])
                        write_json(attempt/'parsed/parse_manifest.json',report)
                        job.update(status='success',report=report,guard_event=event,
                            engineering_fix='empty_g1_move_guard',parsed_dir=str((attempt/'parsed').relative_to(output)))
                        files=[p for p in attempt.rglob('*') if p.is_file()]
                        if any(not p.resolve().is_relative_to(attempt.resolve()) for p in files):raise ValueError('Guarded output escapes attempt')
                        job['files_sha256']={str(p.relative_to(output)):sha256(p) for p in files}
                    except (ProcessCleanupError,KeyboardInterrupt) as exc:
                        job.update(status='interrupted_or_cleanup_unconfirmed',error=f'{type(exc).__name__}: {exc}')
                        write_json(job_path,job);state['jobs'].append(job)
                        state['status']='interrupted_or_cleanup_unconfirmed';write_json(state_path,state);raise
                    except Exception as exc:
                        job.update(status='timed_out' if isinstance(exc,JobTimedOut) else 'failed',
                            failure_kind='guarded_first_pass_unresolved',error=f'{type(exc).__name__}: {exc}')
                state['successful_groups']+=job['status']=='success'
                state['failed_groups']+=job['status']!='success'
                state['timed_out_groups']+=job['status']=='timed_out'
            state['jobs'].append(job);write_json(job_path,job);write_json(state_path,state)
            print(f'[GUARDED {number}/{len(index)}] {entry["motif_uid"]}: {job["status"]}',flush=True)
        verify_qualification(qualification,gibbs_root=root)
        _preparation(prep)
        if _signature(prep,fasta,qualification)!=signature:raise ValueError('Guarded inputs/code changed during execution')
        return publish_batch(output,index,state)
