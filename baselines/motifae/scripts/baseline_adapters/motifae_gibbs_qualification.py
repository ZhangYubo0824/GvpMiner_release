"""Portable, family-independent qualification of the fixed Gibbs guard.

Qualification executes only the bundled regression inputs. Family mapping is
deliberately absent; sampling must run and map each family's own peptides.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import shutil

from motifae_gibbs_results import parse_native_run,sha256,write_json,_shared_hashes
from motifae_native_guard import (INSTALL_SHA,code_signature,compare_native,
    run_guarded_native,verified_installation)
from motifae_process_control import batch_lock,termination_signals

FIXTURES=Path(__file__).resolve().parents[2]/'qualification_fixtures'
FIXTURE_MANIFEST_SHA256='9630fd72187126e3eedab160316e5d093f12cbf71fa2667b063e0e69215ba9ec'
SINGLETON_PEPTIDES=['QPSHRLPSEDE']


def _inventory(directory,exclude=()):
    directory=Path(directory).resolve();result={}
    for path in directory.rglob('*'):
        if path.is_symlink():raise ValueError('Qualification artifact inventory cannot contain symbolic links')
        if path.is_file() and path.relative_to(directory).as_posix() not in exclude:
            result[path.relative_to(directory).as_posix()]=sha256(path)
    return result


def _fixtures():
    path=FIXTURES/'fixture_manifest.json'
    if not path.is_file() or sha256(path)!=FIXTURE_MANIFEST_SHA256:
        raise ValueError('Fixed qualification fixture manifest hash differs from this code version')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if _inventory(FIXTURES,('fixture_manifest.json','README.md'))!=manifest['files_sha256']:
        raise ValueError('Fixed qualification fixture input/reference inventory hash mismatch')
    normal=(FIXTURES/'normal_peptides.txt').read_text(encoding='ascii').splitlines()
    fault=(FIXTURES/'fault_peptides.txt').read_text(encoding='ascii').splitlines()
    native_input=[line.split()[0] for line in (FIXTURES/'normal_reference/data/pepdata.txt').read_text(encoding='ascii').splitlines()]
    if sorted(set(normal))!=native_input or Counter(fault)!=Counter({'AIGSE':23,'SSAPQ':1}):
        raise ValueError('Qualification normal native input or fault regression input differs')
    parse_native_run(FIXTURES/'normal_reference',normal)
    return normal,fault


def _signature():
    code=code_signature();code.update(_shared_hashes())
    for name in ('motifae_gibbs_qualification.py','motifae_output_safety.py','motifae_guarded_batch.py'):
        code[name]=sha256(Path(__file__).with_name(name))
    return dict(implementation="motifae-guarded-first",code_sha256=code,installation_sha256=INSTALL_SHA,
        singleton_peptides=SINGLETON_PEPTIDES,
        fixture_manifest_sha256=FIXTURE_MANIFEST_SHA256,
        fixture_files_sha256=json.loads((FIXTURES/'fixture_manifest.json').read_text(encoding='utf-8'))['files_sha256'])


def _case(peptides,root,output,timeout):
    output.mkdir(parents=True)
    (output/'input_peptides.txt').write_text(''.join(p+'\n' for p in peptides),encoding='ascii')
    event=run_guarded_native(peptides,root,output/'native',timeout=timeout)
    parsed=parse_native_run(output/'native',peptides)
    case=dict(guard_event=event,completion_status=parsed['completion_status'],
        retained_unique_sequences=len(parsed['records']),trash_unique_sequences=len(parsed['trash_sequences']),
        sampler_kind='version_locked_gdb_empty_g1_guard')
    write_json(output/'case_manifest.json',case)
    return case


def _hits(case):
    event=case['guard_event'];counts=[event.get(key) for key in ('indel_moves_skipped','phase_moves_skipped')]
    if any(type(n) is not int or n<0 for n in counts):raise ValueError('Qualification guard counters are invalid')
    return sum(counts)


def qualify_installation(gibbs_root,output_dir,*,timeout=300):
    from motifae_output_safety import _fresh,_locked_new_output
    root=verified_installation(gibbs_root);normal,fault=_fixtures()
    output=_fresh(output_dir,[root,FIXTURES])
    state=dict(status='running_installation_qualification',signature=_signature(),
        engineering_fix='version_locked_empty_g1_indel_phase_move_guard',
        scientific_parameters_changed=False,original_binary_modified=False,
        sampler_kind='version_locked_gdb_empty_g1_guard',family_independent=True)
    with batch_lock(output),termination_signals():
        _locked_new_output(output);write_json(output/'qualification_manifest.json',state)
        try:
            shutil.copytree(FIXTURES/'normal_reference',output/'reference_native')
            normal_case=_case(normal,root,output/'normal_case',timeout)
            if _hits(normal_case):raise ValueError('Normal qualification reference unexpectedly triggered the guard')
            comparison=compare_native(output/'reference_native',output/'normal_case/native',normal,blosum_root=root)
            fault_case=_case(fault,root,output/'fault_case',timeout)
            if _hits(fault_case)<1:raise ValueError('Fault qualification did not exercise the runtime guard')
            singleton_case=_case(SINGLETON_PEPTIDES,root,output/'singleton_case',timeout)
            _hits(singleton_case)
            if singleton_case['retained_unique_sequences']+singleton_case['trash_unique_sequences']!=1:
                raise ValueError('Singleton qualification lost its input sequence')
            verified_installation(root);_fixtures()
            if _signature()!=state['signature']:raise ValueError('Qualification code/fixtures changed during execution')
            state.update(status='installation_qualification_passed',normal_case=normal_case,
                fault_case=fault_case,singleton_case=singleton_case,normal_comparison=comparison,
                files_sha256=_inventory(output,('qualification_manifest.json','.batch.lock')))
            write_json(output/'qualification_manifest.json',state)
            return state
        except BaseException as exc:
            state.update(status='installation_qualification_failed',error=f'{type(exc).__name__}: {exc}')
            write_json(output/'qualification_manifest.json',state);raise


def verify_qualification(directory,*,gibbs_root=None):
    directory=Path(directory).resolve();normal,fault=_fixtures()
    state=json.loads((directory/'qualification_manifest.json').read_text(encoding='utf-8'))
    if state.get('status')!='installation_qualification_passed' or state.get('signature')!=_signature():
        raise ValueError('Qualification is missing, failed or stale for the current code/fixture signature')
    files=state.get('files_sha256',{})
    actual=_inventory(directory,('qualification_manifest.json','.batch.lock'))
    required={'reference_native/res/gibbs.1g.out','reference_native/res/gibbs.1g.ds.out',
        'reference_native/cores/gibbs.1g.cores','normal_case/case_manifest.json','fault_case/case_manifest.json',
        'normal_case/input_peptides.txt','fault_case/input_peptides.txt',
        'singleton_case/case_manifest.json','singleton_case/input_peptides.txt','singleton_case/native/res/gibbs.1g.out',
        'normal_case/native/res/gibbs.1g.out','fault_case/native/res/gibbs.1g.out'}
    if not required.issubset(files) or files!=actual:
        raise ValueError('Qualification artifact inventory/hash mismatch (missing, extra or changed files)')
    for label,peptides in (('normal_case',normal),('fault_case',fault),('singleton_case',SINGLETON_PEPTIDES)):
        if (directory/label/'input_peptides.txt').read_text(encoding='ascii').splitlines()!=peptides:
            raise ValueError('Qualification case input differs from the fixed fixture')
        state[label]=json.loads((directory/label/'case_manifest.json').read_text(encoding='utf-8'))
        parsed=parse_native_run(directory/label/'native',peptides)
        if state[label]['completion_status']!=parsed['completion_status']:
            raise ValueError('Qualification case native completion status differs')
        if (state[label]['retained_unique_sequences']!=len(parsed['records'])
                or state[label]['trash_unique_sequences']!=len(parsed['trash_sequences'])):
            raise ValueError('Qualification case sequence accounting differs')
        _hits(state[label])
    if _hits(state['normal_case']) or _hits(state['fault_case'])<1:
        raise ValueError('Qualification normal/fault guard counters do not verify')
    # Reference remains fixed evidence even if someone edits the run inventory.
    for relative in ('res/gibbs.1g.out','res/gibbs.1g.ds.out','cores/gibbs.1g.cores','data/pepdata.txt'):
        if sha256(directory/'reference_native'/relative)!=sha256(FIXTURES/'normal_reference'/relative):
            raise ValueError('Qualification native reference hash differs from the fixed fixture')
    compare_native(directory/'reference_native',directory/'normal_case/native',normal,blosum_root=gibbs_root)
    return state
