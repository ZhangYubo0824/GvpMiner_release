"""Version-locked engineering guard for a confirmed Gibbs empty-cluster loop.

Not an unmodified official release. Only impossible g=1 indel/phase moves
are skipped, before their RNG draw and scratch-state changes. No motif
thresholds, input sampling or native binary bytes are changed.
"""
from __future__ import annotations
import math
from pathlib import Path
import platform
import re
import shlex
import shutil
import sys
import time

from motifae_gibbs_results import (_records,parse_native_run,sha256,write_json)
from motifae_process_control import (run_external,termination_signals,_cleanup_signals,
    attempt_has_live_processes,ProcessCleanupError)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'diagnostics'))
from probe_pidfd_compat import PidfdBackend,check_backend,cleanup_probe

INSTALL_SHA={
    'bin/gibbs_cluster_1.2.2_DB.Linux_x86_64':'1acc1a12da275bc62b30ccb18dee1edfc30a4947eda4b7a1a5245420a7a4f6b6',
    'GibbsCluster-2.0e_SA.pl':'4e5fd229a75198847bc04bc4c4c3fde63787da11c2f4a236e0e33167ffba2c73',
    'data/blosum62.freq_rownorm':'0be49a877645efbf23f65b9c8cf9223fedc96d480436f087b7a03fb3a99c9d0d'}


def guard_decision(ordinary_groups,sizes,unique_inputs):
    if (ordinary_groups!=1 or len(sizes)!=2 or any(not isinstance(n,int) or n<0 for n in sizes)
            or sum(sizes)!=unique_inputs or unique_inputs<1):
        raise ValueError('Guard requires one ordinary group and verified full sequence accounting')
    return sizes[0]==0


def native_command(root,input_path):
    root=Path(root)
    return [str(root/'bin/gibbs_cluster_1.2.2_DB.Linux_x86_64'),'-trash','-l','5','-g','1','-ts','1.5',
        '-blf',str(root/'data/blosum%i.freq_rownorm'),'-blm','62','-base','1','-bg','1',
        '-i','10','-s','1','-nt','20','-fr','20','-fs','100','-wlc','200','-swt','0','-lambda','0.8',
        '-dlen','1','-ilen','1','-findels','10','-sigma','5',str(input_path)]


def guard_commands(arguments,unique_inputs,stdout_path,stderr_path):
    if unique_inputs<1:raise ValueError('Guarded sampling requires at least one unique input')
    text=['set pagination off','set confirm off','set startup-with-shell on',
          'set $guard_indel = 0','set $guard_phase = 0']
    ordinary='*(int*)($rbp-0x5fc)'
    sizes='*(unsigned long*)($rbp-0x618)'
    for name,address,target in [('indel','0x402b7a','0x402fb4'),('phase','0x402fdb','0x4031ad')]:
        text += [f'break *{address}',f'condition $bpnum {ordinary} == 1 && *(int*)({sizes}) == 0',
            'commands','silent',f'if *(int*)(({sizes})+4) != {unique_inputs}',
            'printf "GUARD_REFUSED_SEQUENCE_ACCOUNTING\\n"','kill','quit 63','end',
            f'set $guard_{name} = $guard_{name} + 1',f'set $pc = {target}','continue','end']
    run=' '.join(shlex.quote(str(x)) for x in arguments)
    text += [f'run {run} > {shlex.quote(Path(stdout_path).as_posix())} 2> {shlex.quote(Path(stderr_path).as_posix())}',
             'printf "MOTIFAE_GUARD_COUNTS %d %d\\n", $guard_indel, $guard_phase']
    return '\n'.join(text)+'\n'


def materialize_auxiliary(native,expected):
    """Reproduce the official Perl raw->display/core copying, with provenance."""
    native=Path(native);raw=native/'res/gibbs.1g.out';text=raw.read_text(encoding='utf-8')
    finals=re.findall(r'^### Final KLD sum after (\d+) iterations: (\S+)\s*$',text,re.M)
    last=next((s.strip() for s in reversed(text.splitlines()) if s.strip()),'')
    if len(finals)!=1 or int(finals[0][0])<=0 or last!=f'### Final KLD sum after {finals[0][0]} iterations: {finals[0][1]}':
        raise ValueError('Native sampler did not produce a terminal completed result')
    records=_records(text,5)
    if records:
        if not math.isfinite(float(finals[0][1])):raise ValueError('Retained native result has nonfinite KLD')
        lines=[line for line in text.splitlines() if line.startswith('G ') and not line.startswith('G Gn')]
        (native/'res/gibbs.1g.ds.out').write_text(
            'G Gn Num Sequence Core o of ip IP il IL dp DP dl DL Annotation sS Self bgG bgG bgS bgScor cS cScore\n'+
            '\n'.join(lines)+'\n',encoding='utf-8')
        (native/'cores').mkdir(exist_ok=True)
        (native/'cores/gibbs.1g.cores').write_text('## Alignment cores for group 1\n'+
            ''.join(r['core']+'\n' for r in records),encoding='utf-8')
    return parse_native_run(native,expected)


def compare_native(reference,trial,expected,*,blosum_root=None):
    def parameter_lines(folder):
        lines=[line.split() for line in (Path(folder)/'res/gibbs.1g.out').read_text(encoding='utf-8').splitlines()
               if re.match(r'^#\s+(?:-\w+\s|\[-)',line)]
        if blosum_root is not None:
            for line in lines:
                if len(line)>2 and line[1] in ('-blf','[-blf'):
                    index=3 if line[1]=='[-blf' else 2
                    path=line[index].replace('\\','/')
                    if not path.endswith('/data/blosum%i.freq_rownorm'):
                        raise ValueError('Unexpected BLOSUM parameter filename')
                    line[index]='<verified-installation>/data/blosum%i.freq_rownorm'
        return lines
    if blosum_root is not None:verified_installation(blosum_root)
    if parameter_lines(reference)!=parameter_lines(trial):
        raise ValueError('Guarded reference changed native parameters')
    def full_rows(folder,name):
        return [tuple(line.split()) for line in (Path(folder)/'res'/name).read_text(encoding='utf-8').splitlines()
                      if line.startswith('G ') and not line.startswith('G Gn')]
    for name in ('gibbs.1g.out','gibbs.1g.ds.out'):
        if full_rows(reference,name)!=full_rows(trial,name):
            raise ValueError('Guarded reference changed complete native rows, including all background/corrected scores')
    def cores(folder):
        return [line.strip() for line in (Path(folder)/'cores/gibbs.1g.cores').read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.startswith('#')]
    if cores(reference)!=cores(trial):raise ValueError('Guarded reference changed native cores/order')
    reference=parse_native_run(reference,expected);trial=parse_native_run(trial,expected)
    if blosum_root is not None:
        for result in (reference,trial):
            if 'blf' in result['parameters']:
                result['parameters']['blf']='<verified-installation>/data/blosum%i.freq_rownorm'
    if (sorted(reference['records'],key=lambda r:r['sequence'])!=sorted(trial['records'],key=lambda r:r['sequence'])
            or reference['trash_sequences']!=trial['trash_sequences']
            or reference['parameters']!=trial['parameters']):
        raise ValueError('Guarded reference run changed native assignments/cores/offsets/indels/scores/parameters')
    if reference['kld'] is None or trial['kld'] is None or reference['kld']!=trial['kld']:
        raise ValueError('Reference run lacks matching finite KLD')
    return dict(retained_unique_sequences=len(reference['records']),kld=reference['kld'],
                full_native_assignments_match=True)


def verified_installation(root):
    root=Path(root).resolve()
    for name,digest in INSTALL_SHA.items():
        path=root/name
        if not path.is_file() or sha256(path)!=digest:raise ValueError(f'Unsupported or changed Gibbs installation: {name}')
    return root


def code_signature():
    import motifae_gibbs_results,probe_pidfd_compat,motifae_process_control
    result={Path(module.__file__).name:sha256(Path(module.__file__))
            for module in (sys.modules[__name__],motifae_gibbs_results,probe_pidfd_compat,motifae_process_control)}
    result.update(motifae_gibbs_results._shared_hashes())
    return result


def run_guarded_native(peptides,root,output,*,timeout=300,heartbeat=30):
    if platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64'):
        raise RuntimeError('Guarded native execution requires the verified Linux x86_64 binary')
    root=verified_installation(root);output=Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('Keep guard output separate from Gibbs installation')
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('A finite positive native time budget is required')
    peptides=list(peptides);unique=sorted(set(peptides))
    if len(unique)<1 or any(not 5<=len(s)<=30 or set(s)-set('ACDEFGHIKLMNPQRSTVWY') for s in unique):
        raise ValueError('Guarded sampler requires at least one valid unique peptide sequence')
    gdb=shutil.which('gdb')
    if not gdb:raise FileNotFoundError('gdb is required for guarded native execution')
    backend=PidfdBackend();check_backend(backend)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Guarded native output must be new/empty')
    (output/'data').mkdir(parents=True,exist_ok=True);(output/'res').mkdir()
    input_path=output/'data/pepdata.txt'
    input_path.write_text(''.join(s+'\tPeplist\n' for s in unique),encoding='ascii')
    native=native_command(root,input_path)
    script=output/'guard.gdb';raw=output/'res/gibbs.1g.out';stderr=output/'native.stderr.log'
    script.write_text(guard_commands(native[1:],len(unique),raw,stderr),encoding='utf-8')
    command=[gdb,'--nx','--batch','--return-child-result','--command',str(script),native[0]]
    event=dict(status='running_guarded_sampler',engineering_fix='skip_impossible_empty_g1_indel_phase_moves',
        sampler_kind='version_locked_gdb_empty_g1_guard',
        native_command=native,gdb_command=command,input_sha256=sha256(input_path),
        original_instances=len(peptides),unique_inputs=len(unique),scientific_parameters_changed=False,
        original_binary_modified=False,installation_sha256=INSTALL_SHA,code_sha256=code_signature())
    write_json(output/'native_guard_manifest.json',event)
    try:
        with termination_signals():
            try:
                with (output/'gdb.log').open('wb') as stdout,(output/'gdb.stderr.log').open('wb') as errors:
                    process=run_external(command,cwd=output,stdout=stdout,stderr=errors,
                        timeout_seconds=timeout,heartbeat_seconds=heartbeat,label=output.name)
            finally:
                with _cleanup_signals():
                    try:
                        event['remaining_samplers_signalled']=cleanup_probe(input_path,backend)
                        deadline=time.monotonic()+2
                        while attempt_has_live_processes(output) and time.monotonic()<deadline:time.sleep(.05)
                        if attempt_has_live_processes(output):raise RuntimeError('owned sampler still alive')
                    except Exception as exc:raise ProcessCleanupError(f'Guarded sampler cleanup unconfirmed: {exc}') from exc
        if process.returncode:raise RuntimeError(f'Guarded debugger/native exited {process.returncode}; inspect gdb logs')
        counts=re.findall(r'^MOTIFAE_GUARD_COUNTS (\d+) (\d+)\s*$',(output/'gdb.log').read_text(encoding='utf-8'),re.M)
        if len(counts)!=1:raise ValueError('Guard did not complete with an unambiguous counter record')
        parsed=materialize_auxiliary(output,peptides)
        verified_installation(root)
        event.update(status='completed_guarded_sampler',indel_moves_skipped=int(counts[0][0]),
            phase_moves_skipped=int(counts[0][1]),result_status=parsed['completion_status'],
            auxiliary_files_origin='raw sequence/core copying equivalent to official Perl postprocessor',
            native_files_sha256=parsed['native_files_sha256'])
        write_json(output/'native_guard_manifest.json',event)
        return event
    except BaseException as exc:
        event.update(status='unresolved_guarded_sampler',error=f'{type(exc).__name__}: {exc}')
        write_json(output/'native_guard_manifest.json',event)
        raise
