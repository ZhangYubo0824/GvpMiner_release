#!/usr/bin/env python3
"""Validate native GibbsCluster 2.0f results and map selected instances.

No inference, feature-count filters, new sequence scanning or licensed binaries
are included. Coordinates cover only the selected alignment input instances.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil

from adapter_common import read_fasta_records
from motifae_candidate_filter import canonical_protein_id, sha256


AA='ACDEFGHIKLMNPQRSTVWY'
SCOPE='selected_top1000_alignment_instances'
OCC_FIELDS=['family','method','motif_uid','parent_id','sequence_id','occ_start','occ_end',
    'motif_sequence','aligned_core','core_residue_positions_0based','alignment_width',
    'mapped_residue_count','span_length','candidate_start','candidate_end','candidate_sequence',
    'selection_rank','score','gibbs_score','localization_scope']
ASSIGN_FIELDS=['sequence','status','core','offset','ip','il','dp','dl','self_score',
               'relative_core_positions_0based','selected_instance_count']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def core_positions(sequence,core,offset,*,ip=-99,il=-99,dp=-99,dl=-99):
    """Reconstruct reported aligned columns, retaining gaps and deleted spans.

    Offset/deletion semantics are checked against all 128 actual user pilot
    rows; both insertion and deletion also reconstruct all 197 official demo
    rows. Never substitute a substring search for reported offsets.
    """
    insertion=il>0;deletion=dl>0
    if (ip>=0)!=insertion or (dp>=0)!=deletion:
        raise ValueError('Inconsistent native indel position/length fields')
    if insertion and deletion:
        raise ValueError('Simultaneous insertion/deletion mapping is not verified')
    if insertion and not 0<=ip<ip+il<=len(core):
        raise ValueError('Invalid reported insertion coordinates')
    if deletion and not 0<=dp<len(core):
        raise ValueError('Invalid reported deletion coordinates')
    if set(sequence)-set(AA) or set(core)-set(AA+'-'):
        raise ValueError('Unsupported sequence/core alphabet')
    cursor=offset;positions=[];rebuilt=[]
    for column in range(len(core)):
        if insertion and ip<=column<ip+il:
            positions.append(None);rebuilt.append('-');continue
        if deletion and column==dp:
            cursor+=dl
        valid=0<=cursor<len(sequence)
        positions.append(cursor if valid else None)
        rebuilt.append(sequence[cursor] if valid else '-')
        cursor+=1
    if ''.join(rebuilt)!=core or not any(p is not None for p in positions):
        raise ValueError('Reported core does not reconstruct from offset/indel fields')
    return positions


ROW=re.compile(r'^G\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+o\s+(-?\d+)'
    r'\s+ip\s+(-?\d+)\s+il\s+(-?\d+)\s+dp\s+(-?\d+)\s+dl\s+(-?\d+)'
    r'\s+.*?\s+sS\s+(\S+)\s+bgG\s+-?\d+\s+bgS\s+\S+\s+cS\s+\S+\s*$')


def _records(text,core_length):
    records=[]
    for line in text.splitlines():
        if not line.startswith('G ') or line.startswith('G Gn'):
            continue
        match=ROW.fullmatch(line)
        if not match:
            raise ValueError('Unrecognized native Gibbs sequence row')
        group,number,sequence,core,offset,ip,il,dp,dl,score=match.groups()
        if int(group)!=0 or len(core)!=core_length or not math.isfinite(float(score)):
            raise ValueError('Unexpected group, core width or native score')
        record=dict(sequence=sequence,core=core,offset=int(offset),ip=int(ip),il=int(il),
                    dp=int(dp),dl=int(dl),self_score=float(score))
        record['positions']=core_positions(sequence,core,record['offset'],
                           **{k:record[k] for k in ('ip','il','dp','dl')})
        records.append(record)
    if len({r['sequence'] for r in records})!=len(records):
        raise ValueError('Retained sequence appears more than once in native output')
    return records


def _all_trash_result(run_dir,raw,expected,parameters,core_length):
    """Accept only a completed sampler's explicit, exhaustive trash decision."""
    if not expected or any(set(s)-set(AA) for s in expected):
        raise ValueError('Invalid all-trash selected input set')
    lines=[line for line in raw.splitlines() if line.startswith('#Trash')]
    matches=[re.fullmatch(r'#Trash([0-9]+)\s+([A-Z]+)(?:\s+.*)?',line) for line in lines]
    if any(match is None for match in matches):raise ValueError('Malformed explicit native trash record')
    trash=[match[2] for match in matches]
    numbers=[int(match[1]) for match in matches]
    counts=re.findall(r'^# Trash cluster: removed (\d+) outliers\s*$',raw,re.M)
    clusters=re.findall(r'^# Final number of non-empty clusters: (\d+)\s*$',raw,re.M)
    finals=re.findall(r'^### Final KLD sum after (\d+) iterations: (\S+)\s*$',raw,re.M)
    if (set(trash)!=expected or len(trash)!=len(expected) or len(set(numbers))!=len(numbers)
            or counts!=[str(len(expected))] or clusters!=['0'] or len(finals)!=1
            or int(finals[0][0])<=0):
        raise ValueError('Native all-trash decision is incomplete or does not match selected inputs')
    last=next((line.strip() for line in reversed(raw.splitlines()) if line.strip()),'')
    if last!=f'### Final KLD sum after {finals[0][0]} iterations: {finals[0][1]}':
        raise ValueError('Native all-trash final marker is not terminal')
    value=float(finals[0][1])
    if not math.isfinite(value) and not re.fullmatch(r'[+-]?nan',finals[0][1],re.I):
        raise ValueError('Unexpected nonfinite all-trash KLD')
    files=[run_dir/'res/gibbs.1g.out']
    display=run_dir/'res/gibbs.1g.ds.out';cores=run_dir/'cores/gibbs.1g.cores'
    if display.exists():
        if _records(display.read_text(encoding='utf-8'),core_length):
            raise ValueError('All-trash raw output contradicts formatted retained rows')
        files.append(display)
    if cores.exists():
        if any(line.strip() and not line.startswith('#') for line in cores.read_text(encoding='utf-8').splitlines()):
            raise ValueError('All-trash raw output contradicts nonempty core file')
        files.append(cores)
    return dict(records=[],trash_sequences=sorted(trash),kld=None,kld_raw=finals[0][1],
                kld_status='undefined_no_retained_sequences',completion_status='all_sequences_trash',
                parameters=parameters,native_files_sha256={str(p.relative_to(run_dir)):sha256(p) for p in files})


def parse_native_run(run_dir,expected_peptides,core_length=5):
    run_dir=Path(run_dir)
    raw_path=run_dir/'res/gibbs.1g.out'
    display_path=run_dir/'res/gibbs.1g.ds.out'
    core_path=run_dir/'cores/gibbs.1g.cores'
    raw=raw_path.read_text(encoding='utf-8')
    parameters=dict(re.findall(r'^#\s+-(\w+)\s+(\S+)',raw,re.M))
    for name,expected in dict(l=core_length,g=1,trash=1,base=1,ilen=1,dlen=1).items():
        if name not in parameters or float(parameters[name])!=expected:
            raise ValueError(f'Native output does not verify published parameter: {name}')
    records=_records(raw,core_length)
    if not records:
        return _all_trash_result(run_dir,raw,set(expected_peptides),parameters,core_length)
    display=display_path.read_text(encoding='utf-8')
    if records!=_records(display,core_length):
        raise ValueError('Raw and formatted native sequence records differ')
    cores=[line.strip() for line in core_path.read_text(encoding='utf-8').splitlines()
           if line.strip() and not line.startswith('#')]
    if Counter(cores)!=Counter(r['core'] for r in records):
        raise ValueError('Native core file differs from formatted sequence records')
    expected=set(expected_peptides);retained={r['sequence'] for r in records}
    if not expected or retained-expected:
        raise ValueError('Native retained sequence absent from selected inputs')
    trash_matches=re.findall(r'^# Trash cluster: removed (\d+) outliers\s*$',raw,re.M)
    if len(trash_matches)!=1 or int(trash_matches[0])!=len(expected-retained):
        raise ValueError('Retained/trash sequence accounting does not match selected inputs')
    kld_match=re.findall(r'^### Final KLD sum after \d+ iterations: (\S+)\s*$',raw,re.M)
    if len(kld_match)!=1 or not math.isfinite(float(kld_match[0])):
        raise ValueError('Native run lacks a finite final KLD result')
    return dict(records=records,trash_sequences=sorted(expected-retained),
                kld=float(kld_match[0]),kld_raw=kld_match[0],kld_status='finite',
                completion_status='retained_alignment',parameters=parameters,
                native_files_sha256={str(p.relative_to(run_dir)):sha256(p)
                                    for p in [raw_path,display_path,core_path]})


def _preparation(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'gibbs_preparation_manifest.json').read_text(encoding='utf-8'))
    if manifest['core_length']!=5 or manifest['published_gibbs_parameters']!=['-g','1','-l','5','-T','-j','1','-I','1','-D','1']:
        raise ValueError('This validated importer currently supports the published default 5-aa core')
    for name,digest in manifest['files_sha256'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or sha256(path)!=digest:
            raise ValueError(f'Changed or unsafe preparation file: {name}')
    index=read_csv(directory/'feature_index.csv')
    for entry in index:
        for field in ('mapping_file','peptide_file'):
            name=entry[field]
            if name and name not in manifest['files_sha256']:
                raise ValueError('Index references an unverified preparation file')
    return directory,manifest,index


def _sequences(fasta):
    result={}
    for header,sequence in read_fasta_records(Path(fasta)):
        protein=canonical_protein_id(header)
        if not protein or protein in result:
            raise ValueError('FASTA identifiers are missing or ambiguous after canonicalization')
        result[protein]=sequence.upper()
    if not result:
        raise ValueError('Empty FASTA input')
    return result


def _import(prep,entry,alignment,sequences,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Parsed output must be new/empty')
    mapping=read_csv(prep/entry['mapping_file'])
    peptide_lines=(prep/entry['peptide_file']).read_text(encoding='ascii').splitlines()
    if peptide_lines!=[row['motif_sequence'] for row in mapping] or not 1<=len(mapping)<=1000:
        raise ValueError('Selected input lines and original instance mapping differ')
    if [int(row['selection_rank']) for row in mapping]!=list(range(1,len(mapping)+1)):
        raise ValueError('Invalid selection ranks')
    by_sequence=defaultdict(list)
    for row in mapping:
        protein=row['parent_id'];start=int(row['occ_start']);end=int(row['occ_end'])
        if (row['motif_uid']!=entry['motif_uid'] or row['family']!=entry['family']
                or row['method']!=entry['method'] or protein not in sequences
                or not 0<=start<end<=len(sequences[protein])
                or sequences[protein][start:end]!=row['motif_sequence']):
            raise ValueError('Selected instance does not match feature or complete protein sequence')
        by_sequence[row['motif_sequence']].append(row)
    native=parse_native_run(alignment,peptide_lines)
    assignments=[];occurrences=[]
    counts=[Counter() for _ in range(5)];gaps=[0]*5
    for record in native['records']:
        seq=record['sequence']
        assignments.append(dict(record,status='retained',
            relative_core_positions_0based=json.dumps(record['positions']),
            selected_instance_count=len(by_sequence[seq])))
        for column,aa in enumerate(record['core']):
            if aa=='-':gaps[column]+=1
            else:counts[column][aa]+=1
        for source in by_sequence[seq]:
            origin=int(source['occ_start'])
            positions=[origin+p if p is not None else None for p in record['positions']]
            actual=[p for p in positions if p is not None]
            start=min(actual);end=max(actual)+1
            if ''.join(sequences[source['parent_id']][p] if p is not None else '-' for p in positions)!=record['core']:
                raise ValueError('Aligned core does not reconstruct on complete protein')
            occurrences.append(dict(source,occ_start=start,occ_end=end,
                motif_sequence=sequences[source['parent_id']][start:end],aligned_core=record['core'],
                core_residue_positions_0based=json.dumps(positions),alignment_width=5,
                mapped_residue_count=len(actual),span_length=end-start,candidate_start=source['occ_start'],
                candidate_end=source['occ_end'],candidate_sequence=seq,gibbs_score=record['self_score'],
                localization_scope=SCOPE))
    for seq in native['trash_sequences']:
        assignments.append(dict(sequence=seq,status='trash',selected_instance_count=len(by_sequence[seq])))
    frequencies=[]
    for i,column in enumerate(counts):
        n=sum(column.values())
        frequencies.append(dict(position=i+1,non_gap_sequences=n,gap_sequences=gaps[i],
                                 **{aa:column[aa]/n if n else '' for aa in AA}))
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'aligned_occurrences.csv',occurrences,OCC_FIELDS)
    write_csv(output/'alignment_assignments.csv',assignments,ASSIGN_FIELDS)
    write_csv(output/'frequency_matrix.csv',frequencies,['position','non_gap_sequences','gap_sequences',*AA])
    pattern_built=bool(native['records']) and all(sum(x.values()) for x in counts)
    report=dict(family=entry['family'],method=entry['method'],motif_uid=entry['motif_uid'],
        status=('all_sequences_trash' if not native['records'] else
                'aligned_pattern_built' if pattern_built else 'alignment_has_empty_frequency_columns'),
        total_candidate_instances=int(entry['total_peptides']),selected_instances=len(mapping),
        unassessed_instances=int(entry['total_peptides'])-len(mapping),
        selected_unique_sequences=len(by_sequence),retained_unique_sequences=len(native['records']),
        trash_unique_sequences=len(native['trash_sequences']),
        mapped_instances=len(occurrences),mapped_proteins=len({r['parent_id'] for r in occurrences}),
        trash_instances=sum(len(by_sequence[s]) for s in native['trash_sequences']),
        deletion_unique_sequences=sum(r['dl']>0 for r in native['records']),
        insertion_unique_sequences=sum(r['il']>0 for r in native['records']),
        gapped_span_instances=sum(r['span_length']!=5 or r['mapped_residue_count']!=5 for r in occurrences),
        pattern_built=bool(pattern_built),
        kld=native['kld'],kld_raw=native['kld_raw'],kld_status=native['kld_status'],localization_scope=SCOPE,
        frequency_definition='Each retained unique Gibbs peptide once; normalize non-gap amino-acid counts per column; gap counts separate; no pseudocounts',
        native_parameters=native['parameters'],native_files_sha256=native['native_files_sha256'],
        biological_function_validated=False,all_feature_candidates_assessed=int(entry['total_peptides'])==len(mapping))
    report['pattern_built']=bool(report['pattern_built'])
    write_json(output/'parse_manifest.json',report)
    return report


def import_alignment(preparation_dir,feature_stem,alignment_dir,fasta,output_dir):
    prep,manifest,index=_preparation(preparation_dir)
    entries=[e for e in index if e['peptide_file'] and Path(e['peptide_file']).stem==feature_stem]
    if len(entries)!=1:
        raise ValueError('Feature file stem does not uniquely identify a prepared input')
    result=_import(prep,entries[0],Path(alignment_dir).resolve(),_sequences(fasta),Path(output_dir))
    result['provenance']=dict(preparation_sha256=sha256(prep/'gibbs_preparation_manifest.json'),
        preparation_dir=str(prep),alignment_dir=str(Path(alignment_dir).resolve()),
        fasta=str(Path(fasta).resolve()),fasta_sha256=sha256(Path(fasta)),
        parser_sha256=sha256(Path(__file__)),shared_code_sha256=_shared_hashes())
    write_json(Path(output_dir)/'parse_manifest.json',result)
    return result


def _shared_hashes():
    return {name:sha256(Path(__file__).with_name(name))
            for name in ['adapter_common.py','motifae_candidate_filter.py','motifae_process_control.py']}


def _resume_job(old,signature,entry,stem,folder,output):
    try:
        if old.get('status')!='success' or old.get('signature')!=signature or old.get('feature_stem')!=stem:
            return None
        attempt=(output/old['attempt']).resolve()
        parsed=(output/old['parsed_dir']).resolve()
        if (attempt.parent!=folder.resolve() or not re.fullmatch(r'attempt_[0-9]{4}',attempt.name)
                or parsed!=attempt/'parsed'):
            return None
        files=old['files_sha256']
        required=[parsed/name for name in ['parse_manifest.json','aligned_occurrences.csv',
                                            'frequency_matrix.csv','alignment_assignments.csv']]
        if any(str(path.relative_to(output)) not in files for path in required):return None
        for name,digest in files.items():
            path=(output/name).resolve()
            if not path.is_relative_to(attempt) or not path.is_file() or sha256(path)!=digest:
                return None
        # The job manifest is mutable control state; derive numerical results
        # only from the verified parse artifact, never its redundant copy.
        report=json.loads((parsed/'parse_manifest.json').read_text(encoding='utf-8'))
        if any(report[field]!=entry[field] for field in ['family','method','motif_uid']):return None
        if report['localization_scope']!=SCOPE:return None
        old['report']=report
        return old
    except (OSError,ValueError,TypeError,KeyError):
        return None


def publish_batch(output,index,state):
    """Publish parsed successes and preserve all unassessed candidate counts."""
    output=Path(output);state_path=output/'batch_manifest.json'
    successful=[job for job in state['jobs'] if job['status']=='success']
    summaries=[dict(job['report'],matrix_file=f"frequency_matrices/{job['feature_stem']}.csv") for job in successful]
    summary_fields=['family','method','motif_uid','status','total_candidate_instances','selected_instances',
        'unassessed_instances','selected_unique_sequences','retained_unique_sequences','trash_unique_sequences',
        'mapped_instances','mapped_proteins','trash_instances','deletion_unique_sequences','insertion_unique_sequences',
        'gapped_span_instances','pattern_built','kld','kld_raw','kld_status','localization_scope','all_feature_candidates_assessed','matrix_file']
    write_csv(output/'feature_summary.csv',summaries,summary_fields)
    def occurrences():
        for job in successful:
            yield from read_csv(output/job['parsed_dir']/'aligned_occurrences.csv')
    write_csv(output/'aligned_occurrences.csv',occurrences(),OCC_FIELDS)
    successful_stems={job['feature_stem'] for job in successful}
    # Invalidate only managed leaf matrices for groups no longer successful.
    # Preserve attempt history and unrelated files in the aggregate directory.
    matrix_dir=output/'frequency_matrices'
    for entry in index:
        stem=Path(entry['mapping_file']).stem
        if not re.fullmatch(r'feature_[0-9]{6}',stem):raise ValueError('Unsafe managed matrix stem')
        stale=matrix_dir/f'{stem}.csv'
        if not stale.resolve().is_relative_to(output):raise ValueError('Managed matrix escapes batch output')
        if stem not in successful_stems:stale.unlink(missing_ok=True)
    for job in successful:
        target=matrix_dir/f"{job['feature_stem']}.csv"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(output/job['parsed_dir']/'frequency_matrix.csv',target)
    total_instances=sum(int(entry['total_peptides']) for entry in index)
    mapped=sum(r['mapped_instances'] for r in summaries)
    discarded=sum(r['trash_instances'] for r in summaries)
    state.update(status='alignment_samples_parsed' if not state['failed_groups'] and not state['blocked_groups'] else 'incomplete_alignment',
        pattern_built_groups=sum(r['pattern_built'] for r in summaries),
        all_trash_groups=sum(r['status']=='all_sequences_trash' for r in summaries),
        total_candidate_instances=total_instances,mapped_instances=mapped,trash_instances=discarded,
        unassessed_instances=total_instances-mapped-discarded,
        biological_function_validated=False)
    write_json(state_path,state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-dir',required=True,type=Path)
    parser.add_argument('--fasta',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--feature-stem',required=True)
    parser.add_argument('--alignment-dir',required=True,type=Path)
    result=import_alignment(**vars(parser.parse_args()))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
