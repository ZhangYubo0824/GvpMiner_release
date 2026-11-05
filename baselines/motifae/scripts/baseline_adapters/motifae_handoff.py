"""Strict sequence-bound MotifAE result handoff; standard library only.

The FASTA is the background universe. Presence calls concern only retained,
selected alignment instances; unselected candidates have unknown status.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile
import os
import inspect
import functools
import shutil
from collections import Counter
from motifae_process_control import batch_lock

SCOPE = 'selected_top1000_alignment_instances'
OCC_FIELDS = ['family','method','motif_uid','parent_id','sequence_id','occ_start','occ_end',
 'motif_sequence','aligned_core','core_residue_positions_0based','alignment_width',
 'mapped_residue_count','span_length','candidate_start','candidate_end','candidate_sequence',
 'selection_rank','score','gibbs_score','localization_scope','sampler_kind','guard_provenance',
 'source_model','consensus_or_label','occ_core_start','occ_core_end','significance']
CAT_FIELDS = ['family','method','motif_uid','consensus_or_label','n_occurrences','n_proteins',
 'score','significance','source_model','localization_scope','sampler_kind','guard_provenance']
SEQ_FIELDS = ['sequence_id','protein_id','family','sequence','sequence_length','sequence_sha256','fasta_path','description']


def _empty_output(path):
    if any(p.name!='.batch.lock' for p in path.iterdir()):
        raise FileExistsError('Output must remain new/empty; another result or concurrent publication exists: '+str(path))


def _publish_new(stage,output):
    _empty_output(output)
    # Manifests are published after their data. Root files use exclusive create,
    # so an unrelated writer cannot have its existing manifest replaced.
    paths=sorted(stage.iterdir(),key=lambda p:('manifest' in p.name,p.name))
    for source in paths:
        target=output/source.name
        if not source.resolve().is_relative_to(stage.resolve()) or not target.resolve().is_relative_to(output.resolve()):
            raise ValueError('Publication path escapes its staging/output directory')
        if source.is_dir():
            if target.exists():raise FileExistsError('Publication target already exists: '+str(target))
            os.rename(source,target)
        else:
            with source.open('rb') as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)


def _new_output_transaction(function):
    signature=inspect.signature(function)
    @functools.wraps(function)
    def run(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        raw_output=Path(bound.arguments['output_dir'])
        if raw_output.is_symlink():raise ValueError('Output directory cannot be a symlink')
        output=raw_output.resolve();inputs=[]
        for name in ['handoff_dir','term2gene','annotation_fasta','annotation_sequence_hashes']:
            if bound.arguments.get(name) is not None:inputs.append(Path(bound.arguments[name]).resolve())
        families=bound.arguments.get('family_runs')
        if isinstance(families,list):
            for description in families:
                if isinstance(description,dict):
                    inputs.extend(Path(description[k]).resolve() for k in ['fasta','preparation_dir','batch_dir'] if k in description)
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
            raise ValueError('Keep output separate from all source input directories/files')
        with batch_lock(output):
            _empty_output(output)
            with tempfile.TemporaryDirectory(prefix='.motifae-publish-',dir=output.parent) as temporary:
                stage=Path(temporary);bound.arguments['output_dir']=stage
                try:
                    result=function(*bound.args,**bound.kwargs)
                except Exception:
                    failed=stage/'binding_manifest.json'
                    if failed.is_file() and load(failed).get('status')=='blocked_sequence_binding':
                        _publish_new(stage,output)
                    raise
                _publish_new(stage,output)
                return result
    return run

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def seqsha(seq): return hashlib.sha256(seq.encode('ascii')).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        sample=f.read(8192);f.seek(0)
        if not sample.strip():raise ValueError('CSV must contain a header: '+str(path))
        return list(csv.DictReader(f,delimiter='\t' if '\t' in sample.splitlines()[0] else ','))
def write_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def canonical_id(header):
    token=header.strip().split()[0] if header.strip() else ''
    parts=token.split('|')
    return parts[1] if len(parts)>1 and parts[0] in {'sp','tr','ref','gb','emb','dbj'} else parts[0]

def fasta_records(path):
    result={};header=None;chunks=[]
    def save():
        if header is None:return
        protein=canonical_id(header);sequence=''.join(chunks).upper()
        if not protein or protein in result:raise ValueError('Duplicate or ambiguous FASTA identifier: '+protein)
        if not sequence or not re.fullmatch('[A-Z]+',sequence):raise ValueError('Invalid FASTA sequence: '+protein)
        result[protein]=(sequence,header)
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            line=line.strip()
            if not line:continue
            if line.startswith('>'):save();header=line[1:];chunks=[]
            elif header is None:raise ValueError('FASTA sequence without header')
            else:chunks.append(line)
    save()
    if not result:raise ValueError('Empty FASTA background')
    return result

def safe_file(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Unsafe source path: '+str(name))
    return path

def verify_hashes(root,hashes,required=()):
    if not isinstance(hashes,dict):raise ValueError('Incomplete source files_sha256')
    normalized={str(name).replace('\\','/'):digest for name,digest in hashes.items()}
    if len(normalized)!=len(hashes) or not {str(n).replace('\\','/') for n in required}<=set(normalized):
        raise ValueError('Incomplete or duplicate source files_sha256')
    for name,digest in hashes.items():
        path=safe_file(root,name)
        if not path.is_file() or sha256(path)!=digest:raise ValueError('Changed or missing source hash: '+str(path))

def validate_slice(row,sequences,start_field,end_field,seq_field):
    protein=row['parent_id']
    if protein not in sequences:raise ValueError('Unknown unmapped protein ID: '+protein)
    if row.get('sequence_id')!=protein:raise ValueError('Sequence/parent ID ambiguity: '+protein)
    start=int(row[start_field]);end=int(row[end_field]);seq=sequences[protein][0]
    if not 0<=start<end<=len(seq) or seq[start:end]!=row[seq_field]:
        raise ValueError('Protein sequence slice mismatch: '+protein)
    return start,end

def _family(run):
    family=str(run['family'])
    if family in ('.','..') or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',family):raise ValueError('Unsafe family name')
    fasta=Path(run['fasta']);batch=Path(run['batch_dir']);prep=Path(run['preparation_dir'])
    sequences=fasta_records(fasta);manifest=load(batch/'batch_manifest.json')
    signature=manifest.get('signature',{})
    if signature.get('fasta_sha256')!=sha256(fasta):raise ValueError('FASTA hash mismatch or missing provenance')
    if signature.get('preparation_sha256')!=sha256(prep/'gibbs_preparation_manifest.json'):
        raise ValueError('Preparation manifest hash mismatch or missing provenance')
    pm=load(prep/'gibbs_preparation_manifest.json');verify_hashes(prep,pm.get('files_sha256'),['feature_index.csv'])
    entries=read_csv(prep/'feature_index.csv');by_uid={}
    for entry in entries:
        uid=entry['motif_uid']
        if uid in by_uid or entry['family']!=family or entry['method']!='MotifAE':raise ValueError('Duplicate or incorrect preparation feature')
        by_uid[uid]=entry
        for field in ['mapping_file','peptide_file']:
            if entry.get(field) and entry[field] not in pm['files_sha256']:raise ValueError('Unverified preparation reference')
        mappings=read_csv(prep/entry['mapping_file']);seen=set()
        for row in mappings:
            if (row['family'],row['method'],row['motif_uid'])!=(family,'MotifAE',uid):raise ValueError('Preparation feature identity mismatch')
            validate_slice(row,sequences,'occ_start','occ_end','motif_sequence')
            key=(row['parent_id'],int(row['occ_start']),int(row['occ_end']))
            if key in seen:raise ValueError('Duplicate selected occurrence')
            seen.add(key)
        if len(mappings)!=int(entry['selected_peptides']):raise ValueError('Preparation selected instance count mismatch')
        if entry.get('peptide_file'):
            peptides=(prep/entry['peptide_file']).read_text(encoding='ascii').splitlines()
            if Counter(peptides)!=Counter(r['motif_sequence'] for r in mappings):raise ValueError('Preparation peptide/mapping mismatch')
        entry['_mapping']={str(r['selection_rank']):r for r in mappings}
        if len(entry['_mapping'])!=len(mappings):raise ValueError('Duplicate selection rank')
    jobs=manifest.get('jobs',[]);job_ids=[j['motif_uid'] for j in jobs]
    if len(set(job_ids))!=len(jobs):raise ValueError('Duplicate batch feature')
    ready={u for u,e in by_uid.items() if e.get('peptide_file')}
    if set(job_ids)!=ready:raise ValueError('Batch/preparation feature background mismatch')
    if manifest.get('failed_groups',0) or any(j.get('status')!='success' for j in jobs):raise ValueError('Batch contains incomplete groups')
    if int(manifest.get('total_groups',len(jobs)))!=len(jobs):raise ValueError('Batch group count mismatch')
    occurrences=[];catalog=[];seen_occ=set();reports=[]
    for job in jobs:
        uid=job['motif_uid'];parsed=safe_file(batch,job['parsed_dir'])
        names=[str((parsed/name).relative_to(batch.resolve())).replace('\\','/') for name in ['parse_manifest.json','aligned_occurrences.csv']]
        verify_hashes(batch,job.get('files_sha256'),names)
        report=load(parsed/'parse_manifest.json')
        for key in ['family','method','motif_uid','pattern_built','mapped_instances']:
            if report.get(key)!=job.get('report',{}).get(key):raise ValueError('Batch/parsed report mismatch: '+key)
        if (report.get('family'),report.get('method'),report.get('motif_uid'))!=(family,'MotifAE',uid):raise ValueError('Incorrect parsed feature identity')
        if report.get('localization_scope')!=SCOPE:raise ValueError('Unknown localization scope')
        reports.append(report);rows=read_csv(parsed/'aligned_occurrences.csv')
        if len(rows)!=int(report['mapped_instances']):raise ValueError('Mapped instance count mismatch')
        if bool(rows)!=bool(report['pattern_built']):raise ValueError('Pattern/occurrence inconsistency')
        guard=job.get('guard_event',{})
        sampler=job.get('sampler_kind') or ('native_empty_g1_move_guard' if guard else 'native_gibbscluster')
        provenance=json.dumps(dict(guard_event=guard,engineering_fix=job.get('engineering_fix'),
            batch_engineering_fix=manifest.get('engineering_fix'),source_attempt=job.get('source_attempt'),
            source_job_manifest_sha256=job.get('source_job_manifest_sha256')),sort_keys=True,separators=(',',':'))
        for row in rows:
            if (row['family'],row['method'],row['motif_uid'])!=(family,'MotifAE',uid):raise ValueError('Occurrence feature mismatch')
            start,end=validate_slice(row,sequences,'occ_start','occ_end','motif_sequence')
            validate_slice(row,sequences,'candidate_start','candidate_end','candidate_sequence')
            origin=by_uid[uid]['_mapping'].get(str(row['selection_rank']))
            if origin is None or any(str(origin[k])!=str(row[v]) for k,v in [('parent_id','parent_id'),('occ_start','candidate_start'),('occ_end','candidate_end'),('motif_sequence','candidate_sequence')]):raise ValueError('Occurrence missing from selected preparation mapping')
            pos=json.loads(row['core_residue_positions_0based']);core=row['aligned_core'];seq=sequences[row['parent_id']][0]
            actual=[p for p in pos if p is not None]
            if len(pos)!=int(pm['core_length']) or len(core)!=len(pos) or not actual or any(type(p) is not int or not 0<=p<len(seq) for p in actual):raise ValueError('Invalid exact core positions')
            if actual!=sorted(set(actual)) or (min(actual),max(actual)+1)!=(start,end):raise ValueError('Core/bounding coordinates mismatch')
            if any(not int(row['candidate_start'])<=p<int(row['candidate_end']) for p in actual):raise ValueError('Core escapes candidate')
            if ''.join('-' if p is None else seq[p] for p in pos)!=core:raise ValueError('Exact core sequence mismatch')
            if row['localization_scope']!=SCOPE:raise ValueError('Occurrence scope mismatch')
            key=(uid,row['parent_id'],row['selection_rank'])
            if key in seen_occ:raise ValueError('Duplicate aligned occurrence')
            seen_occ.add(key)
            row.update(sampler_kind=sampler,guard_provenance=provenance,source_model='MotifAE_GibbsCluster',consensus_or_label=uid,
                       alignment_width=len(pos),mapped_residue_count=len(actual),span_length=end-start)
            # A gapped core has no contiguous core interval: do not manufacture one.
            occurrences.append(row)
        if rows:
            catalog.append(dict(family=family,method='MotifAE',motif_uid=uid,consensus_or_label=uid,
                n_occurrences=len(rows),n_proteins=len({r['parent_id'] for r in rows}),
                score=max(float(r['score']) for r in rows),source_model='MotifAE_GibbsCluster',
                localization_scope=SCOPE,sampler_kind=sampler,guard_provenance=provenance))
    summary=dict(status='completed' if catalog else ('all_trash' if jobs else 'no_candidates'),
        motifs=len(catalog),occurrences=len(occurrences),background_proteins=len(sequences),
        foreground_proteins=len({r['parent_id'] for r in occurrences}),groups=len(jobs),
        unassessed_instances=sum(int(r.get('unassessed_instances',0)) for r in reports),
        unassessed_status='unknown_not_negative',localization_scope=SCOPE,
        batch_manifest_sha256=sha256(batch/'batch_manifest.json'),preparation_sha256=signature['preparation_sha256'],
        fasta_sha256=signature['fasta_sha256'],source_verification='complete')
    return family,sequences,occurrences,catalog,summary

@_new_output_transaction
def export_handoff(family_runs,output_dir):
    """Validate all runs before writing a new, fully sequence-bound handoff."""
    if not isinstance(family_runs,list) or not family_runs:
        raise ValueError('Handoff requires a complete family-runs list; resolve incomplete alignment families first')
    output=Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Handoff output must be new/empty')
    if any(not isinstance(r,dict) for r in family_runs):raise ValueError('Invalid family run description')
    if len({str(r.get('family','')) for r in family_runs})!=len(family_runs):raise ValueError('Duplicate family run')
    for run in family_runs:
        for field in ['fasta','batch_dir','preparation_dir']:
            source=Path(run[field]).resolve()
            if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Keep handoff output separate from source inputs')
    global_seqs={};sequence_rows=[];summary={}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.handoff-',dir=output.parent) as temp:
        stage=Path(temp)
        for run in family_runs:
            family,sequences,occurrences,catalog,status=_family(run)
            summary[family]=status
            for protein,(seq,header) in sequences.items():
                if protein in global_seqs and global_seqs[protein]!=seq:raise ValueError('Conflicting cross-family sequences: ambiguous ID '+protein)
                global_seqs[protein]=seq
                sequence_rows.append(dict(sequence_id=protein,protein_id=protein,family=family,sequence=seq,
                    sequence_length=len(seq),sequence_sha256=seqsha(seq),fasta_path=str(Path(run['fasta']).resolve()),description=header))
            write_csv(stage/family/'MotifAE/motif_occurrences_MotifAE.csv',occurrences,OCC_FIELDS)
            write_csv(stage/family/'MotifAE/motif_catalog_MotifAE.csv',catalog,CAT_FIELDS)
            dump(stage/family/'MotifAE/handoff_status.json',status)
            del occurrences,catalog,sequences
        for name in ['sequence_index.csv','sequence_table.csv']:write_csv(stage/name,sequence_rows,SEQ_FIELDS)
        write_csv(stage/'protein_sequence_sha256.csv',[dict(protein_id=p,sequence_sha256=seqsha(s)) for p,s in sorted(global_seqs.items())],['protein_id','sequence_sha256'])
        (stage/'all_gvp_proteins.fasta').write_text(''.join('>'+p+'\n'+s+'\n' for p,s in sorted(global_seqs.items())),encoding='ascii')
        result=dict(status='handoff_complete_annotations_pending',families=summary,unique_background_proteins=len(global_seqs),
            family_background_memberships=len(sequence_rows),localization_scope=SCOPE,biological_function_validated=False,
            minimum_support_filter_applied=False,cross_feature_merge=False,exporter_sha256=sha256(Path(__file__)),
            files_sha256={str(p.relative_to(stage)).replace('\\','/'):sha256(p) for p in stage.rglob('*') if p.is_file()})
        dump(stage/'handoff_manifest.json',result)
        output.mkdir(parents=True,exist_ok=True)
        for p in list(stage.iterdir()):os.replace(p,output/p.name)
    return result

def verify_handoff(handoff_dir):
    root=Path(handoff_dir);manifest=load(root/'handoff_manifest.json')
    verify_hashes(root,manifest.get('files_sha256'),['sequence_index.csv','sequence_table.csv','all_gvp_proteins.fasta','protein_sequence_sha256.csv'])
    return manifest


@_new_output_transaction
def prepare_exp5(handoff_dir,output_dir):
    """Make sequence-bound sets with an explicit sampling-scope sidecar."""
    handoff=Path(handoff_dir).resolve();manifest=verify_handoff(handoff);output=Path(output_dir).resolve()
    if output.is_relative_to(handoff) or handoff.is_relative_to(output):raise ValueError('Keep Exp5 output separate from the handoff')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Exp5 preparation output must be new/empty')
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from src.exp5_external_functional_enrichment.package.prepare import read_sequence_index
    from src.exp5_external_functional_enrichment.package.motif_sets import build_motif_sets
    from src.exp5_external_functional_enrichment.package.clusterprofiler import write_clusterprofiler_r_script
    proteins=read_sequence_index(handoff/'sequence_index.csv')
    backgrounds={f:{r.protein_id for r in proteins if r.family==f} for f in manifest['families']}
    output.parent.mkdir(parents=True,exist_ok=True)
    scope_fields=['family','method','motif_cluster','localization_scope','sampler_kind','unselected_status']
    set_fields=['family','method','motif_cluster','set_type','protein_id']
    counts={};total_rows=0
    with tempfile.TemporaryDirectory(prefix='.exp5-prepare-',dir=output.parent) as temp:
        stage=Path(temp)
        with (stage/'motif_sets_long.csv').open('w',encoding='utf-8',newline='') as sh,(stage/'motif_set_scope.csv').open('w',encoding='utf-8',newline='') as ch:
            writer=csv.DictWriter(sh,fieldnames=set_fields);writer.writeheader()
            scope_writer=csv.DictWriter(ch,fieldnames=scope_fields);scope_writer.writeheader()
            for family in manifest['families']:
                occurrences=read_csv(handoff/family/'MotifAE/motif_occurrences_MotifAE.csv')
                expected={}
                for row in occurrences:
                    if row['parent_id'] not in backgrounds[family]:raise ValueError('Foreground ID absent from current family universe')
                    expected.setdefault(row['motif_uid'],set()).add(row['parent_id'])
                sets=build_motif_sets(occurrences,proteins)
                if len(sets)!=len(expected) or any(set(s.foreground_ids)!=expected[s.motif_cluster] or set(s.universe_ids)!=backgrounds[family] for s in sets):
                    raise ValueError('Exp5 foreground/universe membership changed during preparation')
                catalog={r['motif_uid']:r for r in read_csv(handoff/family/'MotifAE/motif_catalog_MotifAE.csv')}
                for s in sets:
                    for kind,ids in [('foreground',s.foreground_ids),('background',s.background_ids),('universe',s.universe_ids)]:
                        for protein in ids:
                            writer.writerow(dict(family=family,method='MotifAE',motif_cluster=s.motif_cluster,set_type=kind,protein_id=protein));total_rows+=1
                    scope_writer.writerow(dict(family=family,method='MotifAE',motif_cluster=s.motif_cluster,localization_scope=SCOPE,sampler_kind=catalog[s.motif_cluster]['sampler_kind'],unselected_status='unknown_not_negative'))
                counts[family]=dict(patterns=len(sets),background_proteins=len(backgrounds[family]))
        write_clusterprofiler_r_script(stage/'run_clusterprofiler_enricher.R')
        result=dict(status='motif_sets_prepared_annotations_pending',families=counts,rows=total_rows,
            localization_scope=SCOPE,inference_target='membership_in_selected_retained_alignment_instances',
            complement_interpretation='not_selected_retained_is_not_proof_of_motif_absence',
            biological_function_validated=False,handoff_manifest_sha256=sha256(handoff/'handoff_manifest.json'),
            files_sha256={p.name:sha256(p) for p in stage.iterdir() if p.is_file()})
        dump(stage/'exp5_preparation_manifest.json',result)
        output.mkdir(parents=True,exist_ok=True)
        for path in list(stage.iterdir()):os.replace(path,output/path.name)
    return result

@_new_output_transaction
def bind_annotations(handoff_dir,term2gene,output_dir,*,annotation_fasta=None,annotation_sequence_hashes=None):
    """Bind an annotation table using its original FASTA or per-ID SHA256 table.

    A complete older annotation universe may be subsetted only when every current
    background ID has the identical original sequence. An empty term table is
    valid evidence of no terms and never creates enrichment positives.
    """
    verify_handoff(handoff_dir);output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Annotation output must be new/empty')
    if annotation_fasta is not None and annotation_sequence_hashes is not None:raise ValueError('Choose one annotation sequence provenance source')
    current={r['protein_id']:r['sequence_sha256'] for r in read_csv(Path(handoff_dir)/'sequence_index.csv')}
    source={};provenance={}
    if annotation_fasta is not None:
        source={p:seqsha(s) for p,(s,h) in fasta_records(annotation_fasta).items()}
        provenance=dict(annotation_input_fasta_sha256=sha256(annotation_fasta))
    elif annotation_sequence_hashes is not None:
        for row in read_csv(annotation_sequence_hashes):
            p=row['protein_id'];digest=row['sequence_sha256']
            if p in source or not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Duplicate ID or invalid annotation sequence hash')
            source[p]=digest
        provenance=dict(annotation_sequence_hashes_sha256=sha256(annotation_sequence_hashes))
    missing=[dict(protein_id=p,sequence_sha256=h,reason=('unknown_annotation_source' if not provenance else 'missing_in_annotation_source' if p not in source else 'sequence_mismatch')) for p,h in sorted(current.items()) if source.get(p)!=h]
    if missing:
        write_csv(output/'reannotation_required.csv',missing,['protein_id','sequence_sha256','reason'])
        dump(output/'binding_manifest.json',dict(status='blocked_sequence_binding',required_reannotation=len(missing),annotation_table_sha256=sha256(term2gene),source=provenance))
        raise ValueError('Annotation source provenance / sequence binding failed; see reannotation_required.csv')
    rows=read_csv(term2gene);kept=[];seen=set()
    for row in rows:
        gene=str(row.get('protein_id',row.get('gene',''))).strip();term=str(row.get('term_id',row.get('term',''))).strip()
        if not gene or not term or gene not in source:raise ValueError('Annotation table has missing term/gene or unknown source ID')
        if (term,gene) in seen:raise ValueError('Duplicate annotation term/gene pair')
        seen.add((term,gene))
        if gene in current:kept.append(dict(term_id=term,protein_id=gene))
    write_csv(output/'term2gene.csv',kept,['term_id','protein_id'])
    write_csv(output/'term2name.csv',[dict(term_id=t,term_name=t) for t in sorted({r['term_id'] for r in kept})],['term_id','term_name'])
    write_csv(output/'reannotation_required.csv',[],['protein_id','sequence_sha256','reason'])
    result=dict(status='annotations_bound' if kept else 'annotations_bound_no_terms',source=provenance,
        annotation_table_sha256=sha256(term2gene),handoff_manifest_sha256=sha256(Path(handoff_dir)/'handoff_manifest.json'),
        bound_background_proteins=len(current),original_source_proteins=len(source),terms=len({r['term_id'] for r in kept}),
        term_gene_pairs=len(kept),biological_function_validated=False,
        files_sha256={name:sha256(output/name) for name in ['term2gene.csv','term2name.csv','reannotation_required.csv']})
    dump(output/'binding_manifest.json',result)
    return result


def verify_annotations(handoff_dir,binding_dir):
    manifest=verify_handoff(handoff_dir);root=Path(binding_dir);bound=load(root/'binding_manifest.json')
    if bound.get('status') not in ('annotations_bound','annotations_bound_no_terms'):
        raise ValueError('Annotation sequence binding has not completed')
    if bound.get('handoff_manifest_sha256')!=sha256(Path(handoff_dir)/'handoff_manifest.json'):
        raise ValueError('Annotation binding belongs to another handoff')
    if bound.get('bound_background_proteins')!=manifest['unique_background_proteins']:
        raise ValueError('Annotation/background universe binding differs')
    verify_hashes(root,bound.get('files_sha256'),['term2gene.csv','term2name.csv'])
    return bound

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    exp=sub.add_parser('export');exp.add_argument('--family-runs',type=Path,required=True,help='JSON list with family,batch_dir,preparation_dir,fasta');exp.add_argument('--output-dir',type=Path,required=True)
    ver=sub.add_parser('verify');ver.add_argument('--handoff-dir',type=Path,required=True)
    exp5=sub.add_parser('prepare-exp5');exp5.add_argument('--handoff-dir',type=Path,required=True);exp5.add_argument('--output-dir',type=Path,required=True)
    verify_ann=sub.add_parser('verify-annotations');verify_ann.add_argument('--handoff-dir',type=Path,required=True);verify_ann.add_argument('--binding-dir',type=Path,required=True)
    ann=sub.add_parser('bind-annotations');ann.add_argument('--handoff-dir',type=Path,required=True);ann.add_argument('--term2gene',type=Path,required=True);ann.add_argument('--output-dir',type=Path,required=True);ann.add_argument('--annotation-fasta',type=Path);ann.add_argument('--annotation-sequence-hashes',type=Path)
    args=parser.parse_args()
    if args.command=='export':result=export_handoff(load(args.family_runs),args.output_dir)
    elif args.command=='verify':result=verify_handoff(args.handoff_dir)
    elif args.command=='prepare-exp5':result=prepare_exp5(args.handoff_dir,args.output_dir)
    elif args.command=='verify-annotations':result=verify_annotations(args.handoff_dir,args.binding_dir)
    else:result=bind_annotations(args.handoff_dir,args.term2gene,args.output_dir,annotation_fasta=args.annotation_fasta,annotation_sequence_hashes=args.annotation_sequence_hashes)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

