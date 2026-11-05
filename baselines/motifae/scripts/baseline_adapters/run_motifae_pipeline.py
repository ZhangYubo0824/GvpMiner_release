#!/usr/bin/env python3
"""Official-model inference -> declared peptide extraction -> Gibbs input files.

The public author code supplies the model; preprint-v1 supplies the explicit
segmentation rules. Prepared inputs or successful external commands are never
reported as completed, biologically validated PSSM motifs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from motifae_candidate_filter import export_results, process_candidates, sha256
from motifae_gibbs_inputs import prepare_inputs
from motifae_official_inference import (extract_peptides, load_autoencoder,
                                       load_cached_records, load_fasta_records)

ESM_SHA256='ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c'
ESM_REGRESSION_SHA256='8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575'


def load_local_esm(checkpoint):
    """Load content-verified official files without a network/cache lookup."""
    checkpoint=Path(checkpoint).resolve()
    if not checkpoint.is_file() or sha256(checkpoint)!=ESM_SHA256:
        raise ValueError('Local ESM checkpoint hash does not match the verified official ESM2-650M file')
    regression=checkpoint.with_name('esm2_t33_650M_UR50D-contact-regression.pt')
    if not regression.is_file() or sha256(regression)!=ESM_REGRESSION_SHA256:
        raise ValueError('Local official ESM contact-regression file is missing or has a different hash')
    import torch
    import esm
    # These exact hash-verified FAIR files contain configuration objects.
    # Explicit loading also works with PyTorch 2.6+ weights_only defaults.
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    reg=torch.load(regression,map_location='cpu',weights_only=False)
    model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',data,reg)
    return model,alphabet,dict(checkpoint=str(checkpoint),sha256=ESM_SHA256,
        regression=str(regression),regression_sha256=ESM_REGRESSION_SHA256,network_lookup=False)


def esm_embeddings(sequence,label,model,converter,device):
    import torch
    _,_,tokens=converter([(label,sequence)])
    with torch.inference_mode():
        output=model(tokens.to(device),repr_layers=[33])
    array=output['representations'][33][0,1:len(sequence)+1].cpu().numpy()
    if array.shape!=(len(sequence),1280):
        raise ValueError(f'{label}: unexpected ESM-2 residue embedding shape {array.shape}')
    return array


def _write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def run_pipeline(*,family,checkpoint,output_dir,input_dir=None,fasta=None,device='auto',
                 activation_protocol='preprint-v1',
                 chunk_size=128,long_sequence_policy='full',
                 esm_checkpoint=None):
    if (input_dir is None)==(fasta is None):
        raise ValueError('Choose exactly one of --input-dir or --fasta')
    if activation_protocol!='preprint-v1' or chunk_size<1:
        raise ValueError('Declare preprint-v1 and a positive encode chunk size')
    if long_sequence_policy not in ('error','skip','full'):
        raise ValueError('long_sequence_policy must be full, error or skip')
    output_dir=Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output directory is not empty; use a new run directory')
    checkpoint=Path(checkpoint).resolve()
    inputs=[checkpoint,Path(input_dir or fasta).resolve()]
    if any(p.is_relative_to(output_dir) for p in inputs):
        raise ValueError('Output directory must not contain its source input or checkpoint')
    if input_dir is not None and output_dir.is_relative_to(Path(input_dir).resolve()):
        raise ValueError('Use an output directory outside the input cache')
    if input_dir is not None:
        records,input_metadata=load_cached_records(Path(input_dir))
    else:
        fasta=Path(fasta).resolve();records=load_fasta_records(fasta)
        input_metadata=dict(source_files=[dict(path=str(fasta),sha256=sha256(fasta))])
        if long_sequence_policy=='error' and any(len(r['sequence'])>1022 for r in records):
            raise ValueError('FASTA contains proteins longer than 1022 aa; select explicit skip policy or supply verified full-length embeddings')
        if long_sequence_policy!='full' and not any(len(r['sequence'])<=1022 for r in records):
            raise ValueError('No proteins can be evaluated under the selected length policy')
    model,model_metadata=load_autoencoder(checkpoint,device=device)
    import numpy as np
    output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(status='running_inference',implementation='motifae-guarded-first',family=family,protocol=activation_protocol,
                model=model_metadata,input=input_metadata,
                completed_motif_count=None,feature_count=None,peptide_count=None,
                processed_proteins=0,skipped_proteins=[],numpy_version=np.__version__,
                long_sequence_policy=long_sequence_policy,
                inference_scope='full_protein_no_crop_or_window_stitching' if long_sequence_policy=='full' else long_sequence_policy,
                code_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                             ['run_motifae_pipeline.py','motifae_official_inference.py',
                              'motifae_peptide_protocol.py','motifae_candidate_filter.py',
                              'motifae_gibbs_inputs.py','adapter_common.py']})
    _write_json(output_dir/'pipeline_manifest.json',report)
    try:
        esm_model=converter=None
        if fasta is not None:
            import esm
            if model.activation_dim!=1280:
                raise ValueError('FASTA/ESM-2 mode requires a 1280-dimensional checkpoint')
            if esm_checkpoint is not None:
                esm_model,alphabet,esm_metadata=load_local_esm(esm_checkpoint)
                report['esm_weights']=esm_metadata
            else:
                esm_model,alphabet=esm.pretrained.esm2_t33_650M_UR50D()
            esm_model=esm_model.to(model_metadata['device']).eval()
            converter=alphabet.get_batch_converter()
            report['esm_model']='esm2_t33_650M_UR50D, layer 33, raw embeddings'
        rows=[]
        for index,record in enumerate(records,start=1):
            sequence=record['sequence']
            if fasta is not None:
                if len(sequence)>1022 and long_sequence_policy=='skip':
                    report['skipped_proteins'].append(dict(parent_id=record['parent_id'],reason='length_gt_1022_explicit_skip'))
                    continue
                embedding=esm_embeddings(sequence,record['safe_id'],esm_model,converter,model_metadata['device'])
                cache=output_dir/'esm_embeddings'/f"{record['safe_id']}.npz"
                cache.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(cache,repr=embedding)
            else:
                with np.load(record['embedding_path'],allow_pickle=False) as archive:
                    embedding=archive['repr']
            rows.extend(extract_peptides(model,embedding,sequence,record['parent_id'],family,
                                          protocol=activation_protocol,chunk_size=chunk_size))
            report['processed_proteins']+=1
            if index%100==0 or index==len(records):
                print(f"[PROGRESS] {index}/{len(records)} records; {len(rows)} peptide occurrences",file=sys.stderr,flush=True)
        if report['processed_proteins']==0:
            raise ValueError('No proteins were evaluated; do not report zero discovered motifs')
        inventory=process_candidates(rows)
        feature_manifest=export_results(output_dir/'features','MotifAE',inventory,
                                         dict(mode='official_model_pipeline',model=model_metadata,
                                              activation_protocol=activation_protocol,input=input_metadata))
        report.update(feature_count=inventory['counts']['retained_features'],
                      peptide_count=inventory['counts']['retained_occurrences'],feature_counts=feature_manifest['counts'])
        source_csv=output_dir/'features/motif_occurrences_MotifAE_raw.csv'
        preparation=prepare_inputs(inventory['raw_occurrences'],output_dir/'gibbs_inputs',
                                   source_protocol='legacy_preprint_v1',
                                   source_files=[dict(path=str(source_csv),sha256=sha256(source_csv))])
        report['gibbs_preparation_counts']=preparation['counts']
        report['status']='awaiting_gibbscluster_2_0'
        _write_json(output_dir/'pipeline_manifest.json',report)
        return report
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        _write_json(output_dir/'pipeline_manifest.json',report)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir',type=Path,help='Existing motifae_input.csv, optional id_map.csv and embeddings/*.npz')
    inputs.add_argument('--fasta',type=Path,help='FASTA input; requires fair-esm and ESM-2 650M weights')
    parser.add_argument('--family',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--activation-protocol',choices=['preprint-v1'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--chunk-size',type=int,default=128)
    parser.add_argument('--long-sequence-policy',choices=['full','error','skip'],default='full',
                        help='full preserves the complete target sequence; no truncation or window stitching')
    parser.add_argument('--esm-checkpoint',type=Path,help='Local official ESM2 checkpoint with co-located contact-regression file; avoids download')
    args=parser.parse_args()
    result=run_pipeline(**vars(args))
    print(json.dumps({k:result[k] for k in ['status','feature_count','peptide_count','completed_motif_count']},ensure_ascii=False))


if __name__=='__main__':
    main()
