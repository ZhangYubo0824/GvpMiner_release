"""Prepare the feature-wise peptide inputs described in published MotifAE Methods.

Preparation is not motif discovery: GibbsCluster alignment/trash filtering and
the aligned-core frequency matrices have not run until separately verified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from adapter_common import read_delimited, write_rows
from motifae_candidate_filter import process_candidates, sha256

PAPER = 'https://doi.org/10.1038/s41467-026-77333-2'
AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')


def _prepare_files(rows, output_dir, *, source_protocol, core_length=5, source_files=None):
    if core_length not in (4, 5, 6):
        raise ValueError('Published core lengths are 4, 5, or 6 (default 5)')
    if not source_protocol:
        raise ValueError('Declare the source extraction protocol; do not infer it from CSV contents')
    output_dir = Path(output_dir)
    inventory = process_candidates(rows)
    groups = defaultdict(list)
    for row in inventory['raw_occurrences']:
        groups[(row['family'], row['method'], row['motif_uid'])].append(row)
    index = []
    too_short = selected_count = ready_count = blocked = 0
    for number, (key, members) in enumerate(sorted(groups.items()), start=1):
        eligible = [r for r in members if len(r['motif_sequence']) >= core_length]
        too_short += len(members)-len(eligible)
        # A deterministic tie-break is local bookkeeping, not a new biological
        # filter. Identical peptide sequences from different proteins remain.
        selected = sorted(eligible, key=lambda r: (-float(r['score']), r['parent_id'],
                          int(r['occ_start']), int(r['occ_end']), r['motif_sequence']))[:1000]
        selected_count += len(selected)
        invalid = sum(not set(r['motif_sequence'].upper()) <= AMINO_ACIDS for r in selected)
        stem = f'feature_{number:06d}'
        peptide_file = f'peptides/{stem}.txt' if selected and not invalid else ''
        mapping_file = f'mappings/{stem}.csv'
        status = 'ready_for_gibbscluster' if selected and not invalid else (
            'blocked_noncanonical_amino_acids' if invalid else 'no_peptides_long_enough_for_core')
        if invalid:
            blocked += 1
        if peptide_file:
            path = output_dir/peptide_file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(r['motif_sequence'].upper()+'\n' for r in selected), encoding='ascii')
            ready_count += len(selected)
        mapping = [dict(selection_rank=i, **r) for i, r in enumerate(selected, start=1)]
        write_rows(output_dir/mapping_file, mapping,
                   ['selection_rank','family','method','motif_uid','parent_id','sequence_id',
                    'occ_start','occ_end','motif_sequence','score'])
        index.append(dict(family=key[0],method=key[1],motif_uid=key[2],
                          total_peptides=len(members), eligible_peptides=len(eligible),
                          selected_peptides=len(selected), unsupported_alphabet_peptides=invalid,
                          status=status, peptide_file=peptide_file, mapping_file=mapping_file))
    output_dir.mkdir(parents=True,exist_ok=True)
    columns = ['family','method','motif_uid','total_peptides','eligible_peptides','selected_peptides',
               'unsupported_alphabet_peptides','status','peptide_file','mapping_file']
    write_rows(output_dir/'feature_index.csv',index,columns)
    manifest = dict(paper=PAPER, published_method_section='Activated peptides and PSSM analyses, PDF page 10',
                    source_protocol=source_protocol, source_files=source_files or [],
                    core_length=core_length, max_peptides_per_feature=1000,
                    peptide_ranking='mean activation descending; stable coordinate tie-break',
                    cross_feature_merge=False, minimum_protein_support_filter=False,
                    published_gibbs_parameters=['-g','1','-l',str(core_length),'-T','-j','1','-I','1','-D','1'],
                    status='awaiting_gibbscluster_2_0', completed_motif_count=None,
                    remaining_steps=['Run licensed GibbsCluster 2.0 on each feature input',
                                     'Remove trash-cluster/outlier peptides using actual GibbsCluster assignments',
                                     'Construct 20-amino-acid frequency matrices from retained aligned cores'],
                    counts=dict(features=len(index),features_with_input=sum(bool(r['peptide_file']) for r in index),
                                raw_peptides=len(inventory['raw_occurrences']),
                                too_short_for_core=too_short,selected_peptides=selected_count,
                                ready_peptides=ready_count,blocked_alphabet_features=blocked),
                    files_sha256={})
    paths = [output_dir/'feature_index.csv']
    paths += [output_dir/r['mapping_file'] for r in index]
    paths += [output_dir/r['peptide_file'] for r in index if r['peptide_file']]
    manifest['files_sha256']={str(p.relative_to(output_dir)).replace('\\','/'):sha256(p) for p in paths}
    (output_dir/'gibbs_preparation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def _managed_path(directory, name):
    if not re.fullmatch(r'(feature_index\.csv|peptides/feature_[0-9]{6}\.txt|mappings/feature_[0-9]{6}\.csv)',name):
        raise ValueError(f'Unexpected managed output path: {name}')
    path=directory/name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f'Managed output escapes preparation directory: {name}')
    return path


def prepare_inputs(rows, output_dir, *, source_protocol, core_length=5, overwrite=False,
                   source_files=None):
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError('Preparation directory exists; choose a new directory or --overwrite')
    manifest_path=output_dir/'gibbs_preparation_manifest.json'
    previous_names=set()
    if manifest_path.is_file():
        previous_names=set(json.loads(manifest_path.read_text(encoding='utf-8'))['files_sha256'])
        for name in previous_names:
            _managed_path(output_dir,name)
    output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir,prefix='.gibbs-preparation-') as temporary:
        staging=Path(temporary)
        manifest=_prepare_files(rows,staging,source_protocol=source_protocol,
                                 core_length=core_length,source_files=source_files)
        current_names=set(manifest['files_sha256'])
        for name in current_names:
            target=_managed_path(output_dir,name)
            if target.exists() and name not in previous_names:
                raise FileExistsError(f'Refusing to overwrite an untracked file: {target}')
        # Invalidate completion metadata before publishing a changed managed
        # set. Never delete source/unrelated files or paths outside this root.
        manifest_path.unlink(missing_ok=True)
        for name in previous_names-current_names:
            _managed_path(output_dir,name).unlink(missing_ok=True)
        for name in sorted(current_names):
            target=_managed_path(output_dir,name)
            target.parent.mkdir(parents=True,exist_ok=True)
            os.replace(staging/name,target)
        os.replace(staging/'gibbs_preparation_manifest.json',manifest_path)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--occurrences',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--source-protocol',required=True,
                        choices=['legacy_preprint_v1','published_input_user_supplied'])
    parser.add_argument('--core-length',type=int,choices=[4,5,6],default=5)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.occurrences.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError('Keep source occurrence files outside the preparation directory')
    manifest=prepare_inputs(read_delimited(args.occurrences),args.output_dir,
                            source_protocol=args.source_protocol,core_length=args.core_length,
                            overwrite=args.overwrite,source_files=[dict(path=str(args.occurrences.resolve()),sha256=sha256(args.occurrences))])
    print(json.dumps(dict(status=manifest['status'],counts=manifest['counts']),ensure_ascii=False))


if __name__=='__main__':
    main()
