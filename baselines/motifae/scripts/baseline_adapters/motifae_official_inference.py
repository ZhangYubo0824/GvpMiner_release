"""Direct inference through the pinned, unmodified MotifAE AutoEncoder."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

from adapter_common import find_first, read_delimited
from motifae_candidate_filter import canonical_protein_id, sha256

VENDOR=Path(__file__).parent/'vendor/motifae'
RELEASED_MD5={
    'cabb07008f26705f07e18344607a2d51':'k3',
}
WEIGHTS_SOURCE='https://zenodo.org/records/20547228'


def official_model_class():
    metadata=json.loads((VENDOR/'UPSTREAM.json').read_text(encoding='utf-8'))
    source=VENDOR/'dictionary.py'
    digest=sha256(source)
    if digest!=metadata['sha256']:
        raise ValueError('Vendored official model source differs from the pinned upstream file')
    name='motifae_author_dictionary_'+digest[:16]
    spec=importlib.util.spec_from_file_location(name,source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load official model source {source}')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoEncoder,dict(source_path=str(source.resolve()),source_sha256=digest,
                                   pinned_sha256=metadata['sha256'],upstream_commit=metadata['commit'],
                                   upstream=metadata['upstream'],pinned_source_verified=digest==metadata['sha256'])


def checkpoint_hashes(path):
    md5=hashlib.md5()
    sha=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            md5.update(block);sha.update(block)
    return md5.hexdigest(),sha.hexdigest()


def load_autoencoder(checkpoint,device='cpu'):
    checkpoint=Path(checkpoint)
    md5,digest=checkpoint_hashes(checkpoint)
    variant=RELEASED_MD5.get(md5)
    if variant!='k3' or digest!='e322c859a593c6c3dba461a2fae6c648ccd064a46422221f05f25c6eac03f28d':
        raise ValueError('Not a verified published checkpoint: this release requires official Zenodo k3/80000 weights')
    import torch
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; select --device cpu')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key in ('model_state_dict','state_dict'):
        if isinstance(state,dict) and key in state:
            state=state[key];break
    required={'bias','encoder.weight','encoder.bias','decoder.weight'}
    if not isinstance(state,dict) or set(state)!=required:
        found=set(state) if isinstance(state,dict) else set()
        raise ValueError(f'Official checkpoint schema mismatch: missing {sorted(required-found)}, extra {sorted(found-required)}')
    weight=state['encoder.weight']
    if not isinstance(weight,torch.Tensor) or weight.ndim!=2:
        raise ValueError('encoder.weight must be a two-dimensional tensor')
    features,dimension=weight.shape
    if (features,dimension)!=(80000,1280):
        raise ValueError('Published k3 checkpoint must have 80000 features and 1280 embedding dimensions')
    shapes={'encoder.weight':(features,dimension),'encoder.bias':(features,),
            'bias':(dimension,),'decoder.weight':(dimension,features)}
    for key,shape in shapes.items():
        value=state[key]
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or not value.is_floating_point():
            raise ValueError(f'Invalid {key} shape or dtype; expected {shape}')
        if value.dtype!=weight.dtype or not torch.isfinite(value).all().item():
            raise ValueError(f'Inconsistent dtype or non-finite checkpoint tensor: {key}')
    cls,source=official_model_class()
    model=cls(dimension,features,tied=False).to(dtype=weight.dtype)
    model.load_state_dict(state,strict=True)
    model=model.to(device).eval()
    return model,dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest,
                      checkpoint_md5=md5,published_checkpoint_verified=variant is not None,
                      published_variant=variant,weights_source=WEIGHTS_SOURCE,
                      source=source,activation_dim=dimension,feature_dim=features,
                      device=str(next(model.parameters()).device),torch_version=torch.__version__)


def residue_view(array,sequence,activation_dim):
    import numpy as np
    if array.ndim==3:
        if array.shape[0]!=1:
            raise ValueError('Expected one protein per embedding cache')
        array=array[0]
    if array.ndim!=2 or array.shape[1]!=activation_dim:
        raise ValueError(f'Expected embedding dimension {activation_dim}; received {array.shape}')
    if array.shape[0]==len(sequence)+2:
        array=array[1:-1]
    elif array.shape[0]!=len(sequence):
        raise ValueError('Embedding length does not match protein; truncated cache has no verified crop offset')
    if not np.issubdtype(array.dtype,np.floating) or not np.isfinite(array).all():
        raise ValueError('Embedding must contain finite floating-point values')
    return array


def extract_peptides(model,embeddings,sequence,protein_id,family,*,protocol='preprint-v1',chunk_size=128):
    if protocol!='preprint-v1':
        raise ValueError('Only the explicitly documented preprint-v1 extraction protocol is implemented')
    if chunk_size<1:
        raise ValueError('chunk_size must be positive')
    import torch
    from motifae_peptide_protocol import _paper_peptide_points
    array=residue_view(embeddings,sequence,model.activation_dim)
    parameter=next(model.parameters())
    sparse=[]
    with torch.inference_mode():
        for start in range(0,len(sequence),chunk_size):
            x=torch.as_tensor(array[start:start+chunk_size],dtype=parameter.dtype,device=parameter.device)
            # Call the actual author class. Do not reimplement its matrix math.
            features=model.encode(x)
            if features.shape!=(len(x),model.dict_size) or not torch.isfinite(features).all().item():
                raise ValueError(f'{protein_id}: invalid official model activations')
            positions,indices=torch.nonzero(features>0,as_tuple=True)
            values=features[positions,indices]
            sparse.extend(zip((positions+start).cpu().tolist(),indices.cpu().tolist(),values.cpu().tolist()))
    protein_id=canonical_protein_id(protein_id)
    return [dict(family=family,method='MotifAE',motif_uid=f'{family}_MotifAE_feature_{feature}',
                 sequence_id=protein_id,parent_id=protein_id,occ_start=start,occ_end=end,
                 score=mean,significance='',motif_sequence=sequence[start:end],
                 consensus_or_label=f'feature_{feature}',source_model='MotifAE')
            for feature,start,end,mean in _paper_peptide_points(sparse)]


def _sequence(value):
    sequence=str(value).strip().upper()
    if not sequence or not re.fullmatch('[ACDEFGHIKLMNPQRSTVWYBXZJUO]+',sequence):
        raise ValueError('Empty or unsupported protein sequence')
    return sequence


def load_cached_records(input_dir):
    input_dir=Path(input_dir)
    sequence_file=find_first(input_dir,['motifae_input.csv'])
    if sequence_file is None:
        raise FileNotFoundError('Cached input needs motifae_input.csv and embeddings/*.npz')
    map_file=find_first(input_dir,['id_map.csv'])
    mapping={}
    if map_file:
        for row in read_delimited(map_file):
            safe=row.get('safe_id','').strip()
            parent=canonical_protein_id(row.get('original_id') or row.get('parent_id'))
            if not safe or not parent or safe in mapping:
                raise ValueError('Missing or duplicate identifier in id_map.csv')
            mapping[safe]=parent
    records=[];seen_safe=set();parents={};duplicate_parents=[]
    sources=[sequence_file]+([map_file] if map_file else [])
    for row in read_delimited(sequence_file):
        safe=(row.get('uniprot') or row.get('safe_id') or row.get('sequence_id') or '').strip()
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',safe) or safe in ('.','..') or safe in seen_safe:
            raise ValueError('Unsafe, empty or duplicate embedding identifier')
        seen_safe.add(safe)
        sequence=_sequence(row.get('sequence') or row.get('random_subregion') or '')
        parent=mapping.get(safe) or canonical_protein_id(safe)
        if parent in parents:
            if parents[parent]!=sequence:
                raise ValueError(f'Conflicting sequences for canonical parent {parent}')
            duplicate_parents.append(safe)
            continue
        path=input_dir/'embeddings'/f'{safe}.npz'
        if not path.is_file():
            raise FileNotFoundError(path)
        parents[parent]=sequence
        records.append(dict(safe_id=safe,parent_id=parent,sequence=sequence,embedding_path=path))
        sources.append(path)
    if not records:
        raise ValueError('No proteins in cached input')
    orphan=sorted(p.name for p in (input_dir/'embeddings').glob('*.npz') if p.stem not in seen_safe)
    return records,dict(source_files=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in sources],
                        duplicate_parent_aliases_skipped=duplicate_parents,orphan_embedding_files=orphan)


def load_fasta_records(path):
    from adapter_common import read_fasta_records
    records=[];seen={}
    for index,(header,sequence) in enumerate(read_fasta_records(Path(path)),start=1):
        parent=canonical_protein_id(header);sequence=_sequence(sequence)
        if not parent or parent in seen:
            raise ValueError('FASTA must have unique canonical protein IDs')
        seen[parent]=sequence
        records.append(dict(safe_id=f'protein_{index:06d}',parent_id=parent,sequence=sequence))
    if not records:
        raise ValueError('No protein records in FASTA')
    return records
