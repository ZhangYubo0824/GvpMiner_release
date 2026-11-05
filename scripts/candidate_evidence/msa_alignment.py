"""Validated full-family MAFFT alignment cache; no conservation interpretation."""
from __future__ import annotations
import hashlib
import gzip
import json
import os
from pathlib import Path
import subprocess
import shutil
from scripts.paths import ROOT, table_path, content_sha256
import tempfile
import pandas as pd


def atomic_write_bytes(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode('utf-8'))


def write_csv(path, frame):
    atomic_write_text(path, frame.to_csv(index=False))

def residue_to_column(sequence):
    return [i for i, char in enumerate(sequence) if char != '-']

def validate_alignment(inputs, alignment):
    if set(inputs) != set(alignment):
        raise ValueError('Alignment identifiers do not equal input identifiers')
    if len(set(map(len, alignment.values()))) != 1:
        raise ValueError('Alignment has unequal row lengths')
    for identifier, sequence in inputs.items():
        if alignment[identifier].replace('-', '').upper() != sequence.upper():
            raise ValueError(f'Ungapped alignment differs: {identifier}')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_fasta(path):
    sequences, key = {}, None
    path=Path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as handle:text=handle.read()
    for line in text.splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]
            if key in sequences:
                raise ValueError('Duplicate FASTA identifier')
            sequences[key] = ''
        elif line.strip():
            sequences[key] += line.strip().upper()
    return sequences

def write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))

def alignment_reusable(input_path, alignment_path, manifest_path, fasta, threads):
    if not (manifest_path.exists() and alignment_path.exists() and input_path.exists()):
        return False
    old = json.loads(manifest_path.read_text())
    return (input_path.read_text(encoding='ascii') == fasta and
            old.get('input_sha256') == sha256(input_path) and
            old.get('alignment_sha256') == sha256(alignment_path) and
            old.get('threads') == threads)

def align_family(family, metadata, folder, mafft, threads):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    inputs={f's{i:06d}':str(row.sequence).upper() for i,row in enumerate(metadata.itertuples(index=False))}
    identities={f's{i:06d}':str(row.protein_id) for i,row in enumerate(metadata.itertuples(index=False))}
    input_path,alignment_path=folder/'input.fasta',folder/'alignment.fasta'
    fasta=''.join(f'>{key}\n{value}\n' for key,value in inputs.items())
    manifest_path=folder/'alignment_manifest.json'
    reusable=alignment_reusable(input_path,alignment_path,manifest_path,fasta,threads)
    frozen=ROOT/'data/reference/alignments'/family
    restored=False
    if not reusable and frozen.is_dir():
        record=json.loads((frozen/'alignment_manifest.json').read_text(encoding='utf-8'))
        frozen_input=table_path(frozen/'input.fasta');frozen_alignment=table_path(frozen/'alignment.fasta')
        if content_sha256(frozen_input)!=record['input_sha256'] or content_sha256(frozen_alignment)!=record['alignment_sha256']:
            raise ValueError('Frozen alignment content hash mismatch')
        if read_fasta(frozen_input)==inputs:
            for source,target in [(frozen_input,input_path),(frozen_alignment,alignment_path)]:
                opener=gzip.open if source.suffix=='.gz' else open
                with opener(source,'rb') as src:atomic_write_bytes(target,src.read())
            restored=True
    write_csv(folder/'identifiers.csv',pd.DataFrame([{'alignment_id':k,'protein_id':v} for k,v in identities.items()]))
    command=None
    if not reusable and not restored:
        atomic_write_bytes(input_path,fasta.encode('ascii'))
        executable=shutil.which(str(mafft)) or (str(Path(mafft).resolve()) if Path(mafft).is_file() else None)
        if executable is None:raise FileNotFoundError('MAFFT is required for inputs without a matching frozen alignment; set --mafft.')
        command=[executable,'--amino','--retree','1','--maxiterate','0','--thread',str(threads)]
        if str(executable).lower().endswith(('.bat','.cmd')):
            command+=['--out',str(alignment_path),str(input_path)]
            with (folder/'mafft.log').open('w',encoding='utf-8') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        else:
            command.append(str(input_path))
            with alignment_path.open('w',encoding='ascii') as aligned,(folder/'mafft.log').open('w',encoding='utf-8') as log:
                completed=subprocess.run(command,stdout=aligned,stderr=log)
        if completed.returncode:raise RuntimeError(f'MAFFT failed for {family}; inspect work output log')
    alignment=read_fasta(alignment_path);validate_alignment(inputs,alignment)
    manifest={'family':family,'method':'FFT-NS-1','input_sha256':sha256(input_path),
              'alignment_sha256':sha256(alignment_path),'threads':threads,'n_sequences':len(inputs),
              'alignment_columns':len(next(iter(alignment.values()))),'validated_ungapped_identity':True,
              'reused_validated_alignment':reusable or restored,'restored_frozen_alignment':restored,
              'alignment_generation_command':command}
    write_json(manifest_path,manifest)
    return {identities[key]:value for key,value in alignment.items()},manifest
