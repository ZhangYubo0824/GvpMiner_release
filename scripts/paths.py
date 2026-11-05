"""Portable paths and read-only input handling for the current experiments."""
from pathlib import Path
import csv
import gzip
import hashlib

ROOT=Path(__file__).resolve().parents[1]
SEQUENCE_TABLE=Path('data/reference/preprocessed/cleaned_gvp_dataset.csv')


def table_path(path):
    path=Path(path)
    if path.is_file():return path
    compressed=path.with_name(path.name+'.gz')
    if compressed.is_file():return compressed
    raise FileNotFoundError(f'Required experiment input is missing: {path}')


def read_rows(path):
    path=table_path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def content_sha256(path):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def output_directory(source_root,output_root):
    source,output=Path(source_root).resolve(),Path(output_root).resolve()
    if output==source or output in source.parents:
        raise ValueError('Output must not replace the source tree')
    if source in output.parents and not output.is_relative_to(source/'work'):
        raise ValueError('Outputs inside the project must be below work/')
    output.mkdir(parents=True,exist_ok=True)
    return output
