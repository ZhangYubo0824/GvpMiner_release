"""Extract family accessions from the current NCBI experiment inputs."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def extract(input_dir):
    families={}
    for folder in sorted(Path(input_dir).iterdir()):
        if not folder.is_dir():continue
        ids=set()
        for path in folder.glob('*_metadata.csv'):
            with path.open(encoding='utf-8-sig',newline='') as f:
                ids.update(row['protein_id'].strip() for row in csv.DictReader(f) if row.get('protein_id','').strip())
        for path in folder.glob('*.fasta'):
            with path.open(encoding='utf-8-sig') as f:
                ids.update(line[1:].split()[0] for line in f if line.startswith('>'))
        if ids:families[folder.name]=sorted(ids)
    return families


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'data/ncbi')
    parser.add_argument('--output',type=Path,default=ROOT/'work/ncbi_ids_by_family.json')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.is_relative_to(ROOT/'data') or output.is_relative_to(ROOT/'results'):
        parser.error('Choose derived output below work/ or outside the input snapshot.')
    result=extract(args.input_dir)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(result)} families, {sum(map(len,result.values()))} identifiers: {output}')


if __name__=='__main__':main()
