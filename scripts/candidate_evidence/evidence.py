"""Current candidate evidence validation and export."""
from pathlib import Path
import argparse
import csv
import json
from scripts.paths import ROOT, read_rows, output_directory

def unique_index(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f'duplicate {field}: {key}')
        result[key] = row
    return result


def flag(value):
    return str(value).lower() == 'true'








def write_csv(path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)



def build(source_root, output_root):
    source=Path(source_root).resolve()
    output=output_directory(source,output_root)
    candidates=read_rows(source/'results/catalogues/GvpMiner_candidate_evidence_latest.csv')
    catalogue=unique_index(candidates,'motif_uid')
    sequences={r['protein_id']:r['sequence'] for r in read_rows(source/'data/reference/preprocessed/cleaned_gvp_dataset.csv')}
    occurrences=read_rows(source/'results/evidence/evidence/occurrence_details.csv')
    for row in occurrences:
        start,end=int(row['occ_start']),int(row['occ_end'])
        sequence=sequences[row['parent_id']]
        if row['motif_uid'] not in catalogue or not 0<=start<end<=len(sequence):
            raise ValueError('Invalid candidate membership or occurrence coordinates')
        if row['sequence']!=sequence[start:end]:
            raise ValueError('Occurrence sequence does not match its parent protein')
    if set(row['motif_uid'] for row in occurrences)!=set(catalogue):
        raise ValueError('Catalogue and occurrence key sets differ')
    write_csv(output/'candidate_evidence.csv',candidates)
    write_csv(output/'occurrence_details.csv',occurrences)
    report={'candidates':len(candidates),'occurrences':len(occurrences),
            'annotation_supported':sum(flag(row['annotation_supported']) for row in candidates),
            'scope':'Current frozen evidence export with sequence-coordinate validation; no new statistical test.'}
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path,default=ROOT/'work/candidate_evidence')
    args=parser.parse_args()
    print(json.dumps(build(args.source_root,args.output_root),indent=2))


if __name__=='__main__':main()
