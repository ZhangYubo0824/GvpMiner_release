"""Re-map the current curated experimental sites onto candidate occurrences."""
import argparse
import json
from pathlib import Path
from scripts.paths import ROOT, read_rows, output_directory
from .mapping import map_all
from .io_utils import write_csv


def run(source_root, output_root, families=None, event_ids=None, dry_run=False):
    source=Path(source_root).resolve()
    evidence=json.loads((source/'data/literature/curated_events.json').read_text(encoding='utf-8'))
    if families:evidence['events']=[e for e in evidence['events'] if e['family'] in families]
    if event_ids:evidence['events']=[e for e in evidence['events'] if e['event_id'] in event_ids]
    selected_families={e['family'] for e in evidence['events']}
    occurrences=[r for r in read_rows(source/'data/literature/occurrence_mapping_input.csv.gz') if r['family'] in selected_families]
    for row in occurrences:
        for key in ['occ_start','occ_end','start_1based','end_1based']:row[key]=int(row[key])
        row['support_intervals']=json.loads(row['support_intervals'])
    sequences={r['protein_id']:r for r in read_rows(source/'data/reference/preprocessed/cleaned_gvp_dataset.csv')}
    plan={'families':sorted(selected_families),'events':len(evidence['events']),'occurrences':len(occurrences)}
    if dry_run:return plan
    output=output_directory(source,output_root)
    rules={'max_optimal_alignments':64,'min_identity':.5,'min_reference_coverage':.8,'min_local_identity':.7}
    rows,_=map_all(evidence,occurrences,sequences,rules)
    write_csv(output/'site_mapping.csv',rows)
    summary={**plan,'mapping_rows':len(rows),'positive_mapping_rows':sum(r['positive_support'] for r in rows),
             'scope':'Computational site correspondence; reviewed evidence remains a separate input.', 'rules':rules}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path,default=ROOT/'work/literature')
    parser.add_argument('--families',nargs='+')
    parser.add_argument('--event-ids',nargs='+')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.source_root,args.output_root,args.families,args.event_ids,args.dry_run),indent=2))


if __name__=='__main__':main()
