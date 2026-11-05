"""Read-only checks for the experimental results; Python standard library only."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'GvpMiner': 522, 'GLAM2': 160, 'MEME': 320,
            'STREME': 529, 'MotifAE': 8378, 'InterPLM': 875}


def rows(path):
    if not path.is_file():
        path = path.with_name(path.name + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, hashes=False):
    data = root / 'results/paper/source_data'
    candidates = list(rows(data / 'Table_S14_candidate_status.csv'))
    require(Counter(r['method'] for r in candidates) == EXPECTED,
            'Six-method candidate counts differ from the experimental snapshot')
    require(len({r['motif_uid'] for r in candidates}) == 10784,
            'Duplicate candidate identifiers')
    structure = {r['method']: r for r in rows(data / 'Figure_4_method_summary.csv')}
    annotation = {r['method']: r for r in rows(data / 'Figure_5_method_summary.csv')}
    for method, count in EXPECTED.items():
        selected = [r for r in candidates if r['method'] == method]
        require(int(structure[method]['input']) == count, f'{method}: structure denominator')
        require(int(annotation[method]['input']) == count, f'{method}: annotation denominator')
        for field, table, column in [
            ('structural_testable', structure, 'testable'),
            ('strict_supported', structure, 'strict_supported'),
            ('annotation_has_output', annotation, 'with_output'),
            ('annotation_supported_raw', annotation, 'supported'),
        ]:
            require(sum(r[field].lower() == 'true' for r in selected) == int(table[method][column]),
                    f'{method}: candidate/summary mismatch for {field}')
    counts = {}
    for method, expected in [('M2', 522), ('M3', 464)]:
        paths = sorted((root / 'results/catalogues/GvpMiner_M2_M3').glob(f'*/motif_catalog_{method}.csv'))
        require(len(paths) == 16, f'{method}: missing family catalogue')
        counts[method] = sum(sum(1 for _ in rows(path)) for path in paths)
        require(counts[method] == expected, f'{method}: catalogue count mismatch')
    evidence = list(rows(root / 'results/catalogues/GvpMiner_candidate_evidence_latest.csv'))
    keys = {r['motif_uid'] for r in evidence}
    require(len(evidence) == len(keys) == 522, 'Main candidate evidence count mismatch')
    sequences = {r['protein_id']: r['sequence'] for r in rows(root / 'data/reference/preprocessed/cleaned_gvp_dataset.csv')}
    seen = set()
    occurrence_count = 0
    for row in rows(root / 'results/evidence/evidence/occurrence_details.csv'):
        sequence = sequences[row['parent_id']]
        start, end = int(row['occ_start']), int(row['occ_end'])
        require(row['motif_uid'] in keys and 0 <= start < end <= len(sequence), 'Invalid occurrence membership or coordinates')
        require(sequence[start:end] == row['sequence'], 'Occurrence/parent sequence mismatch')
        seen.add(row['motif_uid'])
        occurrence_count += 1
    require(seen == keys, 'Candidates without an occurrence')
    figures = [root / f'results/figures/Figure_{prefix}{index}.png'
               for prefix in ['', 'S'] for index in range(1, 8)]
    for figure in figures:
        require(figure.is_file(), f'Missing final figure: {figure.name}')
    checked = 0
    if hashes:
        for line in (root / 'manifests/SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            expected, relative = line.split('  ', 1)
            path = (root / relative).resolve()
            require(path.is_relative_to(root), 'Manifest path escapes the release')
            require(digest(path) == expected, f'File hash mismatch: {relative}')
            checked += 1
    return {'status': 'PASS', 'candidates': EXPECTED,
            'evaluated_candidates': len(candidates), 'catalogues': counts,
            'validated_occurrences': occurrence_count, 'final_figures': len(figures),
            'file_hashes_checked': checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--hashes', action='store_true', help='Also check every file in SHA256SUMS.txt')
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), args.hashes), indent=2))


if __name__ == '__main__':
    main()
