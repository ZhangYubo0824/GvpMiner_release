"""Taxonomic label coverage with explicit unresolved and missing metadata states."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd

from .cross_model import file_record, load_frozen, safe_output_root


def parse_organism(organism, lineage=''):
    """Recognize conservative organism-derived binomials; never guess lineage ranks.

    'Recognized' is a reproducible label rule, not external nomenclatural verification.
    Strain/subspecies suffixes collapse to the binomial; Candidatus remains explicit.
    """
    organism = '' if pd.isna(organism) else str(organism).strip()
    lineage = '' if pd.isna(lineage) else str(lineage).strip()
    result = {'recognized_species': None, 'recognized_genus': None,
              'organism_status': 'unresolved_species' if organism else 'missing_organism',
              'domain_label': None, 'has_lineage': bool(lineage)}
    tokens = [x.strip() for x in lineage.split(';') if x.strip()]
    roots = [x for x in tokens if x in {'Bacteria', 'Archaea', 'Eukaryota', 'Viruses'}]
    if roots:
        result['domain_label'] = roots[0]
    if not organism:
        return result
    normalized = re.sub(r'[\[\]"\u201c\u201d]', '', organism)
    words = normalized.split()
    candidatus = words[0].lower() == 'candidatus'
    if candidatus:
        words = words[1:]
    if not words:
        return result
    genus = words[0]
    # Uppercase higher-taxon names can occur as organism labels; exclude explicit rank suffixes.
    rank_suffixes = ('aceae', 'ales', 'ota', 'phyceae', 'phycideae', 'mycetes', 'bacteria')
    genus_ok = bool(re.fullmatch(r'[A-Z][a-z]+', genus)) and not genus.endswith(rank_suffixes)
    genus_ok = genus_ok and genus.lower() not in {'bacteria', 'archaea', 'eukaryota', 'bacterium', 'archaeon'}
    if not genus_ok:
        return result
    if len(words) >= 2 and words[1].lower().rstrip('.') in {'bacterium', 'archaeon', 'cyanobacterium'}:
        return result
    result['recognized_genus'] = genus
    if len(words) < 2:
        return result
    epithet = words[1]
    ambiguous = {'sp', 'spp', 'cf', 'aff', 'nr', 'bacterium', 'archaeon', 'cyanobacterium', 'symbiont',
                 'endosymbiont', 'metagenome', 'uncultured', 'unclassified', 'unidentified', 'candidate', 'group', 'complex'}
    if epithet.rstrip('.').lower() in ambiguous or not re.fullmatch(r'[a-z][a-z-]+', epithet):
        return result
    if any(w.rstrip('.,').lower() in {'group', 'complex', 'cf', 'aff', 'nr'} for w in words[2:]):
        return result
    species = f'{genus} {epithet}'
    result['recognized_species'] = 'Candidatus ' + species if candidatus else species
    result['organism_status'] = 'recognized_species'
    return result


def prepare_metadata(frame):
    required = {'family', 'protein_id', 'organism', 'taxonomy'}
    if not required <= set(frame.columns):
        raise ValueError('Metadata missing required columns: ' + str(required - set(frame.columns)))
    frame = frame[['family', 'protein_id', 'organism', 'taxonomy']].copy().fillna('')
    if frame.duplicated(['family', 'protein_id']).any():
        raise ValueError('Duplicate family/protein metadata keys')
    if (frame.protein_id == '').any() or (frame.family == '').any():
        raise ValueError('Empty family/protein metadata key')
    parsed = [parse_organism(r.organism, r.taxonomy) for r in frame.itertuples(index=False)]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(parsed)], axis=1)


def _dominance(labels, n_all):
    counts = Counter(x for x in labels if pd.notna(x) and x)
    if not counts:
        return None, 0, None, None
    label, count = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0]
    return label, count, count / n_all if n_all else None, count / sum(counts.values())


def candidate_coverage(family, motif_uid, proteins, metadata):
    proteins = set(map(str, proteins))
    background = metadata.loc[metadata.family == family]
    present = background.loc[background.protein_id.isin(proteins)]
    n = len(proteins)
    species = set(present.recognized_species.dropna()) - {''}
    genera = set(present.recognized_genus.dropna()) - {''}
    background_species = set(background.recognized_species.dropna()) - {''}
    background_genera = set(background.recognized_genus.dropna()) - {''}
    n_resolved = int(present.recognized_species.notna().sum())
    row = {'family': family, 'motif_uid': motif_uid, 'n_proteins': n,
           'n_metadata_proteins': len(present), 'n_missing_metadata': n - len(present),
           'n_missing_organism': int((present.organism_status == 'missing_organism').sum()),
           'n_unresolved_species': int((present.organism_status == 'unresolved_species').sum()),
           'n_recognized_species_proteins': n_resolved,
           'n_recognized_species': len(species), 'n_recognized_genera': len(genera),
           'n_missing_lineage': int((~present.has_lineage).sum()),
           'n_unrecognized_domain': int(present.domain_label.isna().sum()),
           'n_background_proteins': len(background), 'n_background_species': len(background_species),
           'n_background_genera': len(background_genera),
           'protein_background_coverage': len(present) / len(background) if len(background) else None,
           'species_background_coverage': len(species) / len(background_species) if background_species else None,
           'genus_background_coverage': len(genera) / len(background_genera) if background_genera else None,
           'recognized_species_protein_fraction': n_resolved / n if n else None,
           'status': 'evaluated' if len(present) else 'no_metadata' if n else 'empty_occurrences'}
    for rank, column in [('species', 'recognized_species'), ('genus', 'recognized_genus'), ('domain', 'domain_label')]:
        label, count, all_share, resolved_share = _dominance(present[column], n)
        row[f'dominant_{rank}'] = label
        row[f'dominant_{rank}_n_proteins'] = count
        row[f'dominant_{rank}_share_all_proteins'] = all_share
        row[f'dominant_{rank}_share_resolved_proteins'] = resolved_share
    return row


def _candidate_summary(frame, families=None):
    rows = []
    for family in families or sorted(frame.family.unique()):
        group = frame.loc[frame.family == family]
        rows.append({'family': family, 'n_candidates': len(group),
                     'n_with_two_or_more_species': int((group.n_recognized_species >= 2).sum()),
                     'n_with_two_or_more_genera': int((group.n_recognized_genera >= 2).sum()),
                     'n_with_no_recognized_species': int((group.n_recognized_species == 0).sum()),
                     'n_candidates_missing_metadata': int((group.n_missing_metadata > 0).sum()),
                     'median_species_richness': float(group.n_recognized_species.median()) if len(group) else None,
                     'median_genus_richness': float(group.n_recognized_genera.median()) if len(group) else None,
                     'median_species_background_coverage': float(group.species_background_coverage.median()) if len(group) else None,
                     'median_genus_background_coverage': float(group.genus_background_coverage.median()) if len(group) else None,
                     'median_dominant_species_share_all_proteins': float(group.dominant_species_share_all_proteins.median()) if len(group) else None,
                     'median_recognized_species_protein_fraction': float(group.recognized_species_protein_fraction.median()) if len(group) else None,
                     'n_single_genus_candidates': int((group.n_recognized_genera == 1).sum())})
    return pd.DataFrame(rows)


def run(source_root, output_root):
    output = safe_output_root(source_root, output_root)
    metadata_path = Path(source_root) / 'data/reference/preprocessed/cleaned_gvp_dataset.csv'
    metadata = prepare_metadata(pd.read_csv(metadata_path, keep_default_na=False))
    families, data, inputs = load_frozen(source_root, methods=('M2',))
    inputs.append(file_record(metadata_path))
    rows, memberships, backgrounds = [], [], []
    for family in families:
        catalog, occurrences, unions = data[family, 'M2']
        bg = metadata.loc[metadata.family == family]
        bgrow = candidate_coverage(family, '__BACKGROUND__', bg.protein_id, metadata)
        bgrow.pop('motif_uid')
        bgrow['n_candidates'] = len(catalog)
        backgrounds.append(bgrow)
        for uid, per_protein in unions.items():
            r = candidate_coverage(family, uid, per_protein, metadata)
            rows.append(r)
            memberships.extend({'family': family, 'motif_uid': uid, 'protein_id': p} for p in sorted(per_protein))
    frame = pd.DataFrame(rows)
    family_summary = _candidate_summary(frame, families=families)
    overall = _candidate_summary(frame.assign(family='ALL'))
    frame.to_csv(output / 'candidate_taxonomy.csv', index=False)
    pd.DataFrame(memberships).to_csv(output / 'candidate_protein_membership.csv', index=False)
    metadata.to_csv(output / 'protein_taxonomy_metadata.csv', index=False)
    family_summary.to_csv(output / 'family_summary.csv', index=False)
    pd.DataFrame(backgrounds).to_csv(output / 'family_background.csv', index=False)
    overall.to_csv(output / 'overall_summary.csv', index=False)
    (output / 'candidate_taxonomy.json').write_text(frame.to_json(orient='records', indent=2, force_ascii=False), encoding='utf-8')
    summary = {'n_candidates': len(frame), 'n_families': len(families), 'n_candidate_bearing_families': int(frame.family.nunique()),
               'n_unique_candidate_protein_keys': len(pd.DataFrame(memberships).drop_duplicates(['family', 'protein_id'])),
               'overall': json.loads(overall.to_json(orient='records')), 'families': json.loads(family_summary.to_json(orient='records'))}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps({'module': 'taxonomy', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(Path(source_root).resolve()), 'inputs': inputs,
        'parameters': {'species_rule': 'organism-derived conservative binomial; strain/subspecies collapsed; Candidatus retained',
                       'lineage_rank_inference': False, 'metadata_key': ['family', 'protein_id'], 'method': 'M2'}}, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'methods.md').write_text('''# Taxonomic coverage and metadata limits

All frozen M2 candidates are retained. Proteins are unique family/protein-ID keys, joined to cleaned_gvp_dataset.csv. The original organism and unranked semicolon-delimited taxonomy strings are retained in protein_taxonomy_metadata.csv; candidate_protein_membership.csv makes every membership auditable. Same-family backgrounds include all cleaned proteins in each evaluated family. Missing metadata, missing organism, unresolved species labels, missing lineage, and unrecognized domains are counted separately.

Recognized species are operational organism-derived binomial labels, not externally validated accepted taxonomic names. Genus is a syntactically recognizable first organism word, excluding explicit higher-rank suffixes (-aceae, -ales, -ota, -phyceae, -phycideae, -mycetes, -bacteria); bracketed genus names are normalized. Generic X bacterium/archaeon/cyanobacterium labels resolve neither species nor genus. Species requires a lower-case epithet. Strain and subspecies suffixes collapse to the binomial; Candidatus prefixes are retained in species labels. sp., spp., cf., aff., nr., uncultured, unclassified, generic bacterium/archaeon and group/complex labels do not resolve species. A genus-only or Genus sp. organism can resolve a genus while species remains unresolved. Uncertain prefixes are not stripped to fabricate a genus. Synonyms, renamed taxa, and mislabeled records are not reconciled. Raw labels remain available for later curation. Higher-rank suffix rejection is conservative rather than a nomenclatural registry lookup.

No rank is guessed from the last lineage token. Only explicit Bacteria/Archaea/Eukaryota/Viruses tokens provide domain labels. Richness counts unique recognized species/genus labels. Coverage divides represented recognized labels by unique recognized labels in the same-family background. Protein coverage divides metadata-matched member proteins by all same-family background proteins; absent metadata is separately reported and never enters the numerator. Dominance is based on protein counts, with both all-member-protein and resolved-protein denominators. Unresolved labels are neither discarded from protein denominators nor treated as distinct species. Candidate summaries are medians of candidate-level values, not pooled global species coverage.

Taxonomic breadth provides descriptive sampling coverage and does not establish phylogenetic independence, evolutionary conservation, motif specificity, function, or experimental validation. Label uncertainty and unequal sampling limit these comparisons.
''', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.output_root), indent=2))


if __name__ == '__main__':
    main()
