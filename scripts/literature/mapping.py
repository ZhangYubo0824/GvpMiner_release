"""Conservative affine-gap reference-residue mapping; no automatic validation claim."""
from collections import defaultdict


def align_reference(reference,target,max_alignments=64):
    from Bio.Align import PairwiseAligner, substitution_matrices
    if not reference or not target: raise ValueError('Empty alignment sequence')
    aligner=PairwiseAligner(mode='global')
    aligner.substitution_matrix=substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score=-10; aligner.extend_gap_score=-0.5
    # Global alignment keeps terminal truncations visible; no local short-hit inflation.
    alignments=aligner.align(reference,target)
    variants=[]; overflow=False; first=None
    for i,aln in enumerate(alignments):
        if i>=max_alignments: overflow=True; break
        mapping={}; matched=0
        for (a,b),(c,d) in zip(*aln.aligned):
            for r,t in zip(range(a,b),range(c,d)):
                mapping[int(r)+1]=int(t)+1
                matched+=reference[r]==target[t]
        variants.append(mapping)
        if first is None:
            first={'reference_aligned':str(aln[0]),'target_aligned':str(aln[1]),
                   'identity':matched/len(mapping) if mapping else 0,
                   'reference_coverage':len(mapping)/len(reference),
                   'target_coverage':len(mapping)/len(target),'score':float(aln.score)}
    return {**first,'variants':variants,'overflow':overflow}


def map_event(event,reference,target,start,end,alignment,min_identity=0.5,min_reference_coverage=0.8,min_local_identity=0.7):
    positions=event['positions']; expected=event['expected_residues']
    if not positions or len(positions)!=len(expected): raise ValueError('Invalid experimental positions/residues')
    if any(not 1<=p<=len(reference) for p in positions): raise ValueError('Experimental position outside reference')
    if ''.join(reference[p-1] for p in positions)!=expected:
        raise ValueError(f'Experimental wild-type residue mismatch: {event.get("event_id","")}')
    variants={tuple(mapping.get(p) for p in positions) for mapping in alignment['variants']}
    unique=len(variants)==1 and not alignment['overflow']
    mapped=list(next(iter(variants))) if variants else []
    nongap=bool(mapped) and all(p is not None for p in mapped)
    inside=nongap and all(start<p<=end for p in mapped)
    conserved=nongap and ''.join(target[p-1] for p in mapped)==expected
    mp=alignment['variants'][0]
    region=list(range(max(1,min(positions)-5),min(len(reference),max(positions)+5)+1))
    local_identity=sum(p in mp and reference[p-1]==target[mp[p]-1] for p in region)/len(region)
    quality=alignment['identity']>=min_identity and alignment['reference_coverage']>=min_reference_coverage and local_identity>=min_local_identity
    verified=event.get('sequence_verified') is True
    localized=event.get('source_scope')=='local_perturbation'
    positive=event.get('effect') in ('loss','reduced','altered','increased')
    compatible=unique and inside and conserved and quality and verified and localized
    provisional=('A' if reference==target else 'B') if compatible else 'D'
    reasons=[]
    for condition,label in [(not unique,'ambiguous_alignment'),(not nongap,'site_gap'),(not inside,'outside_window'),
                            (not conserved,'residue_not_conserved'),(not quality,'alignment_quality'),
                            (not verified,'source_sequence_unverified'),(not localized,'not_local_perturbation'),
                            (not positive,'null_or_context_evidence')]:
        if condition: reasons.append(label)
    return {'target_positions':mapped,'positions_unambiguous':unique,'inside_window':inside,'conserved':conserved,
            'identity':alignment['identity'],'reference_coverage':alignment['reference_coverage'],
            'local_identity':local_identity,'provisional_grade':provisional,
            'evidence_grade':'pending_review' if compatible else 'D',
            'positive_support':compatible and positive,'compatible_mapping':compatible,
            'reason':';'.join(reasons)}


def map_all(evidence,occurrences,sequences,config):
    refs=evidence.get('references',{}); events=evidence.get('events',[])
    by_family=defaultdict(list);seen=set()
    for event in events:
        if not event.get('event_id') or event['event_id'] in seen: raise ValueError('Missing or duplicate event ID')
        seen.add(event['event_id'])
        if event['reference_id'] not in refs: raise ValueError('Event references missing protein')
        ref=refs[event['reference_id']]
        if ref['family']!=event['family']: raise ValueError('Experimental reference family mismatch')
        positions=event['positions']
        if not positions or len(set(positions))!=len(positions) or any(type(p) is not int or not 1<=p<=len(ref['sequence']) for p in positions):
            raise ValueError('Invalid experimental positions')
        if len(positions)!=len(event['expected_residues']): raise ValueError('Position/residue count mismatch')
        if not isinstance(ref.get('sequence_verified',False),bool): raise ValueError('sequence_verified must be boolean')
        # Validate source coordinates even when the family has no occurrence.
        if ''.join(ref['sequence'][p-1] for p in event['positions'])!=event['expected_residues']:
            raise ValueError(f'Source residue conflict: {event["event_id"]}')
        by_family[event['family']].append(event)
    rows=[]; cache={}
    for o in occurrences:
        target_record=sequences.get(o['parent_id'])
        if not target_record: continue
        target=target_record['sequence']
        for event in by_family[o['family']]:
            ref=refs[event['reference_id']]; sequence=ref['sequence']
            key=(sequence,target)
            if key not in cache:
                cache[key]=align_reference(sequence,target,config['max_optimal_alignments'])
            event_input={**event,'sequence_verified':ref.get('sequence_verified',False)}
            result=map_event(event_input,sequence,target,o['occ_start'],o['occ_end'],cache[key],
                             config['min_identity'],config['min_reference_coverage'],config['min_local_identity'])
            matching_windows=[interval for interval in o.get('support_intervals',[[o['occ_start'],o['occ_end']]])
                              if result['positions_unambiguous'] and result['target_positions']
                              and all(p is not None and interval[0]<p<=interval[1] for p in result['target_positions'])]
            if not matching_windows and result['compatible_mapping']:
                result.update(compatible_mapping=False,positive_support=False,provisional_grade='D',evidence_grade='D',
                              reason=result['reason']+';no_single_source_window_covers_all_sites')
            rows.append({'motif_uid':o['motif_uid'],'family':o['family'],'occurrence_id':o['occurrence_id'],
                         'protein_id':o['parent_id'],'event_id':event['event_id'],'reference_id':event['reference_id'],
                         'doi':event['doi'],'reference_positions':event['positions'],'expected_residues':event['expected_residues'],
                         'effect':event['effect'],'experimental_change':event['experimental_change'],
                         'function':event['function'],'source_anchor':event['source_anchor'],
                         'motif_start':o['start_1based'],'motif_end':o['end_1based'],
                         'matching_source_windows':matching_windows,'matching_source_window_count':len(matching_windows),**result})
    return rows,cache
