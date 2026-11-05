"""Strict MotifAE handoff and annotation provenance regression tests."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ADAPTERS = Path(__file__).resolve().parents[1] / 'scripts' / 'baseline_adapters'
sys.path.insert(0, str(ADAPTERS))

def api():
    spec = importlib.util.find_spec('motifae_handoff')
    assert spec is not None, 'strict MotifAE handoff API must exist'
    import motifae_handoff
    return motifae_handoff

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')

def table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def fixture(root, family='GvpA', seq='ACDEFGHIKL', empty=False):
    base = root / family; prep = base / 'prep'; batch = base / 'batch'
    fasta = base / 'input.fasta'; base.mkdir(parents=True)
    fasta.write_text('>p1\n' + seq + '\n>background\nLMNPQRSTVW\n', encoding='utf-8')
    uid = family + '_feature_1'
    mapping = dict(selection_rank='1',family=family,method='MotifAE',motif_uid=uid,
        parent_id='p1',sequence_id='p1',occ_start='0',occ_end='7',motif_sequence=seq[:7],score='1')
    table(prep/'mapping.csv', [mapping], list(mapping))
    (prep/'peptide.txt').write_text(seq[:7]+'\n')
    entry = dict(family=family,method='MotifAE',motif_uid=uid,total_peptides=1,
        selected_peptides=1,status='ready_for_gibbscluster',mapping_file='mapping.csv',peptide_file='peptide.txt')
    table(prep/'feature_index.csv', [entry], list(entry))
    dump(prep/'gibbs_preparation_manifest.json', dict(core_length=5,files_sha256={n:sha(prep/n) for n in ['mapping.csv','peptide.txt','feature_index.csv']}))
    occ = dict(mapping, occ_start='0',occ_end='6',motif_sequence=seq[:6],candidate_start='0',candidate_end='7',candidate_sequence=seq[:7],aligned_core=seq[0]+seq[2:6],core_residue_positions_0based='[0,2,3,4,5]',localization_scope='selected_top1000_alignment_instances')
    parsed = batch/'jobs'/'f1'/'parsed'
    table(parsed/'aligned_occurrences.csv', [] if empty else [occ], list(occ))
    report = dict(family=family,method='MotifAE',motif_uid=uid,pattern_built=not empty,status='all_sequences_trash' if empty else 'aligned_pattern_built',mapped_instances=0 if empty else 1,mapped_proteins=0 if empty else 1,total_candidate_instances=1,selected_instances=1,unassessed_instances=0,trash_instances=1 if empty else 0,localization_scope='selected_top1000_alignment_instances')
    dump(parsed/'parse_manifest.json', report)
    job = dict(motif_uid=uid,status='success',parsed_dir='jobs/f1/parsed',report=report,files_sha256={str(p.relative_to(batch)).replace('\\','/'):sha(p) for p in parsed.iterdir()})
    manifest = dict(status='alignment_samples_parsed',total_groups=1,successful_groups=1,failed_groups=0,jobs=[job],signature=dict(preparation_sha256=sha(prep/'gibbs_preparation_manifest.json'),fasta_sha256=sha(fasta)),localization_scope='selected_top1000_alignment_instances')
    dump(batch/'batch_manifest.json', manifest)
    return dict(family=family,batch_dir=batch,preparation_dir=prep,fasta=fasta)

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_keeps_full_background_and_exact_gapped_core(self):
        run=fixture(self.root); m=api().export_handoff([run],self.root/'out')
        rows=read(self.root/'out/GvpA/MotifAE/motif_occurrences_MotifAE.csv')
        self.assertEqual((rows[0]['motif_sequence'],rows[0]['aligned_core']),('ACDEFG','ADEFG'))
        self.assertEqual(len(read(self.root/'out/sequence_index.csv')),2)
        self.assertEqual(m['families']['GvpA']['motifs'],1)
        self.assertEqual(m['families']['GvpA']['foreground_proteins'],1)
    def test_missing_mapping_id_hard_fails(self):
        run=fixture(self.root); run['fasta'].write_text('>other\nACDEFGHIKL\n')
        m=json.loads((run['batch_dir']/'batch_manifest.json').read_text());m['signature']['fasta_sha256']=sha(run['fasta']);dump(run['batch_dir']/'batch_manifest.json',m)
        with self.assertRaisesRegex(ValueError,'unmapped|missing|Unknown'): api().export_handoff([run],self.root/'out')
    def test_sequence_mismatch_hard_fails(self):
        run=fixture(self.root); run['fasta'].write_text('>p1\nACDEYGHIKL\n>background\nLMNPQRSTVW\n')
        m=json.loads((run['batch_dir']/'batch_manifest.json').read_text());m['signature']['fasta_sha256']=sha(run['fasta']);dump(run['batch_dir']/'batch_manifest.json',m)
        with self.assertRaisesRegex(ValueError,'sequence|slice'): api().export_handoff([run],self.root/'out')
    def test_changed_background_fails_fasta_hash(self):
        run=fixture(self.root);run['fasta'].write_text(run['fasta'].read_text()+'>extra\nACDEF\n')
        with self.assertRaisesRegex(ValueError,'FASTA|fasta'): api().export_handoff([run],self.root/'out')
    def test_changed_preparation_fails_hash(self):
        run=fixture(self.root);(run['preparation_dir']/'mapping.csv').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'hash|Changed'): api().export_handoff([run],self.root/'out')
    def test_empty_feature_keeps_headers_and_status(self):
        run=fixture(self.root,empty=True);m=api().export_handoff([run],self.root/'out')
        self.assertEqual(read(self.root/'out/GvpA/MotifAE/motif_catalog_MotifAE.csv'),[])
        self.assertEqual(m['families']['GvpA']['status'],'all_trash')
    def test_same_id_same_sequence_preserves_both_families(self):
        a=fixture(self.root);b=fixture(self.root,'GvpB');api().export_handoff([a,b],self.root/'out')
        self.assertEqual({r['family'] for r in read(self.root/'out/sequence_index.csv') if r['protein_id']=='p1'},{'GvpA','GvpB'})
        self.assertEqual((self.root/'out/all_gvp_proteins.fasta').read_text().count('>'),2)
    def test_same_id_different_sequence_is_ambiguous(self):
        a=fixture(self.root);b=fixture(self.root,'GvpB','ACDEYGHIKL')
        with self.assertRaisesRegex(ValueError,'ambig|Conflicting'): api().export_handoff([a,b],self.root/'out')
    def test_annotation_requires_sequence_source(self):
        run=fixture(self.root);api().export_handoff([run],self.root/'out')
        terms=self.root/'terms.csv';table(terms,[dict(term='GO:1',gene='p1')],['term','gene'])
        with self.assertRaisesRegex(ValueError,'provenance|source'): api().bind_annotations(self.root/'out',terms,self.root/'bound')
        self.assertEqual(len(read(self.root/'bound/reannotation_required.csv')),2)
    def test_annotation_subset_requires_identical_current_sequences(self):
        run=fixture(self.root);api().export_handoff([run],self.root/'out')
        terms=self.root/'terms.csv';table(terms,[dict(term='GO:1',gene='p1'),dict(term='GO:2',gene='old')],['term','gene'])
        old=self.root/'old.fasta';old.write_text(run['fasta'].read_text()+'>old\nACDEF\n')
        m=api().bind_annotations(self.root/'out',terms,self.root/'bound',annotation_fasta=old)
        self.assertEqual(m['status'],'annotations_bound')
        self.assertEqual(read(self.root/'bound/term2gene.csv'),[dict(term_id='GO:1',protein_id='p1')])
    def test_old_shared_id_sequence_mismatch_writes_reannotation(self):
        run=fixture(self.root);api().export_handoff([run],self.root/'out')
        terms=self.root/'terms.csv';table(terms,[],['term','gene'])
        old=self.root/'old.fasta';old.write_text('>p1\nACDEYGHIKL\n')
        with self.assertRaisesRegex(ValueError,'sequence|binding'):api().bind_annotations(self.root/'out',terms,self.root/'bound',annotation_fasta=old)
        self.assertEqual({r['reason'] for r in read(self.root/'bound/reannotation_required.csv')},{'sequence_mismatch','missing_in_annotation_source'})

class AdditionalHandoffTests(unittest.TestCase):
    setUp=HandoffTests.setUp
    tearDown=HandoffTests.tearDown
    def test_family_metadata_fasta_header_uses_accession(self):
        run=fixture(self.root)
        run['fasta'].write_text('>p1|GvpA|Species_name\nACDEFGHIKL\n>background|GvpA|Other_species\nLMNPQRSTVW\n')
        m=json.loads((run['batch_dir']/'batch_manifest.json').read_text());m['signature']['fasta_sha256']=sha(run['fasta']);dump(run['batch_dir']/'batch_manifest.json',m)
        api().export_handoff([run],self.root/'out')
        self.assertEqual(read(self.root/'out/sequence_index.csv')[0]['protein_id'],'p1')
    def test_missing_required_per_job_hash_fails(self):
        run=fixture(self.root);p=run['batch_dir']/'batch_manifest.json';m=json.loads(p.read_text());m['jobs'][0]['files_sha256'].pop('jobs/f1/parsed/aligned_occurrences.csv');dump(p,m)
        with self.assertRaisesRegex(ValueError,'Incomplete'):api().export_handoff([run],self.root/'out')
    def test_exact_core_mismatch_even_after_rehash_fails(self):
        run=fixture(self.root);p=run['batch_dir']/'jobs/f1/parsed/aligned_occurrences.csv';rows=read(p);rows[0]['aligned_core']='AAAAA';table(p,rows,list(rows[0]))
        bp=run['batch_dir']/'batch_manifest.json';m=json.loads(bp.read_text());m['jobs'][0]['files_sha256']['jobs/f1/parsed/aligned_occurrences.csv']=sha(p);dump(bp,m)
        with self.assertRaisesRegex(ValueError,'core sequence'):api().export_handoff([run],self.root/'out')
    def test_verify_rejects_changed_background(self):
        run=fixture(self.root);api().export_handoff([run],self.root/'out');(self.root/'out/sequence_index.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash'):api().verify_handoff(self.root/'out')
    def test_empty_term_table_cannot_claim_positive(self):
        run=fixture(self.root);api().export_handoff([run],self.root/'out');t=self.root/'terms.csv';table(t,[],['term','gene'])
        m=api().bind_annotations(self.root/'out',t,self.root/'bound',annotation_fasta=run['fasta'])
        self.assertEqual(m['status'],'annotations_bound_no_terms');self.assertFalse(m['biological_function_validated'])

    def test_accepts_actual_exp5_annotation_writer_schema(self):
        sys.path.insert(0,str(ADAPTERS.parents[1]))
        from src.exp5_external_functional_enrichment.package.annotations import AnnotationTerm,write_term_tables
        run=fixture(self.root);api().export_handoff([run],self.root/'out')
        outputs=write_term_tables([AnnotationTerm('InterProScan','GO','GO:0000001','p1','test')],self.root/'annotations')
        api().bind_annotations(self.root/'out',outputs['term2gene'],self.root/'bound',annotation_fasta=run['fasta'])
        self.assertEqual(read(self.root/'bound/term2gene.csv'),[dict(term_id='GO:0000001',protein_id='p1')])

    def test_prepare_exp5_keeps_full_universe_and_explicit_selected_scope(self):
        run=fixture(self.root);api().export_handoff([run],self.root/'out')
        self.assertTrue(hasattr(api(),'prepare_exp5'),'strict Exp5 preparation entry missing')
        result=api().prepare_exp5(self.root/'out',self.root/'exp5')
        sets=read(self.root/'exp5/motif_sets_long.csv')
        self.assertEqual({r['protein_id'] for r in sets if r['set_type']=='universe'},{'p1','background'})
        self.assertEqual(result['localization_scope'],'selected_top1000_alignment_instances')
        self.assertEqual(read(self.root/'exp5/motif_set_scope.csv')[0]['unselected_status'],'unknown_not_negative')

    def test_annotation_verification_rejects_changed_bound_table(self):
        run=fixture(self.root);api().export_handoff([run],self.root/'out')
        terms=self.root/'terms.csv';table(terms,[dict(term='GO:1',gene='p1')],['term','gene'])
        api().bind_annotations(self.root/'out',terms,self.root/'bound',annotation_fasta=run['fasta'])
        self.assertTrue(hasattr(api(),'verify_annotations'),'bound annotation verifier missing')
        api().verify_annotations(self.root/'out',self.root/'bound')
        (self.root/'bound/term2gene.csv').write_text('term_id,protein_id\nGO:2,p1\n')
        with self.assertRaisesRegex(ValueError,'hash'):api().verify_annotations(self.root/'out',self.root/'bound')

    def test_handoff_does_not_overwrite_output_created_during_validation(self):
        run=fixture(self.root);target=self.root/'out';original=api()._family
        def concurrent_create(description):
            result=original(description);target.mkdir(exist_ok=True)
            (target/'handoff_manifest.json').write_text('{"preserve":"other run"}')
            return result
        with mock.patch.object(api(),'_family',side_effect=concurrent_create):
            with self.assertRaises((FileExistsError,ValueError)):
                api().export_handoff([run],target)
        self.assertEqual(json.loads((target/'handoff_manifest.json').read_text()),dict(preserve='other run'))


if __name__=='__main__':unittest.main()
