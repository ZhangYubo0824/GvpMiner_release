"""Regression checks for the published experiment layout and paper profile."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ReleasePathTests(unittest.TestCase):
    def test_exp3_defaults_match_current_profile(self):
        tree=ast.parse((ROOT/'src/config.py').read_text(encoding='utf-8-sig'))
        values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in {'EXP3_WINDOW_SIZE','EXP3_WINDOW_STEP'}}
        self.assertEqual((values['EXP3_WINDOW_SIZE'],values['EXP3_WINDOW_STEP']),(20,15))

    def test_baseline_finds_bundled_family_without_old_environment(self):
        spec=importlib.util.spec_from_file_location('current_adapter',ROOT/'baselines/adapters/adapter_common.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        saved={k:os.environ.pop(k,None) for k in ['PROJECT','PREPROCESSED']}
        try:
            self.assertEqual(module.resolve_family_fasta('GvpA').resolve(),
                ROOT/'data/reference/preprocessed/by_family/GvpA_cleaned.fasta')
        finally:
            for k,v in saved.items():
                if v is not None:os.environ[k]=v

    def test_writing_and_retired_maintenance_code_are_absent(self):
        excluded=['scripts/candidate_evidence/verify.py','scripts/candidate_evidence/summary.py',
            'scripts/candidate_evidence/report.py','scripts/motif_representation/audit_delivery.py',
            'scripts/motif_representation/report.py','src/stable_parameter_workflow/decision_reissue.py',
            'src/motif_validation_tables.py','src/exp5_external_functional_enrichment/scripts/plot_precision_yield_exploratory.py',
            'src/exp5_external_functional_enrichment/scripts/run_precision_yield_exploratory.py']
        self.assertEqual([p for p in excluded if (ROOT/p).exists()],[])

    def test_no_active_historical_layout_in_current_source(self):
        failures=[]
        tokens=('results5/','results6/','results7/','result6/','src2/','src3/',
                '.codex_tmp','revision_v5','D:/研究内容/Task1','d:\\研究方向')
        for folder in ['src','scripts','baselines/adapters']:
            for p in (ROOT/folder).rglob('*.py'):
                if 'tests' in p.parts:continue
                tree=ast.parse(p.read_text(encoding='utf-8-sig'))
                docstrings=set()
                for node in ast.walk(tree):
                    if isinstance(node,(ast.Module,ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)) and node.body:
                        first=node.body[0]
                        if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant):docstrings.add(id(first.value))
                for node in ast.walk(tree):
                    if isinstance(node,ast.Constant) and isinstance(node.value,str) and id(node) not in docstrings:
                        if any(t in node.value for t in tokens):failures.append(f'{p.relative_to(ROOT)}:{node.lineno}')
        self.assertEqual(failures,[])


if __name__=='__main__':unittest.main()
