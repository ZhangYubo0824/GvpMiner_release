"""Print the current six-method result counts without scientific dependencies."""
from pathlib import Path
import csv

root=Path(__file__).resolve().parents[1]
data=root/'results/paper/source_data'
def read(name):
    with (data/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
structure={r['method']:r for r in read('Figure_4_method_summary.csv')}
annotation={r['method']:r for r in read('Figure_5_method_summary.csv')}
print('method,input,structure_testable,strict_structure,annotation_output,annotation_supported')
for name in ['GvpMiner','GLAM2','MEME','STREME','MotifAE','InterPLM']:
    s,a=structure[name],annotation[name]
    print(','.join([name,s['input'],s['testable'],s['strict_supported'],a['with_output'],a['supported']]))
print('Total evaluated candidates:',sum(int(r['input']) for r in structure.values()))
