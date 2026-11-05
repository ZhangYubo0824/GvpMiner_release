"""Prepare traceable literature tables and the replacement Figure 7."""
from pathlib import Path
import csv, gzip, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT
DATA=ROOT/'results/paper/source_data'
FIG=ROOT/'work/figures'
FIG.mkdir(parents=True,exist_ok=True)
AA='ACDEFGHIKLMNPQRSTVWY'
NSIG,NOOUT,INK,ACCENT='#247C83','#BBC3CC','#263442','#C98736'

def read(p):
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))



def plot(target,cases):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
                         'font.size':7,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,
                         'text.color':INK,'axes.labelcolor':INK,'figure.facecolor':'white','savefig.facecolor':'white'})
    W,H=170,170
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    def txt(x,y,s,size=7,**kw):return fig.text(x/W,1-y/H,s,fontsize=size,ha='left',va='center',**kw)
    def ax(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    txt(3,5,'a  未获注释富集支持候选的单事件位点对应',9,weight='bold')
    ov=ax(29,13,123,42)
    ys=np.arange(len(target));vals=[r['mapped_fraction']*100 for r in target]
    ov.barh(ys,vals,height=.62,color=[NSIG if r['annotation_output_status']=='output_not_significant' else NOOUT for r in target])
    for y,r,v in zip(ys,target,vals):
        ov.text(v+1,y,f"{r['mapped_proteins']}/{r['all_cluster_proteins']}",va='center',fontsize=6.3)
    ov.set(yticks=ys,yticklabels=[r['display_id'] for r in target],xlim=(0,119),xticks=[0,25,50,75,100],xlabel='满足同一事件映射条件的成员比例（%）')
    ov.invert_yaxis();ov.tick_params(axis='both',labelsize=6.5,length=2,pad=2)
    ov.spines['right'].set_visible(False);ov.spines['top'].set_visible(False)
    ov.set_axisbelow(True);ov.grid(axis='x',color='#E7EAEE',lw=.5)
    txt(32,64,'● 有输出但未显著',6.7,color=NSIG);txt(88,64,'● 无富集结果行',6.7,color='#687785')
    colors={a:'#414B54' for a in AA}
    for chars,color in [('KRH','#3C7FA1'),('DE','#B95C58'),('STNQ','#49866D'),('FWY','#916EAA'),('C','#A78D41'),('GP','#929292')]:colors.update({a:color for a in chars})
    glyphs={a:TextPath((0,0),a,size=1,prop=FontProperties(family='Arial',weight='bold')) for a in AA}
    for i,c in enumerate(cases):
        left=5+56*i
        txt(left,75,f"{'bcd'[i]}  {c['display_id']}",9.5,weight='bold')
        txt(left,81,f"{c['mapped_proteins']}/{c['all_cluster_proteins']}（{c['mapped_fraction']:.2%}）",7.1,color=NSIG)
        txt(left,87,'候选成员的残基频率模式',6.8)
        la=ax(left+3,92,48,29)
        columns=[int(r['msa_column_0based'])+1 for r in c['profile']]
        for j,row in enumerate(c['profile']):
            values=np.array([float(row['p_'+a]) for a in AA]);bottom=0
            if columns[j]==c['site_msa_column_1based']:la.axvspan(j-.48,j+.48,color=ACCENT,alpha=.18,zorder=0)
            for k in np.argsort(values,kind='stable'):
                height=values[k]
                if height<=0:continue
                a=AA[k];g=glyphs[a];box=g.get_extents()
                trans=Affine2D().translate(-box.x0,-box.y0).scale(.87/box.width,height/box.height).translate(j-.435,bottom)
                la.add_patch(PathPatch(g,transform=trans+la.transData,facecolor=colors[a],linewidth=0));bottom+=height
            if j and columns[j]!=columns[j-1]+1:la.axvline(j-.5,color=ACCENT,ls=':',lw=.65)
        la.set(xlim=(-.65,19.65),ylim=(0,1.03),yticks=[0,.5,1],xticks=range(20),xticklabels=columns)
        la.tick_params(axis='x',rotation=90,labelsize=5.3,pad=1,length=2);la.tick_params(axis='y',labelsize=5.7,pad=1,length=2)
        la.set_xlabel('家族比对列（1基）',fontsize=6,labelpad=2)
        if i==0:la.set_ylabel('加权残基频率',fontsize=6,labelpad=1)
        txt(left,134,f"文献参考 {c['reference_start_1based']}–{c['reference_start_1based']+19}",6.3)
        txt(left,145,f"{c['protein_id']}  {c['start_1based']}–{c['end_1based']}",6.1)
        for y,sequence in [(139,c['reference_window']),(150,c['member_window'])]:
            for j,a in enumerate(sequence):
                index=c['member_site_1based']-c['start_1based']
                txt(left+1+j*2.43,y,a,6.4,family='DejaVu Sans Mono',weight='bold' if j==index else 'normal',color=ACCENT if j==index else INK)
        txt(left,158,f"文献位点 {'G' if i==0 else 'K'}{c['reference_site_1based']} → 成员 {'G' if i==0 else 'K'}{c['member_site_1based']}",6.4,color='#A26529')
    for ext in ['svg','pdf']:fig.savefig(FIG/f'Figure_7_literature_sites.{ext}')
    fig.savefig(FIG/'Figure_7_literature_sites_600dpi.png',dpi=600)
    fig.savefig(FIG/'Figure_7_literature_sites_preview.png',dpi=160)
    plt.close(fig)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    t=read(DATA/'Table_S6_10_without_enrichment_support.csv')
    for row in t:
        for field in ['mapped_proteins','all_cluster_proteins']:
            row[field]=int(row[field])
        row['mapped_fraction']=float(row['mapped_fraction'])
    c=json.loads((DATA/'Figure7_cases.json').read_text(encoding='utf-8-sig'));plot(t,c)
    print('Prepared 17 candidates, 10 target candidates, 3 profiles and Figure 7.')
