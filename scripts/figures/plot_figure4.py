"""Figure 4: two aligned structural-support proportions and exact absolute counts.

Run: python plot_figure4.py [--language en]
No hypothesis tests are run here. All values come from audited candidate records.
"""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json, argparse
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'work/figures'
BASE.mkdir(parents=True,exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image
DATA=ROOT/'results/paper/source_data';QA=ROOT/'work/figure_qa'
QA.mkdir(parents=True,exist_ok=True)
METHODS=['GvpMiner','GLAM2','MEME','STREME','MotifAE','InterPLM']
WIDTH_MM,HEIGHT_MM=170,95
INK='#263442'
def pct(n,N):return str((Decimal(int(n))*100/Decimal(int(N))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
def figtext(fig,x,top,s,size=7,**kwargs):
    return fig.text(x/WIDTH_MM,1-top/HEIGHT_MM,s,ha='left',va='center',fontsize=size,**kwargs)
def axes(fig,x,top,w,h):return fig.add_axes([x/WIDTH_MM,1-(top+h)/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM])
def main(language='zh'):
    zh=language=='zh'
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'] if zh else ['Arial','DejaVu Sans'],
        'font.size':7.3,'axes.labelsize':7.2,'xtick.labelsize':6.8,'ytick.labelsize':7.3,
        'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
        'xtick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'figure.facecolor':'white','savefig.facecolor':'white','legend.frameon':False})
    data=pd.read_csv(DATA/'Figure_4_method_summary.csv').set_index('method').loc[METHODS]
    colors=pd.read_csv(DATA/'method_palette.csv').set_index('method').color.to_dict()
    assert data.any_endpoint_supported.tolist()==[509,159,289,515,4717,301]
    assert data.strict_supported.tolist()==[348,102,149,262,2028,93]
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    a=axes(fig,25,28,52,46);b=axes(fig,85,28,52,46);c=axes(fig,145,28,20,46)
    titles=(['至少一个结构终点显著','严格结构支持','严格支持数'] if zh else
        ['At least one significant\nstructural endpoint','Strict structural\nsupport','Strict support\ncount'])
    subtitles=(['分母：结构可检验候选','分母：统一评估输入','候选数'] if zh else
        ['Denominator: structurally testable','Denominator: all evaluated inputs','Candidates (n)'])
    for x,letter,title,subtitle in zip([25,85,145],'abc',titles,subtitles):
        figtext(fig,x-5,10,letter,10,weight='bold')
        figtext(fig,x+1,10,title,7.6 if zh else 7.3)
        figtext(fig,x+1,21,subtitle,6.5 if zh else 6.3,color='#65747D')
    figtext(fig,5,25,'方法' if zh else 'Method',6.7,color='#65747D')
    points=[];counts=[];annotation_boxes=[]
    for j,(ax,numerator,denominator) in enumerate([(a,'any_endpoint_supported','testable'),(b,'strict_supported','input')]):
        ax.set(xlim=(0,1),ylim=(5.65,-.65),yticks=range(6),yticklabels=METHODS if j==0 else [])
        ax.set_xticks([0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='x',color='#E8ECEF',lw=.4)
        for i,method in enumerate(METHODS):
            n=int(data.loc[method,numerator]);N=int(data.loc[method,denominator]);value=n/N
            # Keep exact 0–100% axes; markers near 100% remain fully visible.
            ax.scatter(value,i,s=23,color=colors[method],edgecolors='white',linewidths=.4,
                       zorder=4,clip_on=False)
            on_left=value>=.55
            label=ax.annotate(f'{n}/{N}  ({pct(n,N)}%)',xy=(value,i),
                xytext=(-6 if on_left else 6,0),textcoords='offset points',
                ha='right' if on_left else 'left',va='center',fontsize=6.7,color=INK,
                bbox={'facecolor':'white','edgecolor':'none','pad':.2},zorder=5)
            annotation_boxes.append((ax,label))
            points.append(dict(panel=chr(97+j),method=method,numerator=n,denominator=N,
                numerator_field=numerator,denominator_field=denominator,rate=value,
                display_percent=pct(n,N),color=colors[method],y=i))
        ax.set_xlabel('候选比例' if zh else 'Proportion of candidates',labelpad=6)
    # Exact count cells use no length/area scale, axis break, or data-dependent size.
    c.set(xlim=(0,1),ylim=(5.65,-.65));c.axis('off')
    c.add_patch(Rectangle((0,-.65),1,6.3,color='#F5F7F8',linewidth=0,zorder=0))
    for boundary in np.arange(.5,5,1):
        c.axhline(boundary,color='white',lw=.8,zorder=1)
    for i,method in enumerate(METHODS):
        count=int(data.loc[method,'strict_supported'])
        c.scatter(.15,i,s=15,color=colors[method],edgecolors='none',zorder=3)
        c.text(.83,i,str(count),ha='right',va='center',fontsize=10.5,weight='normal',color=INK,zorder=4)
        counts.append(dict(method=method,count=count,y=i,color=colors[method],font_size=10.5))
    conditions=('严格支持须同时满足：主终点显著、平均 pLDDT ≥ 70、组成匹配对照覆盖率 ≥ 90%' if zh else
        'Strict support requires a significant primary endpoint, mean pLDDT ≥ 70, and composition-matched control coverage ≥ 90%.')
    figtext(fig,6,90,conditions,6.5 if zh else 6.3,color='#596A75')
    assert a.get_xlim()==b.get_xlim()==(0.,1.)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];overflow=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:outside.append(t.get_text())
    for ax,t in annotation_boxes:
        box=t.get_window_extent(renderer)
        if box.x0<ax.bbox.x0 or box.x1>ax.bbox.x1:overflow.append(t.get_text())
    assert not outside,outside
    assert not overflow,overflow
    stem='Figure_4_structural_support'+('_EN' if not zh else '')
    fig.savefig(BASE/f'{stem}.svg')
    fig.savefig(BASE/f'{stem}.pdf')
    fig.savefig(BASE/f'{stem}_600dpi.png',dpi=600)
    with Image.open(BASE/f'{stem}_600dpi.png') as im:
        preview=im.copy();preview.thumbnail((1500,839));preview.save(BASE/f'{stem}_preview.png')
        small=im.copy();small.thumbnail((750,420));small.save(QA/f'{stem}_85mm_reduction.png')
    pd.DataFrame(points).to_csv(QA/f'{stem}_points.csv',index=False)
    pd.DataFrame(counts).to_csv(QA/f'{stem}_counts.csv',index=False)
    audit=dict(language=language,size_mm=[170,95],method_order=METHODS,method_palette=colors,
        fraction_axis_limits={'a':list(a.get_xlim()),'b':list(b.get_xlim())},
        same_row_positions=True,same_point_sizes=True,count_panel='exact numerals; no quantitative size scale',
        points=points,counts=counts,strict_thresholds={'q':'< 0.05','mean_plddt':'>= 70','control_coverage':'>= 0.90'},
        between_method_tests=False,new_statistical_results=False,outside_text=outside,point_label_overflow=overflow,
        min_font_pt=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()))
    (QA/f'{stem}_render_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig);print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--language',choices=['zh','en'],default='zh')
    main(parser.parse_args().language)
