"""Three-panel Figure 5: status composition and two explicitly different denominators.

Run: python plot_figure5.py [--language en]
All 10,784 candidate states come from source_data, recounted from original outputs.
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
from matplotlib.ticker import PercentFormatter
from PIL import Image

DATA=ROOT/'results/paper/source_data';QA=ROOT/'work/figure_qa'
QA.mkdir(parents=True,exist_ok=True)
METHODS=['GvpMiner','GLAM2','MEME','STREME','MotifAE','InterPLM']
STATUS_COLORS=['#247C83','#9EAFBD','#E2E6E9'];INK='#263442'
WIDTH_MM,HEIGHT_MM=170,150
def percent(n,N):return str((Decimal(int(n))*100/Decimal(int(N))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
def figtext(fig,x,top,s,size=7,**kwargs):
    return fig.text(x/WIDTH_MM,1-top/HEIGHT_MM,s,ha='left',va='center',fontsize=size,**kwargs)
def axmm(fig,x,top,w,h):return fig.add_axes([x/WIDTH_MM,1-(top+h)/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM])
def method_axis(ax):
    ax.set(yticks=np.arange(6),yticklabels=METHODS,ylim=(5.65,-.65),xlim=(0,1))
    ax.set_xticks([0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.tick_params(axis='y',length=0,pad=5)
    ax.spines['left'].set_visible(False)
    ax.set_axisbelow(True)

def main(language='zh'):
    zh=language=='zh'
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'] if zh else ['Arial','DejaVu Sans'],
        'font.size':7.3,'axes.labelsize':7.3,'xtick.labelsize':7,'ytick.labelsize':7.3,
        'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
        'xtick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    d=pd.read_csv(DATA/'Figure_5_method_summary.csv').set_index('method').loc[METHODS]
    palette=pd.read_csv(DATA/'method_palette.csv').set_index('method').color.to_dict()
    assert list(d.input)==[522,160,320,529,8378,875]
    assert (d.supported+d.output_not_supported+d.no_output).equals(d.input)
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    figtext(fig,7,8,'a',10,weight='bold')
    figtext(fig,14,8,'注释富集输出与支持状态' if zh else 'Annotation enrichment output and support status',8.5)
    a=axmm(fig,29,26,126,43.5);method_axis(a)
    labels=['显著支持','有输出但未显著','无富集输出'] if zh else ['Significant support','Output, not significant','No enrichment output']
    left=np.zeros(6);bar_records=[];point_records=[]
    for status,label,color in zip(['supported','output_not_supported','no_output'],labels,STATUS_COLORS):
        values=d[status].to_numpy()/d.input.to_numpy()
        a.barh(range(6),values,left=left,height=.62,color=color,label=label,edgecolor='white',linewidth=.35)
        for i,method in enumerate(METHODS):
            n=int(d.loc[method,status]);N=int(d.loc[method,'input'])
            shown=f'{n}'
            if method=='GvpMiner' and status=='no_output':shown+=f' ({percent(n,N)}%)'
            a.text(left[i]+values[i]/2,i,shown,ha='center',va='center',fontsize=7,
                weight='bold' if method=='GvpMiner' and status=='no_output' else 'normal',
                color='white' if status=='supported' else INK)
            bar_records.append(dict(method=method,status=status,count=n,total=N,left=left[i],width=values[i],color=color))
        left+=values
    np.testing.assert_allclose(left,np.ones(6),atol=1e-15)
    a.legend(loc='lower left',bbox_to_anchor=(-.005,1.08),ncol=3,fontsize=7,
             handlelength=1.5,handletextpad=.5,columnspacing=1.7,borderaxespad=0)
    a.set_xlabel('占全部输入候选的比例' if zh else 'Proportion of all input candidates',labelpad=6)
    figtext(fig,158,22,'输入数' if zh else 'Input n',6.8)
    for i,method in enumerate(METHODS):
        a.text(1.025,i,str(int(d.loc[method,'input'])),transform=a.get_yaxis_transform(),va='center',fontsize=7.2)
    b=axmm(fig,29,104,51.5,32.5);c=axmm(fig,109,104,51.5,32.5)
    lower_titles=(['有富集输出候选中的注释支持','全部输入候选中的注释支持'] if zh else
        ['Annotation support among candidates\nwith enrichment output','Annotation support among\nall input candidates'])
    subtitles=(['显著支持数 / 有富集输出候选数','显著支持数 / 全部输入候选数'] if zh else
        ['Significant / with enrichment output','Significant / all input candidates'])
    label_boxes=[]
    for j,(ax,denominator) in enumerate([(b,'with_output'),(c,'input')]):
        method_axis(ax)
        figtext(fig,7+80*j,87,chr(98+j),10,weight='bold')
        figtext(fig,14+80*j,87,lower_titles[j],7.9 if zh else 7.4)
        figtext(fig,14+80*j,97,subtitles[j],6.6,color='#65747D')
        for i,method in enumerate(METHODS):
            S=int(d.loc[method,'supported']);N=int(d.loc[method,denominator]);rate=S/N
            ax.scatter(rate,i,s=23,color=palette[method],edgecolors='white',linewidths=.4,zorder=3)
            left_side=rate>=.55
            label=ax.annotate(f'{S}/{N}  ({percent(S,N)}%)',xy=(rate,i),
                xytext=(-6 if left_side else 6,0),textcoords='offset points',
                ha='right' if left_side else 'left',va='center',fontsize=6.8,color=INK)
            label_boxes.append((ax,label))
            point_records.append(dict(panel=chr(98+j),method=method,significant=S,denominator=N,
                denominator_field=denominator,rate=rate,display_percent=percent(S,N),color=palette[method],y=i))
        ax.set_xlabel('注释支持比例' if zh else 'Annotation support rate',labelpad=6)
    assert b.get_xlim()==c.get_xlim()==(0.,1.)
    assert [p['method'] for p in point_records[:6]]==[p['method'] for p in point_records[6:]]==METHODS
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];label_overflow=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:outside.append(t.get_text())
    for ax,label in label_boxes:
        box=label.get_window_extent(renderer)
        if box.x0<ax.bbox.x0 or box.x1>ax.bbox.x1:label_overflow.append(label.get_text())
    assert not outside,outside
    assert not label_overflow,label_overflow
    stem='Figure_5_annotation_support'+('_EN' if not zh else '')
    fig.savefig(BASE/f'{stem}.svg')
    fig.savefig(BASE/f'{stem}.pdf')
    fig.savefig(BASE/f'{stem}_600dpi.png',dpi=600)
    with Image.open(BASE/f'{stem}_600dpi.png') as master:
        preview=master.copy();preview.thumbnail((1400,1235));preview.save(BASE/f'{stem}_preview.png')
        reduced=master.copy();reduced.thumbnail((700,618));reduced.save(QA/f'{stem}_85mm_reduction.png')
    pd.DataFrame(bar_records).to_csv(QA/f'{stem}_bars.csv',index=False)
    pd.DataFrame(point_records).to_csv(QA/f'{stem}_points.csv',index=False)
    audit=dict(language=language,size_mm=[170,150],points=point_records,methods=METHODS,
        status_colors=STATUS_COLORS,method_colors=palette,axis_limits={'a':list(a.get_xlim()),'b':list(b.get_xlim()),'c':list(c.get_xlim())},
        same_method_order_and_color=True,same_point_size=True,between_method_tests=False,
        no_output_GvpMiner='239/522 (45.79%)',outside_text=outside,point_label_overflow=label_overflow,
        min_font_pt=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()))
    (QA/f'{stem}_render_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig);print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--language',choices=['zh','en'],default='zh')
    main(parser.parse_args().language)
