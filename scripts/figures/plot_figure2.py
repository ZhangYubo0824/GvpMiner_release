"""Figure 2: all paired observations, unchanged values and descriptive summaries.

Run: python plot_figure2.py [--english]. Requires numpy, pandas, matplotlib, Pillow.
No resampling, clustering, significance testing, or missing-to-zero conversion.
"""
from pathlib import Path
import json, argparse
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'work/figures'
BASE.mkdir(parents=True,exist_ok=True)
(ROOT/'work/figure_qa').mkdir(parents=True,exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image
W,H=170,165
TEAL='#247C83';ORANGE='#CE8C45';GRAY='#AAB7C0';INK='#263442'
METRICS=['stability_floor','stable_protein_coverage','stable_window_coverage','noise_ratio']
SETTINGS=['w20_s10','w20_s15'];MODELS=['M2','M3']
def main(english=False):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial' if english else 'Microsoft YaHei','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':7.2,'xtick.labelsize':6.5,'ytick.labelsize':6.2,
        'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'legend.fontsize':6.5,
        'figure.facecolor':'white','savefig.facecolor':'white'})
    d=pd.read_csv(ROOT/'results/paper/figure_inputs/Figure_2/stage_c_setting_family_metrics.csv')
    summary=pd.read_csv(ROOT/'results/paper/figure_inputs/Figure_2/panel_summary.csv').set_index(['model','metric'])
    nested=pd.read_csv(ROOT/'results/paper/figure_inputs/Figure_2/cross_representation_nested_medians.csv').set_index('setting')
    assert len(d)==64 and not d[METRICS].isna().any().any()
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    titles=(['Stability lower bound','Stable protein coverage','Stable window coverage','Noise fraction'] if english else
        ['稳定性下界','稳定蛋白覆盖率','稳定窗口覆盖率','噪声比例'])
    def text(x,y,s,**kw):return fig.text(x/W,1-y/H,s,color=INK,va='center',**kw)
    def axes(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    handles=[Line2D([0],[0],color=GRAY,lw=.75,marker='o',markersize=2.4,alpha=.6),
        Line2D([0],[0],color=ORANGE,lw=.9,marker='o',markersize=2.5),Line2D([0],[0],color=TEAL,lw=2.5)]
    labels=['Other families','GvpH / GvpO','Family median'] if english else ['其他家族','GvpH / GvpO','家族中位数']
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.53,.993),handlelength=1.7,columnspacing=2.1)
    text(88,13,'20/10 → 20/15: increases in the first three metrics and decreases in noise count as better' if english else
         '20/10 → 20/15：前三项上升、噪声下降计为改善',ha='center',fontsize=6.5)
    points=[];medians=[];panels=[];direct=[];lines=[]
    for row,model in enumerate(MODELS):
        top=30+56*row
        text(3.8,top+15,['ESM-2','ProtBERT-BFD'][row],ha='center',rotation=90,fontsize=8,weight='bold')
        for col,metric in enumerate(METRICS):
            left=18+37.5*col;width=32
            s=summary.loc[(model,metric)];letter=chr(97+4*row+col)
            # Two-line EN titles keep the same physical title slot as the CN figure.
            title=titles[col]
            if english:title=title.replace(' lower ',' lower\n').replace(' protein ',' protein\n').replace(' window ',' window\n')
            text(left-4,top-7,letter,weight='bold',fontsize=9,ha='left')
            text(left+width/2,top-7,title,ha='center',fontsize=7.1,linespacing=1.15)
            ax=axes(left,top,width,30)
            p=d[d.method==model].pivot(index='family',columns='setting',values=metric).sort_index()
            for family in sorted(p.index,key=lambda f:f in ['GvpH','GvpO']):
                values=p.loc[family,SETTINGS].to_numpy(dtype=float)
                highlight=family in ['GvpH','GvpO'];color=ORANGE if highlight else GRAY
                alpha=.95 if highlight else .43;z=5 if highlight else 2
                # No jitter or value transformation. Finite pairs remain directly connected.
                finite=np.isfinite(values)
                if finite.all():
                    artist,=ax.plot([0,1],values,color=color,lw=.9 if highlight else .65,alpha=alpha,
                        marker='o',ms=2.5 if highlight else 2.2,mew=0,zorder=z,clip_on=False)
                    lines.append(dict(panel=letter,model=model,metric=metric,family=family,
                        x0=float(artist.get_xdata()[0]),x1=float(artist.get_xdata()[1]),
                        y0=float(artist.get_ydata()[0]),y1=float(artist.get_ydata()[1])))
                elif finite.any():
                    ax.plot(np.arange(2)[finite],values[finite],ls='none',marker='o',color=color,ms=2.5)
                for x,v in enumerate(values):
                    if finite[x]:points.append(dict(panel=letter,model=model,metric=metric,family=family,setting=SETTINGS[x],x=x,value=v))
            for x,setting in enumerate(SETTINGS):
                median=float(p[setting].median())
                a,=ax.plot([x-.11,x+.11],[median,median],lw=2.6,color=TEAL,zorder=4,solid_capstyle='butt',clip_on=False)
                medians.append(dict(panel=letter,model=model,metric=metric,setting=setting,value=float(a.get_ydata()[0])))
            ymax=.30 if metric=='noise_ratio' else 1.
            ax.set_ylim(0,ymax);ax.set_yticks([0,.1,.2,.3] if metric=='noise_ratio' else [0,.25,.5,.75,1])
            ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.set_xlim(-.22,1.67);ax.set_xticks([0,1],['20/10','20/15'] if row else [])
            ax.tick_params(axis='x',length=2 if row else 0,pad=3);ax.tick_params(axis='y',length=2,pad=2)
            ax.grid(axis='y',color='#E8ECEF',lw=.5);ax.set_axisbelow(True)
            # Direct endpoint labels with leaders; labels move, observation coordinates do not.
            named=['GvpH','GvpO',s.worst_ordinary_family]
            observed=np.array([p.loc[f,SETTINGS[1]] for f in named]);order=np.argsort(observed,kind='stable')
            label_y=observed[order].copy();gap=.085*ymax
            for k in range(1,len(label_y)):label_y[k]=max(label_y[k],label_y[k-1]+gap)
            if label_y[-1]>.985*ymax:label_y-=label_y[-1]-.985*ymax
            if label_y[0]<.025*ymax:label_y+=.025*ymax-label_y[0]
            for idx,yy in zip(order,label_y):
                family=named[idx];color=ORANGE if family in ['GvpH','GvpO'] else '#647784'
                ax.annotate(family,xy=(1,observed[idx]),xytext=(1.17,yy),ha='left',va='center',fontsize=6,
                    color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.45,shrinkA=0,shrinkB=2),
                    annotation_clip=False,zorder=8)
                direct.append(dict(panel=letter,family=family,endpoint=float(observed[idx]),label_y=float(yy)))
            counts=(f"Better {s.improved} / worse {s.worse} / same {s.unchanged}" if english else
                    f"改善 {s.improved} / 变差 {s.worse} / 不变 {s.unchanged}")
            text(left+width/2,top+36+4*row,counts,ha='center',fontsize=6.1)
            # Set color separately to keep one text helper for all editable labels.
            t=text(left+width/2,top+40.5+4*row,('Median ' if english else '中位数 ')+
                f'{s.median_before*100:.2f} → {s.median_after*100:.2f}%',ha='center',fontsize=6.1)
            t.set_color(TEAL)
            worse=s.worse_families.split(';') if isinstance(s.worse_families,str) else []
            panels.append(dict(panel=letter,model=model,metric=metric,ylim=list(ax.get_ylim()),
                yticks=list(ax.get_yticks()),families=len(p),paired_families=16,
                counts={'improved':int(s.improved),'worse':int(s.worse),'unchanged':int(s.unchanged)},
                worse_families=worse,x_labels_visible=bool(row)))
    # This band is separate from the eight single-model panels. Nested medians are not pooled observations.
    fig.add_artist(Rectangle((11/W,1-161/H),154/W,26/H,transform=fig.transFigure,
        facecolor='#F2F6F7',edgecolor='none',zorder=-1))
    text(88,139,'Across representations: 20/10 → 20/15 (separate from either model)' if english else
        '跨两种表示汇总：20/10 → 20/15',ha='center',fontsize=6.5,weight='bold')
    for col,c in enumerate(METRICS):
        x=18+37.5*col+16
        short=(['Stability','Protein coverage','Window coverage','Noise'] if english else ['稳定性下界','稳定蛋白覆盖率','稳定窗口覆盖率','噪声比例'])[col]
        text(x,145,short,ha='center',fontsize=6.3)
        text(x,151,f'{nested.loc[SETTINGS[0],c]:.3f} → {nested.loc[SETTINGS[1],c]:.3f}',ha='center',fontsize=7,weight='bold')
    text(88,158,'Median across the two representations within each family, then across 16 families' if english else
        '每个家族先对两种表示取中位数，再跨 16 个家族取中位数',ha='center',fontsize=6)
    stem='Figure_2_parameter_comparison'+('_EN' if english else '')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bbox=fig.bbox
    outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_visible():
            b=t.get_window_extent(renderer)
            if b.x0<-.5 or b.y0<-.5 or b.x1>bbox.x1+.5 or b.y1>bbox.y1+.5:outside.append(t.get_text())
    fig.savefig(BASE/f'{stem}.svg')
    fig.savefig(BASE/f'{stem}.pdf')
    fig.savefig(BASE/f'{stem}_600dpi.png',dpi=600)
    with Image.open(BASE/f'{stem}_600dpi.png') as im:
        im.thumbnail((1200,1600));im.save(BASE/f'{stem}_preview.png')
    for name,records in [('points',points),('paired_lines',lines),('medians',medians),('direct_labels',direct)]:
        pd.DataFrame(records).to_csv(ROOT/'work/figure_qa'/f'{stem}_{name}.csv',index=False)
    (ROOT/'work/figure_qa'/f'{stem}_render_audit.json').write_text(json.dumps(dict(
        dimensions_mm=[W,H],panels=panels,outside_text=outside,observations=len(points),paired_lines=len(lines),
        median_markers=len(medians),minimum_font_pt=6,analyses_rerun=False,statistical_tests_added=False,
        axis_data_jitter=False,missing_to_zero=False,cross_route_summary_is_separate=True),ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
    print(json.dumps(dict(stem=stem,points=len(points),lines=len(lines),medians=len(medians),outside_text=outside),ensure_ascii=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--english',action='store_true');main(parser.parse_args().english)
