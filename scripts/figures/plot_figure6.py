"""Figure 7: descriptive conservation, using every scored source observation.

Run: python plot_figure7.py
Dependencies: numpy, pandas, matplotlib, Pillow. Use --language en for English.
All scores are read from source_data; display offsets never modify score values.
"""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'work/figures'
BASE.mkdir(parents=True,exist_ok=True)
(BASE/'qa').mkdir(parents=True,exist_ok=True)
# Optional existing project dependencies; ordinary installed packages also work.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import findfont, FontProperties

DATA = ROOT/'results/paper/figure_inputs/Figure_6'
CASES = ['GvpJ_M2_motif_018', 'GvpM_M2_motif_002']
LABELS = ['GvpJ-018', 'GvpM-002']
TEAL, BLUE, LIGHT, INK = '#247C83', '#788FA6', '#DCE9EE', '#263442'

def offsets(n, span):
    """Deterministic display-only offsets; no subsampling or score jitter."""
    return (((np.arange(n) + .5) * .6180339887498949) % 1 - .5) * 2 * span

def spread_ties(values):
    """Spread identical scores evenly horizontally, retaining exact vertical values."""
    x = np.zeros(len(values))
    for value in np.unique(values):
        ix = np.flatnonzero(values == value)
        x[ix] = np.linspace(-.32, .32, len(ix)) if len(ix) > 1 else 0
    return x

def build(language='zh'):
    zh = language == 'zh'
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'Arial', 'DejaVu Sans'] if zh else ['Arial','DejaVu Sans'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    candidates = pd.read_csv(DATA/'Figure_7ab_candidates.csv')
    proteins = pd.read_csv(DATA/'Figure_7c_proteins.csv')
    family = pd.read_csv(DATA/'family_summary.csv').set_index('family')
    catalogue = pd.read_csv(DATA/'family_catalogue.csv').set_index('family').sort_index()
    scored = candidates.loc[candidates.mean_conservation.notna()]
    v = scored.mean_conservation.to_numpy()
    assert len(candidates)==522 and len(scored)==492 and np.isfinite(v).all()
    assert ((v>=0)&(v<=1)).all() and not candidates.motif_uid.duplicated().any()
    families = catalogue.index.tolist()
    assert len(families)==16 and int(catalogue.loc['GvpH','catalog_count'])==0
    assert f'{np.median(v):.4f}'=='0.5634'

    # Quantitative grid: a above b/c, fixed 170 x 160 mm publication size.
    fig = plt.figure(figsize=(170/25.4, 160/25.4))
    a = fig.add_axes([.205,.465,.765,.445])
    b = fig.add_axes([.105,.090,.365,.255])
    c = fig.add_axes([.625,.090,.345,.255])
    title_a = '各 Gvp 家族的候选保守性' if zh else 'Conservation across Gvp families'
    title_b = '候选保守性评分的总体分布' if zh else 'Distribution of candidate conservation scores'
    title_c = 'GvpJ-018 与 GvpM-002 的保守性' if zh else 'Conservation of GvpJ-018 and GvpM-002'
    # Place panel letters separately so titles use the same convention as Figures 2–5.
    for ax,letter,title in [(a,'a',title_a),(b,'b',title_b),(c,'c',title_c)]:
        title_size = (8 if ax is c else 8.5) if zh or ax is a else 7.5
        ax.set_title(title,loc='left',pad=12,fontsize=title_size,fontweight='normal')
        ax.text(-.065 if ax is a else -.125,1.05,letter,transform=ax.transAxes,
                ha='left',va='bottom',weight='bold',fontsize=10,color='black')
        ax.set_axisbelow(True)
    ticks=[]; point_rows=[]; box_rows=[]
    for y,f in enumerate(families,1):
        group=scored.loc[scored.family==f]
        values=group.mean_conservation.to_numpy()
        total=int(catalogue.loc[f,'catalog_count'])
        assert total==int((candidates.family==f).sum())
        ticks.append(f'{f}   {len(values)}/{total}')
        if f in family.index:
            assert len(values)==int(family.loc[f,'n_candidates_with_score'])
        if len(values)>1:
            a.boxplot([values],positions=[y],orientation='horizontal',widths=.54,
                whis=(0,100),showfliers=False,patch_artist=True,manage_ticks=False,
                boxprops={'facecolor':LIGHT,'edgecolor':TEAL,'linewidth':.65},
                medianprops={'color':INK,'linewidth':1.15},
                whiskerprops={'color':TEAL,'linewidth':.65},
                capprops={'color':TEAL,'linewidth':.65},zorder=2)
            q=np.quantile(values,[0,.25,.5,.75,1])
            box_rows.append(dict(family=f,n=len(values),minimum=q[0],q25=q[1],median=q[2],q75=q[3],maximum=q[4]))
        if len(values):
            yy=y+(offsets(len(values),.16) if len(values)>1 else 0)
            a.scatter(values,np.broadcast_to(yy,values.shape),s=5 if len(values)>1 else 12,
                      color=TEAL,alpha=.28 if len(values)>1 else .9,edgecolors='none',zorder=3)
            for uid,x,yp in zip(group.motif_uid,values,np.broadcast_to(yy,values.shape)):
                point_rows.append(dict(panel='a',identifier=uid,x_score=x,y_display=yp))
        else:
            # Annotation uses y-axis transform: no score or placeholder point at zero.
            text='无候选（不适用）' if total==0 and zh else ('No candidates (not applicable)' if total==0 else ('无可评分候选' if zh else 'No scored candidates'))
            a.text(.025,y,text,transform=a.get_yaxis_transform(),va='center',fontsize=7,color='#65747D')
    a.set(yticks=np.arange(1,17),yticklabels=ticks,xlim=(0,1),ylim=(16.6,.4),
          xlabel='候选区域保守性评分' if zh else 'Candidate region conservation score',
          ylabel='家族（可评分数 / 全部候选数）' if zh else 'Family (scored / total candidates)')
    a.set_xticks(np.linspace(0,1,6)); a.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    a.grid(axis='x',color='#E7ECEF',linewidth=.45)

    edges=np.linspace(0,1,26)
    counts,_,patches=b.hist(v,bins=edges,color='#6D9D9E',edgecolor='white',linewidth=.45)
    assert int(counts.sum())==492 and len(patches)==25
    b.axvline(np.median(v),color=INK,linewidth=.95,linestyle=(0,(3,2)),zorder=4)
    b.set(xlim=(0,1),ylim=(0,82),xlabel='候选区域保守性评分' if zh else 'Candidate region conservation score',
          ylabel='候选数' if zh else 'Candidates')
    b.set_xticks(np.linspace(0,1,6)); b.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    b.set_yticks([0,20,40,60,80])
    b.text(.035,.96,('可评分：492/522\n中位数：0.5634' if zh else 'Scored: 492/522\nMedian: 0.5634'),
           transform=b.transAxes,ha='left',va='top',fontsize=7,linespacing=1.5)

    c.set(ylim=(.68,.82),xlim=(-.56,1.56),xticks=[0,1],xticklabels=LABELS,
          ylabel='成员级区间保守性评分' if zh else 'Member-level region conservation score')
    c.set_yticks(np.arange(.68,.82001,.02)); c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    c.grid(axis='y',color='#E7ECEF',linewidth=.45)
    c.text(.5,1.005,'均值 ± 样本标准差' if zh else 'Mean ± sample SD',
           transform=c.transAxes,ha='center',va='bottom',fontsize=6.8,color='#52616B')
    case_stats=[]
    for i,uid in enumerate(CASES):
        g=proteins.loc[proteins.motif_uid==uid]
        values=g.median_conservation.to_numpy()
        assert len(values)==[261,247][i] and np.isfinite(values).all()
        assert ((values>=.68)&(values<=.82)).all()
        xx=i+spread_ties(values)
        c.scatter(xx,values,s=3.5,alpha=.30,color=[TEAL,BLUE][i],edgecolors='none',zorder=3)
        mean,sd=values.mean(),values.std(ddof=1)
        c.errorbar(i,mean,yerr=sd,fmt='D',ms=3.4,color=INK,capsize=3,
                   linewidth=.8,capthick=.8,zorder=5)
        # Summaries sit outside the data area so that the highest observations remain visible.
        c.text(i,-.17,f'n = {len(values)}\n{mean:.4f} ± {sd:.4f}',
               transform=c.get_xaxis_transform(),ha='center',va='top',fontsize=6.8,linespacing=1.35)
        case_stats.append(dict(candidate=LABELS[i],n=len(values),mean=mean,sample_sd=sd))
        for pid,x,y in zip(g.protein_id,xx,values):
            point_rows.append(dict(panel='c',identifier=uid+'|'+pid,x_display=x,y_score=y))
    stem='Figure_6_conservation'+('_EN' if not zh else '')
    fig.canvas.draw()
    # Ensure all rendered text is inside the canvas; preserve exact physical dimensions.
    renderer=fig.canvas.get_renderer()
    outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        bb=t.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.x1+.5 or bb.y1>fig.bbox.y1+.5:
            outside.append(t.get_text())
    assert not outside, outside
    fig.savefig(BASE/f'{stem}.svg')
    fig.savefig(BASE/f'{stem}.pdf')
    fig.savefig(BASE/f'{stem}_600dpi.png',dpi=600)
    # Previews are derived from the 600-dpi master; they are not submission files.
    with Image.open(BASE/f'{stem}_600dpi.png') as master:
        preview=master.copy(); preview.thumbnail((1338,1260))
        preview.save(BASE/f'{stem}_preview.png')
        reduced=master.copy(); reduced.thumbnail((669,630))
        reduced.save(BASE/'qa'/f'{stem}_85mm_reduction.png')
    plotted=pd.DataFrame(point_rows)
    plotted.to_csv(BASE/'qa'/f'{stem}_plotted_observations.csv',index=False)
    pd.DataFrame(box_rows).to_csv(BASE/'qa'/f'{stem}_box_statistics.csv',index=False)
    pd.DataFrame({'left':edges[:-1],'right':edges[1:],'count':counts.astype(int)}).to_csv(BASE/'qa'/f'{stem}_histogram.csv',index=False)
    # The source coordinates, histogram and physical output are auditable without image digitization.
    assert len(plotted.query("panel=='a'"))==492 and len(plotted.query("panel=='c'"))==508
    np.testing.assert_array_equal(plotted.loc[plotted.panel=='a','x_score'],
        np.concatenate([scored.loc[scored.family==f,'mean_conservation'].to_numpy() for f in families]))
    np.testing.assert_array_equal(plotted.loc[plotted.panel=='c','y_score'],
        np.concatenate([proteins.loc[proteins.motif_uid==uid,'median_conservation'].to_numpy() for uid in CASES]))
    qa=dict(width_mm=170,height_mm=160,language=language,dpi=600,
            font_file=findfont(FontProperties(family=plt.rcParams['font.sans-serif'][0])),
            axes_limits={'a':list(a.get_xlim()),'b':list(b.get_xlim()),'c':list(c.get_ylim())},
            panel_a_points=492,panel_c_points=508,box_count=len(box_rows),single_point_family='GvpL',
            empty_family='GvpH',bins=25,histogram_total=int(counts.sum()),
            all_score_coordinates_exact=True,display_offsets_only=True,tests_performed=False,
            case_statistics=case_stats,outside_text=outside,
            minimum_font_pt=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()))
    (BASE/'qa'/f'{stem}_render_audit.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
    print(json.dumps(qa,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--language',choices=['zh','en'],default='zh')
    build(parser.parse_args().language)
