"""Figure 3: unchanged family counts, original post-step CDF and five-seed recovery.

Run: python plot_figure3.py [--language en]
No clustering, resampling, PCA, null modelling or inference is performed here.
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
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from PIL import Image
DATA=ROOT/'results/paper/figure_inputs/Figure_3';QA=ROOT/'work/figure_qa'
QA.mkdir(parents=True,exist_ok=True)
TEAL,ORANGE,RAW,BLUEGRAY,INK='#247C83','#CE8C45','#D5DDE2','#9EAFBD','#263442'
WIDTH_MM,HEIGHT_MM=170,165
def pct(n,N):return str((Decimal(int(n))*100/Decimal(int(N))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
def text(fig,x,top,s,size=7,**kwargs):return fig.text(x/WIDTH_MM,1-top/HEIGHT_MM,s,ha='left',va='center',fontsize=size,**kwargs)
def axes(fig,x,top,w,h):return fig.add_axes([x/WIDTH_MM,1-(top+h)/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM])
def main(language='zh'):
    zh=language=='zh'
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'] if zh else ['Arial','DejaVu Sans'],
        'font.size':7.3,'axes.labelsize':7.2,'xtick.labelsize':6.8,'ytick.labelsize':6.8,
        'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':2.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'svg.fonttype':'none','pdf.fonttype':42,'text.color':INK,'axes.labelcolor':INK,
        'xtick.color':INK,'ytick.color':INK,'legend.frameon':False,
        'figure.facecolor':'white','savefig.facecolor':'white'})
    family=pd.read_csv(DATA/'family_counts.csv').sort_values('family').reset_index(drop=True)
    raw=pd.read_csv(DATA/'fig3_raw_cluster_stability.csv')
    cat=pd.read_csv(DATA/'stage_c_catalogue_522.csv')
    seed=pd.read_csv(DATA/'seed_recovery_counts.csv')
    assert family.raw.sum()==len(raw)==1244 and family.retained.sum()==len(cat)==522
    assert seed.candidates.tolist()==[2,2,3,14,501]
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    text(fig,7,8,'a',10,weight='bold')
    text(fig,14,8,'各家族的原始簇与保留候选' if zh else 'Candidate composition across families',8.5)
    text(fig,19,15,('总体保留：522/1,244 = 41.96%' if zh else 'Overall retention: 522/1,244 = 41.96%'),7.5)
    text(fig,119,15,'GvpH：0 个稳定候选' if zh else 'GvpH: no retained candidates',6.8,color='#65747D')
    labels=(['原始非噪声簇（1,244）','保留候选（522）','其中 GvpO：探索性回退'] if zh else
        ['Non-noise clusters (1,244)','Retained candidates (522)','GvpO subset: exploratory fallback'])
    fig.legend([Patch(facecolor=c) for c in [RAW,TEAL,ORANGE]],labels,loc='center left',
        bbox_to_anchor=(19/WIDTH_MM,1-23/HEIGHT_MM),ncol=3,fontsize=6.6,
        handlelength=1.5,handletextpad=.5,columnspacing=1.2,borderaxespad=0)
    a=axes(fig,19,31,146,43);x=np.arange(16);bar_records=[]
    a.bar(x-.19,family.raw,width=.36,color=RAW,zorder=3)
    retained_colors=[ORANGE if f=='GvpO' else TEAL for f in family.family]
    a.bar(x+.19,family.retained,width=.36,color=retained_colors,zorder=3)
    for i,r in enumerate(family.itertuples()):
        for status,n,xloc,color in [('raw',int(r.raw),i-.19,'#65747D'),('retained',int(r.retained),i+.19,retained_colors[i])]:
            a.annotate(str(n),xy=(xloc,n),xytext=(0,2),textcoords='offset points',
                ha='center',va='bottom',fontsize=6.2,color=color)
            bar_records.append(dict(family=r.family,status=status,count=n,x=xloc,
                color=RAW if status=='raw' else retained_colors[i]))
    a.set(xlim=(-.65,15.65),ylim=(0,310),ylabel='簇 / 候选数' if zh else 'Clusters / candidates')
    a.set_yticks([0,100,200,300]);a.set_xticks(x,family.family,rotation=45,ha='right',fontsize=6.8)
    a.set_axisbelow(True);a.grid(axis='y',color='#E8ECEF',lw=.4)

    text(fig,7,94,'b',10,weight='bold')
    text(fig,14,94,'条件稳定比例的累积分布' if zh else 'Conditional stability distribution',8.2)
    text(fig,88,94,'c',10,weight='bold')
    text(fig,95,94,'五个 UMAP 种子下的恢复' if zh else 'Recovery across five UMAP seeds',8.2)
    b=axes(fig,19,115,61,32.5);c=axes(fig,106,115,59,32.5)
    cdf_rows=[]
    # Exactly the original score sorting, rank normalization and where='post' rule.
    # No smoothing, tie aggregation or extension of the original curve arrays.
    for retained,label,color in [(True,'保留候选' if zh else 'Retained candidates',TEAL),
                                  (False,'未保留簇' if zh else 'Excluded clusters','#7D95A9')]:
        group=raw[raw.stable==retained].sort_values('stable_repeat_fraction',kind='stable')
        values=np.sort(raw.loc[raw.stable==retained,'stable_repeat_fraction'].to_numpy())
        cumulative=np.arange(1,len(values)+1)/len(values)
        assert len(values)==(522 if retained else 722)
        b.step(values,cumulative,where='post',color=color,lw=1.3,label=f'{label} (n = {len(values)})')
        for rank,(row,score,cy) in enumerate(zip(group.itertuples(),values,cumulative),1):
            cdf_rows.append(dict(group='retained' if retained else 'excluded',family=row.family,
                reference_label=int(row.reference_label),score=score,rank=rank,n=len(values),cumulative=cy))
    b.axvline(.8,color=ORANGE,ls='--',lw=.85)
    b.text(.775,.06,'阈值 0.8' if zh else 'Threshold 0.8',rotation=90,ha='right',va='bottom',
           fontsize=6.5,color='#AD713C',bbox={'facecolor':'white','edgecolor':'none','pad':.3})
    b.set(xlim=(0,1),ylim=(0,1),xlabel='条件稳定比例' if zh else 'Conditional stability',
          ylabel='累积比例' if zh else 'Cumulative proportion')
    b.set_xticks(np.linspace(0,1,6));b.set_yticks(np.linspace(0,1,6))
    b.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));b.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    b.legend(loc='lower left',bbox_to_anchor=(0,1.13),fontsize=6.8,handlelength=1.8,labelspacing=.4,borderaxespad=0)

    text(fig,106,103,'501/522 (95.98%)',8.2,weight='bold',color=TEAL)
    text(fig,106,108,'五个种子均恢复' if zh else 'recovered in all five seeds',6.8,color=INK)
    c.bar(range(1,6),seed.candidates,width=.64,color=[BLUEGRAY]*4+[TEAL],zorder=3)
    for k,n in zip(range(1,6),seed.candidates):
        c.annotate(str(int(n)),xy=(k,n),xytext=(0,3),textcoords='offset points',
            ha='center',va='bottom',fontsize=8 if k==5 else 6.8,color=TEAL if k==5 else INK)
    c.set(xlim=(.4,5.6),ylim=(0,600),xticks=range(1,6),xticklabels=[f'{k}/5' for k in range(1,6)],
          yticks=[0,200,400,600],xlabel='UMAP 种子恢复次数' if zh else 'Recovered UMAP seeds',
          ylabel='候选数' if zh else 'Candidates')
    c.set_axisbelow(True);c.grid(axis='y',color='#E8ECEF',lw=.4)
    inset=c.inset_axes([.09,.39,.61,.45]);inset.set_facecolor('white')
    inset.bar(range(1,5),seed.candidates.iloc[:4],width=.6,color=BLUEGRAY)
    for k,n in zip(range(1,5),seed.candidates.iloc[:4]):
        inset.annotate(str(int(n)),xy=(k,n),xytext=(0,1.5),textcoords='offset points',
                       ha='center',va='bottom',fontsize=6.2)
    inset.set(xlim=(.4,4.6),ylim=(0,20),xticks=range(1,5),xticklabels=[f'{k}/5' for k in range(1,5)],yticks=[0,10,20])
    inset.tick_params(labelsize=6,length=1.8,pad=1)
    for spine in inset.spines.values():spine.set_visible(True);spine.set_linewidth(.5);spine.set_color('#B7C2C9')
    inset.set_title('局部放大：1/5–4/5' if zh else 'Detail: 1/5–4/5',fontsize=6.2,pad=3)
    independent=('其他稳健性维度：PCA 匹配 342/522；匹配零分布 FDR < 0.05：431/522' if zh else
                 'Other robustness dimensions: PCA match, 342/522; matching-null FDR < 0.05, 431/522.')
    text(fig,9,161,independent,6.5,color='#65747D')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:outside.append(t.get_text())
    assert not outside,outside
    stem='Figure_3_stable_catalogue'+('_EN' if not zh else '')
    fig.savefig(BASE/f'{stem}.svg')
    fig.savefig(BASE/f'{stem}.pdf')
    fig.savefig(BASE/f'{stem}_600dpi.png',dpi=600)
    with Image.open(BASE/f'{stem}_600dpi.png') as master:
        preview=master.copy();preview.thumbnail((1400,1359));preview.save(BASE/f'{stem}_preview.png')
        small=master.copy();small.thumbnail((700,680));small.save(QA/f'{stem}_85mm_reduction.png')
    pd.DataFrame(bar_records).to_csv(QA/f'{stem}_family_bars.csv',index=False)
    pd.DataFrame(cdf_rows).to_csv(QA/f'{stem}_CDF.csv',index=False)
    seed.to_csv(QA/f'{stem}_recovery_bars.csv',index=False)
    seed.iloc[:4].to_csv(QA/f'{stem}_inset_bars.csv',index=False)
    audit=dict(language=language,size_mm=[170,165],family_order=family.family.tolist(),raw_total=1244,retained_total=522,
        family_bars=32,family_count_y=list(a.get_ylim()),GvpH_retained=0,GvpO_color=ORANGE,retention_percent=pct(522,1244),
        CDF_style='post',CDF_score_field='stable_repeat_fraction',CDF_counts={'retained':522,'excluded':722},
        CDF_limits={'x':list(b.get_xlim()),'y':list(b.get_ylim())},threshold=.8,
        recovery_counts=seed.candidates.tolist(),main_recovery_y=list(c.get_ylim()),inset_recovery_y=list(inset.get_ylim()),
        inset_categories=[1,2,3,4],inset_repeats_existing_counts=True,all_five_percent=pct(501,522),
        independent_summaries={'PCA_matches':342,'null_FDR_below_0_05':431,'denominator':522,'not_exclusive_categories':True},
        analyses_rerun=False,between_group_inference=False,outside_text=outside,
        minimum_font_pt=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()))
    (QA/f'{stem}_render_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig);print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--language',choices=['zh','en'],default='zh')
    main(parser.parse_args().language)
