from pathlib import Path


import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'work'
(BASE/'qa').mkdir(parents=True,exist_ok=True)
OUT=BASE/'figures';OUT.mkdir(parents=True,exist_ok=True);DATA=ROOT/'results/paper/source_data'
METHODS=['GvpMiner','GLAM2','MEME','STREME','MotifAE','InterPLM']
COLORS=['#247C83','#788FA6','#A895B5','#BC996F','#759F91','#A77F83']
TEAL=COLORS[0];ORANGE='#CE8C45';GRAY='#C7CDD3';INK='#263442'
FAMILIES=sorted(pd.read_csv(DATA/'stage_c_family_catalogue.csv')['family'])
CMAP=LinearSegmentedColormap.from_list('support',['#F6F8FA','#B6D7D8','#247C83'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
 'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
 'legend.frameon':False,'legend.fontsize':7,'figure.facecolor':'white','savefig.facecolor':'white'})
MANIFEST=[]
def load(n):return pd.read_csv(DATA/n)

def save(fig,name,claim):
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}_preview.png',dpi=300,bbox_inches='tight',pad_inches=.08)
    MANIFEST.append({'figure':name,'conclusion':claim,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),'formats':['svg','pdf','png'],'dpi':600})
    fig.savefig(OUT/f'{name}.tiff',dpi=600,bbox_inches='tight',pad_inches=.08,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def panel(ax,label,title):
    ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-14,10),textcoords='offset points',weight='bold',fontsize=10,va='bottom')
    ax.set_title(title,loc='left',pad=10)

def pct(ax,axis='x'):
    if axis=='x':ax.set_xlim(0,1.04);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    else:ax.set_ylim(-.025,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))


def supp5():
    summ=load('exp4_method_summary.csv').set_index('method').reindex(METHODS);fam=load('exp4_family_summary.csv');fig=plt.figure(figsize=(7.1,7.0),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.2])
    ax=fig.add_subplot(gs[0,0]);x=np.arange(6);features=[('contact_sig','接触密度'),('sasa_sig','相对 SASA'),('longrange_sig','长程接触数')]
    endpoint=[]
    for j,(col,label) in enumerate(features):
        v=summ[col]/summ.testable_motifs;ax.bar(x+(j-1)*.24,v,width=.22,color=[TEAL,'#8BA6B8',ORANGE][j],label=label)
        for m in METHODS:endpoint.append({'method':m,'endpoint':col,'significant':int(summ.loc[m,col]),'testable':int(summ.loc[m,'testable_motifs'])})
    pd.DataFrame(endpoint).to_csv(DATA/'supp5_endpoint_counts.csv',index=False,encoding='utf-8-sig')
    ax.set_xticks(x,METHODS,rotation=40,ha='right');pct(ax,'y');ax.set_ylim(0,1.28);ax.set_yticks(np.arange(0,1.01,.2));ax.set_ylabel('显著候选 / 可检验候选');ax.legend(loc='upper right',fontsize=6);panel(ax,'a','三个预设结构终点')
    ax=fig.add_subplot(gs[0,1]);
    for j,(sig,direction,label) in enumerate([('contact_sig','positive_contact','接触密度增加'),('sasa_sig','negative_sasa','相对 SASA 降低'),('longrange_sig','positive_longrange','长程接触数增加')]):
        v=summ[direction]/summ[sig];ax.plot(x,v,marker=['o','s','^'][j],ms=3,lw=.8,color=[TEAL,'#8BA6B8',ORANGE][j],label=label)
    ax.set_xticks(x,METHODS,rotation=40,ha='right');pct(ax,'y');ax.set_ylabel('指定方向 / 该终点显著候选');ax.legend(loc='lower left',fontsize=6);panel(ax,'b','显著效应的方向')
    ax.text(4, .06, 'NA', ha='center', fontsize=6, color=INK)
    ax=fig.add_subplot(gs[1,:]);mat=fam.pivot(index='family',columns='method',values='strict').reindex(index=FAMILIES,columns=METHODS)
    den=load('Table_S14_family_status.csv').pivot(index='family',columns='method',values='input').reindex_like(mat)
    rat=mat/den
    im=ax.imshow(rat.T,cmap=CMAP,vmin=0,vmax=1,aspect='auto')
    for i,m in enumerate(METHODS):
        for j,f in enumerate(mat.index):
            v=rat.loc[f,m];txt='NA' if pd.isna(v) else f'{int(mat.loc[f,m])}/\n{int(den.loc[f,m])}'
            ax.text(j,i,txt,ha='center',va='center',fontsize=5.7,color='white' if v>.65 else INK)
    ax.set_xticks(range(len(FAMILIES)),mat.index,rotation=45,ha='right');ax.set_yticks(range(6),METHODS);panel(ax,'c','家族分层的严格支持比例（分子 / 分母）');fig.colorbar(im,ax=ax,shrink=.65,label='严格支持比例',fraction=.025)
    save(fig,'Figure_S5_structural_endpoints','各预设结构终点的显著性、效应方向和家族构成共同界定结构支持。')


def supp6():
    d=load('exp5_gvpminer_motif_states.csv')
    r=load('supp6_family_annotation.csv')
    fig=plt.figure(figsize=(7.1,6.6),layout='constrained')
    gs=fig.add_gridspec(3,1,height_ratios=[1,1,1.1])
    for k,(num,den,title) in enumerate([('with_output','input','富集输出覆盖（有输出 / 输入）'),('supported','with_output','注释支持（获支持 / 有输出）')]):
        ax=fig.add_subplot(gs[k])
        n=r.pivot(index='method',columns='family',values=num).reindex(index=METHODS,columns=FAMILIES)
        N=r.pivot(index='method',columns='family',values=den).reindex_like(n)
        v=n/N
        ax.imshow(v,cmap=CMAP,vmin=0,vmax=1,aspect='auto')
        for i in range(6):
            for j in range(len(FAMILIES)):
                label='NA' if pd.isna(N.iloc[i,j]) or not N.iloc[i,j] else f'{int(n.iloc[i,j])}/\n{int(N.iloc[i,j])}'
                ax.text(j,i,label,ha='center',va='center',fontsize=5.7,color='white' if v.iloc[i,j]>.65 else INK)
        ax.set_xticks(range(len(FAMILIES)),FAMILIES,rotation=40,ha='right',fontsize=6)
        ax.set_yticks(range(6),METHODS,fontsize=6)
        panel(ax,chr(97+k),title)
    # This panel describes the same GvpMiner catalogue as in the existing draft.
    ax=fig.add_subplot(gs[2])
    universe=load('Table_S14_candidate_status.csv')
    ids=set(universe.loc[(universe.method=='GvpMiner')&~universe.annotation_has_output,'motif_uid'])
    d=d[d.motif_cluster.isin(ids)]
    assert len(d)==239
    reasons=[('unevaluable_insufficient_annotated_foreground','前景注释蛋白不足'),('unevaluable_no_eligible_gene_sets','无合格基因集'),('unevaluable_no_family_annotations','家族无注释')]
    left=np.zeros(len(FAMILIES))
    reason_counts={}
    for (status,label),co in zip(reasons,[TEAL,'#A7BBC8',ORANGE]):
        vals=d[d.status==status].groupby('family').size().reindex(FAMILIES,fill_value=0)
        reason_counts[status]=int(sum(vals))
        ax.bar(range(len(FAMILIES)),vals,bottom=left,color=co,label=f'{label}（{sum(vals)}）')
        left+=vals.to_numpy()
    assert sum(reason_counts.values())==239
    ax.set_xticks(range(len(FAMILIES)),FAMILIES,rotation=40,ha='right',fontsize=6)
    ax.set_ylabel('无富集输出候选数')
    ax.legend(fontsize=6,ncol=1,loc='upper left');ax.margins(y=.25)
    panel(ax,'c','GvpMiner 无输出候选的历史注释覆盖状态')
    save(fig,'Figure_S6_annotation_coverage','输出覆盖和条件支持均按家族保留；无输出候选另列覆盖状态。')

if __name__=='__main__':
    supp5();supp6()
    (BASE/'qa/supplement_figure_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
