"""Complete probability-logo atlas; all displayable columns, no member sampling."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
TEAL, ORANGE, GRAY = "#247C83", "#CE8C45", "#C7CDD3"
COLORS = {a: "#464646" for a in AA}
COLORS.update({a:"#3C7FA1" for a in "KRH"})
COLORS.update({a:"#B95C58" for a in "DE"})
COLORS.update({a:"#49866D" for a in "STNQ"})
COLORS.update({a:"#916EAA" for a in "FWY"})
COLORS.update({a:"#A78D41" for a in "C"})
COLORS.update({a:"#929292" for a in "GP"})
plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":8, "svg.fonttype":"none", "pdf.fonttype":42,
                     "axes.spines.top":False, "axes.spines.right":False,
                     "axes.linewidth":.6, "legend.frameon":False})
GLYPHS = {a:TextPath((0,0), a, size=1, prop=FontProperties(family="DejaVu Sans", weight="bold")) for a in AA}
CASE_SITES = {"GvpJ_M2_motif_018":{1024:"D22*",1025:"K23*"},
              "GvpM_M2_motif_002":{13:"H9*",15:"I11*"}}


def draw_logo(ax, frame, sites=None):
    freqs = frame[["p_"+a for a in AA]].to_numpy()
    columns = frame.msa_column_0based.to_numpy()
    for x, values in enumerate(freqs):
        bottom = 0.
        for j in np.argsort(values, kind="stable"):
            height = values[j]
            if height <= 0:
                continue
            glyph = GLYPHS[AA[j]]
            box = glyph.get_extents()
            transform = (Affine2D().translate(-box.x0, -box.y0)
                         .scale(.86/box.width, height/box.height)
                         .translate(x-.43, bottom))
            ax.add_patch(PathPatch(glyph, transform=transform+ax.transData, facecolor=COLORS[AA[j]], linewidth=0))
            bottom += height
        if x and columns[x] != columns[x-1]+1:
            ax.axvline(x-.5, color=ORANGE, linewidth=.75, linestyle=":")
        if sites and columns[x] in sites:
            ax.text(x,1.04,sites[columns[x]],ha="center",va="bottom",fontsize=7,color="#9C4A34")
    ax.set_xlim(-.7, max(len(frame)-.3, 1))
    ax.set_ylim(0, 1.18 if sites else 1.02)
    ax.set_yticks([0,.5,1])
    ax.set_ylabel("Probability")
    ax.set_xticks(range(len(frame)), [str(c+1) for c in columns], rotation=90, fontsize=6)
    ax.set_xlabel("Family MSA column (1-based; dotted dividers mark omitted columns)", fontsize=7)


def candidate_figure(row, frame):
    display = frame[frame.display_column].reset_index(drop=True)
    chunks = max(1, math.ceil(len(display)/40))
    height = 3.5 + 1.55*chunks
    fig = plt.figure(figsize=(7.4, height))
    gs = fig.add_gridspec(chunks+1, 1, height_ratios=[1.1]+[1]*chunks,
                          left=.095, right=.975, bottom=1.05/height, top=1-.9/height,
                          hspace=.85)
    fig.text(.095, 1-.24/height, row.motif_uid, fontsize=12, fontweight="bold")
    fig.text(.095, 1-.44/height,
             f"{row.n_proteins} proteins | {row.n_occurrences} occurrences | "
             f"{row.display_columns} displayed columns | residue mass retained: {row.retained_residue_mass:.1%}",
             fontsize=8)
    warnings = []
    if row.n_proteins < 2:
        warnings.append("ONE PROTEIN: no cross-protein conservation claim")
    if row.retained_residue_mass < .5:
        warnings.append("Displayed columns cover <50% of weighted residue mass")
    fig.text(.095, 1-.66/height, "; ".join(warnings) if warnings else
             "Descriptive profile; probability heights are conditional on canonical residues.",
             fontsize=7, color="#9C4A34" if warnings else "#444444")
    ax = fig.add_subplot(gs[0])
    x = frame.msa_column_0based+1
    ax.fill_between(x, 1-frame.outside_fraction, color=GRAY, label="Interval span", step="mid")
    ax.plot(x, frame.canonical_coverage, color=TEAL, linewidth=1, label="Canonical residue")
    ax.axhline(.5, color=ORANGE, linestyle="--", linewidth=.7, label="Display threshold")
    ax.set_ylim(0,1.06)
    ax.set_xlim(x.iloc[0]-.5, x.iloc[-1]+.5)
    ticks = MaxNLocator(nbins=9, integer=True).tick_values(x.iloc[0], x.iloc[-1])
    ax.set_xticks(ticks[(ticks >= x.iloc[0]) & (ticks <= x.iloc[-1])])
    ax.set_ylabel("Protein-weighted\ncoverage")
    ax.set_xlabel("Family MSA column (1-based; full candidate span)")
    ax.legend(loc="upper right", fontsize=6, ncol=3, bbox_to_anchor=(1,1.28))
    for k in range(chunks):
        ax = fig.add_subplot(gs[k+1])
        if len(display):
            draw_logo(ax, display.iloc[k*40:(k+1)*40], CASE_SITES.get(row.motif_uid))
        else:
            ax.axis("off")
            ax.text(.5,.65,"No column reaches 50% canonical-residue coverage.", ha="center", fontsize=10)
            ax.text(.5,.3,"All occurrences and the complete profile remain in the downloadable files.", ha="center", fontsize=7)
    footnote = ("Each protein has total weight 1, shared across its occurrences. Full profiles retain all columns.\n"
                "Logo columns may be discontinuous; this display is not a contiguous sequence-search model.")
    if row.motif_uid in CASE_SITES:
        footnote += "\n* Site label refers to the literature-linked member, not the consensus residue."
    fig.text(.095,.20/height, footnote,
             fontsize=7, linespacing=1.5)
    return fig


def layout_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    bad = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0-2 or box.y0 < bounds.y0-2 or box.x1 > bounds.x1+2 or box.y1 > bounds.y1+2:
            bad.append(text.get_text())
    footer = fig.texts[-1].get_window_extent(renderer)
    for ax in fig.axes:
        for label in [ax.xaxis.label, *ax.get_xticklabels()]:
            if label.get_visible() and label.get_text() and footer.overlaps(label.get_window_extent(renderer)):
                bad.append("footer_overlap:"+label.get_text())
    return bad


def overview(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.6,3.5))
    axes[0].hist(data.display_columns, bins=np.arange(0,361,10), color=TEAL)
    axes[0].set(xlabel="Displayed columns per candidate", ylabel="Candidate count")
    axes[1].hist(data.retained_residue_mass, bins=np.linspace(0,1,21), color=TEAL)
    axes[1].set(xlabel="Weighted residue mass retained", ylabel="Candidate count", xlim=(0,1))
    axes[2].scatter(data.n_proteins, data.retained_residue_mass, s=12, alpha=.55, color=TEAL, edgecolor="none")
    axes[2].set(xscale="log", xlabel="Member proteins (log scale)", ylabel="Weighted residue mass retained", ylim=(-.03,1.03))
    for ax, label in zip(axes,"abc"):
        ax.text(-.17,1.06,label,transform=ax.transAxes,fontweight="bold",fontsize=12)
    fig.suptitle("All 522 frozen candidates; descriptive representation diagnostics", fontsize=11)
    fig.tight_layout(rect=(0,.02,1,.94), w_pad=2)
    for ext in ["png","svg","pdf","tiff"]:
        fig.savefig(output/f"representation_overview.{ext}", dpi=300)
    plt.close(fig)


def gallery(data, output):
    records = []
    for r in data.itertuples():
        records.append(dict(uid=r.motif_uid, family=r.family, proteins=int(r.n_proteins),
                            columns=int(r.display_columns), retained=round(r.retained_residue_mass*100,2),
                            status=r.representation_status))
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>GvpMiner：522 个候选的序列表示</title>
<style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.6 system-ui;color:#263238}
h1{font-size:28px}input,select{padding:9px;margin-right:12px}article{border-top:1px solid #ddd;padding:15px 0}
img{display:block;max-width:100%;height:auto}a{color:#247c83;margin-right:14px}.note{background:#f2f5f5;padding:16px}
small{color:#555}button{padding:8px 16px;margin:12px}</style>
<h1>GvpMiner：全部 522 个冻结候选的序列表示</h1>
<p class="note">每个候选保留完整成员与坐标。图中字母高度为残基条件概率，非富集显著性。
logo 展示蛋白权重覆盖率 ≥50% 的列，橙色虚线表示省略了中间比对列。
不同图中的家族 MSA 坐标不能跨家族直接比较。单蛋白候选不能证明跨蛋白保守性。</p>
<p><a href="论文适用性评估.md">方法与论文适用性评估</a><a href="candidate_summary.csv">完整汇总表</a>
<a href="all_candidates_atlas.pdf">522 页图册</a><a href="coordinate_consensus.fasta">保留比对坐标的共识摘要</a></p>
<input id="query" placeholder="输入候选编号，例如 GvpJ_M2_motif_018" size="43">
<select id="family"><option value="">全部家族</option></select>
<select id="mode"><option value="">全部候选</option><option value="single">单蛋白</option>
<option value="low">残基保留比例 &lt;50%</option><option value="none">无可展示列</option></select>
<p id="count"></p><div id="cards"></div><button id="more">继续显示</button>
<script>const rows=DATA;let limit=12;
for(const f of [...new Set(rows.map(r=>r.family))])family.add(new Option(f,f));
function render(){const q=query.value.toLowerCase();const a=rows.filter(r=>
r.uid.toLowerCase().includes(q)&&(!family.value||family.value===r.family)&&
(!mode.value||(mode.value==='single'&&r.proteins===1)||(mode.value==='low'&&r.retained<50)||
(mode.value==='none'&&r.columns===0)));
count.textContent=`符合条件 ${a.length} 个；当前展示 ${Math.min(limit,a.length)} 个`;
cards.innerHTML=a.slice(0,limit).map(r=>`<article><h2>${r.uid}</h2>
<small>${r.proteins} 个蛋白；${r.columns} 个显示列；加权残基保留比例 ${r.retained}%</small>
<p><a href="figures/${r.uid}.svg">SVG</a><a href="profiles/${r.uid}.csv.gz">完整位置矩阵</a>
<a href="alignments/${r.uid}.masked.fasta.gz">带坐标掩码的比对</a>
<a href="sequences/${r.uid}.fasta.gz">全部实际片段</a></p>
<img loading="lazy" src="figures/${r.uid}.png" alt="${r.uid} 的概率 logo 与覆盖率"></article>`).join('');
more.hidden=limit>=a.length}
for(const el of [query,family,mode])el.addEventListener('input',()=>{limit=12;render()});
more.onclick=()=>{limit+=12;render()};render();</script></html>""".replace("DATA",json.dumps(records))
    (output/"index.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("work/motif_representation_v1"))
    parser.add_argument("--preview",action="store_true")
    args=parser.parse_args()
    out=args.output
    data=pd.read_csv(out/"candidate_summary.csv")
    (out/"figures").mkdir(exist_ok=True)
    (out/"paper_figures").mkdir(exist_ok=True)
    cases={"GvpJ_M2_motif_018","GvpM_M2_motif_002"}
    selection=cases|{"GvpD_M2_motif_055","GvpD_M2_motif_056","GvpC_M2_motif_012"}
    subset=data[data.motif_uid.isin(selection)] if args.preview else data
    failures=[]
    with PdfPages(out/("preview_atlas.pdf" if args.preview else "all_candidates_atlas.pdf")) as pdf:
        for i,r in enumerate(subset.itertuples(),1):
            frame=pd.read_csv(out/"profiles"/f"{r.motif_uid}.csv.gz")
            fig=candidate_figure(r,frame)
            bad=layout_check(fig)
            if bad: failures.append({"motif_uid":r.motif_uid,"outside_text":bad})
            fig.savefig(out/"figures"/f"{r.motif_uid}.png",dpi=300)
            fig.savefig(out/"figures"/f"{r.motif_uid}.svg")
            pdf.savefig(fig)
            if r.motif_uid in cases:
                for ext in ["pdf","svg","png","tiff"]:
                    fig.savefig(out/"paper_figures"/f"{r.motif_uid}.{ext}",dpi=300)
            plt.close(fig)
            if i%25==0:print(f"{i}/{len(subset)} figures rendered",flush=True)
    (out/("preview_layout_check.json" if args.preview else "layout_check.json")).write_text(
        json.dumps(dict(figures_checked=len(subset),failures=failures),indent=2),encoding="utf-8")
    if not args.preview:
        overview(data,out/"paper_figures")
        gallery(data,out)
    print(f"Finished {len(subset)} figures; layout failures: {len(failures)}",flush=True)


if __name__=="__main__":
    main()
