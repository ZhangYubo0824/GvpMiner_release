#!/usr/bin/env python3
"""
数据预处理 (Exp0)
==================
从 data3/ncbi 加载所有 Gvp 家族数据 → 质控过滤 → 去冗余 → 输出统一数据集。

用法:
    python data_preprocessing.py
    python data_preprocessing.py --skip-cdhit   # 跳过CD-HIT (直接用序列去重)
    python data_preprocessing.py --families GvpA GvpJ GvpN  # 仅处理指定家族

输出:
    results/preprocessed/
        cleaned_gvp_dataset.csv     — 全量清洗后数据
        family_statistics.csv       — 各家族统计摘要
        filtered_sequences.fasta    — 去冗余后的FASTA
"""

import sys
import csv
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import Counter, defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    NCBI_DIR, PREPROCESSED_DIR, GVP_FAMILIES,
    MIN_SEQ_LENGTH, MAX_SEQ_LENGTH, CDHIT_THRESHOLD, MIN_FAMILY_SIZE,
    STANDARD_AA, INVALID_AA_CHARS,
)


def load_ncbi_metadata(families=None):
    """从 data3/ncbi 各家族目录读取 _metadata.csv"""
    all_families = families if families else GVP_FAMILIES
    all_records = []
    load_stats = {}

    for family_name in all_families:
        family_dir = NCBI_DIR / family_name
        if not family_dir.exists():
            print(f"  [跳过] {family_name}: 目录不存在")
            continue

        csv_files = list(family_dir.glob("*_metadata.csv"))
        if not csv_files:
            print(f"  [跳过] {family_name}: 无 metadata.csv")
            continue

        csv_path = csv_files[0]
        count = 0
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["family"] = family_name  # 确保家族标签正确
                    all_records.append(row)
                    count += 1
        except Exception as e:
            print(f"  [错误] {family_name}: {e}")
            continue

        load_stats[family_name] = count
        print(f"  [加载] {family_name}: {count} 条记录")

    print(f"\n总计加载: {len(all_records)} 条记录, 来自 {len(load_stats)} 个家族")
    return all_records, load_stats


def quality_filter(records):
    """序列质量过滤"""
    filtered = []
    stats = Counter()

    for rec in records:
        seq = rec.get("sequence", "").strip().upper()

        # 过滤空序列
        if not seq:
            stats["empty_sequence"] += 1
            continue

        # 过滤过短/过长
        if len(seq) < MIN_SEQ_LENGTH:
            stats["too_short"] += 1
            continue
        if len(seq) > MAX_SEQ_LENGTH:
            stats["too_long"] += 1
            continue

        # 过滤含非标准氨基酸的序列
        invalid_count = sum(1 for c in seq if c in INVALID_AA_CHARS)
        if invalid_count / len(seq) > 0.05:  # 超过5%非标准字符
            stats["too_many_invalid_aa"] += 1
            continue

        # 清理非标准字符 (少量的直接替换为X不影响)
        seq_clean = "".join(c if c in STANDARD_AA else "" for c in seq)
        if len(seq_clean) < MIN_SEQ_LENGTH:
            stats["clean_too_short"] += 1
            continue

        rec["sequence"] = seq_clean
        rec["sequence_length"] = str(len(seq_clean))
        filtered.append(rec)

    stats["passed"] = len(filtered)
    print("\n质量过滤结果:")
    for reason, count in sorted(stats.items()):
        print(f"  {reason}: {count}")

    return filtered, dict(stats)


def deduplicate_sequences(records):
    """基于序列精确去重 (同一家族内相同序列只保留1条)"""
    seen = set()
    deduped = []

    for rec in records:
        key = (rec["family"], rec["sequence"])
        if key not in seen:
            seen.add(key)
            deduped.append(rec)

    removed = len(records) - len(deduped)
    print(f"\n精确去重: {len(records)} -> {len(deduped)} (去除 {removed} 条)")
    return deduped


def run_cdhit(records, threshold=0.90):
    """
    使用 CD-HIT 去冗余 (如果已安装)。
    如果 CD-HIT 不可用，回退为精确去重。
    """
    # 检查 CD-HIT 是否可用
    try:
        subprocess.run(["cd-hit", "--help"], capture_output=True, timeout=5)
        cdhit_available = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        cdhit_available = False

    if not cdhit_available:
        print("\n[警告] CD-HIT 未安装, 使用精确去重代替")
        print("  安装 CD-HIT: conda install -c bioconda cd-hit")
        return deduplicate_sequences(records)

    print(f"\n运行 CD-HIT (阈值={threshold})...")
    deduped = []

    # 按家族分组运行
    family_groups = defaultdict(list)
    for rec in records:
        family_groups[rec["family"]].append(rec)

    for family, recs in family_groups.items():
        # 写入临时FASTA
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_in:
            for i, rec in enumerate(recs):
                tmp_in.write(f">{rec.get('protein_id', f'seq_{i}')}\n")
                tmp_in.write(f"{rec['sequence']}\n")
            tmp_input = tmp_in.name

        tmp_output = tmp_input + ".out"

        # 运行 CD-HIT
        cmd = [
            "cd-hit", "-i", tmp_input, "-o", tmp_output,
            "-c", str(threshold), "-n", "5",
            "-M", "2000", "-T", "4", "-d", "0",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=300)

            # 读取结果: 保留的序列ID
            kept_ids = set()
            with open(tmp_output) as f:
                for line in f:
                    if line.startswith(">"):
                        kept_ids.add(line.strip().lstrip(">").split()[0])

            for rec in recs:
                pid = rec.get("protein_id", "")
                if pid in kept_ids:
                    deduped.append(rec)

            print(f"  {family}: {len(recs)} -> {len(kept_ids)}")

        except Exception as e:
            print(f"  [警告] {family} CD-HIT 失败 ({e}), 使用精确去重")
            seen = set()
            for rec in recs:
                if rec["sequence"] not in seen:
                    seen.add(rec["sequence"])
                    deduped.append(rec)

        # 清理临时文件
        for suffix in ["", ".clstr"]:
            try:
                Path(tmp_output + suffix).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            Path(tmp_input).unlink(missing_ok=True)
        except Exception:
            pass

    print(f"\nCD-HIT 去冗余完成: {len(records)} -> {len(deduped)}")
    return deduped


def filter_small_families(records, min_size=None):
    """过滤样本过少的家族"""
    if min_size is None:
        min_size = MIN_FAMILY_SIZE

    family_counts = Counter(rec["family"] for rec in records)
    small_families = {f for f, c in family_counts.items() if c < min_size}

    if small_families:
        print(f"\n以下家族样本过少 (<{min_size}), 标记但保留:")
        for f in sorted(small_families):
            print(f"  {f}: {family_counts[f]} 条")

    return records, small_families


def save_outputs(records, small_families, load_stats, filter_stats):
    """保存预处理结果"""
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "protein_id", "protein_name", "description", "organism", "taxonomy",
        "sequence_length", "sequence", "accessions", "db_source", "date",
        "family", "domains",
    ]

    # 1. 保存合并数据集 CSV (实验流水线输入)
    csv_path = PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\n已保存合并数据: {csv_path} ({len(records)} 条)")

    # 2. 按家族分别保存 CSV (检查/调试/论文附表)
    by_family_dir = PREPROCESSED_DIR / "by_family"
    by_family_dir.mkdir(parents=True, exist_ok=True)

    family_groups = defaultdict(list)
    for rec in records:
        family_groups[rec["family"]].append(rec)

    for family, recs in sorted(family_groups.items()):
        # 分家族 CSV
        family_csv = by_family_dir / f"{family}_cleaned.csv"
        with open(family_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(recs)
        # 分家族 FASTA
        family_fasta = by_family_dir / f"{family}_cleaned.fasta"
        with open(family_fasta, "w") as f:
            for rec in recs:
                pid = rec.get("protein_id", "unknown")
                org = rec.get("organism", "").replace(" ", "_")
                f.write(f">{pid}|{family}|{org}\n")
                f.write(f"{rec['sequence']}\n")
    print(f"已保存分家族数据: {by_family_dir}/ ({len(family_groups)} 个家族, CSV+FASTA)")

    # 3. 保存合并 FASTA
    fasta_path = PREPROCESSED_DIR / "filtered_sequences.fasta"
    with open(fasta_path, "w") as f:
        for rec in records:
            pid = rec.get("protein_id", "unknown")
            family = rec.get("family", "unknown")
            f.write(f">{pid}|{family}\n")
            f.write(f"{rec['sequence']}\n")
    print(f"已保存 FASTA: {fasta_path}")

    # 4. 保存统计摘要
    stats_path = PREPROCESSED_DIR / "family_statistics.csv"
    family_counts = Counter(rec["family"] for rec in records)
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "count", "is_small", "original_count"])
        writer.writeheader()
        for family in sorted(family_counts.keys()):
            writer.writerow({
                "family": family,
                "count": family_counts[family],
                "is_small": family in small_families,
                "original_count": load_stats.get(family, 0),
            })
    print(f"已保存统计: {stats_path}")

    # 5. 打印最终统计
    print(f"\n{'='*50}")
    print("最终数据集统计:")
    print(f"{'='*50}")
    print(f"  总序列数: {len(records)}")
    print(f"  家族数: {len(family_counts)}")
    print(f"  序列长度: {min(int(r['sequence_length']) for r in records)}"
          f" - {max(int(r['sequence_length']) for r in records)} aa")
    print("\n各家族分布:")
    for family in sorted(family_counts.keys()):
        marker = " ⚠" if family in small_families else ""
        print(f"  {family:20s}: {family_counts[family]:6d}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Gvp数据预处理")
    parser.add_argument("--families", nargs="+", default=None,
                        help="指定处理的家族, 默认全部")
    parser.add_argument("--skip-cdhit", action="store_true",
                        help="跳过CD-HIT, 仅做精确去重")
    parser.add_argument("--cdhit-threshold", type=float, default=CDHIT_THRESHOLD,
                        help=f"CD-HIT阈值 (默认{CDHIT_THRESHOLD})")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: 数据加载")
    print("=" * 60)
    records, load_stats = load_ncbi_metadata(args.families)
    if not records:
        print("错误: 未加载到任何数据")
        return

    print("\n" + "=" * 60)
    print("Step 2: 质量过滤")
    print("=" * 60)
    records, filter_stats = quality_filter(records)

    print("\n" + "=" * 60)
    print("Step 3: 序列去冗余")
    print("=" * 60)
    if args.skip_cdhit:
        records = deduplicate_sequences(records)
    else:
        records = run_cdhit(records, threshold=args.cdhit_threshold)

    print("\n" + "=" * 60)
    print("Step 4: 家族筛选")
    print("=" * 60)
    records, small_families = filter_small_families(records)

    print("\n" + "=" * 60)
    print("Step 5: 保存结果")
    print("=" * 60)
    save_outputs(records, small_families, load_stats, filter_stats)

    print("\n数据预处理完成! ✓")


if __name__ == "__main__":
    main()
