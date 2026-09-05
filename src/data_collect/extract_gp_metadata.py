#!/usr/bin/env python3
"""
从 data/ncbi 中各 Gvp 家族目录下的 .gp (GenPept) 文件中提取元数据，
生成 Gvp{family}_metadata.csv 格式的文件。

用法:
    python extract_gp_metadata.py                     # 处理所有有 .gp 文件的家族
    python extract_gp_metadata.py --families GvpA GvpN # 仅处理指定家族
    python extract_gp_metadata.py --overwrite          # 覆盖 work 中已存在的 CSV
"""

import csv
import argparse
from pathlib import Path
from io import StringIO

from Bio import SeqIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NCBI_DIR = PROJECT_ROOT / "data" / "ncbi"
OUTPUT_DIR = PROJECT_ROOT / "work" / "results" / "ncbi_metadata"

CSV_FIELDS = [
    "protein_id",
    "protein_name",
    "description",
    "organism",
    "taxonomy",
    "sequence_length",
    "sequence",
    "accessions",
    "db_source",
    "date",
    "family",
    "domains",
]


def parse_gp_file(gp_path, family_name):
    """解析一个 .gp (GenPept/GenBank格式) 文件，返回元数据行列表。"""
    metadata_rows = []
    
    with open(gp_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    record_count = 0
    error_count = 0
    
    for record in SeqIO.parse(StringIO(text), "genbank"):
        try:
            row = {
                "protein_id": record.id,
                "protein_name": record.name,
                "description": record.description,
                "organism": record.annotations.get("organism", ""),
                "taxonomy": "; ".join(record.annotations.get("taxonomy", [])),
                "sequence_length": len(record.seq),
                "sequence": str(record.seq),
                "accessions": record.annotations.get("db_source", "") or "; ".join(record.annotations.get("accessions", [])),
                "db_source": record.annotations.get("db_source", ""),
                "date": record.annotations.get("date", ""),
                "family": family_name,
            }
            
            # 提取 Region 特征 (domains)
            domains = []
            for feat in record.features:
                if feat.type == "Region":
                    note = feat.qualifiers.get("region_name", [""])[0]
                    db_xref = feat.qualifiers.get("db_xref", [""])[0]
                    domains.append(f"{note}|{db_xref}")
            row["domains"] = "; ".join(domains)
            
            metadata_rows.append(row)
            record_count += 1
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  警告: 解析记录时出错: {e}")
    
    print(f"  成功解析 {record_count} 条记录, {error_count} 条出错")
    return metadata_rows


def find_gp_families(ncbi_dir, output_dir):
    """扫描 ncbi 目录，找出所有包含 .gp 文件的家族目录。
    返回 [(family_name, gp_file_path, csv_output_path), ...]
    """
    results = []
    if not ncbi_dir.exists():
        print(f"错误: 目录 {ncbi_dir} 不存在")
        return results
    
    for family_dir in sorted(ncbi_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        
        family_name = family_dir.name
        if "hybrid" in family_name.lower():
            continue
        gp_files = list(family_dir.glob("*.gp"))
        
        if gp_files:
            gp_file = gp_files[0]  # 取第一个 .gp 文件
            csv_path = output_dir / family_name / f"{family_name}_metadata.csv"
            results.append((family_name, gp_file, csv_path))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="从 .gp 文件提取 Gvp 家族元数据到 CSV")
    parser.add_argument("--families", nargs="+", default=None,
                        help="指定处理的家族名称，如 GvpA GvpN (默认处理所有有.gp文件的家族)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的 _metadata.csv 文件")
    parser.add_argument("--ncbi-dir", type=Path, default=NCBI_DIR,
                        help="只读 NCBI 原始数据目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="元数据输出目录（默认位于 work/results）")
    args = parser.parse_args()
    
    all_families = find_gp_families(args.ncbi_dir, args.output_dir)
    
    if not all_families:
        print("未找到任何包含 .gp 文件的家族目录")
        return
    
    print(f"共发现 {len(all_families)} 个含 .gp 文件的家族:")
    for name, gp, csv_path in all_families:
        gp_size_mb = gp.stat().st_size / (1024 * 1024)
        has_csv = "✓" if csv_path.exists() else "✗"
        print(f"  {name}: {gp.name} ({gp_size_mb:.1f} MB) | CSV已存在: {has_csv}")
    
    # 筛选要处理的家族
    if args.families:
        target_names = set(args.families)
        families_to_process = [(n, g, c) for n, g, c in all_families if n in target_names]
        not_found = target_names - {n for n, _, _ in families_to_process}
        if not_found:
            print(f"\n警告: 以下家族未找到 .gp 文件: {', '.join(not_found)}")
    else:
        families_to_process = all_families
    
    if not families_to_process:
        print("没有需要处理的家族")
        return
    
    print(f"\n将处理 {len(families_to_process)} 个家族:")
    
    for family_name, gp_path, csv_path in families_to_process:
        if csv_path.exists() and not args.overwrite:
            print(f"\n[跳过] {family_name}: CSV 已存在 ({csv_path.name})，使用 --overwrite 覆盖")
            continue
        
        print(f"\n[处理] {family_name}: 解析 {gp_path.name} ...")
        rows = parse_gp_file(gp_path, family_name)
        
        if rows:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  ✓ 已保存 {len(rows)} 条记录 -> {csv_path.name}")
        else:
            print("  ✗ 未能解析出任何记录")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
