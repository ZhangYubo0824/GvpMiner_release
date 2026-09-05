import csv
import json
from pathlib import Path

def main():
    base_dir = Path(r"d:\研究方向\Task1\data3\ncbi")
    output_file = Path(r"d:\研究方向\Task1\data3\ncbi_ids_by_family.json")
    
    family_ids = {}
    total_ids = 0
    
    for family_dir in base_dir.iterdir():
        if family_dir.is_dir():
            family = family_dir.name
            csv_file = family_dir / f"{family}_metadata.csv"
            
            ids = set()
            if csv_file.exists():
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pid = row.get("protein_id", "").strip()
                        if pid:
                            ids.add(pid)
            
            # 另外也可以检查 fasta 文件如果 csv 不存在或者不全
            fasta_file = family_dir / f"{family}_proteins.fasta"
            if fasta_file.exists():
                with open(fasta_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith(">"):
                            pid = line[1:].split()[0]
                            ids.add(pid)
            
            id_list = sorted(list(ids))
            if id_list:
                family_ids[family] = id_list
                total_ids += len(id_list)
                print(f"Family {family}: {len(id_list)} IDs found.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(family_ids, f, indent=4)
        
    print(f"\nTotal families with IDs: {len(family_ids)}")
    print(f"Total unique IDs extracted: {total_ids}")
    print(f"Saved to: {output_file}")

if __name__ == '__main__':
    main()
