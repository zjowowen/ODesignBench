"""
检查 CSV 中 release_date 在 2021-09-01 之前和之后的数量。
"""
import csv
from datetime import date

CSV_PATH = "/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/indices/pdb_reso2_fixed_clean.csv"
CUTOFF = date(2021, 9, 1)

before = 0
after = 0
missing = 0

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw = row.get("release_date", "").strip()
        if not raw:
            missing += 1
            continue
        try:
            d = date.fromisoformat(raw)
        except ValueError:
            missing += 1
            continue
        if d < CUTOFF:
            before += 1
        else:
            after += 1

total = before + after + missing
print(f"截止日期：{CUTOFF}（不含当天算 before）")
print(f"总行数   : {total}")
print(f"Before   : {before}  ({before/total*100:.1f}%)")
print(f"After    : {after}   ({after/total*100:.1f}%)")
if missing:
    print(f"缺失/无效: {missing}")
