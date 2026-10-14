"""
generate_sample_csv.py — standalone script to produce sample_power_quality.csv
Run once: python generate_sample_csv.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from modules.data_loader import get_sample_csv

output = pathlib.Path(__file__).parent / "sample_power_quality.csv"
output.write_text(get_sample_csv(), encoding="utf-8")
print(f"Written: {output}")
