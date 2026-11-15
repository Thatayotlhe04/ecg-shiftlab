"""Download the MIT-BIH Arrhythmia Database (PhysioNet, ODC-By 1.0) into data/mitdb
and build the beat-level cache at data/processed/beats.npz.

    python scripts/fetch_data.py            # download + build
    python scripts/fetch_data.py --no-download   # data/mitdb already populated
"""
import os
import sys

sys.path.insert(0, ".")
import wfdb

from shiftlab import config as C, data as D

RAW = "data/mitdb"
if "--no-download" not in sys.argv:
    os.makedirs(RAW, exist_ok=True)
    wfdb.dl_database("mitdb", RAW, records=C.RECORDS)
d = D.build(RAW, "data/processed/beats.npz")
print(f"{len(d['y'])} beats from {len(set(d['rec']))} records")
