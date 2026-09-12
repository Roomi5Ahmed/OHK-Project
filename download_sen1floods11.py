"""Download Sen1Floods11 via GCS HTTPS - fixed path construction."""
import urllib.request
import csv
from pathlib import Path

DATA_DIR = Path("C:/Git/OHK Project/data/sen1floods11")
(DATA_DIR / "s1").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "labels").mkdir(parents=True, exist_ok=True)

GCS = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"

# Parse CSVs
all_s1 = set()
all_labels = set()
for s in ["flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv"]:
    fpath = DATA_DIR / s
    if not fpath.exists():
        continue
    with open(fpath) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                all_s1.add(row[0])
                all_labels.add(row[1])

print("S1 files: %d, Label files: %d" % (len(all_s1), len(all_labels)))

# Download S1
s1_dir = DATA_DIR / "s1"
ok = 0
fail = 0
for f in sorted(all_s1):
    local = s1_dir / f
    if local.exists():
        ok += 1
        continue
    url = GCS + "/S1Hand/" + f
    try:
        urllib.request.urlretrieve(url, local)
        ok += 1
        if ok % 50 == 0:
            print("  %d/%d..." % (ok, len(all_s1)))
    except Exception as e:
        fail += 1
        if fail <= 3:
            print("  FAIL %s: %s" % (f, str(e)[:60]))

print("S1: %d ok, %d fail" % (ok, fail))

# Download labels
label_dir = DATA_DIR / "labels"
ok = 0
fail = 0
for f in sorted(all_labels):
    local = label_dir / f
    if local.exists():
        ok += 1
        continue
    url = GCS + "/LabelHand/" + f
    try:
        urllib.request.urlretrieve(url, local)
        ok += 1
        if ok % 50 == 0:
            print("  %d/%d..." % (ok, len(all_labels)))
    except Exception as e:
        fail += 1
        if fail <= 3:
            print("  FAIL %s: %s" % (f, str(e)[:60]))

print("Labels: %d ok, %d fail" % (ok, fail))
print("Done!")
