#!/usr/bin/env python3
import argparse, json, shutil, urllib.request, zipfile
from pathlib import Path

SOURCES={
 "middlebury":("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/cones-png-2.zip","cones-png-2.zip"),
 "dlslp":("https://zenodo.org/records/10404434/files/GDD_dataset_package.zip","GDD_dataset_package.zip"),
 "sfnet":("https://www.dropbox.com/scl/fo/pb3nlmwpxmmcbn9ma7q05/ABO-rcmSAmSl3XNQn9kjsUs?rlkey=9sejey926zz1n3d7d5shc5g34&dl=1","SynthFringe.zip")}
p=argparse.ArgumentParser(); p.add_argument("dataset",choices=SOURCES); p.add_argument("--root",default="data/raw")
p.add_argument("--extract",action="store_true"); p.add_argument("--force",action="store_true"); a=p.parse_args()
url,name=SOURCES[a.dataset]; dst=Path(a.root)/a.dataset; dst.mkdir(parents=True,exist_ok=True); archive=dst/name
if archive.exists() and not a.force: print("exists:",archive)
else:
    print("downloading",url,"to",archive)
    with urllib.request.urlopen(url) as r, archive.open("wb") as f: shutil.copyfileobj(r,f)
if a.extract:
    if not zipfile.is_zipfile(archive): raise SystemExit(f"{archive} is not a zip (provider may require browser download)")
    with zipfile.ZipFile(archive) as z: z.extractall(dst)
manifest={"dataset":a.dataset,"source":url,"archive":str(archive),"bytes":archive.stat().st_size}
(dst/"download_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
print(json.dumps(manifest,indent=2))
