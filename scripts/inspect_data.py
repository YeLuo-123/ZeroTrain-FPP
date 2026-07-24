#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from PIL import Image
import numpy as np
p=argparse.ArgumentParser(); p.add_argument("root"); a=p.parse_args(); root=Path(a.root)
exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".pgm",".ppm"}
files=[f for f in root.rglob("*") if f.suffix.lower() in exts]
sizes={}; bad=0
for f in files:
    try: sizes[str(Image.open(f).size)]=sizes.get(str(Image.open(f).size),0)+1
    except Exception: bad+=1
names=[f.name.lower() for f in root.rglob("*") if f.is_file()]
out={"root":str(root),"images":len(files),"resolutions":sizes,"unreadable":bad,
     "phase_candidates":sum("phase" in n for n in names),"depth_candidates":sum(any(x in n for x in ("depth","height","disp")) for n in names)}
print(json.dumps(out,ensure_ascii=False,indent=2))
