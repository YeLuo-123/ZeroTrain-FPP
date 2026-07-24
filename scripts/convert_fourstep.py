#!/usr/bin/env python3
"""Convert a four-step public/captured dataset to the benchmark NPZ contract."""
import argparse
from pathlib import Path
import cv2, numpy as np
from fringe_repair.physics import psp4

p=argparse.ArgumentParser()
p.add_argument("--root",required=True); p.add_argument("--glob",default="**/I1.*")
p.add_argument("--names",nargs=4,default=["I1.png","I2.png","I3.png","I4.png"])
p.add_argument("--phase-name",default="phase.npy"); p.add_argument("--depth-name",default="depth.npy")
p.add_argument("--out",default="data/processed/train"); a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True); count=0
for first in sorted(Path(a.root).glob(a.glob)):
    parent=first.parent; ims=[]
    for name in a.names:
        im=cv2.imread(str(parent/name),cv2.IMREAD_GRAYSCALE)
        if im is None: break
        ims.append(im.astype(np.float32)/255)
    if len(ims)!=4: continue
    fringe=np.stack(ims); pp=parent/a.phase_name; dp=parent/a.depth_name
    phase=np.load(pp).astype(np.float32) if pp.exists() else psp4(fringe).astype(np.float32)
    depth=np.load(dp).astype(np.float32) if dp.exists() else phase.copy()
    valid=np.isfinite(depth).astype(np.uint8)
    np.savez_compressed(out/f"sample_{count:06d}.npz",fringe=fringe,phase=phase,depth=np.nan_to_num(depth),valid=valid)
    count+=1
print(f"converted {count} four-step samples")
