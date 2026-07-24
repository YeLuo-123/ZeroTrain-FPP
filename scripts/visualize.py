#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from fringe_repair.dataset import FringeDataset
p=argparse.ArgumentParser(); p.add_argument("--root",default="data/processed"); p.add_argument("--out",default="results/sample.png"); a=p.parse_args()
b=FringeDataset(a.root,"test",crop_size=None)[0]; fig,ax=plt.subplots(3,4,figsize=(12,8))
for k in range(4):
    ax[0,k].imshow(b["clean"][k],cmap="gray",vmin=0,vmax=1); ax[0,k].set_title(f"clean I{k+1}")
    ax[1,k].imshow(b["damaged"][k],cmap="gray",vmin=0,vmax=1); ax[1,k].set_title(f"damaged I{k+1}")
ax[2,0].imshow(b["mask"].amax(0),cmap="gray"); ax[2,0].set_title("damage mask")
ax[2,1].imshow(b["phase"],cmap="twilight",vmin=-3.1416,vmax=3.1416); ax[2,1].set_title("wrapped phase GT")
ax[2,2].imshow(b["depth"],cmap="viridis"); ax[2,2].set_title("depth GT"); ax[2,3].axis("off")
for x in ax.ravel(): x.axis("off")
fig.tight_layout(); Path(a.out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.out,dpi=160); print(a.out)
