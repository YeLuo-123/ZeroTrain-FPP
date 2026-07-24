#!/usr/bin/env python3
"""Deterministic smoke dataset; not a substitute for the requested public datasets."""
import argparse
from pathlib import Path
import numpy as np
from fringe_repair.physics import synthesize_fringe

p=argparse.ArgumentParser(); p.add_argument("--out",default="data/processed"); p.add_argument("--samples",type=int,default=24)
p.add_argument("--height",type=int,default=128); p.add_argument("--width",type=int,default=160); a=p.parse_args()
rng=np.random.default_rng(42); root=Path(a.out)
for i in range(a.samples):
    split="train" if i<int(.7*a.samples) else ("val" if i<int(.85*a.samples) else "test")
    y,x=np.mgrid[-1:1:complex(a.height),-1:1:complex(a.width)]
    depth=.3*np.sin((2+i%3)*x)+.2*np.cos(3*y)+.5*np.exp(-((x-rng.uniform(-.4,.4))**2+(y-rng.uniform(-.4,.4))**2)/.15)
    phase=(10*np.pi*x+5*depth).astype(np.float32)
    fringe=np.clip(synthesize_fringe(phase)+rng.normal(0,.01,(4,a.height,a.width)),0,1).astype(np.float32)
    d=root/split; d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/f"synthetic_{i:04d}.npz",fringe=fringe,phase=((phase+np.pi)%(2*np.pi)-np.pi),depth=depth.astype(np.float32),valid=np.ones_like(depth,np.uint8))
print(f"wrote {a.samples} samples to {root}")
