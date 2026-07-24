#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np, torch, yaml
from fringe_repair.dataset import FringeDataset
from fringe_repair.models import create_model
from fringe_repair.baselines import run_classical
from fringe_repair.physics import phase_to_depth
from fringe_repair.metrics import image_metrics,phase_metrics,depth_metrics,chamfer_depth,profile_model

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--checkpoint"); p.add_argument("--method",default="all",choices=["all","psp","telea","ns","unet","phase","physics"])
p.add_argument("--out",default="results/metrics.json"); a=p.parse_args(); c=yaml.safe_load(open(a.config))
ds=FringeDataset(c["data"]["root"],"test",c["data"]["damage_ratio"],None,c["seed"]); methods=["psp","telea","ns"] if a.method=="all" else [a.method]
model=None
if a.checkpoint:
    ck=torch.load(a.checkpoint,map_location="cpu"); name=a.method if a.method in ("unet","phase","physics") else ck["config"]["model"]["name"]
    model=create_model(name,c["model"]["base_channels"]); model.load_state_dict(ck["model"]); model.eval(); methods=[name]
rows=[]
for method in methods:
    agg=[]
    for i in range(len(ds)):
        b=ds[i]; damaged,clean,mask=[b[k].numpy() for k in ("damaged","clean","mask")]
        if model:
            with torch.no_grad(): o=model(b["damaged"][None]); o={"fringe":torch.sigmoid(o),"phase":None} if isinstance(o,torch.Tensor) else o
            restored=o.get("fringe",b["damaged"][None])[0].numpy()
            phase=o["phase"][0].numpy() if o.get("phase") is not None else run_classical(restored,mask,"psp")[1]
        else: restored,phase=run_classical(damaged,mask,method)
        gt,depth,valid=[b[k].numpy() for k in ("phase","depth","valid")]
        # Without calibration, phase-derived depth is explicitly a normalized surrogate.
        pd=phase_to_depth(phase); pd=(pd-pd.min())/(pd.max()-pd.min()+1e-8)*(depth.max()-depth.min())+depth.min()
        agg.append({**image_metrics(restored,clean),**phase_metrics(phase,gt,valid),**depth_metrics(pd,depth,valid),"chamfer":chamfer_depth(pd,depth,valid)})
    rows.append({"method":method,**{k:float(np.mean([x[k] for x in agg])) for k in agg[0]}})
if model: rows[0].update(profile_model(model))
dst=Path(a.out); dst.parent.mkdir(parents=True,exist_ok=True); dst.write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
