#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from fringe_repair.dataset import FringeDataset
from fringe_repair.models import create_model, PatchDiscriminator
from fringe_repair.losses import physics_losses

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--model"); p.add_argument("--epochs",type=int); p.add_argument("--ablate",choices=["physics","phase","geometry"]); a=p.parse_args()
c=yaml.safe_load(open(a.config)); seed=c["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
dev="cuda" if c["train"]["device"]=="auto" and torch.cuda.is_available() else ("cpu" if c["train"]["device"]=="auto" else c["train"]["device"])
name=a.model or c["model"]["name"]; model=create_model(name,c["model"]["base_channels"]).to(dev)
ds=FringeDataset(c["data"]["root"],"train",c["data"]["damage_ratio"],c["data"]["crop_size"],seed)
dl=DataLoader(ds,batch_size=c["train"]["batch_size"],shuffle=True,num_workers=c["data"]["workers"])
opt=torch.optim.AdamW(model.parameters(),lr=c["train"]["lr"]); weights=c["loss"].copy()
disc=PatchDiscriminator().to(dev) if name=="gan" else None
opt_d=torch.optim.Adam(disc.parameters(),lr=c["train"]["lr"],betas=(.5,.999)) if disc else None
if a.ablate=="physics": weights["shift"]=0
if a.ablate=="phase": weights["phase"]=0
if a.ablate=="geometry": weights["depth"]=0
out=Path(c["train"]["output"]+("" if not a.ablate else "_no_"+a.ablate)); out.mkdir(parents=True,exist_ok=True)
epochs=a.epochs or c["train"]["epochs"]; history=[]
for epoch in range(epochs):
    model.train(); sums={}
    for b in dl:
        b={k:v.to(dev) for k,v in b.items()}; opt.zero_grad(set_to_none=True)
        pred=model(b["damaged"])
        if isinstance(pred,torch.Tensor):
            pred={"fringe":torch.sigmoid(pred),"phase":torch.atan2(pred[:,3]-pred[:,1],pred[:,0]-pred[:,2])}
        loss,terms=physics_losses(pred,b,weights)
        if disc:
            # Pix2Pix: conditional PatchGAN plus strongly weighted paired L1/physics terms.
            opt_d.zero_grad(set_to_none=True)
            real=disc(b["damaged"],b["clean"]); fake=disc(b["damaged"],pred["fringe"].detach())
            dloss=.5*(torch.nn.functional.binary_cross_entropy_with_logits(real,torch.ones_like(real))+
                      torch.nn.functional.binary_cross_entropy_with_logits(fake,torch.zeros_like(fake)))
            dloss.backward(); opt_d.step()
            adv=torch.nn.functional.binary_cross_entropy_with_logits(
                disc(b["damaged"],pred["fringe"]),torch.ones_like(fake))
            loss=loss+.01*adv
            terms.update(adversarial=adv,discriminator=dloss)
        loss.backward(); opt.step()
        for k,v in {"total":loss,**terms}.items(): sums[k]=sums.get(k,0)+float(v.detach())
    row={"epoch":epoch+1,**{k:v/len(dl) for k,v in sums.items()}}; history.append(row); print(json.dumps(row))
    state={"model":model.state_dict(),"config":c,"epoch":epoch+1}
    if disc: state["discriminator"]=disc.state_dict()
    torch.save(state,out/"last.pt")
(out/"history.json").write_text(json.dumps(history,indent=2))
