import time
import numpy as np
from skimage.metrics import structural_similarity
from .physics import wrap_phase


def image_metrics(pred, target):
    mse = float(np.mean((pred-target)**2))
    psnr = float(-10*np.log10(max(mse, 1e-12)))
    ssim = float(np.mean([structural_similarity(target[k], pred[k], data_range=1) for k in range(4)]))
    return {"psnr": psnr, "ssim": ssim}


def phase_metrics(pred, target, valid=None):
    valid = np.ones_like(target, bool) if valid is None else valid.astype(bool)
    e = wrap_phase(pred-target)[valid]
    return {"phase_rmse": float(np.sqrt(np.mean(e*e))),
            "wrapped_phase_mae": float(np.mean(np.abs(e)))}


def depth_metrics(pred, target, valid=None):
    valid = np.isfinite(target) if valid is None else valid.astype(bool) & np.isfinite(target)
    e = pred[valid]-target[valid]
    return {"depth_rmse": float(np.sqrt(np.mean(e*e))),
            "point_mae": float(np.mean(np.abs(e)))}


def chamfer_depth(a, b, valid=None, max_points=20000):
    from scipy.spatial import cKDTree
    yy, xx = np.indices(a.shape)
    valid = np.ones_like(a, bool) if valid is None else valid.astype(bool)
    pa=np.c_[xx[valid],yy[valid],a[valid]]; pb=np.c_[xx[valid],yy[valid],b[valid]]
    if len(pa)>max_points:
        idx=np.linspace(0,len(pa)-1,max_points).astype(int); pa,pb=pa[idx],pb[idx]
    return float(cKDTree(pb).query(pa)[0].mean()+cKDTree(pa).query(pb)[0].mean())/2


def profile_model(model, shape=(1,4,256,256), device="cpu"):
    import torch
    x=torch.zeros(shape,device=device); model=model.to(device).eval()
    with torch.no_grad():
        for _ in range(3): model(x)
        t=time.perf_counter()
        for _ in range(10): model(x)
        if device.startswith("cuda"): torch.cuda.synchronize()
    sec=(time.perf_counter()-t)/10
    out={"parameters":sum(p.numel() for p in model.parameters()),"inference_ms":sec*1000,"fps":1/sec}
    try:
        from fvcore.nn import FlopCountAnalysis
        out["flops"]=int(FlopCountAnalysis(model,x).total())
    except Exception: out["flops"]=None
    return out
