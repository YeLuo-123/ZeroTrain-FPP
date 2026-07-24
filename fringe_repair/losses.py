import torch
import torch.nn.functional as F
from .physics import wrap_phase


def gradient_loss(pred, target, mask, valid):
    total = pred.new_tensor(0.)
    for dim in (-1, -2):
        dp, dt = torch.diff(pred, dim=dim), torch.diff(target, dim=dim)
        m = torch.diff(mask.float(), dim=dim).abs().clamp(0, 1)
        v = torch.diff(valid.float(), dim=dim).abs().logical_not()
        total += (wrap_phase(dp-dt).abs() * (m + .1) * v).sum() / ((m + .1)*v).sum().clamp_min(1)
    return total


def physics_losses(out, batch, weights):
    phase, gt, valid = out["phase"], batch["phase"], batch["valid"]
    mask = batch["mask"].amax(1)
    circular = (1 - torch.cos(phase - gt))
    l_phase = (circular * valid).sum() / valid.sum().clamp_min(1)
    deltas = torch.arange(4, device=phase.device) * (torch.pi/2)
    target = batch["clean"]
    a = target.mean(1, keepdim=True)
    b = ((target-a).square().mean(1, keepdim=True)*2).sqrt()
    reproj = a + b * torch.cos(phase[:,None] + deltas[None,:,None,None])
    l_shift = F.l1_loss(reproj * valid[:,None], target * valid[:,None])
    l_grad = gradient_loss(phase, gt, mask, valid)
    # Dataset-specific calibrated phase-to-depth mapping can replace this normalized relation.
    def norm(z):
        lo=z.flatten(1).amin(1)[:,None,None]; hi=z.flatten(1).amax(1)[:,None,None]
        return (z-lo)/(hi-lo).clamp_min(1e-6)
    l_depth = F.smooth_l1_loss(norm(phase)*valid, norm(batch["depth"])*valid)
    l_fringe = F.l1_loss(out.get("fringe", reproj), target)
    terms = {"phase":l_phase,"shift":l_shift,"gradient":l_grad,"depth":l_depth,"fringe":l_fringe}
    return sum(weights.get(k, 0)*v for k,v in terms.items()), terms
