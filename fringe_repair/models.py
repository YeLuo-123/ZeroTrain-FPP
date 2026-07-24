import torch
from torch import nn


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(min(8, b), b),
                         nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(min(8, b), b), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, base=32):
        super().__init__()
        self.e1, self.e2, self.e3 = block(in_ch, base), block(base, base*2), block(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.d2, self.d1 = block(base*6, base*2), block(base*3, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); z = self.e3(self.pool(e2))
        z = self.d2(torch.cat([nn.functional.interpolate(z, e2.shape[-2:], mode="bilinear"), e2], 1))
        z = self.d1(torch.cat([nn.functional.interpolate(z, e1.shape[-2:], mode="bilinear"), e1], 1))
        return self.out(z)


class FringeToPhase(nn.Module):
    """Predict sin/cos to avoid the wrapped-phase branch cut."""
    def __init__(self, base=32):
        super().__init__(); self.net = UNet(4, 2, base)
    def forward(self, x):
        sc = self.net(x)
        phase = torch.atan2(sc[:, 0], sc[:, 1])
        return {"phase": phase, "sin_cos": sc}


class PhysicsPhaseNet(nn.Module):
    """Shared encoder with fringe reconstruction and phase heads."""
    def __init__(self, base=32):
        super().__init__(); self.backbone = UNet(4, 6, base)
    def forward(self, x):
        y = self.backbone(x)
        fringe = torch.sigmoid(y[:, :4])
        phase = torch.atan2(y[:, 4], y[:, 5])
        return {"fringe": fringe, "phase": phase, "sin_cos": y[:, 4:6]}


class PatchDiscriminator(nn.Module):
    def __init__(self, ch=8, base=32):
        super().__init__()
        layers = []
        for a, b, s in [(ch,base,2),(base,base*2,2),(base*2,base*4,2),(base*4,1,1)]:
            layers += [nn.Conv2d(a,b,4,stride=s,padding=1), nn.LeakyReLU(.2)] if b != 1 else [nn.Conv2d(a,b,4,stride=s,padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, damaged, restored): return self.net(torch.cat([damaged, restored], 1))


def create_model(name, base=32):
    if name == "unet": return UNet(4, 4, base)
    if name == "phase": return FringeToPhase(base)
    if name in ("physics", "gan"): return PhysicsPhaseNet(base)
    raise ValueError(name)
