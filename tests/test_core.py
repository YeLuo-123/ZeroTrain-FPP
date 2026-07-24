import numpy as np
from fringe_repair.physics import synthesize_fringe, psp4, wrap_phase
from fringe_repair.degradation import degrade, DamageConfig

def test_psp_roundtrip():
    phase=np.linspace(-np.pi,np.pi,1024,endpoint=False).reshape(32,32)
    err=wrap_phase(psp4(synthesize_fringe(phase))-phase)
    assert np.max(np.abs(err)) < 1e-5

def test_degradation_is_reproducible():
    x=synthesize_fringe(np.zeros((64,64)))
    a,m=degrade(x,DamageConfig(ratio=.2),7); b,n=degrade(x,DamageConfig(ratio=.2),7)
    assert np.array_equal(a,b) and np.array_equal(m,n) and m.any()
