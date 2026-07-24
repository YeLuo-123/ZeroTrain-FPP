import cv2
import numpy as np
from .physics import psp4


def inpaint(fringe, mask, method="telea", radius=3):
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    out = np.empty_like(fringe)
    for k in range(4):
        im=np.clip(fringe[k]*255,0,255).astype(np.uint8)
        out[k]=cv2.inpaint(im,(mask[k]>0).astype(np.uint8)*255,radius,flag)/255.
    return out


def run_classical(damaged, mask, method):
    restored=damaged if method=="psp" else inpaint(damaged,mask,method)
    return restored, psp4(restored)
