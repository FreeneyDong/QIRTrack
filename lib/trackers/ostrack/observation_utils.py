from pathlib import Path
import os,json,hashlib,time,math,re,random,copy
import numpy as np
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def valid(b):
    return bool(np.isfinite(b).all() and np.asarray(b)[2] > 0 and (np.asarray(b)[3] > 0))

def quantize(a):
    a = np.asarray(a, np.float32)
    scale = np.maximum(np.max(np.abs(a), axis=0) / 127, 1e-08).astype(np.float32)
    q = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
    return (q, scale)
