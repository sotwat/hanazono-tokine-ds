"""Check time maps and exact bypass without relying on diffusion draws."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def check(folder):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    maps = ort.InferenceSession(str(folder/'maps.onnx'), opts, providers=['CPUExecutionProvider'])
    restore = ort.InferenceSession(str(folder/'restore.onnx'), opts, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20)
    cases = []
    for length in (1, 23, 39, 40, 53, 120):
        for token in (3, 9, 14, 20, 28, 16, 1, 2):
            d = np.array([[8, length, 7]], np.int64)
            tokens = np.array([[1, token, 16]], np.int64)
            f = np.full((1, d.sum()), 233., np.float32)
            for moving in (False, True):
                if moving:
                    f[0,8:8+length] = np.linspace(233, 330, length)
                dd, ff, back, eligible = maps.run(None, dict(tokens=tokens,durations=d,f0=f))
                expected = token in (3,9,14,20,28) and length >= 40 and not moving
                assert dd[0,1] == (24 if expected else length)
                assert ff.shape[1] == dd.sum() and back.shape == f.shape
                assert np.all(eligible[0,:8] == 0) and np.all(eligible[0,8+length:] == 0)
                assert np.all(eligible[0,8:8+length] == expected)
                assert back[0,8] == 8 and abs(back[0,8+length-1]-(8+dd[0,1]-1)) < 1e-4
                short = rng.normal(size=(1,int(dd.sum()),128)).astype(np.float32)
                baseline = rng.normal(size=(1,int(d.sum()),128)).astype(np.float32)
                weight = rng.uniform(size=(1,int(d.sum()),1)).astype(np.float32)
                y = restore.run(None, dict(short=short,back=back,eligible=eligible,weight=weight,baseline=baseline))[0]
                inactive = np.broadcast_to(eligible == 0, y.shape)
                assert np.array_equal(y[inactive], baseline[inactive])
                zero = restore.run(None, dict(short=short,back=back,eligible=eligible,weight=np.zeros_like(weight),baseline=baseline))[0]
                assert np.array_equal(zero, baseline)
                assert np.isfinite(y).all()
                cases.append([length,token,moving])
    # Zero-length phones must not introduce NaN or move their neighbours.
    d = np.array([[8,0,53,0,7]],np.int64)
    t = np.array([[1,16,3,14,1]],np.int64)
    result = maps.run(None,dict(tokens=t,durations=d,f0=np.full((1,d.sum()),233.,np.float32)))
    assert np.isfinite(result[2]).all() and result[0].tolist() == [[8,0,24,0,7]]
    data = dict(cases=len(cases),zero_duration=True,zero_weight_bit_exact=True,unaffected_samples_bit_exact=True,boundaries_preserved=True)
    (folder/'checks.json').write_text(json.dumps(data,indent=2))
    print(data)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    check(p.parse_args().folder)
