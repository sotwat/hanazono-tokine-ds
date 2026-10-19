"""Check bypass, pitch preservation and phone-boundary isolation of SustainCycles."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def check(layer: Path, output: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(layer), options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20261020)
    frames, hop = 86, 512
    n = frames*hop
    pos = np.arange(n)
    alpha = (pos % hop)/hop
    index = pos//hop
    rows = []
    for hz in (110., 233.08188, 440., 880.):
        for vibrato in (0., .8):
            f = (hz*2**(vibrato*np.sin(np.arange(frames)*hop/44100*2*np.pi*5)/12)).astype(np.float32)[None]
            freq = f[0,index]*(1-alpha)+f[0,np.minimum(index+1,frames-1)]*alpha
            phase = np.cumsum(freq/44100)
            clean = (.1*np.sin(2*np.pi*phase)+.03*np.sin(4*np.pi*phase)).astype(np.float32)[None]
            control = np.zeros((1,frames,128), np.float32)
            control[:,:,2] = frames
            feed = {'wave':clean, 'f0':f, 'control':control}
            assert np.array_equal(session.run(None,feed)[0],clean)
            control[:,:,6] = 1
            y = session.run(None,feed)[0]
            middle = slice(4096,-4096)
            relative = float(np.sqrt(np.mean((y[:,middle]-clean[:,middle])**2)/np.mean(clean[:,middle]**2)))
            assert relative < .02, (hz,vibrato,relative)
            noisy = clean+rng.normal(0,.015,clean.shape).astype(np.float32)
            feed['wave'] = noisy
            y = session.run(None,feed)[0]
            ratio = float(np.mean((y[:,middle]-clean[:,middle])**2)/np.mean((noisy[:,middle]-clean[:,middle])**2))
            assert ratio < .2, (hz,vibrato,ratio)
            assert np.isfinite(y).all()
            rows.append({'f0':hz,'vibrato_semitones':vibrato,'clean_relative_error':relative,'noise_power_ratio':ratio})
    # An active phone must not borrow another phone's samples.
    f = np.full((1,frames),233.08188,np.float32)
    control = np.zeros((1,frames,128),np.float32)
    control[:,:43,2] = 43
    control[:,43:,1] = 43
    control[:,43:,2] = frames
    control[:,:43,6] = 1
    wave = np.zeros((1,n),np.float32)
    wave[:,43*hop:] = rng.normal(0,.1,(1,n-43*hop))
    y = session.run(None,{'wave':wave,'f0':f,'control':control})[0]
    assert np.count_nonzero(y[:,:42*hop]) == 0
    assert np.array_equal(y[:,43*hop:],wave[:,43*hop:])
    output.write_text(json.dumps({'cases':rows,'phone_isolation':True,'zero_weight_bit_exact':True},indent=2))
    print(len(rows),'pitch cases, bypass and phone isolation passed')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('layer',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    check(args.layer,args.output)
