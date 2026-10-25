"""Check exported leveling against Torch, including untouched articulation and bypasses."""
import sys
import numpy as np
import onnxruntime as ort
import torch
from sustain_envelope import SustainEnvelope
s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider'])
cases=0
for frames in [24,36,80,200]:
    for hz in [146.83,293.66,493.88,880]:
        for vowel in [0,1]:
            for swell in [False,True]:
                t=np.arange(frames*512)/44100
                envelope=np.ones(len(t))*.08
                if swell:envelope*=1+.6*np.sin(np.pi*np.arange(len(t))/len(t))**2
                wave=(envelope*np.sin(2*np.pi*hz*t)).astype('float32')[None]
                control=np.zeros((frames,128),'float32');control[:,2]=frames;control[:,3]=hz;control[:,6]=vowel
                y=s.run(None,dict(wave=wave,control=control))[0]
                expected=SustainEnvelope()(torch.from_numpy(wave),torch.from_numpy(control)).numpy()
                assert np.max(np.abs(y-expected))<1e-5
                assert np.isfinite(y).all() and y.shape==wave.shape
                assert np.array_equal(y[:,:2646],wave[:,:2646])
                assert np.array_equal(y[:,-2646:],wave[:,-2646:])
                if not vowel or frames<36:assert np.array_equal(y,wave)
                if swell and vowel and frames>=80:
                    def spread(x):
                        p=x[0].reshape(-1,512).astype('float64')**2
                        rms=np.sqrt(np.convolve(p.mean(1),np.ones(11)/11,'same'))
                        return np.ptp(20*np.log10(rms[13:-13]))
                    assert spread(y)<spread(wave)*.5
                cases+=1
print(f'{cases} cases passed')
from sustain_envelope import SustainSource
source=ort.InferenceSession(str(__import__('pathlib').Path(sys.argv[1]).with_name('source-layer.onnx')),providers=['CPUExecutionProvider'])
source_cases=0
for duration in [12,35,36,100]:
    for pitch in [48,55,56,56.5,57,62,69,72,72.5,73,80]:
        for vowel in [0,1]:
            for previous in [0,.5,1]:
                c=np.zeros((duration,128),'float32');c[:,2]=duration;c[:,3]=440*2**((pitch-69)/12);c[:,6]=vowel
                prior=np.full((1,duration),previous,'float32')
                y=source.run(None,dict(previous=prior,control=c))[0]
                expected=SustainSource()(torch.from_numpy(prior),torch.from_numpy(c)).numpy()
                assert np.max(np.abs(y-expected))<1e-5
                assert np.all(y>=prior) and np.all(y<=1)
                assert np.ptp(y)<1e-6
                if duration<36 or not vowel or pitch<=56 or pitch>=73:assert np.allclose(y,prior,atol=1e-6)
                source_cases+=1
print(f'{source_cases} source conditions passed')
