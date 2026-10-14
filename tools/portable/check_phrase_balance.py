"""Verify scalar dynamics, silence bypass, headroom and dynamic phrase lengths."""
import sys
import numpy as np
import onnxruntime as ort
s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider'])
rng=np.random.default_rng(52);checks=0
for frames in [1,2,9,32,261,340]:
    for level in [0,.001,.02,.06,.1,.18,.5]:
        x=rng.normal(0,level,(1,frames*512)).astype(np.float32)
        for voiced in [0,1]:
            mask=np.full((1,frames),voiced,np.float32)
            y=s.run(None,{'wave':x,'mask':mask})[0]
            count=mask.sum()*512
            rms=np.sqrt((x*x).sum()/max(count,1)+1e-10) if voiced else 1e-5
            gain=np.clip((.1/rms)**.7,.7079458,2.) if rms>.004 and count>=1024 else 1.
            gain=min(gain,.95/max(np.abs(x).max(),1e-6))
            assert np.allclose(y,x*gain,rtol=3e-5,atol=2e-7)
            assert y.shape==x.shape and np.isfinite(y).all() and abs(y).max()<=.95001
            if (not voiced or level<.003) and abs(x).max()<=.95:assert np.array_equal(x,y)
            checks+=1
print(f'{checks} cases passed')
