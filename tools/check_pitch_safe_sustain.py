"""Check pitch-dependent hold bypass and protected short/stop/silent phones."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper


def check(before,after,output):
    rows=[];sustained={2,3,9,10,12,14,18,19,20,24,25,28,31}
    for prefix in ['original_','donor_']:
        a,b=wrapper(before,prefix),wrapper(after,prefix)
        for midi in [48,55,60,67,72,77,84]:
            hz=440*2**((midi-69)/12)
            for token in range(1,32):
                for d in [1,12,39,40,54,90,171,172,275]:
                    F=d+16;raw=np.random.default_rng(9).normal(-4,.5,(1,F,128)).astype(np.float32)
                    feed=dict(tokens=np.array([[1,token,1]],np.int64),durations=np.array([[8,d,8]],np.int64),f0=np.full((1,F),hz,np.float32));feed[prefix+'su_raw']=raw
                    x,y=a.run(None,feed)[0],b.run(None,feed)[0];assert np.isfinite(y).all()
                    assert np.array_equal(x[:,:8],y[:,:8]) and np.array_equal(x[:,8+d:],y[:,8+d:])
                    if token not in sustained or d<40 or d>=172:assert np.array_equal(x,y)
                    rows.append(dict(branch=prefix,midi=midi,token=token,duration=d))
        # The strong stationary donor is at the interior plateau; distant F0
        # must use frame-local raw spectra rather than stale donor harmonics.
        d=90;raw=np.full((1,106,128),-5,np.float32);raw[:,48:65]=-2
        f=np.full((1,106),392,np.float32);f[:,28:42]*=2**(4/12)
        feed=dict(tokens=np.array([[1,20,1]],np.int64),durations=np.array([[8,d,8]],np.int64),f0=f);feed[prefix+'su_raw']=raw
        y=b.run(None,feed)[0];assert np.array_equal(y[:,30:40],raw[:,30:40]);assert np.exp(y[:,70:80]).mean()>np.exp(raw[:,70:80]).mean()*5
        rows.append(dict(branch=prefix,pitch_bend_bypass=True,stationary_sustain_retained=True))
    output.write_text(json.dumps(rows,indent=2));print(f'{len(rows)} checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.before,a.after,a.output)
