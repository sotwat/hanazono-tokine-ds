"""Check healthy-vowel bypass, decaying-vowel support and protected phones."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper


def check(before, after, output):
    rows=[]
    for prefix in ['original_', 'donor_']:
        old,new=wrapper(before,prefix),wrapper(after,prefix)
        wet=wrapper(after,prefix,prefix+'su_wet')
        for token in [1,3,9,14,20,28,24,22]:
            for duration in [30,40,60,86,172,275]:
                frames=duration+32
                raw=np.full((1,frames,128),-4,np.float32)
                raw[0,8:8+duration] += (.025*np.sin(np.arange(duration)/2))[:,None]
                base=dict(tokens=np.array([[1,token,1]],np.int64),durations=np.array([[8,duration,24]],np.int64),f0=np.full((1,frames),293.66476,np.float32))
                for decay in [False,True]:
                    x=raw.copy()
                    if decay:x[:,8+duration//2:8+duration]-=3
                    feed=dict(base,**{prefix+'su_raw':x})
                    a=old.run(None,feed)[0];b=new.run(None,feed)[0]
                    assert np.isfinite(b).all()
                    if not decay:
                        w=wet.run(None,feed)[0]
                        assert np.count_nonzero(w) == 0, (prefix,token,duration,float(abs(w).max()),np.flatnonzero(np.max(abs(w),axis=(0,2))>0).tolist())
                    if token in [1,22] or duration<40:
                        assert np.array_equal(a,b)
                    if decay and token in [3,9,14,20,28] and duration>=60:
                        ix=8+duration*3//4
                        assert np.mean(np.exp(b[:,ix]))>2*np.mean(np.exp(x[:,ix]))
                    rows.append(dict(branch=prefix,token=token,frames=duration,decay=decay,finite=True))
    output.write_text(json.dumps(rows,indent=2));print(len(rows),'conditions passed')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['before','after','output']:p.add_argument(name,type=Path)
    a=p.parse_args();check(a.before,a.after,a.output)
