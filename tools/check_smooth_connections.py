"""Verify boundary morphing and exact preservation elsewhere."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper


def check(baseline,candidate,output):
    vowels=[3,9,14,20,28]
    cases=[(a,b,207,275) for a in vowels for b in vowels]
    cases += [(9,3,30,275),(9,3,207,30),(8,3,90,90),(9,8,90,90),
              (9,1,90,90),(24,3,90,90)]
    rows=[]
    for prefix in ['original_','donor_']:
        before,after=wrapper(baseline,prefix),wrapper(candidate,prefix)
        for a,b,da,db in cases:
            ds=np.array([[12,da,db,16]],np.int64)
            frames,boundary=int(ds.sum()),12+da
            raw=np.random.default_rng(72).normal(-4,1,(1,frames,128)).astype(np.float32)
            feed=dict(tokens=np.array([[1,a,b,1]],np.int64),durations=ds,
                      f0=np.full((1,frames),440,np.float32))
            feed[prefix+'su_raw']=raw
            x,y=before.run(None,feed)[0],after.run(None,feed)[0]
            eligible=a in vowels and b in vowels and min(da,db)>=40
            mask=np.zeros(frames,bool)
            if eligible:
                mask[boundary-7:boundary+8]=True
                t=np.arange(1,16,dtype=np.float32)/16
                w=(t*t*(3-2*t))[None,:,None]
                expected=(1-w)*x[:,boundary-8:boundary-7]+w*x[:,boundary+8:boundary+9]
                np.testing.assert_allclose(y[:,mask],expected,atol=2e-6,rtol=2e-6)
            assert np.array_equal(x[:,~mask],y[:,~mask])
            assert np.isfinite(y).all()
            rows.append(dict(branch=prefix,tokens=[a,b],durations=[da,db],
                             unaffected_exact=True,interpolation_verified=eligible))
    output.write_text(json.dumps(rows,indent=2))
    print(f'{len(rows)} cases passed.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['baseline','candidate','output']:p.add_argument(name,type=Path)
    a=p.parse_args();check(a.baseline,a.candidate,a.output)
