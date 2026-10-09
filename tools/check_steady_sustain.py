"""Check constant donor spectra and preservation of unsupported/short phones."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper


def check(before,after,output):
    rows=[]
    for prefix in ['original_','donor_']:
        a,b=wrapper(before,prefix),wrapper(after,prefix)
        for token,duration in [(3,30),(8,90),(3,48),(3,90),(20,90),(24,90),(3,260)]:
            ds=np.array([[12,8,duration,16]],np.int64);frames=int(ds.sum())
            raw=np.random.default_rng(9).normal(-4,1,(1,frames,128)).astype(np.float32)
            feed=dict(tokens=np.array([[1,8,token,1]],np.int64),durations=ds,f0=np.full((1,frames),349.228,np.float32))
            feed[prefix+'su_raw']=raw
            x,y=a.run(None,feed)[0],b.run(None,feed)[0]
            mask=np.zeros(frames,bool)
            eligible=duration>=40 and token!=8
            start,fade=(74,26) if duration>=172 else ((16,10) if duration>=60 else (12,8))
            if eligible:mask[20+start:20+duration]=True
            assert np.array_equal(x[:,~mask],y[:,~mask])
            assert np.isfinite(y).all()
            if eligible:
                plateau=y[:,20+start+fade:20+duration-12]
                assert plateau.shape[1]>1
                assert np.max(np.ptp(plateau,axis=1))<2e-6
            rows.append(dict(branch=prefix,token=token,duration=duration,unchanged_outside_sustain=True,constant_plateau=eligible))
    output.write_text(json.dumps(rows,indent=2));print(len(rows),'cases passed')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['before','after','output']:p.add_argument(k,type=Path)
    a=p.parse_args();check(a.before,a.after,a.output)
