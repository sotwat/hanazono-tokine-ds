"""Check onset locality, pitch guards, energy preservation and fixed-raw playback."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper


def check(before,after,output):
    rows=[]
    for pr in ['original_','donor_']:
        a,b=wrapper(before,pr),wrapper(after,pr)
        for previous in [1,3,8,14,20]:
            for current in [3,9,14,20,28,8,22]:
                for duration in [1,2,3,8,40,180]:
                    for bend in [0,100]:
                        ds=np.array([[8,8,duration,8]],np.int64);length=int(ds.sum());raw=np.random.default_rng(12).normal(-4,1,(1,length,128)).astype(np.float32)
                        f=np.full((1,length),440,np.float32)
                        if bend:f[0]=440*2**(np.arange(length)*bend/1200)
                        feed=dict(tokens=np.array([[1,previous,current,1]],np.int64),durations=ds,f0=f,**{pr+'su_raw':raw})
                        x,y=a.run(None,feed)[0],b.run(None,feed)[0];allowed=np.zeros(length,bool)
                        # First previous token also follows SP if it is a vowel.
                        for start,t,prev,d in [(8,previous,1,8),(16,current,previous,duration)]:
                            if t in [3,9,14,20,28] and prev==1 and not bend and d>1:allowed[start:start+min(3,d)]=True
                        assert np.array_equal(x[:,~allowed],y[:,~allowed]), (pr,previous,current,duration,bend,np.where(np.max(abs(x-y),axis=(0,2))>0)[0]);assert np.isfinite(y).all()
                        assert np.allclose(np.mean(np.exp(x),axis=2),np.mean(np.exp(y),axis=2),rtol=2e-6,atol=1e-9)
                        rows.append(dict(branch=pr,previous=previous,current=current,duration=duration,bend=bend,passed=True))
    output.write_text(json.dumps(dict(cases=len(rows),all_passed=True),indent=2));print(len(rows),'cases passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.before,a.after,a.output)
