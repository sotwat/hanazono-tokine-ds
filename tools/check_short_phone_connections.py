"""Verify short-note joins, bypass cases and long-note regression on fixed spectra."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper


def check(before, after, output):
    count = changed = 0
    vowels = [3,9,14,20,28]
    sustained = [2,3,9,10,12,14,18,19,20,24,25,28,31]
    consonants = [i for i in range(4,32) if i not in vowels]
    fricatives = [6,10,12,13,15,24,25,27,31]
    for prefix in ['original_', 'donor_']:
        old, new = wrapper(before,prefix), wrapper(after,prefix)
        for prev in sustained + [1,8]:
            for nxt in consonants + [1,3,9]:
                for da in [0,1,3,4,7,8,9,16,39,40,86,258]:
                    for db in [0,1,2,3,4,8,12]:
                        d=np.array([[8,da,db,24,8]],np.int64)
                        frames=int(d.sum());boundary=8+da
                        raw=np.random.default_rng(10).normal(-4,1,(1,frames,128)).astype(np.float32)
                        feed=dict(tokens=np.array([[1,prev,nxt,3,1]],np.int64),durations=d,f0=np.full((1,frames),698.4555,np.float32))
                        feed[prefix+'su_raw']=raw
                        x=old.run(None,feed)[0];y=new.run(None,feed)[0]
                        assert y.shape==x.shape and np.isfinite(y).all()
                        allowed=np.zeros(frames,bool)
                        eligible=prev in sustained and nxt in consonants and 4<=da<40 and db>=2
                        if eligible:
                            left=min(2 if nxt in fricatives else 1,da//4)
                            right=min(left,(db-1)//2)
                            allowed[boundary-left+1:boundary+right]=True
                        assert np.array_equal(x[:,~allowed],y[:,~allowed]),(prefix,prev,nxt,da,db)
                        if allowed.any():
                            assert np.any(x!=y),(prev,nxt,da,db)
                            changed+=1
                        count+=1
    result=dict(cases=count,changed_cases=changed,finite=True,outside_boundary_bit_exact=True,long_note_bit_exact=True)
    output.write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.before,a.after,a.output)
