"""Check donor coverage, protected phones, and unchanged long-note processing."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper


def check(before, after, output):
    rows=[]
    sustained={2,3,9,10,12,14,18,19,20,24,25,28,31}
    for prefix in ['original_','donor_']:
        a,b=wrapper(before,prefix),wrapper(after,prefix)
        for token in range(1,32):
            for duration in [1,4,12,27,39,40,54,60,96,171,172,275]:
                frames=duration+16
                raw=np.full((1,frames,128),-5,np.float32)
                # A strong interior plateau beyond the old fixed search range.
                raw[:,8+int(duration*.65):8+int(duration*.8)]=-2
                feed=dict(tokens=np.array([[1,token,1]],np.int64),durations=np.array([[8,duration,8]],np.int64),f0=np.full((1,frames),392,np.float32))
                feed[prefix+'su_raw']=raw
                x,y=a.run(None,feed)[0],b.run(None,feed)[0]
                assert np.isfinite(y).all()
                assert np.array_equal(x[:,:8],y[:,:8]) and np.array_equal(x[:,8+duration:],y[:,8+duration:])
                if token not in sustained or duration<40 or duration>=172:
                    assert np.array_equal(x,y),(prefix,token,duration)
                if token in sustained and duration==54:
                    assert np.exp(y[:,8+28:8+45]).mean()>np.exp(x[:,8+28:8+45]).mean()*2
                rows.append(dict(branch=prefix,token=token,duration=duration,changed=bool(np.any(x!=y))))
    output.write_text(json.dumps(rows,indent=2))
    print(f'{len(rows)} checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.before,a.after,a.output)
