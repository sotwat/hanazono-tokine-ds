"""Validate fricative boundary blending and exact preservation elsewhere."""
import argparse,json
from pathlib import Path
import numpy as np
from check_vowel_connections import wrapper

def check(before,after,output):
 rows=[]
 cases=[(a,b,70,8) for a in [3,9,14,20,28] for b in [10,12,24,25,31]]
 cases += [(3,8,70,8),(3,1,70,8),(3,3,70,80),(3,10,30,8),(3,10,70,3),(19,10,70,8)]
 for prefix in ['original_','donor_']:
  a,b=wrapper(before,prefix),wrapper(after,prefix)
  for prev,cur,dp,dc in cases:
   ds=np.array([[12,dp,dc,60,16]],np.int64);frames=int(ds.sum());boundary=12+dp
   raw=np.random.default_rng(3).normal(-4,1,(1,frames,128)).astype(np.float32)
   feed=dict(tokens=np.array([[1,prev,cur,28,1]],np.int64),durations=ds,f0=np.full((1,frames),349.228,np.float32));feed[prefix+'su_raw']=raw
   x,y=a.run(None,feed)[0],b.run(None,feed)[0]
   eligible=prev in [3,9,14,20,28] and cur in [10,12,24,25,31] and dp>=40 and dc>=4
   mask=np.zeros(frames,bool)
   if eligible:
    mask[boundary-7:boundary+2]=True
    t=np.arange(1,10,dtype=np.float32)/10;w=(t*t*(3-2*t))[None,:,None]
    expected=(1-w)*x[:,boundary-8:boundary-7]+w*x[:,boundary+2:boundary+3]
    np.testing.assert_allclose(y[:,mask],expected,atol=2e-6)
   assert np.array_equal(x[:,~mask],y[:,~mask]);assert np.isfinite(y).all()
   rows.append(dict(branch=prefix,previous=prev,current=cur,eligible=eligible,unchanged_elsewhere=True))
 output.write_text(json.dumps(rows,indent=2));print(len(rows),'cases passed')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['before','after','output']:p.add_argument(k,type=Path)
 a=p.parse_args();check(a.before,a.after,a.output)
