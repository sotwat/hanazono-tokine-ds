"""Check vowel coverage, brief-note bounds and unchanged consonant contexts."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h,TensorProto as T
from check_vowel_connections import wrapper


def check(before,after,output):
    rows=[];vowels=[3,9,14,20,28]
    for prefix in ['original_','donor_']:
        old,new=wrapper(before,prefix),wrapper(after,prefix)
        cases=[(a,b,da,db) for a in vowels for b in vowels for da,db in [(9,29),(29,14),(57,86),(86,14),(207,275),(4,6),(3,5)]]
        cases += [(3,14,da,db) for da,db in [(0,0),(0,8),(8,0),(1,1)]]
        cases += [(a,b,57,86) for a,b in [(16,3),(12,3),(3,16),(3,1),(3,10),(19,3)]]
        for a,b,da,db in cases:
            ds=np.array([[12,da,db,16]],np.int64);frames=int(ds.sum());boundary=12+da
            raw=np.random.default_rng(72).normal(-4,.6,(1,frames,128)).astype(np.float32)
            feed=dict(tokens=np.array([[1,a,b,1]],np.int64),durations=ds,f0=np.full((1,frames),349.228,np.float32));feed[prefix+'su_raw']=raw
            x,y=old.run(None,feed)[0],new.run(None,feed)[0]
            assert np.isfinite(y).all() and y.shape==x.shape
            allowed=np.zeros(frames,bool)
            if a in vowels and b in vowels:
                allowed[max(0,boundary-8):boundary+max(10,min(24,int(db*.5)))+1]=True
            assert np.array_equal(x[:,~allowed],y[:,~allowed]),(prefix,a,b,da,db)
            if a in vowels and b in vowels and da>=4 and db>=6:
                assert np.any(x!=y),(a,b,da,db)
            rows.append(dict(branch=prefix,tokens=[a,b],durations=[da,db],finite=True,unchanged_elsewhere=True))
    # A weak repeated vowel may borrow only from its own continuous run.
    for prefix in ['original_','donor_']:
        session=wrapper(after,prefix)
        ds=np.array([[12,70,70,70,16]],np.int64);f0=np.full((1,int(ds.sum())),440,np.float32)
        raw=np.full((1,int(ds.sum()),128),-3,np.float32);raw[:,12:82]=-8
        feed=dict(tokens=np.array([[1,3,3,3,1]],np.int64),durations=ds,f0=f0);feed[prefix+'su_raw']=raw
        y=session.run(None,feed)[0];assert np.all(y[:,32:62]>-3.01)
        feed['tokens']=np.array([[1,3,1,3,1]],np.int64)
        y=session.run(None,feed)[0];assert np.all(y[:,32:62]<-7.99)
    m=onnx.load(after)
    for name in ['gv_tokens','gv_durations']:
        m.graph.output.append(h.make_tensor_value_info(name,T.INT64,[1,'N']))
    o=ort.SessionOptions();o.log_severity_level=3;o.intra_op_num_threads=2
    session=ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider'])
    for tokens,durations,expected_t,expected_d in [([1,3,3,3,1],[12,86,14,72,16],[1,3,1],[12,172,16]),([1,3,3,3,3,3,1],[12,86,14,72,29,86,16],[1,3,3,1],[12,172,115,16]),([1,3,1,3,1],[12,14,8,14,16],[1,3,1,3,1],[12,14,8,14,16]),([1,16,16,14,3,1],[12,3,3,14,29,16],[1,16,16,14,3,1],[12,3,3,14,29,16])]:
        got=session.run(['gv_tokens','gv_durations'],dict(tokens=np.array([tokens],np.int64),durations=np.array([durations],np.int64),f0=np.full((1,sum(durations)),440,np.float32),depth=np.array(.1,np.float32),steps=np.array(20,np.int64)))
        assert got[0].tolist()==[expected_t] and got[1].tolist()==[expected_d]
    output.write_text(json.dumps(dict(wrapper_cases=rows,merge_cases=4,run_donor_cases=4),indent=2));print(len(rows),'wrapper cases; 4 merge and 4 run-donor cases passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();check(a.before,a.after,a.output)
