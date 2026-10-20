"""Check scope, level contour and unchanged trained weights in the onset trial."""
import argparse,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from check_a_vowel_rearticulation import check as reference_check

def check(source,folder):
    reference_check(source,folder/'reference')
    opt=ort.SessionOptions();opt.log_severity_level=3;opt.intra_op_num_threads=2
    c=ort.InferenceSession(str(folder/'longer/context.onnx'),opt);j=ort.InferenceSession(str(folder/'longer/join.onnx'),opt);l=ort.InferenceSession(str(folder/'contour.onnx'),opt)
    rng=np.random.default_rng(20261020);count=0;error=0
    for prev in range(32):
      for before in [39,40,87]:
       for duration in [11,12,24,52,90]:
        t=np.array([[1,prev,3,1]],np.int64);d=np.array([[8,before,duration,8]],np.int64);n=int(d.sum());f=np.full((1,n),311,np.float32);b=rng.normal(-4,.5,(1,n,128)).astype(np.float32)
        nt,nd,nf=c.run(None,dict(tokens=t,durations=d,f0=f));fresh=rng.normal(-4,.5,(1,int(nd.sum()),128)).astype(np.float32)
        alt=j.run(None,dict(tokens=t,durations=d,base=b,fresh=fresh))[0];out=l.run(None,dict(reference=b,candidate=alt))[0]
        eligible=prev in [9,14,20,28] and before>=40 and duration>=12
        mask=np.zeros(n,bool)
        if eligible:
            off=np.arange(n)-(8+before);mask=(off>-2)&(off<duration*.75)
            assert nd[0,2]==24 and nt[0,1]==1
        else:assert np.array_equal(nd,d) and np.array_equal(nt,t)
        assert np.array_equal(out[:,~mask],b[:,~mask]) and np.isfinite(out).all()
        e=float(np.max(abs(np.logaddexp.reduce(out,-1)-np.logaddexp.reduce(b,-1))));error=max(error,e);assert e<2e-5
        count+=1
    a=onnx.load(source);b=onnx.load(folder/'acoustic.onnx');weights={v.name:v.SerializeToString() for v in b.graph.initializer};assert all(weights[v.name]==v.SerializeToString() for v in a.graph.initializer)
    result=dict(cases=count,original_check_cases=6146,max_contour_log_error=error,non_target_exact=True,trained_tensors_identical=True,listening='required')
    (folder/'checks.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('folder',type=Path);a=p.parse_args();check(a.source,a.folder)
