"""Validate independent context and join scope on fixed learned spectra."""
import argparse,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort


def check(source,folder):
    opt=ort.SessionOptions();opt.log_severity_level=3;opt.intra_op_num_threads=2
    context=ort.InferenceSession(str(folder/'context.onnx'),opt);join=ort.InferenceSession(str(folder/'join.onnx'),opt)
    rng=np.random.default_rng(20261020);count=0
    cases=[([1,a,b,1],[8,da,db,8]) for a in range(32) for b in range(32) for da,db in [(39,52),(40,52),(87,12),(87,11),(0,52),(87,90)]]
    cases += [([1,14,3,9,3,1],[8,60,40,60,52,8]),([1,14,3,3,3,1],[8,60,60,60,60,8])]
    for ts,ds in cases:
        t=np.array([ts],np.int64);d=np.array([ds],np.int64);f=int(d.sum());starts=np.cumsum(ds)-ds
        f0=np.linspace(220,440,f,dtype=np.float32)[None];base=rng.normal(-4,.5,(1,f,128)).astype(np.float32)
        nd=np.array(ds).copy();rt=ts.copy();eligible=[]
        for i in range(1,len(ts)):
            if ts[i]==3 and ts[i-1] in (9,14,20,28) and ds[i-1]>=40 and ds[i]>=12:
                rt[i-1]=1;nd[i]=12;eligible.append(i)
        cs=np.cumsum(nd)-nd
        fresh=rng.normal(-4,.5,(1,int(nd.sum()),128)).astype(np.float32)
        nt,cd,nf=context.run(None,dict(tokens=t,durations=d,f0=f0))
        out=join.run(None,dict(tokens=t,durations=d,base=base,fresh=fresh))[0]
        mask=np.zeros(f,bool);positions=[]
        for i in range(len(ts)):
            pos=starts[i]+np.arange(nd[i])*max(ds[i]-1,0)/max(nd[i]-1,1)
            values=np.interp(pos,np.arange(f),f0[0]).astype(np.float32)
            if i+1 in eligible:values[:]=0
            positions.extend(values.tolist())
        for i in eligible:
            off=np.arange(f)-starts[i];mask|=(off>-2)&(off<ds[i]*.75)
            assert np.array_equal(out[0,starts[i]],fresh[0,cs[i]+5])
        assert np.array_equal(nt,np.array([rt])) and np.array_equal(cd,np.array([nd]))
        assert np.allclose(nf,np.array([positions]),rtol=1e-6,atol=1e-4)
        assert np.array_equal(out[:,~mask],base[:,~mask]);assert np.isfinite(out).all() and out.shape==base.shape
        count+=1
    a=onnx.load(source);b=onnx.load(folder/'acoustic.onnx');weights={v.name:v.SerializeToString() for v in b.graph.initializer};assert all(weights[v.name]==v.SerializeToString() for v in a.graph.initializer)
    result=dict(cases=count,context_only_previous_vowel_reset=True,original_inputs_unmodified=True,exact_join_bypass=True,unchanged_learned_tensors=True,perceptual_quality='requires listening')
    (folder/'check.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('folder',type=Path);a=p.parse_args();check(a.source,a.folder)
