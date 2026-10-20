"""Check join-only scope with identical spectra; no waveform gain correction."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort


def check(source,folder):
    opt=ort.SessionOptions();opt.log_severity_level=3;opt.intra_op_num_threads=2
    session=ort.InferenceSession(str(folder/'connection.onnx'),opt)
    rng=np.random.default_rng(127);cases=0
    for a in range(32):
      for b in range(32):
       for da,db in [(39,52),(40,52),(87,6),(87,5),(87,90),(0,52)]:
        d=np.array([[8,da,db,8]],np.int64);f=int(d.sum());s=8+da
        base=rng.normal(-4,.4,(1,f,128)).astype(np.float32);old=rng.normal(-4,.4,(1,f,128)).astype(np.float32)
        feed=dict(tokens=np.array([[1,a,b,1]],np.int64),durations=d,base=base,existing=old)
        y=session.run(None,feed)[0];active=b==3 and a in (9,14,20,28) and da>=40 and db>=6
        allowed=np.zeros(f,bool)
        if active:allowed[max(0,s-5):s+int(np.ceil(min(db*.5,24)))]=True
        assert np.array_equal(y[:,~allowed],old[:,~allowed])
        assert y.shape==old.shape and np.isfinite(y).all()
        if active:
         assert np.allclose(y[0,s],base[0,s+2],atol=1e-6)
         assert np.allclose(y[0,s+6],base[0,s+6],atol=1e-6) if db>=14 else True
        cases+=1
    a=onnx.load(source);b=onnx.load(folder/'acoustic.onnx');weights={v.name:v.SerializeToString() for v in b.graph.initializer}
    assert all(weights[v.name]==v.SerializeToString() for v in a.graph.initializer)
    report=dict(cases=cases,outside_join_bit_exact=True,unchanged_initializers=True,advance_ms=2*512/44.1,perceptual_quality='requires listening; uninstalled trial')
    (folder/'check.json').write_text(json.dumps(report,indent=2));print(report)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('folder',type=Path);a=p.parse_args();check(a.source,a.folder)
