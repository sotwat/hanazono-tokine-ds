"""Replay a captured failing donor waveform through WORLD analysis/synthesis."""
import argparse,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h


def world_session(path):
    m=onnx.load(path);needed={'world_wave','world_div_27'};selected=[];boundaries={'donor_wave','source_f0','f0'}
    for n in reversed(m.graph.node):
        if any(v in needed and v not in boundaries for v in n.output):selected.append(n);needed.update(n.input)
    g=h.make_graph(list(reversed(selected)),'world-regression',[h.make_tensor_value_info(k,onnx.TensorProto.FLOAT,[1,'N']) for k in boundaries],[h.make_tensor_value_info(k,onnx.TensorProto.FLOAT,None) for k in ['world_wave','world_div_27']],[v for v in m.graph.initializer if v.name in needed]);model=h.make_model(g,opset_imports=m.opset_import);model.ir_version=m.ir_version;o=ort.SessionOptions();o.intra_op_num_threads=4;o.log_severity_level=3
    return ort.InferenceSession(model.SerializeToString(),o)


def check(before,after,fixture,output):
    z=np.load(fixture);feed=dict(donor_wave=z['donor_wave'],f0=z['f0'],source_f0=np.clip(z['f0'],220,523.2511306));a,b=world_session(before),world_session(after);rows=[]
    for run in range(3):
        x,rx=a.run(None,feed);y,ry=b.run(None,feed)
        assert not np.isfinite(x).all(), 'Fixture no longer reproduces failure'
        assert np.isfinite(y).all() and np.isfinite(ry).all()
        valid=np.isfinite(rx)&(rx>=1e-12)&(rx<=1)
        assert np.array_equal(rx[valid],ry[valid])
        rows.append(dict(run=run,before_nan_samples=int((~np.isfinite(x)).sum()),after_nan_samples=int((~np.isfinite(y)).sum()),valid_ratios_exact=True,peak=float(abs(y).max())))
    output.write_text(json.dumps(rows,indent=2));print(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('fixture',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.before,a.after,a.fixture,a.output)
