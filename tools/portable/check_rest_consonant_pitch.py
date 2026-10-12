"""Check F0 conditioning independently of diffusion sampling."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

p=argparse.ArgumentParser();p.add_argument('acoustic',type=Path);args=p.parse_args();m=onnx.load(args.acoustic);needed={'rp_normalized'};nodes=[]
for n in reversed(m.graph.node):
    if any(x in needed and x not in {'tokens','durations','f0'} for x in n.output):nodes.append(n);needed.update(n.input)
nodes.reverse();init=[x for x in m.graph.initializer if any(x.name in n.input for n in nodes)]
g=onnx.helper.make_graph(nodes,'pitch-condition-only',[onnx.helper.make_tensor_value_info('tokens',7,[1,'N']),onnx.helper.make_tensor_value_info('durations',7,[1,'N']),onnx.helper.make_tensor_value_info('f0',1,[1,'F'])],[onnx.helper.make_tensor_value_info('rp_normalized',1,[1,'F'])],init);q=onnx.helper.make_model(g,opset_imports=m.opset_import,ir_version=m.ir_version);opts=ort.SessionOptions();opts.intra_op_num_threads=1;s=ort.InferenceSession(q.SerializeToString(),opts)
unvoiced=[6,7,10,12,13,16,17,21,24,25,26,27];cases=0
for phone in unvoiced:
    for vowel in [3,9,14,20,28]:
        for duration in [1,2,4,8,16]:
            for connected in [False,True]:
                for cluster in [False,True]:
                    tokens=[1]+([3] if connected else [])+[phone]+([16] if cluster else [])+[vowel,1]
                    durs=[8]+([12] if connected else [])+[duration]+([3] if cluster else [])+[8,8]
                    f=np.linspace(1200,220,sum(durs),dtype=np.float32).reshape(1,-1);starts=np.r_[0,np.cumsum(durs)];first=2 if connected else 1;vindex=len(tokens)-2;ref=starts[vindex];expected=f.copy()
                    if not connected:expected[:,starts[first]:ref]=f[:,ref:ref+1]
                    result=s.run(None,{'tokens':np.array([tokens],np.int64),'durations':np.array([durs],np.int64),'f0':f})[0]
                    assert np.array_equal(result,expected)
                    assert np.isfinite(result).all()
                    cases+=1
print(json.dumps({'cases':cases,'exact_voiced_and_connected_bypass':True,'next_vowel_reference':True}))
