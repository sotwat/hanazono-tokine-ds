"""Apply one peak-bounded gain to a whole rendered phrase."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N

def build(source, target, gain_db=3.0, ceiling=.75):
    model=onnx.load(source)
    if any(n.name.startswith('phrase_gain_') for n in model.graph.node):
        raise ValueError('Output gain is already installed')
    output=model.graph.output[0].name
    raw='phrase_gain_raw'
    for node in model.graph.node:
        for i,name in enumerate(node.output):
            if name==output:node.output[i]=raw
        for i,name in enumerate(node.input):
            if name==output:node.input[i]=raw
    for name,value in [('target',10**(gain_db/20)),('ceiling',ceiling),('one',1.),('epsilon',1e-8)]:
        model.graph.initializer.append(N.from_array(np.array(value,dtype=np.float32),'phrase_gain_'+name))
    specs=[
        ('Abs',[raw],['phrase_gain_abs'],{}),
        ('ReduceMax',['phrase_gain_abs'],['phrase_gain_peak'],{'keepdims':0}),
        ('Max',['phrase_gain_peak','phrase_gain_epsilon'],['phrase_gain_safe_peak'],{}),
        ('Div',['phrase_gain_ceiling','phrase_gain_safe_peak'],['phrase_gain_headroom'],{}),
        ('Min',['phrase_gain_target','phrase_gain_headroom'],['phrase_gain_bounded'],{}),
        ('Max',['phrase_gain_one','phrase_gain_bounded'],['phrase_gain_factor'],{}),
        ('Mul',[raw,'phrase_gain_factor'],[output],{}),
    ]
    for index,(op,inputs,outputs,kwargs) in enumerate(specs):
        model.graph.node.append(H.make_node(op,inputs,outputs,name=f'phrase_gain_{index}',**kwargs))
    onnx.checker.check_model(model)
    onnx.save(model,target)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path);p.add_argument('target',type=Path)
    a=p.parse_args()
    if a.source.resolve()==a.target.resolve():raise ValueError('Use a separate output path')
    build(a.source,a.target)
