"""Keep D4C band-energy ratios finite when a band has no measured energy."""
import argparse
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh


def update(source,output):
    if output.exists():raise FileExistsError(output)
    m=onnx.load(source);target=next(n for n in m.graph.node if n.op_type=='Div' and list(n.output)==['world_div_27'])
    if list(target.input)!=['world_select_8','world_select_9']:raise ValueError('Unexpected D4C graph')
    for k,v in [('zero',0.),('one',1.),('floor',1e-12)]:m.graph.initializer.append(nh.from_array(np.array(v,np.float32),'apg_'+k))
    nodes=[]
    for n in m.graph.node:
        if n is target:
            nodes.extend([
                h.make_node('Greater',['world_select_9','apg_zero'],['apg_energy']),
                h.make_node('Where',['apg_energy','world_select_9','apg_one'],['apg_denominator']),
                h.make_node('Div',['world_select_8','apg_denominator'],['apg_ratio']),
                h.make_node('Where',['apg_energy','apg_ratio','apg_one'],['apg_valid_ratio']),
                h.make_node('Clip',['apg_valid_ratio','apg_floor','apg_one'],['world_div_27']),
            ])
        else:nodes.append(n)
    if not any(n.output and n.output[0]=='apg_ratio' for n in nodes):raise ValueError('D4C node not patched')
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.source,a.output)
