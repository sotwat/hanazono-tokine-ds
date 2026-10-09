"""Hold an averaged donor spectrum instead of scanning it periodically."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh


def update(source,output):
    m=onnx.load(source)
    if output.exists():raise FileExistsError(output)
    prefixes=[v.name[:-len('su_donor_radius')] for v in m.graph.initializer if v.name.endswith('su_donor_radius')]
    if not prefixes or any(v.name.endswith('su_hold_offsets') for v in m.graph.initializer):
        raise ValueError('Expected an unpatched adaptive sustain model')
    for prefix in prefixes:
        s=lambda name:prefix+'su_'+name
        m.graph.initializer.append(nh.from_array(np.array([-1,-.5,0,.5,1],np.float32),s('hold_offsets')))
        for key,value in [('short_start',16.),('short_fade',10.),('brief_start',12.)]:
            item=next(v for v in m.graph.initializer if v.name==s(key))
            item.CopyFrom(nh.from_array(np.array(value,np.float32),s(key)))
        nodes=[]
        def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,[s(out)],**kw))
        for node in m.graph.node:
            if s('loop') not in node.output:
                nodes.append(node);continue
            node.output[0]=s('hold_unused_loop');nodes.append(node)
            n('Mul',[s('selected_radius'),s('hold_offsets')],'hold_span')
            n('Add',[s('selected_center'),s('hold_span')],'hold_local')
            n('Add',[s('startsfloat'),s('hold_local')],'hold_global')
            n('Cast',[s('hold_global')],'hold_idx0',to=onnx.TensorProto.INT64)
            n('Clip',[s('hold_idx0'),s('zeroi'),s('donor_last')],'hold_idx')
            n('Gather',[s('raw2'),s('hold_idx')],'hold_samples',axis=0)
            n('ReduceMean',[s('hold_samples'),s('axis2')],'hold_mean',keepdims=0)
            n('Transpose',[s('activefloat')],'hold_active',perm=[0,2,1])
            n('MatMul',[s('hold_active'),s('hold_mean')],'loop')
        del m.graph.node[:];m.graph.node.extend(nodes)
    onnx.checker.check_model(m);output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();update(a.source,a.output)
