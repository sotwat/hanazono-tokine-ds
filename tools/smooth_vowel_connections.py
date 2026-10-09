"""Morph adjacent sustained vowel spectra across their shared boundary."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    model = onnx.load(source)
    if output.exists():
        raise FileExistsError(output)
    prefixes = [v.name[:-len('su_vv_ids')] for v in model.graph.initializer
                if v.name.endswith('su_vv_ids')]
    if not prefixes or any(v.name.endswith('su_vm_eight') for v in model.graph.initializer):
        raise ValueError('Expected the unsmoothed vowel-connection wrapper')
    for prefix in prefixes:
        s = lambda name: prefix + 'su_' + name
        for key, val, dtype in [('vm_eight',8.,np.float32), ('vm_sixteen',16.,np.float32),
                                ('vm_eighti',8,np.int64), ('vm_minus_eight',-8.,np.float32)]:
            model.graph.initializer.append(nh.from_array(np.array(val,dtype),s(key)))
        nodes=[]
        def n(op, ins, out, **kwargs):
            nodes.append(h.make_node(op,ins,[s(out)],**kwargs))
        final = prefix + 'mel'
        for node in model.graph.node:
            if final not in node.output:
                nodes.append(node)
                continue
            node.output[0] = s('vm_base')
            nodes.append(node)
            n('Squeeze',[s('vm_base'),s('axis0')],'vm_base2')
            n('Sub',[s('starts'),s('vm_eighti')],'vm_left0')
            n('Add',[s('starts'),s('vm_eighti')],'vm_right0')
            n('Clip',[s('vm_left0'),s('zeroi'),s('vv_last')],'vm_left_idx')
            n('Clip',[s('vm_right0'),s('zeroi'),s('vv_last')],'vm_right_idx')
            n('Gather',[s('vm_base2'),s('vm_left_idx')],'vm_left',axis=0)
            n('Gather',[s('vm_base2'),s('vm_right_idx')],'vm_right',axis=0)
            n('Add',[s('offset'),s('vm_eight')],'vm_elapsed')
            n('Div',[s('vm_elapsed'),s('vm_sixteen')],'vm_progress0')
            n('Clip',[s('vm_progress0'),s('zero'),s('one')],'vm_progress')
            n('Mul',[s('vm_progress'),s('vm_progress')],'vm_squared')
            n('Mul',[s('vm_progress'),s('two')],'vm_twice')
            n('Sub',[s('three'),s('vm_twice')],'vm_factor')
            n('Mul',[s('vm_squared'),s('vm_factor')],'vm_smooth')
            n('Greater',[s('offset'),s('vm_minus_eight')],'vm_after_start')
            n('Less',[s('offset'),s('vm_eight')],'vm_before_end')
            n('And',[s('vm_after_start'),s('vm_before_end')],'vm_inside')
            n('Cast',[s('vm_inside')],'vm_inside_float',to=onnx.TensorProto.FLOAT)
            n('Mul',[s('vm_inside_float'),s('vv_eligible')],'vm_mask')
            n('Mul',[s('vm_mask'),s('vm_smooth')],'vm_right_weight')
            n('Sub',[s('vm_mask'),s('vm_right_weight')],'vm_left_weight')
            n('Transpose',[s('vm_right_weight')],'vm_rw',perm=[0,2,1])
            n('Transpose',[s('vm_left_weight')],'vm_lw',perm=[0,2,1])
            n('MatMul',[s('vm_rw'),s('vm_right')],'vm_r')
            n('MatMul',[s('vm_lw'),s('vm_left')],'vm_l')
            n('Add',[s('vm_l'),s('vm_r')],'vm_morph')
            n('ReduceSum',[s('vm_mask'),s('axis1')],'vm_mask2',keepdims=0)
            n('Unsqueeze',[s('vm_mask2'),s('axis2')],'vm_mask3')
            n('Greater',[s('vm_mask3'),s('zero')],'vm_active')
            nodes.append(h.make_node('Where',[s('vm_active'),s('vm_morph'),s('vm_base')],[final]))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,output)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    update(args.source,args.output)
