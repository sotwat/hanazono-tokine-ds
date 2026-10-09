"""Crossfade sustained vowels into fricatives without changing phone timing."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh


def update(source,output):
    m=onnx.load(source)
    if output.exists():raise FileExistsError(output)
    prefixes=[v.name[:-len('su_hold_offsets')] for v in m.graph.initializer if v.name.endswith('su_hold_offsets')]
    if not prefixes or any(v.name.endswith('su_fc_ids') for v in m.graph.initializer):
        raise ValueError('Expected steady sustain model without fricative morph')
    for prefix in prefixes:
        s=lambda key:prefix+'su_'+key
        for k,v,t in [('fc_ids',[10,12,24,25,31],np.int64),('fc_left',8,np.int64),('fc_right',2,np.int64),('fc_min',4.,np.float32),('fc_eight',8.,np.float32),('fc_ten',10.,np.float32),('fc_neg',-8.,np.float32),('fc_two',2.,np.float32)]:
            m.graph.initializer.append(nh.from_array(np.array(v,t),s(k)))
        nodes=[]
        def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,[s(out)],**kw))
        final=prefix+'mel'
        for node in m.graph.node:
            if final not in node.output:nodes.append(node);continue
            node.output[0]=s('fc_base');nodes.append(node)
            n('Equal',[s('tokens3'),s('fc_ids')],'fc_eq')
            n('Cast',[s('fc_eq')],'fc_float',to=onnx.TensorProto.FLOAT)
            n('ReduceMax',[s('fc_float'),s('axis2')],'fc_is',keepdims=1)
            n('GreaterOrEqual',[s('dur'),s('fc_min')],'fc_long')
            n('Cast',[s('fc_long')],'fc_longf',to=onnx.TensorProto.FLOAT)
            n('Mul',[s('fc_is'),s('fc_longf')],'fc_eligible0')
            n('Mul',[s('fc_eligible0'),s('vv_prev_vowel')],'fc_eligible1')
            n('Mul',[s('fc_eligible1'),s('vv_prev_longfloat')],'fc_eligible')
            n('Squeeze',[s('fc_base'),s('axis0')],'fc_base2')
            n('Sub',[s('starts'),s('fc_left')],'fc_li0')
            n('Add',[s('starts'),s('fc_right')],'fc_ri0')
            for key in ['li','ri']:n('Clip',[s('fc_'+key+'0'),s('zeroi'),s('vv_last')],'fc_'+key)
            for key,idx in [('l','li'),('r','ri')]:
                n('Gather',[s('fc_base2'),s('fc_'+idx)],'fc_'+key,axis=0)
                n('Identity',[s('fc_'+key)],'fc_'+key+'amp')
            n('Add',[s('offset'),s('fc_eight')],'fc_elapsed')
            n('Div',[s('fc_elapsed'),s('fc_ten')],'fc_t0')
            n('Clip',[s('fc_t0'),s('zero'),s('one')],'fc_t')
            n('Mul',[s('fc_t'),s('fc_t')],'fc_sq')
            n('Mul',[s('fc_t'),s('two')],'fc_twice')
            n('Sub',[s('three'),s('fc_twice')],'fc_factor')
            n('Mul',[s('fc_sq'),s('fc_factor')],'fc_w')
            n('Greater',[s('offset'),s('fc_neg')],'fc_after')
            n('Less',[s('offset'),s('fc_two')],'fc_before')
            n('And',[s('fc_after'),s('fc_before')],'fc_inside')
            n('Cast',[s('fc_inside')],'fc_in',to=onnx.TensorProto.FLOAT)
            n('Mul',[s('fc_in'),s('fc_eligible')],'fc_mask')
            n('Mul',[s('fc_mask'),s('fc_w')],'fc_rw0')
            n('Sub',[s('fc_mask'),s('fc_rw0')],'fc_lw0')
            for key in ['l','r']:
                n('Transpose',[s('fc_'+key+'w0')],'fc_'+key+'w',perm=[0,2,1])
                n('MatMul',[s('fc_'+key+'w'),s('fc_'+key+'amp')],'fc_'+key+'mix')
            n('Add',[s('fc_lmix'),s('fc_rmix')],'fc_mix')
            n('ReduceSum',[s('fc_mask'),s('axis1')],'fc_mask2',keepdims=0)
            n('Unsqueeze',[s('fc_mask2'),s('axis2')],'fc_mask3')
            n('Greater',[s('fc_mask3'),s('zero')],'fc_active')
            n('Where',[s('fc_active'),s('fc_mix'),s('one')],'fc_safe')
            n('Identity',[s('fc_safe')],'fc_log')
            nodes.append(h.make_node('Where',[s('fc_active'),s('fc_log'),s('fc_base')],[final]))
        del m.graph.node[:];m.graph.node.extend(nodes)
    onnx.checker.check_model(m);output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();update(a.source,a.output)
