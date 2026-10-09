"""Preserve energy at adjacent sustained-vowel boundaries."""
from pathlib import Path
import argparse
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh


def update(source,output):
    m=onnx.load(source)
    if output.exists():raise FileExistsError(output)
    prefixes=[i.name[:-len('su_connected_fade')] for i in m.graph.initializer if i.name.endswith('su_connected_fade')]
    if not prefixes:raise ValueError('Expected contextual sustain model')
    if any(i.name.endswith('su_vv_ids') for i in m.graph.initializer):
        raise ValueError('Vowel connection patch already applied')
    for prefix in prefixes:
        s=lambda x:prefix+'su_'+x
        constants=[('vv_ids',[3,9,14,20,28],np.int64),('vv_two',2.,np.float32),('vv_ten',10.,np.float32),('vv_twelve',12,np.int64),('vv_begin',[0],np.int64),('vv_minusone',[-1],np.int64),('vv_prepad',[0,1,0,0],np.int64),('vv_zeroi',0,np.int64)]
        for k,v,t in constants:m.graph.initializer.append(nh.from_array(np.array(v,t),s(k)))
        nodes=[]
        def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,[s(out)],**kw))
        for node in m.graph.node:
            if s('release') in node.output:
                n('Unsqueeze',[s('next_tokens'),s('axis2')],'vv_next3')
                n('Equal',[s('vv_next3'),s('vv_ids')],'vv_next_eq')
                n('Cast',[s('vv_next_eq')],'vv_next_float',to=onnx.TensorProto.FLOAT)
                n('ReduceMax',[s('vv_next_float'),s('axis2')],'vv_next',keepdims=1)
                n('Equal',[s('tokens3'),s('vv_ids')],'vv_out_eq')
                n('Cast',[s('vv_out_eq')],'vv_out_float',to=onnx.TensorProto.FLOAT)
                n('ReduceMax',[s('vv_out_float'),s('axis2')],'vv_out',keepdims=1)
                n('Mul',[s('vv_out'),s('vv_next')],'vv_out_pair')
                n('Greater',[s('vv_out_pair'),s('zero')],'vv_next_bool')
                n('Where',[s('vv_next_bool'),s('vv_two'),s('context_fade')],'vv_release')
                node.input[1]=s('vv_release')
            if s('wet') in node.input and node.op_type=='Add':
                final=node.output[0];node.output[0]=s('vv_base');nodes.append(node)
                n('Slice',['tokens',s('vv_begin'),s('vv_minusone'),s('next_axis')],'vv_prev_slice')
                n('Pad',[s('vv_prev_slice'),s('vv_prepad'),s('silence_id')],'vv_prev')
                n('Unsqueeze',[s('vv_prev'),s('axis2')],'vv_prev3')
                n('Equal',[s('vv_prev3'),s('vv_ids')],'vv_prev_eq')
                n('Cast',[s('vv_prev_eq')],'vv_prev_float',to=onnx.TensorProto.FLOAT)
                n('ReduceMax',[s('vv_prev_float'),s('axis2')],'vv_prev_vowel',keepdims=1)
                n('Equal',[s('tokens3'),s('vv_ids')],'vv_current_eq')
                n('Cast',[s('vv_current_eq')],'vv_current_float',to=onnx.TensorProto.FLOAT)
                n('ReduceMax',[s('vv_current_float'),s('axis2')],'vv_current',keepdims=1)
                n('Slice',['durations',s('vv_begin'),s('vv_minusone'),s('next_axis')],'vv_prev_durs')
                n('Pad',[s('vv_prev_durs'),s('vv_prepad'),s('vv_zeroi')],'vv_prev_dur2')
                n('Unsqueeze',[s('vv_prev_dur2'),s('axis2')],'vv_prev_dur3')
                n('Cast',[s('vv_prev_dur3')],'vv_prev_dur',to=onnx.TensorProto.FLOAT)
                n('GreaterOrEqual',[s('vv_prev_dur'),s('long')],'vv_prev_long')
                n('Cast',[s('vv_prev_long')],'vv_prev_longfloat',to=onnx.TensorProto.FLOAT)
                n('Mul',[s('vv_prev_vowel'),s('vv_current')],'vv_pair')
                n('Mul',[s('vv_pair'),s('longfloat')],'vv_eligible0')
                n('Mul',[s('vv_eligible0'),s('vv_prev_longfloat')],'vv_eligible')
                n('Div',[s('offset'),s('vv_ten')],'vv_progress')
                n('Sub',[s('one'),s('vv_progress')],'vv_remaining')
                n('Clip',[s('vv_remaining'),s('zero'),s('one')],'vv_ramp')
                n('GreaterOrEqual',[s('offset'),s('zero')],'vv_started')
                n('Cast',[s('vv_started')],'vv_startedfloat',to=onnx.TensorProto.FLOAT)
                n('Mul',[s('vv_ramp'),s('vv_startedfloat')],'vv_window')
                n('Mul',[s('vv_window'),s('vv_eligible')],'vv_weights')
                n('Add',[s('starts'),s('vv_twelve')],'vv_indices0')
                n('Sub',[s('frames'),s('onei')],'vv_last')
                n('Clip',[s('vv_indices0'),s('zeroi'),s('vv_last')],'vv_indices')
                n('Gather',[s('raw2'),s('vv_indices')],'vv_donors',axis=0)
                n('Transpose',[s('vv_weights')],'vv_weights_t',perm=[0,2,1])
                n('MatMul',[s('vv_weights_t'),s('vv_donors')],'vv_selected')
                n('ReduceSum',[s('vv_weights'),s('axis1')],'vv_sum',keepdims=0)
                n('Unsqueeze',[s('vv_sum'),s('axis2')],'vv_weight3')
                n('Sub',[s('one'),s('vv_weight3')],'vv_dry_weight')
                n('Mul',[s('vv_base'),s('vv_dry_weight')],'vv_dry')
                nodes.append(h.make_node('Add',[s('vv_dry'),s('vv_selected')],[final]))
            else:nodes.append(node)
        del m.graph.node[:];m.graph.node.extend(nodes)
    onnx.checker.check_model(m);output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.source,a.output)
