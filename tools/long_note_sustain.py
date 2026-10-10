"""Sustain to connected phone boundaries and use short class-aware joins."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh


def update(source,output):
    if output.exists():
        raise FileExistsError(output)
    m=onnx.load(source)
    nodes=list(m.graph.node)
    for prefix in ['original_','donor_']:
        s=lambda k:prefix+'su_'+k
        g=lambda k:prefix+'lt_'+k
        def c(k,v,dtype=np.float32):m.graph.initializer.append(nh.from_array(np.array(v,dtype),g(k)))
        def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,[g(out)],**kw))
        for key,value in [('start',16.),('fade',10.)]:
            item=next(i for i in m.graph.initializer if i.name==s(key));item.CopyFrom(nh.from_array(np.array(value,np.float32),s(key)))
        c('one',1.);c('two',2.);c('min',40.);c('fricatives',[6,10,12,13,15,24,25,27,31],np.int64)
        for node in nodes:
            if prefix+'gv_base_raw' in node.output:
                node.op_type='Identity';del node.input[:];node.input.append(s('fc_base'));del node.attribute[:]
            if s('release') in node.output or prefix+'gv_weak_release' in node.output:
                node.input[1]=g('release_frames')
        n('Where',[s('ends_phrase3'),s('endfade'),g('one')],'release_frames')
        final=prefix+'mel'
        for node in nodes:
            if final in node.output:node.output[0]=g('base')
        n('Equal',[s('vv_prev3'),s('vowels')],'prev_match')
        n('Cast',[g('prev_match')],'prev_matchf',to=onnx.TensorProto.FLOAT)
        n('ReduceMax',[g('prev_matchf'),s('axis2')],'prev_sustained',keepdims=1)
        n('GreaterOrEqual',[s('vv_prev_dur'),g('min')],'prev_long')
        n('Cast',[g('prev_long')],'prev_longf',to=onnx.TensorProto.FLOAT)
        n('Mul',[g('prev_sustained'),g('prev_longf')],'prev_eligible')
        n('Sub',[s('one'),s('vv_current')],'not_vowel')
        n('Equal',[s('tokens3'),s('silence_id')],'silence')
        n('Not',[g('silence')],'sound')
        n('GreaterOrEqual',[s('dur'),g('two')],'current_long')
        n('And',[g('sound'),g('current_long')],'current_eligible')
        n('Cast',[g('current_eligible')],'currentf',to=onnx.TensorProto.FLOAT)
        n('Mul',[g('prev_eligible'),g('not_vowel')],'eligible0')
        n('Mul',[g('eligible0'),g('currentf')],'eligible')
        n('Equal',[s('tokens3'),g('fricatives')],'fricative')
        n('Cast',[g('fricative')],'fricativef',to=onnx.TensorProto.FLOAT)
        n('ReduceMax',[g('fricativef'),s('axis2')],'fricative2',keepdims=1)
        n('Add',[g('one'),g('fricative2')],'left')
        n('Sub',[s('dur'),s('one')],'available')
        n('Div',[g('available'),g('two')],'half')
        n('Floor',[g('half')],'half_floor')
        n('Min',[g('left'),g('half_floor')],'right0');n('Max',[g('right0'),s('zero')],'right')
        n('Cast',[g('left')],'lefti',to=onnx.TensorProto.INT64)
        n('Cast',[g('right')],'righti',to=onnx.TensorProto.INT64)
        n('Squeeze',[g('lefti'),s('axis2')],'lefti2');n('Squeeze',[g('righti'),s('axis2')],'righti2')
        n('Sub',[s('starts'),g('lefti2')],'leftidx0');n('Add',[s('starts'),g('righti2')],'rightidx0')
        for side in ['left','right']:n('Clip',[g(side+'idx0'),s('zeroi'),s('vv_last')],side+'idx')
        n('Squeeze',[g('base'),s('axis0')],'base2')
        for side in ['left','right']:n('Gather',[g('base2'),g(side+'idx')],side+'mel',axis=0)
        n('Add',[s('offset'),g('left')],'elapsed');n('Add',[g('left'),g('right')],'width')
        n('Div',[g('elapsed'),g('width')],'t0');n('Clip',[g('t0'),s('zero'),s('one')],'t')
        n('Mul',[g('t'),g('t')],'square');n('Mul',[g('t'),s('two')],'twice');n('Sub',[s('three'),g('twice')],'factor');n('Mul',[g('square'),g('factor')],'smooth')
        n('Neg',[g('left')],'negleft');n('Greater',[s('offset'),g('negleft')],'after');n('Less',[s('offset'),g('right')],'before');n('And',[g('after'),g('before')],'inside')
        n('Cast',[g('inside')],'insidef',to=onnx.TensorProto.FLOAT);n('Mul',[g('insidef'),g('eligible')],'mask')
        n('Mul',[g('mask'),g('smooth')],'rightw');n('Sub',[g('mask'),g('rightw')],'leftw')
        for side in ['left','right']:
            n('Transpose',[g(side+'w')],side+'wt',perm=[0,2,1]);n('MatMul',[g(side+'wt'),g(side+'mel')],side+'mix')
        n('Add',[g('leftmix'),g('rightmix')],'mix');n('ReduceSum',[g('mask'),s('axis1')],'mask2',keepdims=0);n('Unsqueeze',[g('mask2'),s('axis2')],'mask3')
        n('Greater',[g('mask3'),s('zero')],'active');nodes.append(h.make_node('Where',[g('active'),g('mix'),g('base')],[final]))
    def dependencies(node):
        out=set(node.input)
        for a in node.attribute:
            if a.type==onnx.AttributeProto.GRAPH:
                local={v.name for v in a.g.input}|{v.name for v in a.g.initializer}|{x for n in a.g.node for x in n.output}
                out.update(set().union(*(dependencies(n) for n in a.g.node))-local)
        return out
    known={v.name for v in m.graph.input}|{v.name for v in m.graph.initializer}|{''};ordered=[]
    while nodes:
        ready=[n for n in nodes if dependencies(n)<=known]
        if not ready:raise ValueError('Unresolved graph dependencies')
        for node in ready:ordered.append(node);known.update(node.output);nodes.remove(node)
    del m.graph.node[:];m.graph.node.extend(ordered)
    onnx.checker.check_model(m);onnx.save(m,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.source,a.output)
