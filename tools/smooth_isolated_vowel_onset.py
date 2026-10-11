"""Smooth brief vowel attacks after silence without mixing different pitches."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    if output.exists(): raise FileExistsError(output)
    m = onnx.load(source)
    nodes = list(m.graph.node)
    for prefix in ['original_', 'donor_']:
        s = lambda k: prefix+'su_'+k
        g = lambda k: prefix+'io_'+k
        def c(k,v,dtype=np.float32):
            m.graph.initializer.append(nh.from_array(np.array(v,dtype),g(k)))
        def n(op,ins,out,**kw): nodes.append(h.make_node(op,ins,[g(out)],**kw))
        for k,v in [('zero',0.),('one',1.),('half',.5),('three',3.),('ratio',2**(.3/12)),('epsilon',1e-12)]: c(k,v)
        for k,v in [('zeroi',0),('onei',1),('axis0',[0]),('axis1',[1]),('axis2',[2]),('prevpad',[1,0]),('begin',[0]),('end',[-1]),('vowels',[3,9,14,20,28])]: c(k,v,np.int64)
        final=prefix+'mel'
        for node in nodes:
            if final in node.output:node.output[0]=g('base')
        n('Squeeze',['tokens',g('axis0')],'tokens1')
        n('Slice',[g('tokens1'),g('begin'),g('end')],'prev0')
        n('Pad',[g('prev0'),g('prevpad'),g('zeroi')],'prev')
        n('Equal',[g('prev'),g('onei')],'after_sp')
        n('Unsqueeze',[g('tokens1'),g('axis1')],'tokens2')
        n('Equal',[g('tokens2'),g('vowels')],'is_vowel0')
        n('Cast',[g('is_vowel0')],'is_vowelf',to=onnx.TensorProto.FLOAT)
        n('ReduceMax',[g('is_vowelf'),g('axis1')],'is_vowel1',keepdims=0)
        n('Greater',[g('is_vowel1'),g('zero')],'is_vowel')
        n('And',[g('after_sp'),g('is_vowel')],'eligible')
        n('Cast',[g('eligible')],'eligiblef',to=onnx.TensorProto.FLOAT)
        n('Unsqueeze',[g('eligiblef'),g('axis0')],'eligible2')
        n('Unsqueeze',[g('eligible2'),g('axis2')],'eligible3')
        n('GreaterOrEqual',[s('offset'),g('zero')],'started')
        n('Less',[s('offset'),g('three')],'early')
        n('And',[g('started'),g('early')],'early_started')
        n('Less',[s('offset'),s('dur')],'before_end')
        n('And',[g('early_started'),g('before_end')],'earlymask')
        n('Cast',[g('earlymask')],'earlyf',to=onnx.TensorProto.FLOAT)
        n('Mul',[g('earlyf'),g('eligible3')],'per_token')
        n('ReduceSum',[g('per_token'),g('axis1')],'mask2',keepdims=0)
        n('Unsqueeze',[g('mask2'),g('axis2')],'mask')
        n('Squeeze',[g('base'),g('axis0')],'base2')
        pitch='source_f0' if prefix=='donor_' else 'f0'
        n('Squeeze',[pitch,g('axis0')],'pitch')
        n('Exp',[g('base')],'amp')
        n('ReduceMean',[g('amp'),g('axis2')],'energy',keepdims=1)
        n('Max',[g('energy'),g('epsilon')],'safeenergy')
        n('Log',[g('safeenergy')],'logenergy')
        n('Sub',[g('base'),g('logenergy')],'shape')
        n('Squeeze',[g('shape'),g('axis0')],'shape2')
        weighted=[g('shape')];weights=[g('one')]
        for tag,op in [('left','Sub'),('right','Add')]:
            n(op,[s('range'),g('onei')],tag+'0')
            n('Clip',[g(tag+'0'),g('zeroi'),s('vv_last')],tag+'idx')
            n('Gather',[g('shape2'),g(tag+'idx')],tag+'shape2',axis=0)
            n('Unsqueeze',[g(tag+'shape2'),g('axis0')],tag+'shape')
            n('Gather',[g('pitch'),g(tag+'idx')],tag+'pitch',axis=0)
            n('Max',[g('pitch'),g(tag+'pitch')],tag+'max')
            n('Min',[g('pitch'),g(tag+'pitch')],tag+'min')
            n('Max',[g(tag+'min'),g('one')],tag+'safe')
            n('Div',[g(tag+'max'),g(tag+'safe')],tag+'ratio')
            n('LessOrEqual',[g(tag+'ratio'),g('ratio')],tag+'compatible')
            # A frame belongs to the current phone only if it is in its span.
            n('Sub',[s('dur'),s('offset')],tag+'remaining')
            if tag=='left':n('GreaterOrEqual',[s('offset'),g('one')],tag+'inside0')
            else:n('Greater',[g(tag+'remaining'),g('one')],tag+'inside0')
            n('Cast',[g(tag+'inside0')],tag+'insidef',to=onnx.TensorProto.FLOAT)
            n('Mul',[g(tag+'insidef'),g('per_token')],tag+'inside_token')
            n('ReduceSum',[g(tag+'inside_token'),g('axis1')],tag+'inside2',keepdims=0)
            n('Greater',[g(tag+'inside2'),g('zero')],tag+'inside')
            n('And',[g(tag+'inside'),g(tag+'compatible')],tag+'ok')
            n('Cast',[g(tag+'ok')],tag+'weight2',to=onnx.TensorProto.FLOAT)
            n('Unsqueeze',[g(tag+'weight2'),g('axis2')],tag+'weight')
            n('Mul',[g(tag+'shape'),g(tag+'weight')],tag+'wet')
            weighted.append(g(tag+'wet'));weights.append(g(tag+'weight'))
        n('Sum',weighted,'sum');n('Sum',weights,'weights')
        n('Div',[g('sum'),g('weights')],'smooth')
        n('Sub',[g('smooth'),g('shape')],'difference')
        n('Mul',[g('mask'),g('half')],'mixweight')
        n('Mul',[g('difference'),g('mixweight')],'wet')
        n('Add',[g('shape'),g('wet')],'mixedshape')
        n('Exp',[g('mixedshape')],'mixedamp')
        n('ReduceMean',[g('mixedamp'),g('axis2')],'mixedenergy',keepdims=1)
        n('Max',[g('mixedenergy'),g('epsilon')],'mixedsafe')
        n('Log',[g('mixedsafe')],'mixedlog')
        n('Sub',[g('mixedshape'),g('mixedlog')],'normalized')
        n('Add',[g('normalized'),g('logenergy')],'processed')
        n('Greater',[g('mask'),g('zero')],'attack_active')
        n('Greater',[g('weights'),g('one')],'has_neighbor')
        n('And',[g('attack_active'),g('has_neighbor')],'active')
        nodes.append(h.make_node('Where',[g('active'),g('processed'),g('base')],[final]))
    def deps(node):
        result=set(node.input)
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.GRAPH:
                graph=attr.g;local={v.name for v in graph.input}|{v.name for v in graph.initializer}|{v for item in graph.node for v in item.output}
                result.update(set().union(*(deps(item) for item in graph.node))-local)
        return result
    known={v.name for v in m.graph.input}|{v.name for v in m.graph.initializer}|{''};ordered=[]
    while nodes:
        ready=[node for node in nodes if deps(node)<=known]
        if not ready: raise ValueError('Unresolved graph dependencies')
        for node in ready:ordered.append(node);known.update(node.output);nodes.remove(node)
    del m.graph.node[:];m.graph.node.extend(ordered)
    onnx.checker.check_model(m);onnx.save(m,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.source,a.output)
