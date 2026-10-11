"""Avoid averaging or holding donor harmonics across incompatible pitches."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh


def update(source, output):
    if output.exists():raise FileExistsError(output)
    m=onnx.load(source)
    prefixes=['original_','donor_']
    nodes=list(m.graph.node)
    for prefix in prefixes:
        s=lambda k:prefix+'su_'+k
        p=lambda k:prefix+'ps_'+k
        def c(k,v,dtype=np.float32):m.graph.initializer.append(nh.from_array(np.array(v,dtype),p(k)))
        def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,[p(out)],**kw))
        item=next(v for v in m.graph.initializer if v.name==s('adaptive_fractions'))
        item.CopyFrom(nh.from_array(np.linspace(.2,.85,32,dtype=np.float32),s('adaptive_fractions')))
        c('one',1.);c('zero',0.);c('ratio30',2**(.3/12));c('ratio20',2**(.2/12));c('ratio70',2**(.7/12));c('span',2**(.7/12)-2**(.2/12));c('axis3',[3],np.int64);c('axis2',[2],np.int64);c('middle',2,np.int64)
        pitch='source_f0' if prefix=='donor_' else 'f0'
        n('Squeeze',[pitch,s('axis0')],'pitch1')
        n('Gather',[p('pitch1'),s('donor_indices')],'candidate_pitch',axis=0)
        n('ReduceMax',[p('candidate_pitch'),p('axis3')],'candidate_max',keepdims=0)
        n('ReduceMin',[p('candidate_pitch'),p('axis3')],'candidate_min',keepdims=0)
        n('Max',[p('candidate_min'),p('one')],'candidate_min_safe')
        n('Div',[p('candidate_max'),p('candidate_min_safe')],'candidate_ratio')
        n('LessOrEqual',[p('candidate_ratio'),p('ratio30')],'stable')
        n('Cast',[p('stable')],'stablef',to=onnx.TensorProto.FLOAT)
        n('ReduceMax',[p('stablef'),p('axis2')],'any_stable',keepdims=1)
        n('Greater',[p('any_stable'),p('zero')],'has_stable')
        n('Mul',[s('donor_scores'),p('stablef')],'stable_scores')
        n('Where',[p('has_stable'),p('stable_scores'),s('donor_scores')],'scores')
        for node in nodes:
            if s('donor_best') in node.output:node.input[0]=p('scores')
        n('Gather',[p('pitch1'),s('hold_idx')],'hold_pitch',axis=0)
        n('ReduceMax',[p('hold_pitch'),p('axis2')],'hold_max',keepdims=0)
        n('ReduceMin',[p('hold_pitch'),p('axis2')],'hold_min',keepdims=0)
        n('Max',[p('hold_min'),p('one')],'hold_min_safe')
        n('Div',[p('hold_max'),p('hold_min_safe')],'hold_ratio')
        n('LessOrEqual',[p('hold_ratio'),p('ratio30')],'hold_stable2')
        n('Unsqueeze',[p('hold_stable2'),s('axis2')],'hold_stable')
        n('Gather',[s('hold_samples'),p('middle')],'center_mel',axis=2)
        n('Where',[p('hold_stable'),s('hold_mean'),p('center_mel')],'safe_mean')
        for node in nodes:
            if s('loop') in node.output:node.input[1]=p('safe_mean')
        n('Gather',[p('hold_pitch'),p('middle')],'reference2',axis=2)
        n('Unsqueeze',[p('reference2'),s('axis2')],'reference')
        n('Unsqueeze',[pitch,s('axis1')],'target')
        n('Max',[p('reference'),p('target')],'max')
        n('Min',[p('reference'),p('target')],'min')
        n('Max',[p('min'),p('one')],'min_safe')
        n('Div',[p('max'),p('min_safe')],'ratio')
        n('Sub',[p('ratio70'),p('ratio')],'remaining')
        n('Div',[p('remaining'),p('span')],'linear0')
        n('Clip',[p('linear0'),p('zero'),p('one')],'linear')
        n('Mul',[p('linear'),p('linear')],'sq')
        n('Mul',[p('linear'),s('two')],'twice')
        n('Sub',[s('three'),p('twice')],'factor')
        n('Mul',[p('sq'),p('factor')],'smooth')
        for node in nodes:
            if s('weight') in node.output:node.output[0]=p('baseweight')
        nodes.append(h.make_node('Mul',[p('baseweight'),p('smooth')],[s('weight')]))
    def deps(node):
        result=set(node.input)
        for a in node.attribute:
            if a.type==onnx.AttributeProto.GRAPH:
                g=a.g;local={v.name for v in g.input}|{v.name for v in g.initializer}|{x for n in g.node for x in n.output}
                result.update(set().union(*(deps(n) for n in g.node))-local)
        return result
    known={v.name for v in m.graph.input}|{v.name for v in m.graph.initializer}|{''};ordered=[]
    while nodes:
        ready=[n for n in nodes if deps(n)<=known]
        if not ready:raise ValueError('Unresolved graph dependencies')
        for n in ready:ordered.append(n);known.update(n.output);nodes.remove(n)
    del m.graph.node[:];m.graph.node.extend(ordered);onnx.checker.check_model(m);onnx.save(m,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.source,a.output)
