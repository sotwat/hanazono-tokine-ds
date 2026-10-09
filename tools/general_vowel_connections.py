"""Merge continuous identical vowels and repair duration-adaptive vowel onsets."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    if output.exists():
        raise FileExistsError(output)
    m = onnx.load(source)
    nodes = []
    def c(k, v, dtype=np.int64):
        m.graph.initializer.append(nh.from_array(np.array(v, dtype), 'gv_'+k))
    def n(op, inputs, out, **kw):
        nodes.append(h.make_node(op, inputs, ['gv_'+out], **kw))
    for k,v in [('axis0',[0]),('axis1',[1]),('axis2',[2]),('zero',0),('one',1),('begin',[0]),('end',[-1]),('pad',[1,0]),('vowels',[3,9,14,20,28])]:c(k,v)
    n('Squeeze',['tokens','gv_axis0'],'t')
    n('Squeeze',['durations','gv_axis0'],'d')
    n('Slice',['gv_t','gv_begin','gv_end'],'prev0')
    n('Pad',['gv_prev0','gv_pad','gv_zero'],'prev')
    n('Unsqueeze',['gv_t','gv_axis1'],'t2')
    n('Equal',['gv_t2','gv_vowels'],'matches')
    n('Cast',['gv_matches'],'matchf',to=onnx.TensorProto.FLOAT)
    n('ReduceMax',['gv_matchf','gv_axis1'],'vowelf',keepdims=0)
    c('zerof',0.,np.float32)
    n('Greater',['gv_vowelf','gv_zerof'],'vowel')
    n('Equal',['gv_t','gv_prev'],'equal')
    n('And',['gv_equal','gv_vowel'],'repeat')
    n('Not',['gv_repeat'],'new')
    n('Cast',['gv_new'],'newi',to=onnx.TensorProto.INT64)
    n('CumSum',['gv_newi','gv_zero'],'groups1')
    n('Sub',['gv_groups1','gv_one'],'groups')
    n('NonZero',['gv_new'],'indices0')
    n('Squeeze',['gv_indices0','gv_axis0'],'indices')
    # Bound merged context so repeated vowels cannot become an arbitrarily long
    # acoustic token whose late frames have already lost voicing.
    c('context_frames',128)
    n('CumSum',['gv_d','gv_zero'],'ends')
    n('Sub',['gv_ends','gv_d'],'starts')
    n('Gather',['gv_starts','gv_indices'],'run_starts',axis=0)
    n('Gather',['gv_run_starts','gv_groups'],'each_run_start',axis=0)
    n('Sub',['gv_starts','gv_each_run_start'],'run_elapsed')
    n('Div',['gv_run_elapsed','gv_context_frames'],'chunk')
    n('Slice',['gv_chunk','gv_begin','gv_end'],'prev_chunk0')
    n('Pad',['gv_prev_chunk0','gv_pad','gv_zero'],'prev_chunk')
    n('Equal',['gv_chunk','gv_prev_chunk'],'same_chunk')
    n('Not',['gv_same_chunk'],'new_chunk')
    n('Or',['gv_new','gv_new_chunk'],'bounded_new')
    n('Cast',['gv_bounded_new'],'bounded_newi',to=onnx.TensorProto.INT64)
    n('CumSum',['gv_bounded_newi','gv_zero'],'bounded_groups1')
    n('Sub',['gv_bounded_groups1','gv_one'],'bounded_groups')
    n('NonZero',['gv_bounded_new'],'bounded_indices0')
    n('Squeeze',['gv_bounded_indices0','gv_axis0'],'bounded_indices')
    n('Gather',['gv_t','gv_bounded_indices'],'merged_t',axis=0)
    n('Shape',['gv_merged_t'],'shape')
    nodes.append(h.make_node('ConstantOfShape',['gv_shape'],['gv_zeros'],value=nh.from_array(np.array([0],np.int64))))
    n('ScatterElements',['gv_zeros','gv_bounded_groups','gv_d'],'merged_d',axis=0,reduction='add')
    n('Unsqueeze',['gv_merged_t','gv_axis0'],'tokens')
    n('Unsqueeze',['gv_merged_d','gv_axis0'],'durations')
    for node in m.graph.node:
        for i,x in enumerate(node.input):
            if x in ('tokens','durations') and any('/fs2/' in y for y in node.output):
                node.input[i]='gv_'+x
    original=list(m.graph.node)
    prefixes=['original_','donor_']
    for prefix in prefixes:
        s=lambda k:prefix+'su_'+k
        # Replace the old fixed-frame onset and long-vowel-only morph together.
        for node in original:
            if s('vm_base') in node.output:
                node.op_type='Identity';del node.input[:];node.input.append(s('vv_base'));del node.attribute[:]
            if s('fc_base') in node.output:
                node.op_type='Identity';del node.input[:];node.input.append(s('vm_base'));del node.attribute[:]
    nodes.extend(original)
    for prefix in prefixes:
        s=lambda k:prefix+'su_'+k
        g=lambda k:prefix+'gv_'+k
        def cn(k,v,dtype=np.float32):m.graph.initializer.append(nh.from_array(np.array(v,dtype),g(k)))
        def gn(op,ins,out,**kw):nodes.append(h.make_node(op,ins,[g(out)],**kw))
        final=prefix+'mel'
        for node in nodes:
            if final in node.output:node.output[0]=g('base_raw')
        for k,v in [('four',4.),('six',6.),('two',2.),('quarter',.25),('half',.5),('twentyfour',24.),('fractions',[.25,.35,.45,.55,.65])]:cn(k,v)
        gn('GreaterOrEqual',[s('dur'),g('six')],'current_long')
        gn('GreaterOrEqual',[s('vv_prev_dur'),g('four')],'previous_long')
        gn('And',[g('current_long'),g('previous_long')],'lengths')
        gn('Greater',[s('vv_pair'),s('zero')],'pair')
        gn('And',[g('pair'),g('lengths')],'eligible')
        gn('Cast',[g('eligible')],'eligiblef',to=onnx.TensorProto.FLOAT)
        # All donor indices stay inside this vowel, including brief notes.
        gn('Mul',[s('dur'),g('fractions')],'local')
        gn('Add',[s('startsfloat'),g('local')],'global')
        gn('Cast',[g('global')],'idx0',to=onnx.TensorProto.INT64)
        gn('Clip',[g('idx0'),s('zeroi'),s('vv_last')],'idx')
        gn('Gather',[s('raw2'),g('idx')],'samples',axis=0)
        gn('Exp',[g('samples')],'amps')
        cn('axis3',[3],np.int64)
        gn('ReduceMean',[g('amps'),g('axis3')],'energy',keepdims=0)
        gn('ArgMax',[g('energy')],'best',axis=2,keepdims=1)
        cn('range',np.arange(5),np.int64)
        gn('Equal',[g('best'),g('range')],'selection')
        gn('Cast',[g('selection')],'selectionf',to=onnx.TensorProto.FLOAT)
        gn('Unsqueeze',[g('selectionf'),g('axis3')],'selection4')
        gn('Mul',[g('samples'),g('selection4')],'selected4')
        gn('ReduceSum',[g('selected4'),s('axis2')],'localanchor',keepdims=0)
        # Reuse voiced material only within the same uninterrupted vowel run,
        # at a nearby pitch. The vocoder still receives the original F0.
        gn('ReduceMax',[g('energy'),s('axis2')],'localenergy',keepdims=0)
        gn('Squeeze',[g('localenergy'),s('axis0')],'energy1')
        gn('Unsqueeze',['gv_groups',s('axis1')],'groupcol')
        gn('Equal',[g('groupcol'),'gv_groups'],'samegroup')
        cn('middle',2,np.int64)
        gn('Gather',[g('idx'),g('middle')],'mididx',axis=2)
        pitch='source_f0' if prefix=='donor_' else 'f0'
        gn('Squeeze',[pitch,s('axis0')],'f01')
        gn('Gather',[g('f01'),g('mididx')],'midpitch',axis=0)
        gn('Squeeze',[g('midpitch'),s('axis0')],'pitch1')
        gn('Max',[g('pitch1'),s('one')],'pitchsafe')
        gn('Unsqueeze',[g('pitchsafe'),s('axis1')],'pitchcol')
        gn('Max',[g('pitchcol'),g('pitchsafe')],'pitchmax')
        gn('Min',[g('pitchcol'),g('pitchsafe')],'pitchmin')
        gn('Div',[g('pitchmax'),g('pitchmin')],'pitchratio')
        cn('near_ratio',2**(2/12))
        gn('LessOrEqual',[g('pitchratio'),g('near_ratio')],'near')
        gn('And',[g('near'),g('samegroup')],'donorok')
        gn('Cast',[g('donorok')],'donorokf',to=onnx.TensorProto.FLOAT)
        gn('Mul',[g('donorokf'),g('energy1')],'donorscores')
        gn('ArgMax',[g('donorscores')],'bestnote',axis=1,keepdims=0)
        gn('Gather',[g('localanchor'),g('bestnote')],'runanchor',axis=1)
        gn('Gather',[g('energy1'),g('bestnote')],'runenergy1',axis=0)
        gn('Unsqueeze',[g('runenergy1'),s('axis0')],'runenergy')
        cn('weak_ratio',1.5)
        gn('Mul',[g('localenergy'),g('weak_ratio')],'weak_threshold')
        gn('Greater',[g('runenergy'),g('weak_threshold')],'weak2')
        gn('Unsqueeze',[g('weak2'),s('axis2')],'weak')
        gn('Where',[g('weak'),g('runanchor'),g('localanchor')],'anchor')
        gn('Div',[s('offset'),g('six')],'weak_attack')
        cn('release',8.)
        gn('Div',[s('remaining'),g('release')],'weak_release')
        gn('Min',[g('weak_attack'),g('weak_release')],'weak_env0')
        gn('Clip',[g('weak_env0'),s('zero'),s('one')],'weak_env')
        gn('Mul',[g('weak_env'),g('weak_env')],'weak_sq')
        gn('Mul',[g('weak_env'),s('two')],'weak_twice')
        gn('Sub',[s('three'),g('weak_twice')],'weak_factor')
        gn('Mul',[g('weak_sq'),g('weak_factor')],'weak_smooth')
        gn('Cast',[g('weak')],'weakf',to=onnx.TensorProto.FLOAT)
        gn('Mul',[g('weak_smooth'),g('weakf')],'weak_weight')
        gn('Transpose',[g('weak_weight')],'weak_weight_t',perm=[0,2,1])
        gn('MatMul',[g('weak_weight_t'),g('anchor')],'weak_mel')
        gn('ReduceSum',[g('weak_weight'),s('axis1')],'weak_weight2',keepdims=0)
        gn('Unsqueeze',[g('weak_weight2'),s('axis2')],'weak_weight3')
        gn('Sub',[s('one'),g('weak_weight3')],'weak_dryweight')
        gn('Mul',[g('base_raw'),g('weak_dryweight')],'weak_dry')
        gn('Add',[g('weak_dry'),g('weak_mel')],'base')
        gn('Mul',[s('vv_prev_dur'),g('quarter')],'left0')
        gn('Min',[g('left0'),g('four')],'left')
        gn('Neg',[g('left')],'leftneg')
        gn('Cast',[g('left')],'lefti',to=onnx.TensorProto.INT64)
        gn('Squeeze',[g('lefti'),s('axis2')],'lefti2')
        gn('Sub',[s('starts'),g('lefti2')],'leftidx0')
        gn('Clip',[g('leftidx0'),s('zeroi'),s('vv_last')],'leftidx')
        gn('Squeeze',[g('base'),s('axis0')],'base2')
        gn('Gather',[g('base2'),g('leftidx')],'leftmel',axis=0)
        gn('Add',[s('offset'),g('left')],'elapsed')
        gn('Add',[g('left'),g('two')],'width')
        gn('Div',[g('elapsed'),g('width')],'t0')
        gn('Clip',[g('t0'),s('zero'),s('one')],'t')
        gn('Mul',[g('t'),g('t')],'sq')
        gn('Mul',[g('t'),s('two')],'twice')
        gn('Sub',[s('three'),g('twice')],'factor')
        gn('Mul',[g('sq'),g('factor')],'w')
        gn('Mul',[s('dur'),g('half')],'end0')
        gn('Min',[g('end0'),g('twentyfour')],'end')
        gn('Mul',[g('end'),g('half')],'hold')
        gn('Sub',[g('end'),s('offset')],'remain')
        gn('Sub',[g('end'),g('hold')],'fade_raw')
        gn('Max',[g('fade_raw'),s('one')],'fade')
        gn('Div',[g('remain'),g('fade')],'fade0')
        gn('Clip',[g('fade0'),s('zero'),s('one')],'fade1')
        gn('Mul',[g('fade1'),g('fade1')],'fsq')
        gn('Mul',[g('fade1'),s('two')],'ftwo')
        gn('Sub',[s('three'),g('ftwo')],'ffactor')
        gn('Mul',[g('fsq'),g('ffactor')],'fadeweight')
        gn('Greater',[s('offset'),g('leftneg')],'after')
        gn('Cast',[g('after')],'afterf',to=onnx.TensorProto.FLOAT)
        gn('Mul',[g('afterf'),g('eligiblef')],'mask')
        gn('Mul',[g('mask'),g('fadeweight')],'weight')
        gn('Mul',[g('weight'),g('w')],'rw')
        gn('Sub',[g('weight'),g('rw')],'lw')
        for side,value in [('r','anchor'),('l','leftmel')]:
            gn('Transpose',[g(side+'w')],side+'wt',perm=[0,2,1])
            gn('MatMul',[g(side+'wt'),g(value)],side+'mel')
        gn('Add',[g('rmel'),g('lmel')],'mix')
        gn('ReduceSum',[g('weight'),s('axis1')],'weight2',keepdims=0)
        gn('Unsqueeze',[g('weight2'),s('axis2')],'weight3')
        gn('Sub',[s('one'),g('weight3')],'dryweight')
        gn('Mul',[g('base'),g('dryweight')],'dry')
        nodes.append(h.make_node('Add',[g('dry'),g('mix')],[final]))
    def dependencies(node):
        result=set(node.input)
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.GRAPH:
                graph=attr.g
                local={v.name for v in graph.input}|{v.name for v in graph.initializer}
                local.update(x for n in graph.node for x in n.output)
                result.update(set().union(*(dependencies(n) for n in graph.node))-local)
        return result
    known={v.name for v in m.graph.input}|{v.name for v in m.graph.initializer}|{''}
    ordered=[]
    while nodes:
        ready=[node for node in nodes if dependencies(node)<=known]
        if not ready:raise ValueError('Unresolved graph dependencies')
        for node in ready:
            ordered.append(node);known.update(node.output);nodes.remove(node)
    del m.graph.node[:];m.graph.node.extend(ordered)
    onnx.checker.check_model(m);output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();update(a.source,a.output)
