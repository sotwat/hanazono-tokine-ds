"""Add an experimental long-vowel mel energy floor; original model is retained."""
import argparse
from pathlib import Path
import numpy as np,onnx
from onnx import helper as H,numpy_helper as N,TensorProto as T
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
if Path(a.output).exists():raise FileExistsError(a.output)
m=onnx.load(a.source)
for n in m.graph.node:
    for i,v in enumerate(n.output):
        if v=='mel':n.output[i]='lv_raw'
def c(name,value,dtype=np.float32):m.graph.initializer.append(N.from_array(np.array(value,dtype),'lv_'+name))
def n(op,ins,out,**kw):m.graph.node.append(H.make_node(op,ins,['lv_'+out],**kw))
c('csaxis',1,np.int64);c('axis1',[1],np.int64);c('axis2',[2],np.int64);c('axis0',0,np.int64);c('zeroi',0,np.int64);c('onei',1,np.int64);c('frameindex',1,np.int64);c('long',129,np.int64);c('anchorlo',17,np.int64);c('anchorhi',62,np.int64);c('applystart',69,np.int64);c('vowels',[3,9,14,20,28],np.int64);c('zero',0);c('one',1);c('drop',.5);c('cap',4.)
n('CumSum',['durations','lv_csaxis'],'ends');n('Sub',['lv_ends','durations'],'starts');n('Shape',['f0'],'shape');n('Gather',['lv_shape','lv_frameindex'],'frames',axis=0);n('Range',['lv_zeroi','lv_frames','lv_onei'],'range')
# Frame axis [1,1,F], token axis [1,T,1].
c('axis01',[0,1],np.int64)
n('Unsqueeze',['lv_range','lv_axis01'],'frame3');n('Unsqueeze',['lv_starts','lv_axis2'],'starts3');n('Sub',['lv_frame3','lv_starts3'],'offset');n('Unsqueeze',['durations','lv_axis2'],'dur3')
n('GreaterOrEqual',['lv_offset','lv_zeroi'],'after');n('Less',['lv_offset','lv_dur3'],'before');n('And',['lv_after','lv_before'],'inside')
n('Unsqueeze',['tokens','lv_axis2'],'token3');n('Equal',['lv_token3','lv_vowels'],'vowelmatch');n('Cast',['lv_vowelmatch'],'vowelnum',to=T.FLOAT);n('ReduceMax',['lv_vowelnum'],'vowelmax',axes=[2],keepdims=1);n('Greater',['lv_vowelmax','lv_zero'],'vowel');n('GreaterOrEqual',['lv_dur3','lv_long'],'longbool');n('And',['lv_vowel','lv_longbool'],'eligible');n('And',['lv_inside','lv_eligible'],'target')
n('ReduceMax',['lv_raw'],'energy',axes=[2],keepdims=0);n('Unsqueeze',['lv_energy','lv_axis1'],'energy3');n('GreaterOrEqual',['lv_offset','lv_anchorlo'],'anchorafter');n('Less',['lv_offset','lv_anchorhi'],'anchorbefore');n('And',['lv_anchorafter','lv_anchorbefore'],'anchorbool');n('Cast',['lv_anchorbool'],'anchor',to=T.FLOAT);n('Mul',['lv_energy3','lv_anchor'],'weighted');n('ReduceSum',['lv_weighted','lv_axis2'],'sum',keepdims=1);n('ReduceSum',['lv_anchor','lv_axis2'],'count',keepdims=1);n('Max',['lv_count','lv_one'],'countsafe');n('Div',['lv_sum','lv_countsafe'],'reference');n('Sub',['lv_reference','lv_drop'],'floor');n('Sub',['lv_floor','lv_energy3'],'difference');n('Clip',['lv_difference','lv_zero','lv_cap'],'gain')
n('GreaterOrEqual',['lv_offset','lv_applystart'],'late');n('And',['lv_target','lv_late'],'latebool');n('Cast',['lv_latebool'],'latefloat',to=T.FLOAT);n('Mul',['lv_gain','lv_latefloat'],'selected');n('ReduceSum',['lv_selected','lv_axis1'],'gain2',keepdims=0);n('Greater',['f0','lv_zero'],'voiced');n('Where',['lv_voiced','lv_gain2','lv_zero'],'voicedgain');n('Unsqueeze',['lv_voicedgain','lv_axis2'],'gain3')
m.graph.node.append(H.make_node('Add',['lv_raw','lv_gain3'],['mel']))
onnx.checker.check_model(m);onnx.save(m,a.output)
