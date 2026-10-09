"""Extend stable vowel spectra with smooth phase and onset/release crossfades."""
from pathlib import Path
import argparse,numpy as np,onnx
from onnx import helper as H,numpy_helper as N,TensorProto as T
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
if Path(a.output).exists():raise FileExistsError(a.output)
m=onnx.load(a.source)
for node in m.graph.node:
 for i,v in enumerate(node.output):
  if v=='mel':node.output[i]='su_raw'
def c(k,v,t=np.float32):m.graph.initializer.append(N.from_array(np.array(v,t),'su_'+k))
def n(op,ins,out,**kw):m.graph.node.append(H.make_node(op,ins,['su_'+out],**kw))
for k,v in [('zero',0),('one',1),('long',172),('start',74),('fade',26),('center',54),('radius',20),('omega',2*np.pi/160)]:c(k,v)
for k,v in [('axis1',[1]),('axis2',[2]),('axis01',[0,1]),('csaxis',1),('zeroi',0),('onei',1),('index',1),('vowels',[3,9,14,20,28])]:c(k,v,np.int64)
n('CumSum',['durations','su_csaxis'],'ends');n('Sub',['su_ends','durations'],'starts');n('Shape',['f0'],'shape');n('Gather',['su_shape','su_index'],'frames',axis=0);n('Range',['su_zeroi','su_frames','su_onei'],'range');n('Unsqueeze',['su_range','su_axis01'],'frames3');n('Unsqueeze',['su_starts','su_axis2'],'starts3');n('Sub',['su_frames3','su_starts3'],'offseti');n('Cast',['su_offseti'],'offset',to=T.FLOAT);n('Unsqueeze',['durations','su_axis2'],'duri');n('Cast',['su_duri'],'dur',to=T.FLOAT)
n('Unsqueeze',['tokens','su_axis2'],'tokens3');n('Equal',['su_tokens3','su_vowels'],'match');n('Cast',['su_match'],'matchfloat',to=T.FLOAT);n('ReduceMax',['su_matchfloat'],'vowel',axes=[2],keepdims=1);n('GreaterOrEqual',['su_dur','su_long'],'longbool');n('Cast',['su_longbool'],'longfloat',to=T.FLOAT);n('Mul',['su_vowel','su_longfloat'],'eligible')
n('Sub',['su_offset','su_start'],'elapsed');n('Mul',['su_elapsed','su_omega'],'phase');n('Cos',['su_phase'],'cos');n('Mul',['su_cos','su_radius'],'swing');n('Add',['su_swing','su_center'],'local');n('Cast',['su_starts3'],'startsfloat',to=T.FLOAT);n('Add',['su_local','su_startsfloat'],'mapped')
n('Div',['su_elapsed','su_fade'],'attack');n('Sub',['su_dur','su_offset'],'remaining');n('Div',['su_remaining','su_fade'],'release');n('Min',['su_attack','su_release'],'envelope');n('Clip',['su_envelope','su_zero','su_one'],'mixlinear')
# Smoothstep has zero derivative at both ends.
c('three',3);c('two',2)
n('Mul',['su_mixlinear','su_mixlinear'],'squared');n('Mul',['su_mixlinear','su_two'],'twox');n('Sub',['su_three','su_twox'],'factor');n('Mul',['su_squared','su_factor'],'smooth');n('Mul',['su_smooth','su_eligible'],'weight')
n('Mul',['su_mapped','su_weight'],'weightedindex');n('ReduceSum',['su_weightedindex','su_axis1'],'sumindex',keepdims=0);n('ReduceSum',['su_weight','su_axis1'],'weight2',keepdims=0);n('Max',['su_weight2','su_one'],'denom');n('Div',['su_sumindex','su_denom'],'safeindex')
# Compute an index without weighting; select the sole active vowel per frame.
n('Greater',['su_weight','su_zero'],'active');n('Cast',['su_active'],'activefloat',to=T.FLOAT);n('Mul',['su_mapped','su_activefloat'],'activeindex');n('ReduceSum',['su_activeindex','su_axis1'],'idx',keepdims=0)
n('Floor',['su_idx'],'floor');n('Cast',['su_floor'],'lo',to=T.INT64);n('Add',['su_lo','su_onei'],'hi')
c('axis0',[0],np.int64)
n('Squeeze',['su_raw','su_axis0'],'raw2');n('Gather',['su_raw2','su_lo'],'lowmel',axis=0);n('Gather',['su_raw2','su_hi'],'highmel',axis=0);n('Sub',['su_idx','su_floor'],'fraction');n('Unsqueeze',['su_fraction','su_axis2'],'frac3');n('Sub',['su_highmel','su_lowmel'],'delta');n('Mul',['su_delta','su_frac3'],'interpdelta');n('Add',['su_lowmel','su_interpdelta'],'loop');n('Unsqueeze',['su_weight2','su_axis2'],'weight3');n('Sub',['su_loop','su_raw'],'difference');n('Mul',['su_difference','su_weight3'],'wet');m.graph.node.append(H.make_node('Add',['su_raw','su_wet'],['mel']))
onnx.checker.check_model(m);onnx.save(m,a.output)
