"""Use the next vowel's onset F0 for consonants following silence."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

p=argparse.ArgumentParser();p.add_argument('acoustic',type=Path);p.add_argument('vocoder',type=Path);p.add_argument('output',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
m=onnx.load(args.acoustic);prefix=[]
def c(name,value,dtype=np.int64):m.graph.initializer.append(nh.from_array(np.array(value,dtype),'rp_'+name))
def n(op,ins,name,**kw):prefix.append(h.make_node(op,ins,['rp_'+name],**kw))
for name,value in [('axis0',[0]),('axis1',[1]),('axis2',[2]),('zero',0),('one',1),('minusone',-1),('unvoiced',[4,5,6,7,8,10,11,12,13,15,16,17,18,19,21,22,23,24,25,26,27,29,30,31]),('vowels',[3,9,14,20,28]),('begin',[0]),('end',[1]),('first3',[3]),('control_index',[2]),('f0_pad',[0,0,3,0,0,124])]:c(name,value)
c('zerof',0.,np.float32)
n('Squeeze',['tokens','rp_axis0'],'tokens');n('Squeeze',['durations','rp_axis0'],'durations');n('Shape',['rp_tokens'],'shape');n('Squeeze',['rp_shape','rp_axis0'],'count');n('Range',['rp_zero','rp_count','rp_one'],'index');n('Sub',['rp_count','rp_one'],'last');n('CumSum',['rp_durations','rp_zero'],'ends');n('Sub',['rp_ends','rp_durations'],'starts')
n('Unsqueeze',['rp_tokens','rp_axis1'],'tokens2');n('Equal',['rp_tokens2','rp_unvoiced'],'matches');n('Cast',['rp_matches'],'matchesf',to=onnx.TensorProto.FLOAT);n('ReduceMax',['rp_matchesf','rp_axis1'],'anyf',keepdims=0);n('Greater',['rp_anyf','rp_zerof'],'unvoiced_mask');n('Not',['rp_unvoiced_mask'],'other')
n('Unsqueeze',['rp_index','rp_axis1'],'rows');n('Unsqueeze',['rp_index','rp_axis0'],'columns');n('Unsqueeze',['rp_other','rp_axis0'],'other_columns')
n('Less',['rp_columns','rp_rows'],'before');n('And',['rp_before','rp_other_columns'],'previous_candidates');n('Where',['rp_previous_candidates','rp_columns','rp_minusone'],'previous_indices');n('ReduceMax',['rp_previous_indices','rp_axis1'],'previous0',keepdims=0);n('Clip',['rp_previous0','rp_zero','rp_last'],'previous');n('Gather',['rp_tokens','rp_previous'],'previous_tokens',axis=0);n('Equal',['rp_previous_tokens','rp_one'],'after_silence')
n('Greater',['rp_columns','rp_rows'],'after');n('And',['rp_after','rp_other_columns'],'next_candidates');n('Where',['rp_next_candidates','rp_columns','rp_count'],'next_indices');n('ReduceMin',['rp_next_indices','rp_axis1'],'next0',keepdims=0);n('Clip',['rp_next0','rp_zero','rp_last'],'next');n('Gather',['rp_tokens','rp_next'],'next_tokens',axis=0);n('Unsqueeze',['rp_next_tokens','rp_axis1'],'next_tokens2');n('Equal',['rp_next_tokens2','rp_vowels'],'vowel_matches');n('Cast',['rp_vowel_matches'],'vowel_matchesf',to=onnx.TensorProto.FLOAT);n('ReduceMax',['rp_vowel_matchesf','rp_axis1'],'vowelf',keepdims=0);n('Greater',['rp_vowelf','rp_zerof'],'next_vowel');n('And',['rp_unvoiced_mask','rp_after_silence'],'eligible0');n('And',['rp_eligible0','rp_next_vowel'],'eligible')
n('Squeeze',['f0','rp_axis0'],'input_f0');n('Shape',['rp_input_f0'],'frame_shape');n('Squeeze',['rp_frame_shape','rp_axis0'],'frames');n('Sub',['rp_frames','rp_one'],'last_frame');n('Gather',['rp_starts','rp_next'],'reference0',axis=0);n('Clip',['rp_reference0','rp_zero','rp_last_frame'],'reference');n('Gather',['rp_input_f0','rp_reference'],'reference_f0',axis=0)
n('Range',['rp_zero','rp_frames','rp_one'],'frame_index');n('Unsqueeze',['rp_starts','rp_axis1'],'starts2');n('Unsqueeze',['rp_ends','rp_axis1'],'ends2');n('GreaterOrEqual',['rp_frame_index','rp_starts2'],'started');n('Less',['rp_frame_index','rp_ends2'],'unfinished');n('And',['rp_started','rp_unfinished'],'inside');n('Unsqueeze',['rp_eligible','rp_axis1'],'eligible2');n('And',['rp_inside','rp_eligible2'],'active');n('Cast',['rp_active'],'activef',to=onnx.TensorProto.FLOAT);n('ReduceSum',['rp_activef','rp_axis0'],'active_sum',keepdims=0);n('Greater',['rp_active_sum','rp_zerof'],'active_frame');n('Unsqueeze',['rp_reference_f0','rp_axis1'],'reference_f02');n('Mul',['rp_activef','rp_reference_f02'],'references');n('ReduceSum',['rp_references','rp_axis0'],'reference_frame',keepdims=0);n('Where',['rp_active_frame','rp_reference_frame','rp_input_f0'],'normalized0');n('Unsqueeze',['rp_normalized0','rp_axis0'],'normalized')
original=list(m.graph.node)
for node in original:
 for i,x in enumerate(node.input):
  if x=='f0':node.input[i]='rp_normalized'
# Control channel 3 carries the same F0 to both neural and WORLD vocoders.
n('Unsqueeze',['rp_normalized','rp_axis2'],'f0_column');n('Pad',['rp_f0_column','rp_f0_pad','rp_zerof'],'extra_control')
for node in original:
 if 'mel' in node.output:
  node.input[:]=['paired_mel','rp_control']
  break
suffix=[h.make_node('Add',['sg_control','rp_extra_control'],['rp_control'])]
# The control must be computed before the final mel concatenation.
concat=next(i for i,node in enumerate(original) if 'mel' in node.output)
del m.graph.node[:];m.graph.node.extend(prefix+original[:concat]+suffix+original[concat:]);onnx.checker.check_model(m);onnx.save(m,args.output/'acoustic.onnx')
m=onnx.load(args.vocoder)
for node in m.graph.node:
 for i,x in enumerate(node.input):
  if x=='f0':node.input[i]='rp_normalized'
m.graph.initializer.append(nh.from_array(np.array(2,np.int64),'rp_control_plane'));m.graph.initializer.append(nh.from_array(np.array(3,np.int64),'rp_f0_channel'))
start=[h.make_node('Gather',['mel','rp_control_plane'],['rp_control'],axis=0),h.make_node('Gather',['rp_control','rp_f0_channel'],['rp_f0_flat'],axis=1),h.make_node('Unsqueeze',['rp_f0_flat','rp_axis0'],['rp_normalized'])];m.graph.initializer.append(nh.from_array(np.array([0],np.int64),'rp_axis0'));old=list(m.graph.node);del m.graph.node[:];m.graph.node.extend(start+old);onnx.checker.check_model(m);onnx.save(m,args.output/'vocoder.onnx')
print('built rest-consonant F0 conditioning')
