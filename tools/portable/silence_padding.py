from pathlib import Path
import sys,torch,onnx,numpy as np
from onnx import helper as h,numpy_helper as nh
from build_pair import append_model
import argparse
parser=argparse.ArgumentParser();parser.add_argument('acoustic',type=Path);parser.add_argument('vocoder',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
class SilenceGate(torch.nn.Module):
 def forward(self,wave,control):
  p=torch.arange(wave.shape[1]).unsqueeze(0);i=(p//512).clamp(max=control.shape[1]-1);flags=control[:,:,0].gather(1,i);start=control[:,:,1].gather(1,i)*512;end=control[:,:,2].gather(1,i)*512;duration=(end-start).clamp(min=1)
  release=(1-(p-start)/torch.minimum(duration*.5,torch.full_like(duration,1102.5))).clamp(0,1);anticipation=(1-(end-p)/torch.minimum(duration*.15,torch.full_like(duration,220.5))).clamp(0,1)
  release=release*release*(3-2*release);anticipation=anticipation*anticipation*(3-2*anticipation)
  release=torch.where((flags.long()&1)>0,release,torch.zeros_like(release));anticipation=torch.where((flags.long()&2)>0,anticipation,torch.zeros_like(anticipation));gain=torch.maximum(release,anticipation)
  return torch.where(flags>=0,wave*gain,wave)
filter_path=root/'silence-filter.onnx'
torch.onnx.export(SilenceGate(),(torch.randn(1,512*32),torch.zeros(1,32,128)),str(filter_path),input_names=['wave','control'],output_names=['clean_wave'],opset_version=18,dynamo=False,dynamic_axes={'wave':{1:'samples'},'control':{1:'frames'},'clean_wave':{1:'samples'}})
m=onnx.load(args.acoustic);nodes=list(m.graph.node)
def c(k,v,dtype=np.int64):m.graph.initializer.append(nh.from_array(np.array(v,dtype),'sg_'+k))
def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,['sg_'+out],**kw))
for k,v in [('axis0',[0]),('axis1',[1]),('zero',0),('one',1),('begin',[0]),('end',[-1]),('nextbegin',[1]),('nextend',[9223372036854775807]),('prevpad',[1,0]),('nextpad',[0,1]),('pads',[0,0,0,0,0,125])]:c(k,v)
c('zerof',0.,np.float32);c('minusonef',-1.,np.float32);c('twof',2.,np.float32)
n('Squeeze',['tokens','sg_axis0'],'tokens');n('Squeeze',['durations','sg_axis0'],'durations');n('CumSum',['sg_durations','sg_zero'],'ends');n('Sub',['sg_ends','sg_durations'],'starts')
n('Slice',['sg_tokens','sg_begin','sg_end'],'prev0');n('Pad',['sg_prev0','sg_prevpad','sg_one'],'prev');n('Slice',['sg_tokens','sg_nextbegin','sg_nextend'],'next0');n('Pad',['sg_next0','sg_nextpad','sg_one'],'next')
for x in ['tokens','prev','next']:n('Equal',['sg_'+x,'sg_one'],x+'sp')
for x in ['prev','next']:n('Not',['sg_'+x+'sp'],x+'voiced');n('Cast',['sg_'+x+'voiced'],x+'f',to=onnx.TensorProto.FLOAT)
n('Mul',['sg_nextf','sg_twof'],'next2f');n('Add',['sg_prevf','sg_next2f'],'bits');n('Where',['sg_tokenssp','sg_bits','sg_minusonef'],'flags')
n('Unsqueeze',['sg_starts','sg_axis1'],'starts2');n('Unsqueeze',['sg_ends','sg_axis1'],'ends2');n('GreaterOrEqual',['original_su_range','sg_starts2'],'started');n('Less',['original_su_range','sg_ends2'],'unfinished');n('And',['sg_started','sg_unfinished'],'inside');n('Cast',['sg_inside'],'insidef',to=onnx.TensorProto.FLOAT)
for src,dst in [('flags','flag'),('starts','start'),('ends','end')]:
 n('Cast',['sg_'+src],dst+'f',to=onnx.TensorProto.FLOAT);n('Unsqueeze',['sg_'+dst+'f','sg_axis1'],dst+'2');n('Mul',['sg_insidef','sg_'+dst+'2'],dst+'all');n('ReduceSum',['sg_'+dst+'all','sg_axis0'],dst+'frame',keepdims=0);n('Unsqueeze',['sg_'+dst+'frame','sg_axis1'],dst+'col')
n('Concat',['sg_flagcol','sg_startcol','sg_endcol'],'control0',axis=1);n('Unsqueeze',['sg_control0','sg_axis0'],'control3');n('Pad',['sg_control3','sg_pads','sg_zerof'],'control')
for node in nodes:
 if 'mel' in node.output:node.output[list(node.output).index('mel')]='paired_mel'
nodes.append(h.make_node('Concat',['paired_mel','sg_control'],['mel'],axis=0));m.graph.output[0].type.tensor_type.shape.dim[0].dim_value=3;del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,root/'silence-acoustic.onnx')
m=onnx.load(args.vocoder);nodes=list(m.graph.node);initializers=list(m.graph.initializer)
for node in nodes:
 if 'waveform' in node.output:node.output[list(node.output).index('waveform')]='silence_before'
initializers.append(nh.from_array(np.array([2],np.int64),'silence_control_index'));nodes.append(h.make_node('Gather',['mel','silence_control_index'],['silence_control'],axis=0));append_model(nodes,initializers,filter_path,'silence_',{'wave':'silence_before','control':'silence_control','clean_wave':'waveform'});m.graph.input[0].type.tensor_type.shape.dim[0].dim_value=3;del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.initializer[:];m.graph.initializer.extend(initializers);onnx.checker.check_model(m);onnx.save(m,root/'silence-vocoder.onnx')
print('built')
