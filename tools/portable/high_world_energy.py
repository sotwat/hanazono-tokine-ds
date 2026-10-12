"""Retain donor energy when high-pitch WORLD harmonic sampling loses power."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
import torch
import torch.nn.functional as F
from build_pair import append_model

class Energy(torch.nn.Module):
    def forward(self,world,donor,mask):
        # A centered 60 ms window avoids cycle-by-cycle gain modulation.
        def power(x):return F.avg_pool1d(F.pad(x.square().unsqueeze(1),(1323,1323),mode='replicate'),2647,stride=1)[:,0]
        ratio=torch.sqrt((power(donor)+1e-8)/(power(world)+1e-8)).clamp(1,8)
        position=torch.arange(world.shape[1],device=world.device)/512
        lo=position.long().clamp(max=mask.shape[1]-1)
        active=mask[:,lo].unsqueeze(1)
        forward=F.avg_pool1d(F.pad(active,(881,0)),882,stride=1)[:,0]
        backward=F.avg_pool1d(F.pad(active,(0,881)),882,stride=1)[:,0]
        weight=active[:,0]*torch.minimum(forward,backward)
        gain=1+weight*(ratio-1)
        return world*gain

def build(acoustic,vocoder,out):
    out.mkdir(parents=True,exist_ok=True)
    m=onnx.load(acoustic);nodes=[]
    def c(name,value,dtype=np.int64):m.graph.initializer.append(nh.from_array(np.array(value,dtype),'he_'+name))
    def n(op,ins,key,**kw):nodes.append(h.make_node(op,ins,['he_'+key],**kw))
    for k,v in [('phones',[2,3,9,14,18,19,20,28]),('axis0',[0]),('axis1',[1]),('axis2',[2]),('pad',[0,0,4,0,0,123])]:c(k,v)
    c('zero',0.,np.float32)
    n('Unsqueeze',['rp_tokens','he_axis1'],'tokens2');n('Equal',['he_tokens2','he_phones'],'matches');n('Cast',['he_matches'],'matchesf',to=1);n('ReduceMax',['he_matchesf','he_axis1'],'phone_mask',keepdims=0)
    n('Unsqueeze',['he_phone_mask','he_axis1'],'phone_mask2');n('Cast',['rp_inside'],'insidef',to=1);n('Mul',['he_insidef','he_phone_mask2'],'active');n('ReduceSum',['he_active','he_axis0'],'frame_mask',keepdims=0);n('Unsqueeze',['he_frame_mask','he_axis0'],'mask2');n('Unsqueeze',['he_mask2','he_axis2'],'column');n('Pad',['he_column','he_pad','he_zero'],'extra');n('Add',['rp_control','he_extra'],'control')
    old=list(m.graph.node);idx=next(i for i,nod in enumerate(old) if 'mel' in nod.output);old[idx].input[1]='he_control';del m.graph.node[:];m.graph.node.extend(old[:idx]+nodes+old[idx:]);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
    x=torch.randn(1,32*512);mask=torch.ones(1,32)
    torch.onnx.export(Energy(),(x,x,mask),out/'energy.onnx',input_names=['world','donor','mask'],output_names=['corrected'],opset_version=18,dynamo=False,dynamic_axes={'world':{1:'samples'},'donor':{1:'samples'},'mask':{1:'frames'},'corrected':{1:'samples'}})
    m=onnx.load(vocoder);nodes=list(m.graph.node);idx=next(i for i,nod in enumerate(nodes) if 'world_wave' in nod.output)+1;extra=[h.make_node('Gather',['rp_control','he_channel'],['he_mask_flat'],axis=1),h.make_node('Unsqueeze',['he_mask_flat','rp_axis0'],['he_mask'])];m.graph.initializer.append(nh.from_array(np.array(4,np.int64),'he_channel'));append_model(extra,m.graph.initializer,out/'energy.onnx','he_energy_',{'world':'world_wave','donor':'donor_wave','mask':'he_mask','corrected':'he_corrected'})
    for nod in nodes[idx:]:
        for i,name in enumerate(nod.input):
            if name=='world_wave':nod.input[i]='he_corrected'
    del m.graph.node[:];m.graph.node.extend(nodes[:idx]+extra+nodes[idx:]);onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('acoustic',type=Path);p.add_argument('vocoder',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.acoustic,a.vocoder,a.out)
