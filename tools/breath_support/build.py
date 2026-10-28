"""Add a separately learned AP decoder without updating singing weights."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
import torch
from train import BreathDecoder
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'portable'))
from build_pair import append_model


class Acoustic(torch.nn.Module):
    def __init__(self,weights):
        super().__init__();self.decoder=BreathDecoder();self.decoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    def forward(self,mel,tokens,durations,f0):
        starts=durations.cumsum(1)-durations
        t=torch.arange(f0.shape[1])[None,None,:]
        inside=(t>=starts[:,:,None])&(t<(starts+durations)[:,:,None])
        active=inside&(tokens[:,:,None]==32)
        mask=active.float().sum(1)[0]
        start=(active*starts[:,:,None]).sum(1)[0]
        dur=(active*durations[:,:,None]).sum(1)[0].clamp(min=1)
        position=((t[0,0]-start).float()/(dur-1).clamp(min=1))
        predicted=self.decoder(position,dur.float()*512/44100)
        original=torch.where(mask[:,None]>0,predicted,mel[0])
        donor=torch.where(mask[:,None]>0,predicted,mel[1])
        channel=torch.arange(128)[None,:]
        # Column 11 marks AP; column 3 is the vocoder F0, zero for an inhale.
        control=torch.where((mask[:,None]>0)&(channel==3),torch.zeros_like(mel[2]),mel[2])
        control=torch.where(channel==11,mask[:,None],control)
        return torch.stack([original,donor,control],0)


class Output(torch.nn.Module):
    def forward(self,previous,raw,mel):
        frame=torch.arange(previous.shape[1])//512
        mask=mel[2,:,11];idx=frame.clamp(max=mask.shape[0]-1)
        active=mask[idx]>0
        # Five ms edge fades avoid discontinuities at the new sound's boundaries.
        width=220
        sm=active.float()[None,None,:]
        smooth=torch.nn.functional.avg_pool1d(torch.nn.functional.pad(sm,(width,width)),2*width+1,stride=1)[0,0]
        gain=(smooth*2-1).clamp(0,1)
        return torch.where(active[None],raw*gain[None],previous)


def build(bank,weights,out):
    out.mkdir(parents=True,exist_ok=True)
    m=onnx.load(bank/'tokine_ds_clarity_hybrid.tokine.onnx')
    def rename(graph):
        for n in graph.node:
            for i,x in enumerate(n.input):
                if x=='tokens':n.input[i]='br_safe_tokens'
            for attr in n.attribute:
                if attr.type==onnx.AttributeProto.GRAPH:rename(attr.g)
    rename(m.graph)
    for n in m.graph.node:
        for i,x in enumerate(n.output):
            if x=='mel':n.output[i]='br_before_mel'
    prefix=[h.make_node('Equal',['tokens','br_ap_id'],['br_ap_tokens']),h.make_node('Where',['br_ap_tokens','br_sp_id','tokens'],['br_safe_tokens'])]
    m.graph.initializer.extend([nh.from_array(np.array(32,np.int64),'br_ap_id'),nh.from_array(np.array(1,np.int64),'br_sp_id')])
    layer=out/'acoustic-layer.onnx';model=Acoustic(weights).eval()
    torch.onnx.export(model,(torch.zeros(3,80,128),torch.tensor([[1,32,3]]),torch.tensor([[20,30,30]]),torch.ones(1,80)*440),layer,input_names=['mel','tokens','durations','f0'],output_names=['out'],opset_version=18,dynamo=False,dynamic_axes={'mel':{1:'F'},'tokens':{1:'N'},'durations':{1:'N'},'f0':{1:'F'},'out':{1:'F'}})
    nodes=prefix+list(m.graph.node);append_model(nodes,m.graph.initializer,layer,'br_ac_',dict(mel='br_before_mel',tokens='tokens',durations='durations',f0='f0',out='mel'))
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
    m=onnx.load(next((bank/'dsvocoder').glob('*.onnx')))
    for n in m.graph.node:
        for i,x in enumerate(n.output):
            if x=='waveform':n.output[i]='br_before_waveform'
    layer=out/'output-layer.onnx'
    torch.onnx.export(Output(),(torch.zeros(1,40960),torch.zeros(1,40960),torch.zeros(3,80,128)),layer,input_names=['previous','raw','mel'],output_names=['wave'],opset_version=18,dynamo=False,dynamic_axes={'previous':{1:'S'},'raw':{1:'S'},'mel':{1:'F'},'wave':{1:'S'}})
    nodes=list(m.graph.node);append_model(nodes,m.graph.initializer,layer,'br_voc_',dict(previous='br_before_waveform',raw='baseline',mel='mel',wave='waveform'))
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('weights',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.bank,a.weights,a.output)
