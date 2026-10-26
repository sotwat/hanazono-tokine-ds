"""Context-limited WORLD corrections for sustained ni and tsu."""
from pathlib import Path
import argparse
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import helper as h, numpy_helper as nh
from build_pair import append_model

class Context(torch.nn.Module):
    def forward(self,tokens,durations,f0):
        starts=durations.cumsum(1)-durations
        t=torch.arange(f0.shape[1])[None,None,:]
        inside=(t>=starts[:,:,None])&(t<(starts+durations)[:,:,None])
        previous=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),1)
        midi=69+12*torch.log2(f0.clamp(min=1)/440)
        pitch=(inside*midi[:,None,:]).sum(2)/durations.clamp(min=1)
        ni=((tokens==14)&(previous==19)&(durations>=20)).float()
        tsu=((tokens==28)&(previous==27)&(durations>=20)).float()
        nw=(pitch-73).clamp(0,1);tw=(pitch-68).clamp(0,1)
        nw=nw*nw*(3-2*nw)*ni;tw=tw*tw*(3-2*tw)*tsu
        return torch.stack(((inside*nw[:,:,None]).sum(1),(inside*tw[:,:,None]).sum(1)),2)

class Correction(torch.nn.Module):
    def __init__(self,spectrum=False):
        super().__init__();self.spectrum=spectrum
    def forward(self,value,control):
        t=torch.arange(control.shape[0])
        distance=torch.minimum(t-control[:,1],control[:,2]-1-t).clamp(min=0)
        w=(distance/4).clamp(0,1);w=w*w*(3-2*w)*control[:,10 if self.spectrum else 9]
        if self.spectrum:
            log=value.clamp(min=1e-12).log()
            smooth=F.avg_pool1d(F.pad(log.transpose(1,2),(3,3),mode='replicate'),7,stride=1).transpose(1,2)
            corrected=torch.exp(log+(smooth-log)*w[None,:,None]*.85)
        else:
            frequency=torch.arange(value.shape[2])*22050/1024
            band=((12000-frequency)/5000).clamp(0,1)
            corrected=value*(1-.7*w[None,:,None]*band)
        return torch.where(w[None,:,None]>0,corrected,value)

def build(bank,out):
    out.mkdir(parents=True,exist_ok=False)
    layer=out/'context.onnx'
    torch.onnx.export(Context(),(torch.tensor([[1,19,14,27,28,1]]),torch.tensor([[8,6,40,7,37,8]]),torch.ones(1,106)*600),layer,input_names=['tokens','durations','f0'],output_names=['mask'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'f0':{1:'F'},'mask':{1:'F'}})
    m=onnx.load(bank/'tokine_ds_clarity_hybrid.tokine.onnx');nodes=[]
    for n in m.graph.node:
        if 'mel' in n.output:
            append_model(nodes,m.graph.initializer,layer,'nts_context_',dict(tokens='tokens',durations='durations',f0='f0',mask='nts_mask'))
            m.graph.initializer.extend([nh.from_array(np.array([0,0,9,0,0,117],np.int64),'nts_pad'),nh.from_array(np.array(0,np.float32),'nts_zero')])
            nodes.extend([h.make_node('Pad',['nts_mask','nts_pad','nts_zero'],['nts_extra']),h.make_node('Add',[n.input[1],'nts_extra'],['nts_control'])]);n.input[1]='nts_control'
        nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
    m=onnx.load(next((bank/'dsvocoder').glob('*.onnx')))
    for spectrum,target in [(False,'world_clamp_13'),(True,'world_clamp_14')]:
        layer=out/('spectrum.onnx' if spectrum else 'ap.onnx');prefix='nts_sp_' if spectrum else 'nts_ap_'
        torch.onnx.export(Correction(spectrum),(torch.ones(1,32,1025)*.5,torch.zeros(32,128)),layer,input_names=['value','control'],output_names=['clean'],opset_version=18,dynamo=False,dynamic_axes={'value':{1:'F'},'control':{0:'F'},'clean':{1:'F'}})
        nodes=[]
        for n in m.graph.node:
            if target in n.output:
                append_model(nodes,m.graph.initializer,layer,prefix,dict(value=n.input[0],control='rp_control',clean=prefix+'clean'));n.input[0]=prefix+'clean'
            nodes.append(n)
        del m.graph.node[:];m.graph.node.extend(nodes)
    onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.bank,a.output)
