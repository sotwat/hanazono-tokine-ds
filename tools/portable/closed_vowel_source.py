"""Trial: extend the existing WORLD route to sustained upper-mid i/u."""
import argparse
from pathlib import Path
import numpy as np
import onnx
import torch
from onnx import helper as h, numpy_helper as nh
from build_pair import append_model

class Mask(torch.nn.Module):
    def forward(self,tokens,durations,f0):
        starts=durations.cumsum(1)-durations
        t=torch.arange(f0.shape[1])[None,None,:]
        active=((tokens==14)|(tokens==28))&(durations>=20)
        inside=(t>=starts[:,:,None])&(t<(starts+durations)[:,:,None])
        return (inside&active[:,:,None]).any(1).float()

class Weight(torch.nn.Module):
    def forward(self,previous,control,f0):
        midi=69+12*torch.log2(f0.clamp(min=1)/440)
        w=((midi-67)/2).clamp(0,1)
        w=w*w*(3-2*w)*control[:,8].unsqueeze(0)
        return torch.maximum(previous,w)

def build(bank,out):
    out.mkdir(parents=True,exist_ok=False)
    layer=out/'mask.onnx'
    torch.onnx.export(Mask(),(torch.tensor([[1,14,27,28,1]]),torch.tensor([[8,108,7,37,8]]),torch.ones(1,168)*440),layer,input_names=['tokens','durations','f0'],output_names=['mask'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'f0':{1:'F'},'mask':{1:'F'}})
    m=onnx.load(bank/'tokine_ds_clarity_hybrid.tokine.onnx');nodes=[]
    assert not any(o.startswith('cvs_') for n in m.graph.node for o in n.output)
    for n in m.graph.node:
        if 'mel' in n.output:
            previous=n.input[1]
            append_model(nodes,m.graph.initializer,layer,'cvs_mask_',dict(tokens='tokens',durations='durations',f0='f0',mask='cvs_mask'))
            m.graph.initializer.extend([nh.from_array(np.array([2],np.int64),'cvs_axis'),nh.from_array(np.array([0,0,8,0,0,119],np.int64),'cvs_pad'),nh.from_array(np.array(0,np.float32),'cvs_zero')])
            nodes.extend([h.make_node('Unsqueeze',['cvs_mask','cvs_axis'],['cvs_column']),h.make_node('Pad',['cvs_column','cvs_pad','cvs_zero'],['cvs_extra']),h.make_node('Add',[previous,'cvs_extra'],['cvs_control'])]);n.input[1]='cvs_control'
        nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
    layer=out/'weight.onnx'
    torch.onnx.export(Weight(),(torch.zeros(1,32),torch.zeros(32,128),torch.ones(1,32)*440),layer,input_names=['previous','control','f0'],output_names=['weight'],opset_version=18,dynamo=False,dynamic_axes={'previous':{1:'F'},'control':{0:'F'},'f0':{1:'F'},'weight':{1:'F'}})
    m=onnx.load(next((bank/'dsvocoder').glob('*.onnx')));nodes=[]
    for n in m.graph.node:
        if 'blend_where_1' in n.output:
            n.output[0]='cvs_previous';nodes.append(n)
            append_model(nodes,m.graph.initializer,layer,'cvs_weight_',dict(previous='cvs_previous',control='rp_control',f0='rp_normalized',weight='blend_where_1'))
        else:nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.bank,a.out)
