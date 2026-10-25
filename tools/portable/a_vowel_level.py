"""Small waveform-only trim for independent /a/ after sustained vowels."""
import argparse
from pathlib import Path
import onnx
import torch
from build_pair import append_model
from a_vowel_rearticulation import targets

class Mark(torch.nn.Module):
    def forward(self, tokens, durations, control):
        start = durations.cumsum(1)-durations
        time = torch.arange(control.shape[1])[None,None,:]
        active = targets(tokens,durations)[:,:,None] & (time>=start[:,:,None]) & (time<(start+durations)[:,:,None])
        flag = active.any(1).float()[:,:,None]
        channel = (torch.arange(128)==7).float()[None,None,:]
        return control + flag*channel

class Trim(torch.nn.Module):
    def forward(self, wave, control):
        sample = torch.arange(wave.shape[1])
        frame = (sample//512).clamp(max=control.shape[0]-1)
        distance = torch.minimum(sample-control[frame,1]*512,control[frame,2]*512-1-sample)
        weight = .5-.5*torch.cos(torch.pi*(distance/882).clamp(0,1))
        weight = torch.where(control[frame,7]>.5,weight,torch.zeros_like(weight))
        gain = torch.exp(weight*(-1.25*2.302585092994046/20))
        return torch.where(weight[None,:]>0,wave*gain[None,:],wave)

def retune_vocoder(source, output):
    """Update an existing -0.5 dB trim without rebuilding other vocoder patches."""
    import numpy as np
    model = onnx.load(source)
    node = next(n for n in model.graph.node if 'avl_/Constant_19_output_0' in n.output)
    value = node.attribute[0]
    if not np.isclose(onnx.numpy_helper.to_array(value.t), -.5*np.log(10)/20):
        raise ValueError('Expected the previous -0.5 dB trim')
    value.t.CopyFrom(onnx.numpy_helper.from_array(np.array(-1.25*np.log(10)/20,dtype=np.float32)))
    onnx.checker.check_model(model)
    onnx.save(model, output)

def build(acoustic,vocoder,out):
    out.mkdir(parents=True,exist_ok=False)
    layer=out/'marker.onnx'
    torch.onnx.export(Mark(),(torch.tensor([[14,3,16]]),torch.tensor([[40,30,5]]),torch.zeros(1,75,128)),layer,input_names=['tokens','durations','control'],output_names=['marked'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'control':{1:'F'},'marked':{1:'F'}})
    model=onnx.load(acoustic);nodes=[]
    assert not any(x.name.startswith(('aa_','avl_')) for x in model.graph.initializer)
    for node in model.graph.node:
        if 'mel' in node.output:
            assert node.input[1]=='vb_control'
            append_model(nodes,model.graph.initializer,layer,'avl_',dict(tokens='tokens',durations='durations',control='vb_control',marked='avl_control'))
            node.input[1]='avl_control'
        nodes.append(node)
    del model.graph.node[:];model.graph.node.extend(nodes);onnx.checker.check_model(model);onnx.save(model,out/'acoustic.onnx')
    layer=out/'trim.onnx'
    torch.onnx.export(Trim(),(torch.randn(1,75*512),torch.zeros(75,128)),layer,input_names=['wave','control'],output_names=['trimmed'],opset_version=18,dynamo=False,dynamic_axes={'wave':{1:'S'},'control':{0:'F'},'trimmed':{1:'S'}})
    model=onnx.load(vocoder);nodes=list(model.graph.node)
    final=next(n for n in nodes if 'waveform' in n.output);final.output[list(final.output).index('waveform')]='avl_before'
    append_model(nodes,model.graph.initializer,layer,'avl_',dict(wave='avl_before',control='rp_control',trimmed='waveform'))
    del model.graph.node[:];model.graph.node.extend(nodes);onnx.checker.check_model(model);onnx.save(model,out/'vocoder.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('acoustic',type=Path);p.add_argument('vocoder',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.acoustic,a.vocoder,a.output)
