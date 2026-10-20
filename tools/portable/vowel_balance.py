"""Bounded, constant-per-phone balancing of voiced vowels."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import torch
from build_pair import append_model

class VowelBalance(torch.nn.Module):
    def forward(self, wave, control):
        frames = control.shape[0]
        frame = torch.arange(frames, device=wave.device)
        starts = control[:, 1].long().clamp(min=0, max=frames-1)
        ends = control[:, 2].long().clamp(min=0, max=frames)
        duration = (ends-starts).clamp(min=1)
        # Whole-phone central power sets a single gain; no tail envelope tracking.
        power = wave[0].reshape(-1,512).square().mean(dim=1)
        integral = torch.cat((torch.zeros_like(power[:1]), power.cumsum(0)))
        lo = (starts + duration//5).clamp(max=frames-1)
        hi = (ends - duration//5).clamp(min=1, max=frames)
        rms = ((integral[hi]-integral[lo]).clamp(min=0)/(hi-lo).clamp(min=1)+1e-10).sqrt()
        sample = torch.arange(wave.shape[1],device=wave.device)
        fi = (sample//512).clamp(max=frames-1)
        lag = (44100/control[fi,3].clamp(min=22)).round().long().clamp(min=1,max=2048)
        delayed = wave[0][(sample-lag).clamp(min=0)]
        cross = (wave[0]*delayed).reshape(-1,512).mean(1)
        delayed_power = delayed.square().reshape(-1,512).mean(1)
        ci = torch.cat((torch.zeros_like(cross[:1]),cross.cumsum(0)))
        di = torch.cat((torch.zeros_like(power[:1]),delayed_power.cumsum(0)))
        correlation = (ci[hi]-ci[lo])/((integral[hi]-integral[lo]).clamp(min=1e-10)*(di[hi]-di[lo]).clamp(min=1e-10)).sqrt()
        voiced = (control[:,6] > .5) & (duration >= 6) & (rms > .008) & (correlation > .25)
        unique = voiced & (frame == starts)
        ordered = torch.sort(torch.where(unique,rms,torch.full_like(rms,1e6)))[0]
        count = unique.long().sum()
        reference = ordered[(count//2)]
        deficit = 20*torch.log10(reference.clamp(min=1e-5)/rms.clamp(min=1e-5))
        boost_db = (torch.sign(deficit)*(deficit.abs()-1.5).clamp(min=0)*.65).clamp(min=-4,max=4)
        boost_db = torch.where(voiced & (count>=2),boost_db,torch.zeros_like(boost_db))
        distance = torch.minimum(sample-starts[fi]*512,ends[fi]*512-1-sample).float()
        fade = (distance/882).clamp(min=0,max=1)
        weight = .5-.5*torch.cos(torch.pi*fade)
        gain = torch.exp(boost_db[fi]*(np.log(10)/20)*weight)
        return wave*gain.unsqueeze(0)

def build(acoustic,vocoder,out):
    out.mkdir(parents=True,exist_ok=True)
    m=onnx.load(acoustic)
    assert not any(x.name.startswith('vb_') for x in m.graph.initializer)
    for name,value in [('vowels',[3,9,14,20,28]),('axis1',[1]),('axis0',[0]),('axis2',[2]),('pad',[0,0,6,0,0,121])]:
        m.graph.initializer.append(nh.from_array(np.array(value,np.int64),'vb_'+name))
    m.graph.initializer.append(nh.from_array(np.array(0,np.float32),'vb_zero'))
    nodes=[]
    def n(op,ins,name,**kw): nodes.append(h.make_node(op,ins,['vb_'+name],**kw))
    n('Equal',['rp_tokens2','vb_vowels'],'match');n('Cast',['vb_match'],'matchf',to=1)
    n('ReduceMax',['vb_matchf','vb_axis1'],'vowel',keepdims=0)
    n('Unsqueeze',['vb_vowel','vb_axis1'],'vowel2');n('Mul',['he_insidef','vb_vowel2'],'inside')
    n('ReduceSum',['vb_inside','vb_axis0'],'frame',keepdims=0)
    n('Unsqueeze',['vb_frame','vb_axis0'],'frame2');n('Unsqueeze',['vb_frame2','vb_axis2'],'column')
    n('Pad',['vb_column','vb_pad','vb_zero'],'extra');n('Add',['he_control','vb_extra'],'control')
    old=list(m.graph.node);idx=next(i for i,x in enumerate(old) if 'mel' in x.output)
    assert old[idx].input[1]=='he_control';old[idx].input[1]='vb_control'
    del m.graph.node[:];m.graph.node.extend(old[:idx]+nodes+old[idx:]);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
    layer=out/'vowel-balance-layer.onnx'
    torch.onnx.export(VowelBalance(),(torch.randn(1,32*512)*.05,torch.zeros(32,128)),layer,input_names=['wave','control'],output_names=['balanced'],opset_version=18,dynamo=False,dynamic_axes={'wave':{1:'samples'},'control':{0:'frames'},'balanced':{1:'samples'}})
    m=onnx.load(vocoder);old=list(m.graph.node);idx=next(i for i,x in enumerate(old) if 'pb_input_wave' in x.output)+1
    # Insert before phrase RMS measurement so the existing peak guard remains last.
    incoming='pb_input_wave';nodes=[]
    append_model(nodes,m.graph.initializer,layer,'vb_',{'wave':incoming,'control':'rp_control','balanced':'vb_balanced'})
    for x in old[idx:]:
        for i,v in enumerate(x.input):
            if v==incoming:x.input[i]='vb_balanced'
    del m.graph.node[:];m.graph.node.extend(old[:idx]+nodes+old[idx:]);onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('acoustic',type=Path);p.add_argument('vocoder',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.acoustic,a.vocoder,a.output)
