"""Slow, bounded vowel sustain leveling; sample phase and phone boundaries stay intact."""
from pathlib import Path
import argparse
import numpy as np
import onnx
import torch
from build_pair import append_model

class SustainEnvelope(torch.nn.Module):
    def forward(self, wave, control):
        frames = control.shape[0]
        frame = torch.arange(frames, device=wave.device)
        start = control[:, 1].long().clamp(0, frames-1)
        end = control[:, 2].long().clamp(1, frames)
        duration = (end-start).clamp(min=1)
        power = wave[0].reshape(-1, 512).square().mean(1)
        integral = torch.cat((torch.zeros_like(power[:1]), power.cumsum(0)))
        # Windows never borrow energy from another phoneme.
        left = torch.maximum(frame-5, start)
        right = torch.minimum(frame+6, end)
        envelope = ((integral[right]-integral[left]).clamp(min=0)/(right-left).clamp(min=1)).clamp(min=1e-9)
        lo = start + duration//5
        hi = end - duration//5
        reference = ((integral[hi]-integral[lo]).clamp(min=0)/(hi-lo).clamp(min=1)).clamp(min=1e-9)
        db = (10*torch.log10(reference/envelope)).clamp(-6, 3)
        eligible = (control[:, 6]>.5) & (duration>=36) & (reference>.008**2)
        db = torch.where(eligible, db, torch.zeros_like(db))
        prefix = torch.cat((torch.zeros_like(db[:1]), db.cumsum(0)))
        l = torch.maximum(frame-4,start)
        r = torch.minimum(frame+5,end)
        db = (prefix[r]-prefix[l])/(r-l).clamp(min=1)
        sample = torch.arange(wave.shape[1], device=wave.device)
        x = (sample.float()/512-.5).clamp(min=0,max=frames-1)
        a=x.long();b=(a+1).clamp(max=frames-1);fraction=x-a
        level=db[a]*(1-fraction)+db[b]*fraction
        fi=(sample//512).clamp(max=frames-1)
        # Preserve the initial articulation and final release/next-phone transition.
        distance=torch.minimum(sample-start[fi]*512, end[fi]*512-1-sample).float()
        ramp=((distance-2646)/3528).clamp(0,1)
        weight=ramp*ramp*(3-2*ramp)
        level=torch.where(eligible[fi],level*weight,torch.zeros_like(level))
        return wave*torch.exp(level*(np.log(10)/20)).unsqueeze(0)

class SustainSource(torch.nn.Module):
    def forward(self, previous, control):
        frames=control.shape[0]
        start=control[:,1].long().clamp(0,frames-1)
        end=control[:,2].long().clamp(1,frames)
        duration=(end-start).clamp(min=1)
        midi=69+12*torch.log2(control[:,3].clamp(min=1)/440)
        prefix=torch.cat((torch.zeros_like(midi[:1]),midi.cumsum(0)))
        pitch=(prefix[end]-prefix[start])/duration
        # One weight per phone avoids repeated source switching during vibrato.
        lower=(pitch-56).clamp(0,1)
        upper=(73-pitch).clamp(0,1)
        weight=lower*lower*(3-2*lower)*upper*upper*(3-2*upper)
        active=(control[:,6]>.5)&(duration>=36)
        return torch.maximum(previous,(weight*active).unsqueeze(0))

def build(bank, out):
    out.mkdir(parents=True, exist_ok=False)
    layer=out/'sustain-layer.onnx'
    torch.onnx.export(SustainEnvelope(),(torch.randn(1,64*512)*.05,torch.zeros(64,128)),layer,input_names=['wave','control'],output_names=['leveled'],opset_version=18,dynamo=False,dynamic_axes={'wave':{1:'S'},'control':{0:'F'},'leveled':{1:'S'}})
    source=out/'source-layer.onnx'
    torch.onnx.export(SustainSource(),(torch.zeros(1,64),torch.zeros(64,128)),source,input_names=['previous','control'],output_names=['weight'],opset_version=18,dynamo=False,dynamic_axes={'previous':{1:'F'},'control':{0:'F'},'weight':{1:'F'}})
    m=onnx.load(next((bank/'dsvocoder').glob('*.onnx')))
    source_nodes=[]
    for node in m.graph.node:
        if 'blend_where_1' in node.output:
            node.output[0]='sle_previous_weight';source_nodes.append(node)
            append_model(source_nodes,m.graph.initializer,source,'sls_',dict(previous='sle_previous_weight',control='rp_control',weight='blend_where_1'))
        else:source_nodes.append(node)
    del m.graph.node[:];m.graph.node.extend(source_nodes)
    old=list(m.graph.node);idx=next(i for i,n in enumerate(old) if 'pb_input_wave' in n.output)+1
    nodes=[]
    append_model(nodes,m.graph.initializer,layer,'sle_',dict(wave='pb_input_wave',control='rp_control',leveled='sle_wave'))
    for n in old[idx:]:
        for i,v in enumerate(n.input):
            if v=='pb_input_wave':n.input[i]='sle_wave'
    del m.graph.node[:];m.graph.node.extend(old[:idx]+nodes+old[idx:])
    onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.bank,a.output)
