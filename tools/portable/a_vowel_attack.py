"""Trial: a bounded gain envelope at /a/ onsets after another vowel."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import torch
from build_pair import append_model


class AAttack(torch.nn.Module):
    def forward(self, wave, control):
        sample = torch.arange(wave.shape[1],device=wave.device)
        fi = (sample//512).clamp(max=control.shape[0]-1)
        start = control[fi,1]*512
        end = control[fi,2]*512
        elapsed = (sample-start)/44100.
        duration = (end-start)/44100.
        # End before the middle of short phones; never reach the next phone.
        hold = (duration*.25).clamp(max=.14)
        finish = (duration*.45).clamp(max=.24)
        rise = (elapsed/.005).clamp(0,1)
        fall = ((finish-elapsed)/(finish-hold).clamp(min=.001)).clamp(0,1)
        rise = rise.square()*(3-2*rise)
        fall = fall.square()*(3-2*fall)
        active = (control[fi,7]>.5) & (duration>=.12)
        weight = torch.where(active,rise*fall,torch.zeros_like(rise))
        headroom = (20*torch.log10(.97/wave.abs().amax().clamp(min=1e-6))).clamp(min=0,max=4)
        gain = torch.exp(weight*headroom*(np.log(10)/20))
        return torch.where(weight>0,wave*gain.unsqueeze(0),wave)


def build(acoustic,vocoder,out):
    out.mkdir(parents=True,exist_ok=True)
    m = onnx.load(acoustic)
    nodes = list(m.graph.node)
    if any(i.name.startswith('aa_') for i in m.graph.initializer):
        raise ValueError('Attack patch already installed')
    def c(name,value,dtype=np.int64):
        m.graph.initializer.append(nh.from_array(np.array(value,dtype),'aa_'+name))
    def n(op,ins,name,**kw):
        nodes.append(h.make_node(op,ins,['aa_'+name],**kw))
    for name,value in [('axis0',[0]),('axis1',[1]),('axis2',[2]),('a',3),('vowels',[3,9,14,20,28]),('begin',[0]),('end',[-1]),('pad',[0,1,0,0]),('zeroi',0),('columnpad',[0,0,7,0,0,120])]:
        c(name,value)
    c('zerof',0.,np.float32)
    n('Equal',['tokens','aa_a'],'is_a')
    n('Slice',['tokens','aa_begin','aa_end','aa_axis1'],'prev0')
    n('Pad',['aa_prev0','aa_pad','aa_zeroi'],'prev')
    n('Unsqueeze',['aa_prev','aa_axis2'],'prev3')
    n('Equal',['aa_prev3','aa_vowels'],'prev_match')
    n('Cast',['aa_prev_match'],'prev_float',to=1)
    n('ReduceMax',['aa_prev_float','aa_axis2'],'prev_vowel_float',keepdims=0)
    n('Greater',['aa_prev_vowel_float','aa_zerof'],'prev_vowel')
    n('And',['aa_is_a','aa_prev_vowel'],'eligible')
    n('Cast',['aa_eligible'],'eligiblef',to=1)
    n('Transpose',['aa_eligiblef'],'eligible_col',perm=[1,0])
    n('Mul',['he_insidef','aa_eligible_col'],'inside')
    n('ReduceSum',['aa_inside','aa_axis0'],'frame',keepdims=0)
    n('Unsqueeze',['aa_frame','aa_axis0'],'frame2')
    n('Unsqueeze',['aa_frame2','aa_axis2'],'column')
    n('Pad',['aa_column','aa_columnpad','aa_zerof'],'extra')
    n('Add',['vb_control','aa_extra'],'control')
    output = next(n for n in nodes if 'mel' in n.output)
    assert output.input[1]=='vb_control'
    nodes.remove(output);output.input[1]='aa_control';nodes.append(output)
    del m.graph.node[:];m.graph.node.extend(nodes)
    onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
    layer = out/'attack-layer.onnx'
    torch.onnx.export(AAttack(),(torch.randn(1,40*512)*.05,torch.zeros(40,128)),layer,input_names=['wave','control'],output_names=['corrected'],opset_version=18,dynamo=False,dynamic_axes={'wave':{1:'samples'},'control':{0:'frames'},'corrected':{1:'samples'}})
    m=onnx.load(vocoder);nodes=list(m.graph.node)
    final = next(n for n in nodes if 'waveform' in n.output)
    final.output[list(final.output).index('waveform')] = 'aa_before'
    append_model(nodes,m.graph.initializer,layer,'aa_',dict(wave='aa_before',control='rp_control',corrected='waveform'))
    del m.graph.node[:];m.graph.node.extend(nodes)
    onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('acoustic',type=Path);p.add_argument('vocoder',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();build(a.acoustic,a.vocoder,a.output)
