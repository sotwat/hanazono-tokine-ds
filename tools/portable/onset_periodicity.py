"""Reduce aperiodic vowel attack residue using pitch-synchronous cycle averaging."""
import argparse
from pathlib import Path
import torch
import onnx
import numpy as np
from onnx import helper as h,numpy_helper as nh
from build_pair import append_model


class CycleAttack(torch.nn.Module):
    def forward(self,wave,f0,control):
        pos=torch.arange(wave.shape[1],device=wave.device).unsqueeze(0)
        fi=(pos//512).clamp(max=f0.shape[1]-1)
        hi=(fi+1).clamp(max=f0.shape[1]-1)
        alpha=(pos%512)/512.
        freq=f0.gather(1,fi)*(1-alpha)+f0.gather(1,hi)*alpha
        # Integrate the sample pitch before looking up equal-phase cycles.
        phase=torch.cumsum(freq/44100.,dim=1)
        flag=control[:,:,0].gather(1,fi)
        eligible=flag>0
        start=control[:,:,1].gather(1,fi)*512
        end=control[:,:,2].gather(1,fi)*512
        elapsed=(pos-start)/44100.
        remaining=(end-pos)/44100.
        attack=(elapsed/.004).clamp(0,1)
        release=((.075-elapsed)/.025).clamp(0,1)
        weight=.8*attack*release*(remaining/.008).clamp(0,1)
        weight=torch.where(eligible & (freq>=60),weight,torch.zeros_like(weight))
        total=wave.clone();count=torch.ones_like(wave)
        radius=torch.ceil(44100/freq.clamp(min=60)*4).long()
        for shift in [-2.,-1.,1.,2.]:
            target=phase+shift
            lo=(pos-radius).clamp(min=0)
            upper=(pos+radius).clamp(max=wave.shape[1]-1)
            for _ in range(13):
                mid=(lo+upper)//2
                lower=phase.gather(1,mid)<target
                lo=torch.where(lower,mid+1,lo)
                upper=torch.where(lower,upper,mid)
            right=lo.clamp(max=wave.shape[1]-1)
            left=(right-1).clamp(min=0)
            p0=phase.gather(1,left);p1=phase.gather(1,right)
            frac=((target-p0)/(p1-p0).clamp(min=1e-6)).clamp(0,1)
            value=wave.gather(1,left)*(1-frac)+wave.gather(1,right)*frac
            valid=(left>=start)&(right<end)&(target>=p0)&(target<=p1)
            valid=valid.to(wave.dtype)
            total=total+value*valid;count=count+valid
        mean=total/count
        mixed=torch.where(weight>0,wave*(1-weight)+mean*weight,wave)
        phone_duration=(end-start)/44100.
        rise_time=(phone_duration*.15).clamp(min=1/44100,max=.008)
        rise=(elapsed/rise_time).clamp(0,1)
        rise=rise*rise*(3-2*rise)
        rest_elapsed=(pos-start)/44100.
        rest_fade=(phone_duration*.2).clamp(min=1/44100,max=.012)
        rest=((2*rest_fade-rest_elapsed)/rest_fade).clamp(0,1)
        rest=rest*rest*(3-2*rest)
        gate=torch.where(flag<0,rest,torch.where(eligible,rise,torch.ones_like(rise)))
        return torch.where(gate<1,mixed*gate,mixed)


def export(path):
    n=32;w=torch.randn(1,n*512);f=torch.full((1,n),440.);c=torch.zeros(1,n,128)
    torch.onnx.export(CycleAttack(),(w,f,c),str(path),input_names=['wave','f0','control'],output_names=['clean_wave'],opset_version=18,dynamo=False,dynamic_axes={'wave':{1:'samples'},'f0':{1:'frames'},'control':{1:'frames'},'clean_wave':{1:'samples'}})


def build(acoustic,vocoder,filter_model,out):
    out.mkdir(parents=True,exist_ok=True)
    m=onnx.load(acoustic);nodes=list(m.graph.node)
    def c(k,v,dtype=np.int64):m.graph.initializer.append(nh.from_array(np.array(v,dtype),'at_'+k))
    def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,['at_'+out],**kw))
    for k,v in [('axis0',[0]),('axis1',[1]),('axis2',[2]),('zero',0),('one',1),('begin',[0]),('end',[-1]),('prevpad',[1,0]),('vowels',[3,9,14,20,28]),('pads',[0,0,0,0,0,125])]:c(k,v)
    n('Squeeze',['tokens','at_axis0'],'tokens');n('Squeeze',['durations','at_axis0'],'durations')
    n('CumSum',['at_durations','at_zero'],'ends');n('Sub',['at_ends','at_durations'],'starts')
    n('Slice',['at_tokens','at_begin','at_end'],'prev0');n('Pad',['at_prev0','at_prevpad','at_zero'],'prev')
    n('Equal',['at_prev','at_one'],'after_sp');n('Unsqueeze',['at_tokens','at_axis1'],'tokens2');n('Equal',['at_tokens2','at_vowels'],'vowels0');n('Cast',['at_vowels0'],'vowelsf',to=onnx.TensorProto.FLOAT);n('ReduceMax',['at_vowelsf','at_axis1'],'vowels1',keepdims=0);c('zerof',0.,np.float32);n('Greater',['at_vowels1','at_zerof'],'vowel');n('And',['at_after_sp','at_vowel'],'eligible');n('Cast',['at_eligible'],'eligiblef',to=onnx.TensorProto.FLOAT)
    c('nextpad',[0,1]);c('nextbegin',[1]);c('nextend',[9223372036854775807]);n('Slice',['at_tokens','at_nextbegin','at_nextend'],'next0');n('Pad',['at_next0','at_nextpad','at_zero'],'next');n('Unsqueeze',['at_next','at_axis1'],'next2');n('Equal',['at_next2','at_vowels'],'nextmatches');n('Cast',['at_nextmatches'],'nextmatchesf',to=onnx.TensorProto.FLOAT);n('ReduceMax',['at_nextmatchesf','at_axis1'],'nextvowelf',keepdims=0);n('Greater',['at_nextvowelf','at_zerof'],'nextvowel');n('Equal',['at_tokens','at_one'],'is_sp');n('And',['at_nextvowel','at_is_sp'],'pre_vowel_sp');n('Cast',['at_pre_vowel_sp'],'pre_spf',to=onnx.TensorProto.FLOAT);n('Sub',['at_eligiblef','at_pre_spf'],'signedf')
    n('Unsqueeze',['at_starts','at_axis1'],'starts2');n('Unsqueeze',['at_ends','at_axis1'],'ends2')
    n('GreaterOrEqual',['original_su_range','at_starts2'],'started');n('Less',['original_su_range','at_ends2'],'unfinished');n('And',['at_started','at_unfinished'],'inside');n('Cast',['at_inside'],'insidef',to=onnx.TensorProto.FLOAT);n('Unsqueeze',['at_signedf','at_axis1'],'eligible2');n('Mul',['at_insidef','at_eligible2'],'active0');n('ReduceSum',['at_active0','at_axis0'],'active',keepdims=0)
    for field in ['starts','ends']:
        n('Cast',['at_'+field+'2'],field+'f',to=onnx.TensorProto.FLOAT);n('Mul',['at_insidef','at_'+field+'f'],field+'all');n('ReduceSum',['at_'+field+'all','at_axis0'],field+'frame',keepdims=0)
    for field in ['active','startsframe','endsframe']:n('Unsqueeze',['at_'+field,'at_axis1'],field+'col')
    n('Concat',['at_activecol','at_startsframecol','at_endsframecol'],'control0',axis=1);n('Unsqueeze',['at_control0','at_axis0'],'control3');n('Pad',['at_control3','at_pads','at_zerof'],'control')
    for node in nodes:
        if 'mel' in node.output:node.output[list(node.output).index('mel')]='paired_mel'
    nodes.append(h.make_node('Concat',['paired_mel','at_control'],['mel'],axis=0));m.graph.output[0].type.tensor_type.shape.dim[0].dim_value=3
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
    m=onnx.load(vocoder);nodes=list(m.graph.node);initializers=list(m.graph.initializer)
    for node in nodes:
        if 'waveform' in node.output:node.output[list(node.output).index('waveform')]='attack_before'
    initializers.append(nh.from_array(np.array([2],np.int64),'attack_control_index'));nodes.append(h.make_node('Gather',['mel','attack_control_index'],['attack_control'],axis=0))
    append_model(nodes,initializers,filter_model,'attack_',{'wave':'attack_before','f0':'f0','control':'attack_control','clean_wave':'waveform'})
    m.graph.input[0].type.tensor_type.shape.dim[0].dim_value=3;del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.initializer[:];m.graph.initializer.extend(initializers);onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('acoustic',type=Path);p.add_argument('vocoder',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);filter_path=a.output/'filter.onnx';export(filter_path);build(a.acoustic,a.vocoder,filter_path,a.output)
