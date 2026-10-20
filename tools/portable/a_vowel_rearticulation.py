"""Trial independent learned /a/ onset at long vowel joins; no gain boost."""
import argparse,copy
from pathlib import Path
import onnx
import torch
from onnx import helper as h
from build_pair import append_model
from short_vowel_resynthesis import rename_graph
from ts_affricate import dependencies


def targets(tokens,durations):
    p=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),1)
    pd=torch.cat((torch.zeros_like(durations[:,:1]),durations[:,:-1]),1)
    vowel=(p==9)|(p==14)|(p==20)|(p==28)
    return (tokens==3)&vowel&(pd>=40)&(durations>=12)


class Context(torch.nn.Module):
    def forward(self,tokens,durations,f0):
        target=targets(tokens,durations)
        reset=torch.cat((target[:,1:],torch.zeros_like(target[:,:1])),1)
        changed=torch.where(reset,torch.ones_like(tokens),tokens)
        compressed=torch.where(target,torch.full_like(durations,12),durations)
        starts=durations.cumsum(1)-durations
        cs=compressed.cumsum(1)-compressed
        t=torch.arange(compressed.sum())[None,None,:]
        inside=(t>=cs[:,:,None])&(t<(cs+compressed)[:,:,None])
        position=starts[:,:,None]+(t-cs[:,:,None])*(durations-1).clamp(min=0)[:,:,None]/(compressed-1).clamp(min=1)[:,:,None]
        pos=(position*inside).sum(1).float().clamp(0,f0.shape[1]-1)
        left=pos.long();right=(left+1).clamp(max=f0.shape[1]-1);alpha=pos-left
        new_f0=f0.gather(1,left)*(1-alpha)+f0.gather(1,right)*alpha
        silence=(reset[:,:,None]&inside).any(1)
        return changed,compressed,torch.where(silence,torch.zeros_like(new_f0),new_f0)


class Join(torch.nn.Module):
    def forward(self,tokens,durations,base,fresh):
        target=targets(tokens,durations)
        starts=durations.cumsum(1)-durations
        compressed=torch.where(target,torch.full_like(durations,12),durations)
        cs=compressed.cumsum(1)-compressed
        t=torch.arange(base.shape[1])[None,None,:]
        off=(t-starts[:,:,None]).float()
        d=durations[:,:,None].float()
        def smooth(x):
            x=x.clamp(0,1)
            return x*x*(3-2*x)
        w=smooth((off+2)/2)*(1-smooth((off-d*.5)/(d*.25).clamp(min=1)))
        w=w*target[:,:,None]
        total=w.sum(1).unsqueeze(2)
        # Keep the moving, voiced core of the short learned vowel, not its breath/release.
        position=cs[:,:,None]+5+4*(off/(d*.75).clamp(min=1)).clamp(0,1)
        pos=(position*w.gt(0)).sum(1).float().clamp(0,fresh.shape[1]-1)
        left=pos.long();right=(left+1).clamp(max=fresh.shape[1]-1);alpha=(pos-left).unsqueeze(2)
        resynth=fresh.gather(1,left.unsqueeze(2).expand(-1,-1,128))*(1-alpha)+fresh.gather(1,right.unsqueeze(2).expand(-1,-1,128))*alpha
        mixed=base*(1-total)+resynth*total
        return torch.where(total>0,mixed,base)


def build(source,output):
    output.mkdir(parents=True,exist_ok=False);m=onnx.load(source)
    args=(torch.tensor([[1,14,3,16,3,1]]),torch.tensor([[8,57,52,6,34,8]]),torch.full((1,165),293.66))
    context=output/'context.onnx'
    torch.onnx.export(Context(),args,context,input_names=['tokens','durations','f0'],output_names=['new_tokens','new_durations','new_f0'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'f0':{1:'F'},'new_tokens':{1:'N'},'new_durations':{1:'N'},'new_f0':{1:'C'}})
    layer=output/'join.onnx'
    torch.onnx.export(Join(),(*args[:2],torch.randn(1,165,128),torch.randn(1,165,128)),layer,input_names=['tokens','durations','base','fresh'],output_names=['joined'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'base':{1:'F'},'fresh':{1:'C'},'joined':{1:'F'}})
    nodes=[];constants=list(m.graph.initializer)
    append_model(nodes,constants,context,'ar_context_',dict(tokens='tokens',durations='durations',f0='f0',new_tokens='ar_tokens',new_durations='ar_durations',new_f0='ar_f0'))
    needed={'original_su_raw','donor_su_raw'};selected=[];boundary={v.name for v in m.graph.input}
    for n in reversed(m.graph.node):
        if (set(n.output)&needed)-boundary:selected.append(copy.deepcopy(n));needed.update(dependencies(n))
    neural=copy.deepcopy(m.graph);del neural.node[:];neural.node.extend(reversed(selected))
    fixed={v.name:v.name for v in m.graph.initializer};fixed.update(tokens='ar_tokens',durations='ar_durations',f0='ar_f0',depth='depth',steps='steps')
    rename_graph(neural,'ar_neural_',fixed);nodes.extend(neural.node)
    for n in m.graph.node:
        if 'paired_mel' in n.output:
            for p in ('original_','donor_'):
                old='ta_'+p+'mel';assert old in n.input
                append_model(nodes,constants,layer,'ar_join_'+p,dict(tokens='tokens',durations='durations',base=old,fresh='ar_neural_'+p+'su_raw',joined='ar_'+p+'mel'))
                for i,k in enumerate(n.input):
                    if k==old:n.input[i]='ar_'+p+'mel'
        nodes.append(n)
    known={v.name for v in m.graph.input}|{v.name for v in constants}|{''};ordered=[]
    while nodes:
        ready=[n for n in nodes if dependencies(n)<=known]
        if not ready:raise ValueError('Unresolved dependencies')
        for n in ready:ordered.append(n);known.update(n.output);nodes.remove(n)
    del m.graph.node[:];m.graph.node.extend(ordered);del m.graph.initializer[:];m.graph.initializer.extend(constants)
    onnx.checker.check_model(m);onnx.save(m,output/'acoustic.onnx')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.source,a.output)
