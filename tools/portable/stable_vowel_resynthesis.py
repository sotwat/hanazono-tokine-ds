"""Neural vowel extension with real-time edges and an independent blend window."""
import argparse
from pathlib import Path
import onnx
import torch
from build_pair import append_model
from ts_affricate import dependencies

class EdgeTimeMaps(torch.nn.Module):
    def forward(self,tokens,durations,f0):
        starts=durations.cumsum(1)-durations;ends=starts+durations
        t=torch.arange(f0.shape[1],device=f0.device)[None,None,:]
        inside=(t>=starts[:,:,None])&(t<ends[:,:,None])
        edge=torch.maximum(durations//5,torch.full_like(durations,6))
        core=inside&(t>=(starts+edge)[:,:,None])&(t<(ends-edge)[:,:,None])
        hi=torch.where(core,f0[:,None,:],torch.zeros_like(f0[:,None,:])).amax(2)
        lo=torch.where(core,f0[:,None,:],torch.full_like(f0[:,None,:],1e6)).amin(2)
        low=torch.where(inside,f0[:,None,:],torch.full_like(f0[:,None,:],1e6)).amin(2)
        vowel=(tokens==3)|(tokens==9)|(tokens==14)|(tokens==20)|(tokens==28)
        eligible=vowel&(durations>=20)&(low>60)&(hi<lo*1.0293)
        compressed=torch.where(eligible,(durations//2).clamp(min=12,max=24),durations)
        cs=compressed.cumsum(1)-compressed;ce=cs+compressed
        ct=torch.arange(compressed.sum(),device=f0.device)[None,None,:]
        ci=(ct>=cs[:,:,None])&(ct<ce[:,:,None])
        e=((compressed-2)//2).clamp(min=0,max=4)[:,:,None]
        d=durations[:,:,None];c=compressed[:,:,None];u=ct-cs[:,:,None]
        middle=e+(u-e)*(d-2*e-1).clamp(min=1)/(c-2*e-1).clamp(min=1)
        mapped=torch.where(u<e,u,torch.where(u>=c-e,d-c+u,middle))
        forward=((starts[:,:,None]+mapped)*ci).sum(1).float().clamp(min=0,max=f0.shape[1]-1)
        left=forward.long();right=(left+1).clamp(max=f0.shape[1]-1);alpha=forward-left
        short_f0=f0.gather(1,left)*(1-alpha)+f0.gather(1,right)*alpha
        u=t-starts[:,:,None]
        middle=e+(u-e)*(c-2*e-1).clamp(min=1)/(d-2*e-1).clamp(min=1)
        mapped=torch.where(u<e,u,torch.where(u>=d-e,c-d+u,middle))
        back=((cs[:,:,None]+mapped)*inside).sum(1).float()
        mask=(inside*eligible[:,:,None]).sum(1).float().unsqueeze(2)
        return compressed,short_f0,back,mask

class RestoreEdges(torch.nn.Module):
    def forward(self,short,back,eligible,baseline,durations):
        left=back.long().clamp(min=0,max=short.shape[1]-1);right=(left+1).clamp(max=short.shape[1]-1)
        a=(back-left).unsqueeze(2);li=left.unsqueeze(2).expand(-1,-1,128);ri=right.unsqueeze(2).expand(-1,-1,128)
        restored=short.gather(1,li)*(1-a)+short.gather(1,ri)*a
        starts=durations.cumsum(1)-durations;ends=starts+durations
        t=torch.arange(back.shape[1],device=back.device)[None,None,:]
        inside=(t>=starts[:,:,None])&(t<ends[:,:,None])
        offset=((t-starts[:,:,None])*inside).sum(1).float()
        remain=((ends[:,:,None]-1-t)*inside).sum(1).float()
        attack=((offset-1)/3).clamp(0,1);release=(remain/6).clamp(0,1)
        attack=attack.square()*(3-2*attack);release=release.square()*(3-2*release)
        w=eligible*(attack*release).unsqueeze(2)
        return torch.where(w>0,baseline*(1-w)+restored*w,baseline)

def build(source,out):
    out.mkdir(parents=True,exist_ok=True);m=onnx.load(source)
    nodes=[n for n in m.graph.node if not any(o.startswith('tm_') or o.startswith('tc_restore_') or o in ('tc_durations','tc_f0','tc_back','tc_eligible','tc_original_mel','tc_donor_mel') for o in n.output)]
    constants=list(m.graph.initializer)
    args=(torch.tensor([[1,3,16,14,1]]),torch.tensor([[8,84,6,26,8]]),torch.full((1,132),293.66))
    maps=out/'maps.onnx'
    torch.onnx.export(EdgeTimeMaps(),args,maps,input_names=['tokens','durations','f0'],output_names=['short_durations','short_f0','back','eligible'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'f0':{1:'F'},'short_durations':{1:'N'},'short_f0':{1:'C'},'back':{1:'F'},'eligible':{1:'F'}})
    append_model(nodes,constants,maps,'tm_',dict(tokens='tokens',durations='durations',f0='f0',short_durations='tc_durations',short_f0='tc_f0',back='tc_back',eligible='tc_eligible'))
    layer=out/'restore.onnx'
    torch.onnx.export(RestoreEdges(),(torch.randn(1,64,128),torch.linspace(0,63,96)[None],torch.ones(1,96,1),torch.randn(1,96,128),torch.tensor([[8,80,8]])),layer,input_names=['short','back','eligible','baseline','durations'],output_names=['restored'],opset_version=18,dynamo=False,dynamic_axes={'short':{1:'C'},'back':{1:'F'},'eligible':{1:'F'},'baseline':{1:'F'},'durations':{1:'N'},'restored':{1:'F'}})
    for p in ('original_','donor_'):
        append_model(nodes,constants,layer,'tc_restore_'+p,dict(short='tc_'+p+'su_raw',back='tc_back',eligible='tc_eligible',baseline=p+'mel',durations='durations',restored='tc_'+p+'mel'))
    known={v.name for v in m.graph.input}|{v.name for v in constants}|{''};ordered=[]
    while nodes:
        ready=[n for n in nodes if dependencies(n)<=known]
        if not ready:raise ValueError('Unresolved graph dependencies')
        for n in ready:ordered.append(n);known.update(n.output);nodes.remove(n)
    del m.graph.node[:];m.graph.node.extend(ordered);del m.graph.initializer[:];m.graph.initializer.extend(constants)
    onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.source,a.output)
