"""Replace inherited short-onset level contour with the vowel's late voiced level."""
import argparse
from pathlib import Path
import torch, onnx
from build_pair import append_model
from a_vowel_rearticulation import targets

class Plateau(torch.nn.Module):
    def forward(self, tokens, durations, reference, candidate):
        starts=durations.cumsum(1)-durations
        t=torch.arange(reference.shape[1])[None,None,:]
        off=t-starts[:,:,None]
        d=durations[:,:,None].float()
        eligible=targets(tokens,durations)[:,:,None]
        # The independent-core crossfade finishes at 75%; sample after that handoff.
        core=(off>=d*.75)&(off<d*.85)
        energy=torch.logsumexp(reference,2)
        level=(energy[:,None,:]*core).sum(2)/core.sum(2).clamp(min=1)
        active=eligible&(off>=0)&(off<d)
        desired=(level[:,:,None]*active).sum(1)
        change=(desired-energy).clamp(-3*2.302585093/20,3*2.302585093/20)
        edge=torch.minimum(off.float(),d-1-off).clamp(min=0)/4
        edge=edge.clamp(0,1);edge=edge*edge*(3-2*edge)
        weight=(edge*active).sum(1)
        candidate_level=torch.logsumexp(candidate,2)
        result=candidate+(energy+weight*change-candidate_level).unsqueeze(2)
        changed=(reference!=candidate).any(2)|active.any(1)
        return torch.where(changed.unsqueeze(2),result,reference)

def build(source,out):
    out.mkdir(parents=True,exist_ok=False)
    layer=out/'plateau.onnx'
    torch.onnx.export(Plateau(),(torch.tensor([[14,3,16]]),torch.tensor([[45,52,6]]),torch.randn(1,103,128),torch.randn(1,103,128)),layer,input_names=['tokens','durations','reference','candidate'],output_names=['result'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'reference':{1:'F'},'candidate':{1:'F'},'result':{1:'F'}})
    m=onnx.load(source);old=list(m.graph.node);nodes=[]
    for branch in ['original','donor']:
        prefix='acr_contour_'+branch+'_'
        old=[n for n in old if not any(o.startswith(prefix) or o=='acr_'+branch+'_result' for o in n.output)]
        append_model(nodes,m.graph.initializer,layer,'apl_'+branch+'_',dict(tokens='tokens',durations='durations',reference='ar_'+branch+'_mel',candidate='acr_ar_'+branch+'_mel',result='acr_'+branch+'_result'))
    from ts_affricate import dependencies
    pending=old+nodes;ordered=[];known={v.name for v in m.graph.input}|{v.name for v in m.graph.initializer}|{''}
    while pending:
        ready=[n for n in pending if dependencies(n)<=known]
        if not ready:raise ValueError('unresolved graph')
        for n in ready:ordered.append(n);known.update(n.output);pending.remove(n)
    del m.graph.node[:];m.graph.node.extend(ordered);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.source,a.out)
