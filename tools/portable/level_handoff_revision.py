"""Revise the existing balance layer and anticipate connected vowel handoffs."""
import argparse
from pathlib import Path
import torch, onnx
from build_pair import append_model
from vowel_balance import VowelBalance

class Handoff(torch.nn.Module):
    def forward(self, tokens, durations, mel):
        previous = torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),1)
        pd = torch.cat((torch.zeros_like(durations[:,:1]),durations[:,:-1]),1)
        def vowel(x): return (x==3)|(x==9)|(x==14)|(x==20)|(x==28)
        eligible = vowel(tokens)&vowel(previous)&(pd>=20)&(durations>=6)
        starts = durations.cumsum(1)-durations
        t = torch.arange(mel.shape[1])[None,None,:]
        off = t-starts[:,:,None]
        # A smooth 20ms anticipation, restored within the incoming phone.
        def smooth(x):
            x=x.clamp(0,1)
            return x*x*(3-2*x)
        width=torch.minimum(durations.float()*.5,torch.full_like(durations,6).float())[:,:,None]
        w=smooth((off+4)/4)*(1-smooth(off/width.clamp(min=1)))
        advance=(w*eligible[:,:,None]).sum(1)*1.72
        pos=(torch.arange(mel.shape[1])[None,:]+advance).clamp(0,mel.shape[1]-1)
        lo=pos.long(); hi=(lo+1).clamp(max=mel.shape[1]-1); a=(pos-lo).unsqueeze(2)
        result=mel.gather(1,lo.unsqueeze(2).expand(-1,-1,128))*(1-a)+mel.gather(1,hi.unsqueeze(2).expand(-1,-1,128))*a
        return torch.where(advance.unsqueeze(2)>0,result,mel)

def build(bank,out):
    out.mkdir(parents=True,exist_ok=False)
    opts=dict(opset_version=18,dynamo=False)
    layer=out/'balance.onnx'
    torch.onnx.export(VowelBalance(),(torch.randn(1,64*512)*.05,torch.zeros(64,128)),layer,input_names=['wave','control'],output_names=['balanced'],dynamic_axes={'wave':{1:'S'},'control':{0:'F'},'balanced':{1:'S'}},**opts)
    model=onnx.load(next((bank/'dsvocoder').glob('*.onnx')))
    assert not any(o.startswith('lvb_') for n in model.graph.node for o in n.output), 'Revision already installed'
    nodes=list(model.graph.node); indices=[i for i,n in enumerate(nodes) if any(o.startswith('vb_') for o in n.output)]
    assert indices and indices==list(range(min(indices),max(indices)+1))
    replacement=[]
    append_model(replacement,model.graph.initializer,layer,'lvb_',{'wave':'pb_input_wave','control':'rp_control','balanced':'vb_balanced'})
    nodes=nodes[:min(indices)]+replacement+nodes[max(indices)+1:]
    del model.graph.node[:];model.graph.node.extend(nodes)
    onnx.checker.check_model(model);onnx.save(model,out/'vocoder.onnx')
    layer=out/'handoff.onnx'
    torch.onnx.export(Handoff(),(torch.tensor([[1,14,3,1]]),torch.tensor([[8,40,20,8]]),torch.randn(1,76,128)),layer,input_names=['tokens','durations','mel'],output_names=['joined'],dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'mel':{1:'F'},'joined':{1:'F'}},**opts)
    model=onnx.load(bank/'tokine_ds_clarity_hybrid.tokine.onnx');nodes=[]
    for n in model.graph.node:
        if 'paired_mel' in n.output:
            for branch in ['original','donor']:
                old='acr_'+branch+'_result';assert old in n.input
                new='lh_'+branch+'_joined'
                append_model(nodes,model.graph.initializer,layer,'lh_'+branch+'_',{'tokens':'tokens','durations':'durations','mel':old,'joined':new})
                for i,x in enumerate(n.input):
                    if x==old:n.input[i]=new
        nodes.append(n)
    del model.graph.node[:];model.graph.node.extend(nodes)
    onnx.checker.check_model(model);onnx.save(model,out/'acoustic.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.bank,a.out)
