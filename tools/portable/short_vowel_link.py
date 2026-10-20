"""Bridge short connected vowels without replaying their breathy transition frames."""
import argparse
from pathlib import Path
import torch,onnx
from build_pair import append_model

class ShortVowelLink(torch.nn.Module):
    def forward(self,tokens,durations,mel):
        prev=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),1)
        pd=torch.cat((torch.zeros_like(durations[:,:1]),durations[:,:-1]),1)
        def vowel(x):return (x==3)|(x==9)|(x==14)|(x==20)|(x==28)
        eligible=vowel(tokens)&vowel(prev)&(pd>=6)&(pd<20)&(durations>=6)
        starts=durations.cumsum(1)-durations
        t=torch.arange(mel.shape[1])[None,None,:]
        off=t-starts[:,:,None]
        active=eligible[:,:,None]&(off>=-3)&(off<2)
        alpha=((off+3).float()/5).clamp(0,1)
        alpha=alpha*alpha*(3-2*alpha)
        left=(starts-3).clamp(0,mel.shape[1]-1)
        right=(starts+2).clamp(0,mel.shape[1]-1)
        li=(left[:,:,None]*active).sum(1)
        ri=(right[:,:,None]*active).sum(1)
        l=mel.gather(1,li.unsqueeze(2).expand(-1,-1,128))
        r=mel.gather(1,ri.unsqueeze(2).expand(-1,-1,128))
        a=(alpha*active).sum(1).unsqueeze(2)
        # Interpolate linear mel amplitudes; avoid a log-domain formant valley.
        peak=torch.maximum(l,r)
        blend=peak+torch.log((1-a)*torch.exp(l-peak)+a*torch.exp(r-peak))
        return torch.where(active.any(1).unsqueeze(2),blend,mel)

def build(source,out):
    out.mkdir(parents=True,exist_ok=False)
    layer=out/'link.onnx'
    torch.onnx.export(ShortVowelLink(),(torch.tensor([[1,14,3,1]]),torch.tensor([[8,14,24,8]]),torch.randn(1,54,128)),layer,input_names=['tokens','durations','mel'],output_names=['joined'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'mel':{1:'F'},'joined':{1:'F'}})
    m=onnx.load(source);nodes=[]
    assert not any(o.startswith('svl_') for n in m.graph.node for o in n.output)
    for n in m.graph.node:
        if 'paired_mel' in n.output:
            for branch in ['original','donor']:
                old='lh_'+branch+'_joined';assert old in n.input
                new='svl_'+branch+'_joined'
                append_model(nodes,m.graph.initializer,layer,'svl_'+branch+'_',{'tokens':'tokens','durations':'durations','mel':old,'joined':new})
                for i,x in enumerate(n.input):
                    if x==old:n.input[i]=new
        nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.source,a.out)
