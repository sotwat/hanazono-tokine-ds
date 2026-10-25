"""Bridge long distinct vowels across an artificial release/rearticulation valley."""
import argparse
from pathlib import Path
import torch,onnx
from build_pair import append_model

class LongVowelLink(torch.nn.Module):
    def forward(self,tokens,durations,mel):
        prev=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),1)
        pd=torch.cat((torch.zeros_like(durations[:,:1]),durations[:,:-1]),1)
        def vowel(x):return (x==3)|(x==9)|(x==14)|(x==20)|(x==28)
        eligible=vowel(tokens)&vowel(prev)&(prev!=tokens)&(pd>=40)&(durations>=12)
        starts=durations.cumsum(1)-durations
        # Repair only a release valley below both connected voiced anchors.
        energy=torch.logsumexp(mel,dim=2)
        le=energy.gather(1,(starts-8).clamp(0,mel.shape[1]-1))
        re=energy.gather(1,(starts+4).clamp(0,mel.shape[1]-1))
        mid=torch.stack([energy.gather(1,(starts-offset).clamp(0,mel.shape[1]-1)) for offset in range(1,7)],dim=2).amin(dim=2)
        eligible=eligible&(mid<torch.minimum(le,re)-0.4)
        t=torch.arange(mel.shape[1])[None,None,:]
        off=t-starts[:,:,None]
        active=eligible[:,:,None]&(off>=-8)&(off<4)
        alpha=((off+8).float()/12).clamp(0,1)
        alpha=alpha*alpha*(3-2*alpha)
        left=(starts-8).clamp(0,mel.shape[1]-1)
        right=(starts+4).clamp(0,mel.shape[1]-1)
        li=(left[:,:,None]*active).sum(1)
        ri=(right[:,:,None]*active).sum(1)
        l=mel.gather(1,li.unsqueeze(2).expand(-1,-1,128))
        r=mel.gather(1,ri.unsqueeze(2).expand(-1,-1,128))
        a=(alpha*active).sum(1).unsqueeze(2)
        # Interpolate linear mel amplitudes; avoid a log-domain formant valley.
        peak=torch.maximum(l,r)
        blend=peak+torch.log((1-a)*torch.exp(l-peak)+a*torch.exp(r-peak))
        # Separate spectral mixing from level interpolation to avoid a loudness bulge.
        level=(1-a)*torch.logsumexp(l,dim=2,keepdim=True)+a*torch.logsumexp(r,dim=2,keepdim=True)
        blend=blend+level-torch.logsumexp(blend,dim=2,keepdim=True)
        return torch.where(active.any(1).unsqueeze(2),blend,mel)

def build(source,out):
    out.mkdir(parents=True,exist_ok=False)
    layer=out/'link.onnx'
    torch.onnx.export(LongVowelLink(),(torch.tensor([[1,14,3,1]]),torch.tensor([[8,14,24,8]]),torch.randn(1,54,128)),layer,input_names=['tokens','durations','mel'],output_names=['joined'],opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'mel':{1:'F'},'joined':{1:'F'}})
    m=onnx.load(source);nodes=[]
    assert not any(o.startswith('lvc_') for n in m.graph.node for o in n.output)
    for n in m.graph.node:
        if 'paired_mel' in n.output:
            for branch in ['original','donor']:
                old='ru_'+branch+'_articulated';assert old in n.input
                new='lvc_'+branch+'_joined'
                append_model(nodes,m.graph.initializer,layer,'lvc_'+branch+'_',{'tokens':'tokens','durations':'durations','mel':old,'joined':new})
                for i,x in enumerate(n.input):
                    if x==old:n.input[i]=new
        nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.source,a.out)
