"""Fit a duration-conditioned breath mel decoder from recording excerpts."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch import nn


class BreathDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(11,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,128))
    def forward(self,position,duration):
        p=position.clamp(0,1)
        features=torch.stack([p,p*p,p*p*p,duration.clamp(.08,1),torch.sin(p*np.pi),torch.cos(p*np.pi),
            torch.sin(p*np.pi*2),torch.cos(p*np.pi*2),torch.sin(p*np.pi*3),torch.cos(p*np.pi*3),torch.ones_like(p)],-1)
        return self.net(features)


def train(dataset,out):
    sys.path.insert(0,'/Users/aaa/Documents/DiffSinger')
    from utils.binarizer_utils import get_mel_torch
    torch.set_num_threads(4);torch.manual_seed(2741);np.random.seed(2741)
    out.mkdir(exist_ok=True,parents=True);rows=json.loads((dataset/'candidates.json').read_text())
    heldout={'Lemon','ray_キー+5','地球最後の告白を','怪物'};groups={False:[],True:[]};clips=[]
    for row in rows:
        y,sr=sf.read(dataset/'clips'/f"{row['name']}.wav",dtype='float32')
        # Equal recording gain; retain each breath's temporal envelope and spectrum.
        gain=min(15.,10**(-33/20)/(np.sqrt(np.mean(y*y))+1e-9));y*=gain
        mel=get_mel_torch(y,sr,device='cpu');frames=len(mel)
        # Three-frame smoothing removes sample noise while retaining breath shape.
        mel=np.stack([np.mean(mel[max(0,i-1):min(frames,i+2)],0) for i in range(frames)])
        pos=np.linspace(0,1,frames,dtype='float32');dur=np.full(frames,len(y)/sr,np.float32)
        groups[row['song'] in heldout].append((pos,dur,mel.astype('float32')))
        clips.append(dict(name=row['name'],song=row['song'],validation=row['song'] in heldout,gain=float(gain),frames=frames))
    def tensors(items):return [torch.from_numpy(np.concatenate([x[i] for x in items])).to('mps') for i in range(3)]
    p,d,y=tensors(groups[False]);vp,vd,vy=tensors(groups[True]);mean=y.mean(0);scale=y.std(0).clamp(min=.8)
    model=BreathDecoder().to('mps')
    with torch.no_grad():
        model.net[-1].weight.zero_();model.net[-1].bias.copy_(mean)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    best=1e9;beststate=None;history=[]
    for step in range(1801):
        model.train();pred=model(p,d);loss=(((pred-y)/scale)**2).mean();opt.zero_grad();loss.backward();opt.step()
        if step%100==0:
            model.eval()
            with torch.no_grad():val=float((((model(vp,vd)-vy)/scale)**2).mean().cpu())
            history.append(dict(step=step,train=float(loss.cpu()),validation=val));print(history[-1],flush=True)
            if val<best:best=val;beststate=copy.deepcopy(model.cpu().state_dict());model.to('mps')
    model.cpu();model.load_state_dict(beststate);model.eval();torch.save(model.state_dict(),out/'decoder.pt')
    torch.onnx.export(model,(torch.linspace(0,1,24),torch.ones(24)*.28),out/'decoder.onnx',input_names=['position','duration'],output_names=['mel'],dynamic_axes={'position':{0:'F'},'duration':{0:'F'},'mel':{0:'F'}},opset_version=18,dynamo=False)
    baseline=float((((mean[None]-vy)/scale)**2).mean().cpu())
    result=dict(clips=clips,history=history,best_validation=best,constant_baseline=baseline,method='duration-conditioned mel MLP, recording-derived breaths only',seed=2741)
    (out/'training.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('final',best,'baseline',baseline,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);a=p.parse_args();train(a.dataset,a.output)
