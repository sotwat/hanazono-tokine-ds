import sys
import torch
import torch.nn.functional as F
class Blend(torch.nn.Module):
 def forward(self,baseline,world,f0):
  midi=69+12*torch.log2(torch.clamp(f0,min=1)/440)
  u=torch.clamp((midi-74)/3,0,1)
  w=torch.where(f0>=698.4555,torch.ones_like(u),u*u*(3-2*u))
  w=torch.where(f0>0,w,torch.zeros_like(w))
  pos=torch.arange(baseline.shape[1],device=f0.device)/512
  lo=pos.long().clamp(max=f0.shape[1]-1);hi=(lo+1).clamp(max=f0.shape[1]-1)
  interp=w[:,lo]*(1-(pos-lo))+w[:,hi]*(pos-lo)
  active=(w[:,lo]>1e-8).to(f0.dtype).unsqueeze(1)
  forward=F.avg_pool1d(F.pad(active,(881,0)),882,stride=1)
  backward=F.avg_pool1d(F.pad(active,(0,881)),882,stride=1)
  weight=(interp*active[:,0]*torch.minimum(forward[:,0],backward[:,0])).clamp(0,1)
  mixed=torch.sqrt(1-weight)*baseline+torch.sqrt(weight)*world
  return torch.where(weight<=0,baseline,torch.where(weight>=1,world,mixed))
if __name__=='__main__':
 n=32;x=torch.randn(1,n*512);f=torch.full((1,n),700.)
 torch.onnx.export(Blend(),(x,x.clone(),f),sys.argv[1],input_names=['baseline','world','f0'],output_names=['waveform'],opset_version=18,dynamo=True,optimize=False,external_data=False,dynamic_shapes=({1:512*torch.export.Dim('frames',min=16)},{1:512*torch.export.Dim('frames',min=16)},{1:torch.export.Dim('frames',min=16)}))
