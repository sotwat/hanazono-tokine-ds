from pathlib import Path
import sys,json
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent))
from vowel_balance import VowelBalance
from level_handoff_revision import Handoff
checks=0
for freq in [73.4,110,220,440,880,1760]:
 for length in [6,12,40,100]:
  lens=[4,length,5,length,4];starts=np.r_[0,np.cumsum(lens)];F=int(starts[-1]);c=np.zeros((F,128),np.float32);x=np.zeros((1,F*512),np.float32)
  for j,(a,b) in enumerate(zip(starts[:-1],starts[1:])):
   c[a:b,1]=a;c[a:b,2]=b;c[a:b,3]=freq;c[a:b,6]=j in [1,3]
   x[0,a*512:b*512]=(.02 if j==1 else .1)*np.sin(2*np.pi*freq*np.arange((b-a)*512)/44100)
  y=VowelBalance()(torch.from_numpy(x),torch.from_numpy(c)).numpy()
  assert np.isfinite(y).all() and abs(y).max()<=abs(x).max()*10**(.2)+1e-6
  for j in [0,2,4]:assert np.array_equal(x[:,starts[j]*512:starts[j+1]*512],y[:,starts[j]*512:starts[j+1]*512])
  assert np.sqrt((y[:,starts[1]*512:starts[2]*512]**2).mean())>np.sqrt((x[:,starts[1]*512:starts[2]*512]**2).mean())
  checks+=1
for previous in range(32):
 for incoming in range(32):
  tok=torch.tensor([[1,previous,incoming,1]]);dur=torch.tensor([[8,40,20,8]]);mel=torch.randn(1,76,128)
  y=Handoff()(tok,dur,mel)
  assert y.shape==mel.shape and torch.isfinite(y).all()
  assert torch.equal(y[:,:44],mel[:,:44]) and torch.equal(y[:,54:],mel[:,54:])
  if previous not in [3,9,14,20,28] or incoming not in [3,9,14,20,28]:assert torch.equal(y,mel)
  checks+=1

print(json.dumps({"checks":checks,"passed":True}))
