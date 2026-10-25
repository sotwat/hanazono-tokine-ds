import sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from closed_vowel_source import Mask,Weight
root=Path(sys.argv[1]);ms=ort.InferenceSession(str(root/'mask.onnx'));ws=ort.InferenceSession(str(root/'weight.onnx'));cases=0
for token in range(32):
 for duration in [0,1,19,20,37,108]:
  t=torch.tensor([[1,token,1]]);d=torch.tensor([[8,duration,8]]);f=torch.full((1,int(d.sum())),440.)
  want=Mask()(t,d,f).numpy();actual=ms.run(None,dict(tokens=t.numpy(),durations=d.numpy(),f0=f.numpy()))[0];np.testing.assert_array_equal(want,actual)
  assert bool(actual.any())==bool(token in [14,28] and duration>=20)
  for midi in [60,67,68,69,71,74,77,84]:
   f=torch.full_like(f,440*2**((midi-69)/12));c=torch.zeros(f.shape[1],128);c[:,8]=torch.from_numpy(actual[0]);p=torch.full_like(f,.3 if midi==74 else 1 if midi>=77 else 0)
   want=Weight()(p,c,f).numpy();got=ws.run(None,dict(previous=p.numpy(),control=c.numpy(),f0=f.numpy()))[0];np.testing.assert_allclose(got,want,atol=1e-6)
   assert np.isfinite(got).all() and got.min()>=0 and got.max()<=1
   np.testing.assert_array_equal(got[actual==0],p.numpy()[actual==0])
   if midi>=77:np.testing.assert_array_equal(got,p.numpy())
   cases+=1
print(cases,'scope/range/ONNX cases passed')
