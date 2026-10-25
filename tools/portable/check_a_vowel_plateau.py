import sys
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from a_vowel_plateau import Plateau
from a_vowel_core_revision import Contour
from a_vowel_rearticulation import targets
s=ort.InferenceSession(str(Path(sys.argv[1])/'plateau.onnx'),providers=['CPUExecutionProvider']);count=0
for prev in [1,3,9,14,20,28,16]:
 for pd in [0,10,39,40,80]:
  for d in [0,6,11,12,24,52,100]:
   tok=torch.tensor([[1,prev,3,16]]);dur=torch.tensor([[8,pd,d,6]]);f=int(dur.sum());ref=torch.randn(1,f,128);cand=ref+.1*torch.randn_like(ref);out=Plateau()(tok,dur,ref,cand);old=Contour()(ref,cand)
   actual=s.run(None,dict(tokens=tok.numpy(),durations=dur.numpy(),reference=ref.numpy(),candidate=cand.numpy()))[0];assert np.allclose(actual,out.numpy(),atol=4e-6)
   starts=dur.cumsum(1)-dur;t=torch.arange(f)[None,None,:];mask=(targets(tok,dur)[:,:,None]&(t>=starts[:,:,None])&(t<(starts+dur)[:,:,None])).any(1)
   assert torch.allclose(out[~mask],old[~mask],atol=2e-6)
   assert torch.max(torch.abs(torch.logsumexp(out,2)-torch.logsumexp(old,2)))<3*np.log(10)/20+2e-6
   assert torch.allclose(out-out.mean(2,keepdim=True),cand-cand.mean(2,keepdim=True),atol=2e-6)
   assert torch.isfinite(out).all();count+=1
print(count,'conditions pass; ONNX, scope, spectral shape, bounded gain')
