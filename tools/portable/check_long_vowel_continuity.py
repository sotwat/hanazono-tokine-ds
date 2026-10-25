"""Check gated vowel bridges and ONNX dynamic duration parity."""
import argparse
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from long_vowel_continuity import LongVowelLink

p=argparse.ArgumentParser();p.add_argument('layer',type=Path);args=p.parse_args()
s=ort.InferenceSession(str(args.layer),providers=['CPUExecutionProvider'])
count=0
for previous in [1,3,9,14,16,20,28]:
 for current in [1,3,9,14,16,20,28]:
  for duration in [0,6,39,40,80]:
   for length in [0,6,12,40]:
    for valley in [False,True]:
     tokens=torch.tensor([[1,previous,current,1]])
     durations=torch.tensor([[8,duration,length,8]])
     frames=int(durations.sum());start=8+duration
     mel=torch.full((1,frames,128),-3.)
     if valley:mel[:,max(0,start-4)]=-5.
     actual=LongVowelLink()(tokens,durations,mel)
     expected=previous in [3,9,14,20,28] and current in [3,9,14,20,28] and previous!=current and duration>=40 and length>=12 and valley
     if not expected:assert torch.equal(actual,mel)
     else:
      assert not torch.equal(actual,mel)
      assert torch.equal(actual[:,:start-8],mel[:,:start-8])
      assert torch.equal(actual[:,start+4:],mel[:,start+4:])
     exported=s.run(None,{'tokens':tokens.numpy(),'durations':durations.numpy(),'mel':mel.numpy()})[0]
     np.testing.assert_allclose(exported,actual.numpy(),atol=1e-6,rtol=1e-6)
     assert np.isfinite(exported).all()
     count+=1
print(f'{count} cases passed')
