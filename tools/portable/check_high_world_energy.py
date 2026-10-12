"""Verify bounded compensation, exact inactive bypass and native nasal gain."""
import argparse,json
from pathlib import Path
import numpy as np
import onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('energy',type=Path);p.add_argument('fixture',type=Path);a=p.parse_args();o=ort.SessionOptions();o.intra_op_num_threads=4;s=ort.InferenceSession(str(a.energy),o);cases=0
for frames in [1,2,8,32,264]:
 for loss in [1.,.5,.125,.01]:
  t=np.arange(frames*512,dtype=np.float32)/44100;donor=(.05*np.sin(2*np.pi*1200*t))[None].astype(np.float32);wave=donor*loss
  for active in [False,True]:
   for is_nasal in [False,True]:
    mask=np.ones((1,frames),np.float32) if active else np.zeros((1,frames),np.float32)
    nasal=mask if is_nasal else np.zeros_like(mask)
    y=s.run(None,{'world':wave,'donor':donor,'mask':mask,'nasal':nasal})[0]
    assert np.isfinite(y).all() and abs(y).max()<=abs(wave).max()*(8.00001 if is_nasal else 2.00001)
    if not active:assert np.array_equal(y,wave)
    cases+=1
z=np.load(a.fixture/'native-vocoder.npz');ac=np.load(a.fixture/'native-acoustic.npz');tokens=ac['dbg_tokens'][0];dur=ac['dbg_durations'][0];mask=np.repeat(np.isin(tokens,[2,3,9,14,18,19,20,28]).astype(np.float32),dur)[None];x=z['world_wave'];y=s.run(None,{'world':x,'donor':z['donor_wave'],'mask':mask,'nasal':np.repeat((tokens==2).astype(np.float32),dur)[None]})[0];assert np.isfinite(y).all();inactive=np.repeat(mask[0]==0,512);assert np.array_equal(y[0,inactive],x[0,inactive]);assert abs(y).max()<1
lo,hi=185*512,200*512;rms=lambda v:float(np.sqrt(np.mean(v[0,lo:hi]**2)));result={'cases':cases,'N_before_rms':rms(x),'N_after_rms':rms(y),'gain_db':20*np.log10(rms(y)/rms(x)),'peak':float(abs(y).max()),'non_target_exact':True};(a.energy.parent/'energy-checks.json').write_text(json.dumps(result,indent=2)+'\n');np.save(a.energy.parent/'corrected-fixed-world.npy',y);print(json.dumps(result))

ap_session=ort.InferenceSession(str(a.energy.parent/'nasal-ap.onnx'),o)
for frames in [1,2,8,32]:
 ap=np.linspace(1e-6,.999999,frames*1025,dtype=np.float32).reshape(1,frames,1025)
 for active in [False,True]:
  mask=np.full((1,frames),float(active),np.float32)
  y=ap_session.run(None,{'ap':ap,'nasal':mask})[0]
  assert np.all(y<=ap) and np.all(y>=ap*.749999)
  if not active:assert np.array_equal(y,ap)
print(json.dumps({'AP_cases':8,'nonN_AP_exact':True,'AP_gain_min':.75}))
