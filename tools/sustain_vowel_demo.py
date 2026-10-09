"""Prototype sustained vowel synthesis from a stable short mel segment."""
import argparse
from pathlib import Path
import numpy as np,onnxruntime as ort,soundfile as sf
p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);p.add_argument('--seconds',type=float,default=20);p.add_argument('--midi',type=int,default=69);a=p.parse_args()
o=ort.SessionOptions();o.intra_op_num_threads=4
ac=ort.InferenceSession(str(next(a.bank.glob('*.onnx'))),o,providers=['CPUExecutionProvider']);voc=ort.InferenceSession(str(next((a.bank/'dsvocoder').glob('*.onnx'))),o,providers=['CPUExecutionProvider'])
freq=440*2**((a.midi-69)/12);short=172
f=np.full((1,short+16),freq,np.float32);f[:,:8]=0;f[:,-8:]=0
mel=ac.run(None,dict(tokens=np.array([[1,3,1]],np.int64),durations=np.array([[8,short,8]],np.int64),f0=f,depth=np.array(.1,np.float32),steps=np.array(20,np.int64)))[0]
n=round(a.seconds*44100/512);assert n>100
# Forward/backward interpolation keeps segment joins continuous in spectral space.
idx=np.arange(n+16,dtype=float);phase=np.arange(n-52)%80;idx[8+26:8+n-26]=34+np.where(phase<=40,phase,80-phase)
idx[:34]=np.arange(34);idx[8+n-26:]=np.linspace(34,short+15,34)
ext=np.stack([np.interp(idx,np.arange(mel.shape[1]),mel[0,:,i]) for i in range(128)],axis=1)[None].astype(np.float32)
f=np.full((1,n+16),freq,np.float32);f[:,:8]=0;f[:,-8:]=0
x=voc.run(None,dict(mel=ext,f0=f))[0].ravel();assert np.isfinite(x).all()
a.output.parent.mkdir(parents=True,exist_ok=True);sf.write(a.output,x,44100,subtype='FLOAT')
print({'seconds':len(x)/44100,'peak':float(abs(x).max()),'first_sustain_rms':float(np.sqrt(np.mean(x[44100:88200]**2))),'last_sustain_rms':float(np.sqrt(np.mean(x[-88200:-44100]**2)))})
