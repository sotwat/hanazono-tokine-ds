from pathlib import Path
import sys,numpy as np,soundfile as sf,torch
from scipy.signal import resample_poly
import argparse
p=argparse.ArgumentParser(description='Extract F0 from an isolated recording with RMVPE.')
p.add_argument('audio',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('rmvpe_code',type=Path);p.add_argument('output',type=Path)
args=p.parse_args()
sys.path.insert(0,str(args.rmvpe_code))
from rmvpe.inference import RMVPE
from rmvpe.model import E2E0
from rmvpe.spec import MelSpectrogram
from rmvpe.constants import *
torch.set_num_threads(4)
r=RMVPE.__new__(RMVPE);r.device='cpu';r.resample_kernel={};r.hop_length=160;r.seg_length=5120
r.model=E2E0(4,1,(2,2)).eval()
# This local fork registers an unused timbre filter absent from the upstream checkpoint.
if hasattr(r.model.unet,'tf'):
 del r.model.unet.tf
r.model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
r.mel_extractor=MelSpectrogram(N_MELS,SAMPLE_RATE,WINDOW_LENGTH,160,None,MEL_FMIN,MEL_FMAX)
x,sr=sf.read(args.audio);x=(x.mean(1) if x.ndim==2 else x);from math import gcd
g=gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
f=[];ts=[]
for start in range(0,len(x),20*16000):
 a=max(0,start-16000);b=min(len(x),start+21*16000)
 y=r.infer_from_audio(x[a:b],16000,thred=.03)
 t=a/16000+np.arange(len(y))*.01
 keep=(t>=start/16000)&(t<min((start+20*16000)/16000,len(x)/16000))
 f.extend(y[keep]);ts.extend(t[keep]);print(round(start/16000),flush=True)
np.savez(args.output,f0=np.array(f),t=np.array(ts))
print('done',len(f),flush=True)
