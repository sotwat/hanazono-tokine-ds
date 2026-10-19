"""Compare processed voicebanks with fixed phoneme lengths and pitch inputs."""
import argparse,json
from pathlib import Path
import numpy as np,soundfile as sf,onnxruntime as ort
R=Path(__file__).parent;B=Path('/Users/aaa/Documents/DiffSinger');BANK=B/'local-voicebanks/花園トキネDS v1.2.4-rc.1'
def session(p):
 o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;ort.set_seed(42);return ort.InferenceSession(str(p),sess_options=o,providers=['CPUExecutionProvider'])
def main(model,tag):
 ac=session(model);vc=session(BANK/'dsvocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx');ph=json.loads((BANK/'tokine_ds_clarity_hybrid.phonemes.json').read_text());cases=[]
 for name,p in [('low_native',B/'artifacts/tokine-low-register-20261015/native.npz'),('high_native',B/'artifacts/tokine-high-balance/ki-before-acoustic.npz')]:
  z=np.load(p);cases.append((name,{k:z['dbg_'+k] for k in ['tokens','durations','f0']}))
 for j,s in enumerate(json.loads((R/'evaluation/new_heldout.ds').read_text())):
  ds=np.array(list(map(float,s['ph_dur'].split())));bounds=np.rint(np.r_[0,np.cumsum(ds)]*44100/512).astype(int);d=np.diff(bounds);assert np.all(d>0)
  f=np.array(list(map(float,s['f0_seq'].split())));f=np.interp(np.arange(sum(d))*512/44100,np.arange(len(f))*s['f0_timestep'],f).astype('float32')[None,:]
  cases.append((f'new_{j+1:02}',dict(tokens=np.array([[ph[x] for x in s['ph_seq'].split()]],dtype='int64'),durations=d.astype('int64')[None,:],f0=f)))
 out=R/'processed-comparison'/tag;out.mkdir(parents=True,exist_ok=True);report=[];aud=[]
 for name,x in cases:
  m=ac.run(None,dict(**x,depth=np.array(.1,dtype='float32'),steps=np.array(20,dtype='int64')))[0];w=vc.run(None,dict(mel=m,f0=x['f0']))[0].reshape(-1);assert np.isfinite(w).all() and np.max(abs(w))>1e-4
  sf.write(out/(name+'.wav'),w,44100,subtype='PCM_24');aud.extend([w,np.zeros(22050)]);np.savez(out/(name+'.npz'),**x)
  report.append(dict(name=name,seconds=len(w)/44100,rms=float(np.sqrt(np.mean(w*w))),peak=float(np.max(abs(w))),over_full_scale=int(np.sum(abs(w)>=1))))
  print(tag,name,report[-1],flush=True)
 sf.write(out/'comparison.wav',np.concatenate(aud),44100,subtype='PCM_24');(out/'metrics.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('tag');a=p.parse_args();main(a.model,a.tag)
