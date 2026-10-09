"""Check finite audio, pitch, and exact bypass using outputs from one run."""
import argparse,json,time
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort,pyworld

def validate(path):
 opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.log_severity_level=3
 a=ort.InferenceSession(str(path/'acoustic.onnx'),opts,providers=['CPUExecutionProvider'])
 m=onnx.load(path/'vocoder.onnx');m.graph.output.append(onnx.helper.make_tensor_value_info('baseline',onnx.TensorProto.FLOAT,[1,'samples']))
 v=ort.InferenceSession(m.SerializeToString(),opts,providers=['CPUExecutionProvider'])
 results=[]
 for midi,n in [(60,64),(74,64),(77,64),(81,64),(77,1723),(0,32)]:
  f0=np.full((1,n),440*2**((midi-69)/12) if midi else 0,np.float32)
  feed=dict(tokens=np.array([[3]],np.int64),durations=np.array([[n]],np.int64),f0=f0,depth=np.array(.1,np.float32),steps=np.array(20,np.int64))
  start=time.time();mel=a.run(None,feed)[0];y,base=v.run(None,dict(mel=mel,f0=f0))
  assert y.shape==(1,n*512) and np.isfinite(y).all(), (midi,y.shape,int((~np.isfinite(y)).sum()),int((~np.isfinite(base)).sum()),int((~np.isfinite(mel)).sum()))
  if midi<=74:assert np.array_equal(y,base)
  if midi>74:assert not np.array_equal(y,base)
  pitch=pyworld.harvest(y[0].astype(np.float64),44100,frame_period=512/44100*1000,f0_floor=50,f0_ceil=1800)[0]
  pitch=pitch[10:-10];pitch=pitch[pitch>0]
  error=float(np.median(abs(1200*np.log2(pitch/f0[0,0])))) if midi and len(pitch) else None
  row=dict(midi=midi,frames=n,finite=True,peak=float(abs(y).max()),bypass_exact=bool(np.array_equal(y,base)),pitch_median_cents=error,seconds=time.time()-start)
  results.append(row);print(row,flush=True)
 (path/'validation.json').write_text(json.dumps(results,indent=2));return results
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('path',type=Path);validate(p.parse_args().path)
