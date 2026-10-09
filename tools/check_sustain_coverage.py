"""Exercise short CV tails and newly supported continuous consonants."""
import argparse,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort

def check(bank,output):
 opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.log_severity_level=3
 m=onnx.load(bank/'tokine_ds_clarity_hybrid.tokine.onnx')
 raw='original_su_raw' if any(i.name=='original_su_long' for i in m.graph.initializer) else 'su_raw'
 m.graph.output.append(onnx.helper.make_tensor_value_info(raw,onnx.TensorProto.FLOAT,[1,'frames',128]))
 ac=ort.InferenceSession(m.SerializeToString(),opts,providers=['CPUExecutionProvider']);voc=ort.InferenceSession(str(next((bank/'dsvocoder').glob('*.onnx'))),opts,providers=['CPUExecutionProvider']);rows=[]
 for token,n,midi in [(3,48,70),(20,48,70),(2,90,70),(24,90,70),(3,30,70),(8,90,70),(3,90,79)]:
  durations=np.array([[12,8,n,12]],np.int64);f0=np.full((1,int(durations.sum())),440*2**((midi-69)/12),np.float32);f0[:,:12]=0;f0[:,-12:]=0
  feed=dict(tokens=np.array([[1,8,token,1]],np.int64),durations=durations,f0=f0,depth=np.array(.1,np.float32),steps=np.array(20,np.int64))
  mel,rawmel=ac.run(None,feed);original=mel[:1];mask=np.ones(mel.shape[1],bool)
  if n>=40 and token!=8:mask[20+(20 if n<60 else 32):20+n]=False
  assert np.array_equal(original[:,mask],rawmel[:,mask])
  y=voc.run(None,dict(mel=mel,f0=f0))[0];assert np.isfinite(y).all()
  row=dict(token=token,frames=n,midi=midi,untouched_mel_exact=True,finite=True,peak=float(abs(y).max()),changed=bool(np.any(original!=rawmel)));rows.append(row)
  assert row['changed']==(n>=40 and token!=8)
 output.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.bank,a.output)
