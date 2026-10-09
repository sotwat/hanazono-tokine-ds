"""Measure pitch on the same interior vowel frames in a range-render comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import pyworld as pw
import soundfile as sf

p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r=np.load(a.inputs)
def get(k):return r[k if k in r else 'dbg_'+k]
f0=get('f0')[0];mask=np.zeros(len(f0),bool);start=0
for tok,dur in zip(get('tokens')[0],get('durations')[0]):
    if tok in [3,9,14,20,28] and dur>8:mask[start+3:start+dur-3]=True
    start+=dur
mask &= (f0>0)&((f0>=698.45)|(f0<=185))
rows={}
for name in ['original','hybrid']:
    x,sr=sf.read(a.output/(name+'.wav'))
    measured,t=pw.harvest(x,sr,f0_floor=55,f0_ceil=1600,frame_period=512/44100*1000)
    measured=measured[:len(f0)];valid=mask&(measured>0)
    errors=1200*np.log2(measured[valid]/f0[valid])
    rows[name]={'vowel_frames':int(mask.sum()),'voiced_frames':int(valid.sum()),
        'median_abs_cents':float(np.median(abs(errors))),
        'p90_abs_cents':float(np.percentile(abs(errors),90)),
        'over_50_cents':int(np.count_nonzero(abs(errors)>50))}
(a.output/'vowel-pitch-check.json').write_text(json.dumps(rows,indent=2));print(rows)
