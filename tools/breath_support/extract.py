"""Detect breaths only inside previously aligned pause intervals."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml


def main(root,source,detector):
    root.mkdir(parents=True,exist_ok=True);(root/'clips').mkdir(exist_ok=True)
    sys.path.insert(0,str(detector/'FoxBreatheLabeler'))
    from Models.CVNT import CVNT
    cfg=yaml.safe_load((detector/'20ms/config.yaml').read_text())
    ck=torch.load(detector/'20ms/model.ckpt',map_location='cpu',weights_only=True)['model']
    model=CVNT(cfg,output_size=1);model.load_state_dict({k[6:]:v for k,v in ck.items() if k.startswith('model.')});model.eval().to('mps')
    manifest={v['name']:v for v in json.loads((source/'dataset_manifest.json').read_text())}
    labels=list(csv.DictReader((source/'raw/tokine/transcriptions.csv').open()));rows=[];seen=set();candidate=0
    torch.set_num_threads(4)
    with torch.inference_mode():
      for ri,row in enumerate(labels):
        meta=manifest[row['name']];wav=source/'raw/tokine/wavs'/f"{row['name']}.wav";x,sr=sf.read(wav,dtype='float32');assert sr==44100
        cursor=0
        for phone,dur in zip(row['ph_seq'].split(),map(float,row['ph_dur'].split())):
          start=cursor;cursor+=dur
          if phone not in {'SP','AP'} or dur<.12:continue
          a=max(0,int(start*sr));b=min(len(x),int(cursor*sr));y=x[a:b]
          if len(y)<6000 or np.sqrt(np.mean(y*y))<.0004:continue
          # Leave context inside the labelled pause, never neighbouring consonants.
          frames=F.pad(torch.from_numpy(y),(71,71)).unfold(0,1024,882)[None].to('mps')
          prob=model(frames).sigmoid().cpu().numpy().reshape(-1)
          active=prob>=.4;indices=np.flatnonzero(active);groups=[]
          for v in indices:
            if not groups or v-groups[-1][-1]>2:groups.append([v])
            else:groups[-1].append(v)
          for group in groups:
            if float(np.mean(prob[group]))<.60 or float(np.max(prob[group]))<.8:continue
            lo=max(0,int(group[0]*882)-441);hi=min(len(y),int((group[-1]+1)*882)+441)
            if hi-lo<.12*sr or hi-lo>1.1*sr:continue
            z=y[lo:hi];db=20*np.log10(np.sqrt(np.mean(z*z))+1e-12);peak=float(np.max(np.abs(z)))
            if db < -55 or peak>.96:continue
            origin_start=float(meta['source_start_sec'])+start+lo/sr;origin_end=origin_start+len(z)/sr
            key=(meta['source_wav'],round(origin_start,2),round(origin_end,2))
            if key in seen:continue
            seen.add(key);name=f'breath_{candidate:04d}';candidate+=1
            sf.write(root/'clips'/f'{name}.wav',z,sr,subtype='PCM_24')
            rows.append(dict(name=name,song=meta['song'],source_wav=meta['source_wav'],start_sec=origin_start,end_sec=origin_end,
                parent=row['name'],label=phone,probability=float(np.mean(prob[group])),min_probability=float(np.min(prob[group])),rms_dbfs=float(db),peak=peak,
                seconds=len(z)/sr,sha256=hashlib.sha256((root/'clips'/f'{name}.wav').read_bytes()).hexdigest()))
        if ri%25==0:print(ri,len(rows),flush=True);(root/'candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (root/'candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('done',len(rows),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('source',type=Path);p.add_argument('detector',type=Path);a=p.parse_args();main(a.output,a.source,a.detector)
