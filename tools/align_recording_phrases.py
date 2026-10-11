from pathlib import Path
import sys,json,re
import torch,numpy as np,soundfile as sf,yaml
import argparse
p=argparse.ArgumentParser(description='Align supplied kana phrases to an isolated recording with SOFA.')
p.add_argument('audio',type=Path);p.add_argument('phrases',type=Path);p.add_argument('dictionary',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('sofa_code',type=Path);p.add_argument('output',type=Path)
args=p.parse_args()
sys.path.insert(0,str(args.sofa_code))
from modules.task.forced_alignment import LitForcedAlignmentTask
from modules.utils.get_melspec import MelSpecExtractor
rows=json.loads(args.phrases.read_text())
dictionary=yaml.safe_load(args.dictionary.read_text());dictionary={e['grapheme']:e['phonemes'] for e in dictionary['entries']}
torch.set_num_threads(4)
ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
m=LitForcedAlignmentTask(**ckpt['hyper_parameters']);m.load_state_dict(ckpt['state_dict']);m.eval();m.set_inference_mode('force')
mel=MelSpecExtractor(**m.melspec_config,device='cpu')
x,sr=sf.read(args.audio,dtype='float32');x=x.mean(1) if x.ndim==2 else x
if sr != m.melspec_config['sample_rate']:
 from scipy.signal import resample_poly
 from math import gcd
 target=m.melspec_config['sample_rate'];g=gcd(sr,target);x=resample_poly(x,target//g,sr//g);sr=target
result=[]
for r in rows:
 kana=re.sub(r'[^ぁ-ゖー]','',r['kana']);syllables=[]
 for c in kana:
  if c in 'ゃゅょぁぃぅぇぉ' and syllables:syllables[-1]+=c
  elif c=='ー' and syllables:pass
  else:syllables.append(c)
 seq=['SP'];word_idx=[-1]
 for i,s in enumerate(syllables):
  for ph in dictionary[s]:
   seq.append(ph if ph in m.vocab else 'SP');word_idx.append(i)
  seq.append('SP');word_idx.append(-1)
 y=torch.from_numpy(x[round(r['start']*sr):round(r['end']*sr)])
 spec=mel(y).detach().unsqueeze(0);spec=(spec-spec.mean())/spec.std();spec=torch.repeat_interleave(spec,m.melspec_config['scale_factor'],dim=-1)
 with torch.inference_mode():z=m._infer_once(spec,len(y)/sr,seq,syllables,word_idx,False,False)
 ph,ints,words,wints,conf=z[:5]
 r.update(syllables=syllables,aligned_words=words.tolist(),intervals=(wints+r['start']).tolist(),phones_aligned=ph.tolist(),phone_intervals=(ints+r['start']).tolist(),alignment_confidence=float(conf))
 result.append(r);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(r['index'],float(conf),len(words),len(syllables),flush=True)
