import sys,csv,json,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np,soundfile as sf,yaml
from scipy.signal import resample_poly
from rapidfuzz import fuzz
sys.path.insert(0,str(Path(__file__).parent))
from build_alignment_package import kana
ROOT=Path(__file__).parent;P=ROOT/'preparation';OLD=ROOT.parent/'tokine-ds-quality25'
def read(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 rows=read(P/'フレーズ候補.csv');qa={r['candidate_id']:r for r in read(P/'音符拘束補正/品質監査.csv')};labs={r['name']:r for r in read(P/'音符拘束補正/transcriptions.csv')};inp={r['candidate_id']:r for r in read(P/'SOFA入力一覧.csv')}
 transcripts=[json.loads(x) for x in (P/'transcripts.jsonl').read_text().splitlines()];audit=[];accepted=[]
 for r in rows:
  cid=r['candidate_id'];song=r['song'];a,b=float(r['start_sec']),float(r['end_sec']);q=qa.get(cid,{});reasons=[]
  refs=[x for x in transcripts if x['song']==song and x['start']<b and x['start']+x['duration']>a]
  sim=fuzz.partial_ratio(kana(r['draft_syllables']),''.join(kana(x['text']) for x in refs)) if refs else 0
  if sim<85:reasons.append('asr_lyric_support_below_85')
  if q.get('status')!='candidate_pass' or int(q.get('unresolved_anchor',1)):reasons.append('boundary_review')
  if float(q.get('confidence',0))<.65:reasons.append('sofa_confidence_below_065')
  if r['review_flags']:reasons.extend(r['review_flags'].split(','))
  lab=labs.get(cid);err=[]
  if lab:
   phones=lab['ph_seq'].split();dur=np.array(list(map(float,lab['ph_dur'].split())));starts=np.r_[0,np.cumsum(dur)[:-1]];spoken=[i for i,p in enumerate(phones) if p not in {'SP','AP'}];anchors=json.loads(inp[cid]['phone_anchors_json'])
   expected=(P/'SOFA候補'/f'{cid}.lab').read_text().split()
   if [phones[i] for i in spoken]!=expected or len(anchors)!=len(spoken):reasons.append('phone_sequence_mismatch')
   else:
    for i,anchor in zip(spoken,anchors):
     if phones[i] in {'a','i','u','e','o','N'}:err.append(abs(starts[i]+a-anchor['start']))
   if err and (np.median(err)>.12 or np.quantile(err,.9)>.25):reasons.append('note_boundary_disagreement')
  else:reasons.append('missing_label')
  world=np.load(P/'features'/f'{song}_world.npz');base=np.load(P/'features'/f'{song}.npz');t=world['t'];wp=world['pitch'];bp=np.interp(t,np.arange(len(base['pitch']))*.02,base['pitch']);mask=(t>=a)&(t<b)&(wp>0)&(bp>0);agree=float(np.mean(abs(wp[mask]-bp[mask])<.5)) if mask.any() else 0
  if agree<.7:reasons.append('independent_pitch_disagreement')
  audit.append(dict(candidate_id=cid,song=song,status='excluded' if reasons else 'accepted',asr_similarity=round(sim,2),sofa_confidence=q.get('confidence',''),pitch_agreement=round(agree,4),boundary_median_sec=round(float(np.median(err)),4) if err else '',boundary_p90_sec=round(float(np.quantile(err,.9)),4) if err else '',reasons=','.join(sorted(set(reasons)))))
  if not reasons:accepted.append(r)
 holdout=set();raytexts=set()
 for song in dict.fromkeys(r['song'] for r in rows):
  items=[r for r in accepted if r['song']==song]
  if len(items)<2:raise RuntimeError(f'{song}: only {len(items)} accepted')
  if song=='ray_キー+5':continue
  v=items[len(items)//2];holdout.add(v['candidate_id'])
  if song=='ray_原曲キー':raytexts.add(kana(v['draft_syllables']))
 # The two Ray takes share a melody: reserve matching lyric material in both keys.
 for r in accepted:
  if r['song']=='ray_キー+5' and any(fuzz.partial_ratio(kana(r['draft_syllables']),s)>=85 for s in raytexts):holdout.add(r['candidate_id'])
 bad={'サムライハート','I beg you','メルト','花になって'};oldmanifest=json.loads((OLD/'dataset_manifest.json').read_text());retained=[r for r in oldmanifest if r['song'] not in bad];retained=[r for r in retained if r['validation'] or not any(v['validation'] and v['source_wav']==r['source_wav'] and float(r['source_start_sec'])<float(v['source_end_sec']) and float(v['source_start_sec'])<float(r['source_end_sec']) for v in retained)];oldnames={r['name'] for r in retained};trans=[r for r in read(OLD/'raw/tokine/transcriptions.csv') if r['name'] in oldnames]
 # Reserve neighbouring clips too when their audio padding overlaps a validation clip.
 for r in accepted:
  if r['candidate_id'] not in holdout and any(v['candidate_id'] in holdout and v['wav']==r['wav'] and float(r['start_sec'])<float(v['end_sec']) and float(v['start_sec'])<float(r['end_sec']) for v in accepted):holdout.add(r['candidate_id'])
 raw=ROOT/'raw/tokine';(raw/'wavs').mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/'dictionary.txt',ROOT/'dictionary.txt')
 manifest=[]
 for r in retained:
  src=OLD/'raw/tokine/wavs'/f"{r['name']}.wav";dst=raw/'wavs'/src.name
  if not dst.exists():dst.symlink_to(src)
  manifest.append(dict(r,origin='retained_quality25'))
 hashes={}
 for r in accepted:
  cid=r['candidate_id'];valid=cid in holdout;name=('validation_' if valid else 'training_')+cid;x,sr=sf.read(P/'確認用WAV'/f'{cid}.wav',dtype='float32');assert sr==48000 and x.ndim==1
  x=resample_poly(x,147,160).astype('float32');sf.write(raw/'wavs'/f'{name}.wav',x,44100,subtype='PCM_24');l=labs[cid];ph=l['ph_seq'].split();d=list(map(float,l['ph_dur'].split()));d[-1]+=len(x)/44100-sum(d);assert min(d)>0 and len(ph)==len(d)
  trans.append(dict(name=name,ph_seq=' '.join(ph),ph_dur=' '.join(f'{v:.8f}' for v in d)))
  src=r['wav']
  if src not in hashes:hashes[src]=hashlib.sha256(Path(src).read_bytes()).hexdigest()
  manifest.append(dict(name=name,song=r['song'],source_candidate_id=cid,source_wav=src,source_start_sec=float(r['start_sec']),source_end_sec=float(r['end_sec']),duration_sec=len(x)/44100,validation=valid,source_sha256=hashes[src],label_method='SOFA phoneme passthrough + note anchors + ASR85 + independent F0',origin='20261019_corrected_or_added'))
 assert len(trans)==len(set(r['name'] for r in trans))
 for r in manifest:
  for v in manifest:
   if r['validation'] and not v['validation'] and r['source_wav']==v['source_wav']:assert float(r['source_end_sec'])<=float(v['source_start_sec']) or float(v['source_end_sec'])<=float(r['source_start_sec']), (r['name'],v['name'])
 for r in manifest:
  if r['song'] in bad:assert r['origin']=='20261019_corrected_or_added'
 for r in trans:
  f=sf.info(raw/'wavs'/f"{r['name']}.wav");assert abs(sum(map(float,r['ph_dur'].split()))-f.duration)<1e-5
 write(raw/'transcriptions.csv',trans);write(P/'selection_audit.csv',audit);(ROOT/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 config=yaml.safe_load((OLD/'config.yaml').read_text());config['dictionaries']['ja']=str(ROOT/'dictionary.txt');config['datasets'][0]['raw_data_dir']=str(raw);config['binary_data_dir']=str(ROOT/'binary');config['finetune_ckpt_path']='/Users/aaa/Documents/DiffSinger/checkpoints/tokine_ds_multispeaker_nit070/model_ckpt_steps_4000.ckpt';config['frozen_params']=['model.fs2.spk_embed'];config.update(max_updates=1500,val_check_interval=500,permanent_ckpt_start=500,permanent_ckpt_interval=500,num_ckpt_keep=4)
 (ROOT/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
 summary=dict(candidates=len(rows),accepted_new=len(accepted),accepted_new_sec=sum(float(r['duration_sec']) for r in accepted),accepted_new_by_song=dict(Counter(r['song'] for r in accepted)),excluded_reasons=dict(Counter(s for r in audit for s in r['reasons'].split(',') if s)),retained_old=len(retained),removed_old=len(oldmanifest)-len(retained),total=len(manifest),training=sum(not r['validation'] for r in manifest),validation=sum(r['validation'] for r in manifest),by_song=dict(Counter(r['song'] for r in manifest)),positive_durations=True,source_split_overlap=False,corrected_four_old_labels_excluded=True)
 (ROOT/'dataset_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
