#!/usr/bin/env python3
"""Prepare separate, recording-relative candidates for the three added songs."""
import csv,json,unicodedata,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,soundfile as sf,yaml
from ust_wav_alignment import audio_features,align,parse_ust,HOP
from local_alignment import window_score
from build_alignment_package import ust_kana,note_metrics
from prepare_singing_candidates import audit
from prepare_sofa_labels import mapping,parse_token
DICTIONARY=mapping()
VOWEL_KANA={"a":"あ","i":"い","u":"う","e":"え","o":"お","N":"ん"}

SRC=Path('/Users/aaa/Downloads/ustストック')
OUT=Path('/Users/aaa/Documents/DiffSinger/artifacts/tokine-ds-added3')
PREP=OUT/'preparation_v3'

def read_score(song):
 if song=='ノーダウト':
  p=SRC/'ノーダウト UST/main.ust';return p,parse_ust(p)[0]
 if song=='サムライハート':
  p=SRC/'サムライハート/サムライハート.ustx';d=yaml.safe_load(p.read_text(encoding='utf-8-sig'));bpm=d['tempos'][0]['bpm'];res=d.get('resolution',480)
  return p,sorted([( (part['position']+n['position'])/res*60/bpm,(part['position']+n['position']+n['duration'])/res*60/bpm,n['tone'],n['lyric']) for part in d['voice_parts'] if part['track_no'] in [1,3] for n in part['notes']])
 p=SRC/'花になって/花になって.vsqx';d=ET.parse(p).getroot()
 for x in d.iter():x.tag=x.tag.split('}')[-1]
 master=d.find('masterTrack');res=int(master.findtext('resolution'));tempos=sorted((int(t.findtext('t')),float(t.findtext('v'))/100) for t in master.findall('tempo'))
 def sec(t):
  total=0.
  for i,(a,bpm) in enumerate(tempos):
   end=min(t,tempos[i+1][0]) if i+1<len(tempos) else t
   if end>a:total+=(end-a)/res*60/bpm
  return total
 notes=[]
 for track in d.findall('vsTrack'):
  if track.findtext('tNo')!='0':continue
  for part in track.findall('vsPart'):
   pos=int(part.findtext('t'))
   for n in part.findall('note'):
    a=pos+int(n.findtext('t'));notes.append((sec(a),sec(a+int(n.findtext('dur'))),int(n.findtext('n')),n.findtext('y')))
 return p,notes

def write_csv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def apply_lyric_corrections(song,rows):
 # The distributed scores omit these closures/syllables; published lyrics
 # and recording ASR agree. Each correction stays on its existing note.
 rules={'花になって':[(['は','ま','な','い'],1,'まん'),(['く','だ','な','い'],1,'だん'),(['つ','つ','ん','あ','げ'],2,'んで')],
        'ノーダウト':[(['き','ら','て','く','れ'],1,'らっ'),(['つ','か','て','か','れ','い'],1,'かっ')]}
 for pattern,index,replacement in rules.get(song,[]):
  for i in range(len(rows)-len(pattern)+1):
   if [x['phonetic_lyric'] for x in rows[i:i+len(pattern)]]==pattern:rows[i+index]['phonetic_lyric']=replacement

def make_clips(rows,duration):
 notes=[r for r in rows if r['phonetic_lyric'] and r['review_flag']!='outside_recording'];groups=[];g=[]
 for i,n in enumerate(notes):
  if g and (n['recording_start_sec']-g[-1]['recording_end_sec']>=.22 or n['recording_end_sec']-g[0]['recording_start_sec']>7.5):groups.append(g);g=[]
  g.append(n)
 if g:groups.append(g)
 clips=[]
 for g in groups:
  first=notes.index(g[0]);last=notes.index(g[-1]);a=g[0]['recording_start_sec'];b=g[-1]['recording_end_sec']
  prev=notes[first-1]['recording_end_sec'] if first else 0;following=notes[last+1]['recording_start_sec'] if last+1<len(notes) else duration
  start=max(0,a-.18,(prev+a)/2);end=min(duration,b+.18,(b+following)/2)
  if end-start<1.5 or len(g)<4:continue
  flags=sorted({x['review_flag'] for x in g if x['review_flag']})
  events=[{'start':n['recording_start_sec'],'end':n['recording_end_sec'],'lyric':n['phonetic_lyric'],'midi':n['midi_note']} for n in g]
  clips.append({'start_sec':round(start,3),'end_sec':round(end,3),'duration_sec':round(end-start,3),'note_count':len(g),'ust_kana_preview':''.join(n['phonetic_lyric'] for n in g)[:70],'draft_syllables':' '.join(n['phonetic_lyric'] for n in g),'note_events_json':json.dumps(events,ensure_ascii=False),'review_flags':','.join(flags),'needs_boundary_review':True})
 return clips

def main():
 PREP.mkdir(parents=True,exist_ok=True);(PREP/'確認用WAV').mkdir(exist_ok=True);(PREP/'notes').mkdir(exist_ok=True)
 clips=[];reports=[]
 for si,song in enumerate(['サムライハート','ノーダウト','花になって'],16):
  wav=next(p for p in (SRC/'DS_wav').glob('*.wav') if unicodedata.normalize('NFC',p.stem)==song)
  p,notes=read_score(song);pitch,rms,duration=audio_features(wav,PREP/'features'/f'{song}.npz')
  trials=[]
  for factor in np.arange(.90,1.151,.01):
   scaled=[(a*factor,b*factor,t,l) for a,b,t,l in notes];match=align(scaled,pitch,duration);trials.append((match['score'],factor,match))
  _,factor,match=max(trials,key=lambda x:x[0]);scaled=[(a*factor,b*factor,t,l) for a,b,t,l in notes]
  print(song,'scale',round(factor,3),'alignment',match,flush=True)
  windows=[]
  for start in np.arange(0,max(n[1] for n in scaled),8):
   result=window_score(scaled,pitch,start,start+12,match['offset_sec'])
   if result:windows.append(result)
  good=[w for w in windows if w['score']>=.45]
  if not good:raise RuntimeError(f'No supported alignment for {song}')
  centers=np.array([(w['ust_start']+w['ust_end'])/2 for w in good]);offsets=np.array([w['offset_sec'] for w in good]);rows=[];previous=''
  for i,(a,b,t,l) in enumerate(scaled,1):
   offset=float(np.interp((a+b)/2,centers,offsets));a+=offset;b+=offset
   lyric=ust_kana(l)
   if l in ['-','ー','+'] and previous:
    ph,_=parse_token(previous,DICTIONARY);lyric=VOWEL_KANA.get(ph[-1],'')
   if lyric:previous=lyric
   voice,median,matched=note_metrics(a,b,t,pitch);flag=''
   if a<0 or b>duration:flag='outside_recording'
   elif b-a>=.16 and voice<.25:flag='low_voicing_review'
   elif b-a>=.16 and voice>.55 and matched is not None and matched<.4:flag='pitch_review'
   rows.append({'source_ust':str(p),'note_index':i,'recording_start_sec':round(a,3),'recording_end_sec':round(b,3),'duration_sec':round(b-a,3),'midi_note':t,'ust_lyric':l,'phonetic_lyric':lyric,'voiced_fraction':voice,'median_pitch_difference_semitones':median,'pitch_match_fraction':matched,'review_flag':flag})
  rows.sort(key=lambda x:x['recording_start_sec']);apply_lyric_corrections(song,rows);write_csv(PREP/'notes'/f'{song}.csv',rows)
  candidates=make_clips(rows,duration)
  for ci,c in enumerate(candidates,1):
   item={'song':song,'wav':str(wav),'candidate_id':f'{si:02d}_{ci:03d}',**c};clips.append(item)
   with sf.SoundFile(wav) as f:
    f.seek(round(c['start_sec']*f.samplerate));x=f.read(round(c['duration_sec']*f.samplerate),dtype='float32',always_2d=True)[:,0];sf.write(PREP/'確認用WAV'/f"{item['candidate_id']}.wav",x,f.samplerate,subtype='PCM_24')
  reports.append({'song':song,'wav':str(wav),'score':str(p),'audio_audit':audit(wav),'time_scale':float(factor),'global_match':match,'local_windows':windows,'note_count':len(rows),'candidate_count':len(candidates),'missing_lyrics':[r['ust_lyric'] for r in rows if not r['phonetic_lyric']]})
 write_csv(PREP/'フレーズ候補.csv',clips);(PREP/'alignment_summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2));print('candidates',len(clips),flush=True)
if __name__=='__main__':main()
