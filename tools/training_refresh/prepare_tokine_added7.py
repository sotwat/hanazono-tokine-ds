#!/usr/bin/env python3
"""Align the seven October recordings and preserve their source scores."""
import csv
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from build_alignment_package import ust_kana, note_metrics
from prepare_tokine_added_songs import make_clips, write_csv, DICTIONARY, VOWEL_KANA
from prepare_sofa_labels import parse_token
from prepare_singing_candidates import audit
from ust_wav_alignment import audio_features, HOP
from local_alignment import window_score

SRC = Path('/Users/aaa/Downloads/ustストック')
OUT = Path('/Users/aaa/Documents/DiffSinger/artifacts/tokine-ds-added7')
PREP = OUT/'preparation'
SONGS = {
    'Tell Your World': ['Tell Your World ust/tell your world.ust'],
    'Don\'t say "lazy"': ["Don't Say Lazy UST (1)/メイン.ust"],
    'I beg you': ['I beg you ust/I beg you main ベタ打ち.ust'],
    'メルト': ['メルト002.ust'],
    '炎': ['炎Lisa UST/炎メイン(ネイロ).ust'],
    '雫': ['雫 スキマスイッチ/雫　スキマスイッチ.ust'],
    '熱情のスペクトラム': ['熱情のスペクトラムustFULL/*.ust'],
}

def nfc(s):
    return unicodedata.normalize('NFC', str(s))

def resolve(pattern):
    matches = [p for p in SRC.rglob('*.ust') if __import__('fnmatch').fnmatch(nfc(p.relative_to(SRC)), pattern)]
    if not matches:
        raise FileNotFoundError(pattern)
    return sorted(matches)

def parse(path):
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp932')
    blocks = re.split(r'(?m)^\[(#[^\]]+)\]\s*\n', text.replace('\r\n','\n'))
    bpm, elapsed, notes = 120., 0., []
    for label, body in zip(blocks[1::2], blocks[2::2]):
        fields = dict(line.split('=',1) for line in body.splitlines() if '=' in line)
        if fields.get('Tempo'):
            bpm = float(fields['Tempo'])
        if not label[1:].isdigit() or 'Length' not in fields:
            continue
        length = int(fields['Length'])/480*60/bpm
        lyric = fields.get('Lyric','').strip()
        if lyric and lyric.upper() != 'R':
            notes.append((elapsed,elapsed+length,int(fields['NoteNum']),lyric))
        elapsed += length
    if not notes:
        raise ValueError(f'Empty score: {path}')
    return notes

def match(notes,pitch,duration,start_range=None):
    eligible=[n for n in notes if n[1]-n[0]>=.09]
    starts=np.array([n[0] for n in eligible]); lengths=np.array([n[1]-n[0] for n in eligible]); tones=np.tile([n[2] for n in eligible],3)
    times=np.concatenate([starts+lengths*f for f in (.3,.5,.7)])
    # Full-range search also covers scores supplied as separate verses.
    def scores(scale, offsets):
        values=[]
        for chunk in np.array_split(offsets,max(1,len(offsets)//80)):
            idx=np.rint((times[None,:]*scale+chunk[:,None])/HOP).astype(int)
            valid=(idx>=0)&(idx<len(pitch)); obs=pitch[np.clip(idx,0,len(pitch)-1)]
            diff=(obs-tones[None,:]+6)%12-6
            values.extend((np.exp(-.5*(diff/.7)**2)*valid*(obs>0)).mean(axis=1))
        return np.array(values)
    best=(-1,None,None)
    for scale in np.arange(.88,1.121,.01):
        lower,upper=start_range or (0,duration)
        first=notes[0][0]*scale
        offsets=np.arange(lower-first,upper-first,.2)
        values=scores(scale,offsets); i=int(values.argmax())
        if values[i]>best[0]: best=(float(values[i]),float(scale),float(offsets[i]))
    for scale in np.arange(best[1]-.01,best[1]+.0101,.001):
        offsets=np.arange(best[2]-.3,best[2]+.301,.02)
        values=scores(scale,offsets); i=int(values.argmax())
        if values[i]>best[0]: best=(float(values[i]),float(scale),float(offsets[i]))
    return {'score':round(best[0],4),'time_scale':best[1],'offset_sec':best[2]}

def correct_lyrics(song, rows):
    rules = {
        '雫': [(['に','あ','た','つ'],1,'あっ'), (['お','わ','て','い'],1,'わっ'),
                (['ま','わ','あ','て'],2,'っ'), (['ね','む','う','た'],2,'っ'),
                (['の','こ','お','た'],2,'っ'), (['き','い','と'],1,'っ'), (['ち','い','た'],1,'っ')],
        'メルト': [(['ま','さ','き'],0,'まっ'), (['お','も','い','き','て'],3,'きっ'),
                  (['ま','え','が','み','お','き','た'],5,'きっ'), (['ぜ','た','い'],0,'ぜっ'),
                  (['わ','た','し','だ','て'],3,'だっ')],
    }
    changes=[]
    for pattern, offset, replacement in rules.get(song,[]):
        for i in range(len(rows)-len(pattern)+1):
            if [r['phonetic_lyric'] for r in rows[i:i+len(pattern)]]==pattern:
                row=rows[i+offset]; changes.append({'song':song,'note_index':row['note_index'],'start_sec':row['recording_start_sec'],'before':row['phonetic_lyric'],'after':replacement})
                row['phonetic_lyric']=replacement
    return changes

def adjust_phrase_ends(rows, pitch, rms, duration):
    changes=[]
    spoken=[r for r in rows if r['phonetic_lyric']]
    for i,row in enumerate(spoken):
        a,b=row['recording_start_sec'],row['recording_end_sec']
        following=spoken[i+1]['recording_start_sec'] if i+1<len(spoken) else duration
        if following-b<.25 or b-a<.1: continue
        limit=min(duration,b+1.5,following-.06)
        begin=max(0,round(a/HOP)); stop=min(len(rms),round(limit/HOP))
        energy=rms[begin:stop]
        if len(energy)<5: continue
        threshold=max(.002,float(np.quantile(energy,.85))*.1)
        active=energy>threshold
        # Require 100 ms of quiet; isolated dips inside a vowel are preserved.
        scan=max(1,round((b-a)*.45/HOP))
        end_index=len(active)
        for j in range(scan,len(active)-4):
            if not active[j:j+5].any(): end_index=j; break
        occupied=np.flatnonzero(active[:end_index])
        if not len(occupied): continue
        revised=min(limit,max(a+.08,(begin+occupied[-1]+1)*HOP+.025))
        if abs(revised-b)<.08: continue
        row['recording_end_sec']=round(revised,4)
        row['duration_sec']=round(revised-a,4)
        changes.append({'note_index':row['note_index'],'source_ust':row['source_ust'],'before_end_sec':b,'after_end_sec':revised})
    return changes

def main():
    for folder in ['notes','確認用WAV','対応UST','修正版USTX','features']:
        (PREP/folder).mkdir(parents=True,exist_ok=True)
    all_clips=[]; reports=[]
    for song_index,(song,patterns) in enumerate(SONGS.items(),19):
        wav=next(p for p in (SRC/'DS_wav').glob('*.wav') if nfc(p.stem)==song)
        probe=subprocess.check_output(['/Users/aaa/Documents/DiffSinger/venv/bin/python','/Users/aaa/.codex/skills/ffmpeg-skill/scripts/probe.py',str(wav)],text=True)
        (PREP/'features'/f'{song}_probe.json').write_text(probe)
        pitch,rms,duration=audio_features(wav,PREP/'features'/f'{song}.npz')
        rows=[]; score_reports=[]; source_hashes={str(wav):hashlib.sha256(wav.read_bytes()).hexdigest()}
        for path in [p for pattern in patterns for p in resolve(pattern)]:
            notes=parse(path)
            section_range=None
            if song=='熱情のスペクトラム':
                label=nfc(path.name)
                section_range=((95,155) if '２番' in label else (150,190) if 'Cメロ' in label else (185,238) if 'ラスト' in label else (0,60))
            fit=match(notes,pitch,duration,section_range)
            print(song,path.name,fit,flush=True)
            scaled=[(a*fit['time_scale'],b*fit['time_scale'],t,l) for a,b,t,l in notes]
            windows=[]
            for start in np.arange(scaled[0][0],scaled[-1][1],8):
                w=window_score(scaled,pitch,start,start+12,fit['offset_sec'])
                if w: windows.append(w)
            good=[w for w in windows if w['score']>=.48]
            centers=np.array([(w['ust_start']+w['ust_end'])/2 for w in good]); offsets=np.array([w['offset_sec'] for w in good])
            source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            dest=PREP/'対応UST'/song/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
            previous=''
            for i,(a,b,t,l) in enumerate(scaled,1):
                # Use one monotone mapping for both boundaries to avoid overlaps.
                def warp(x):
                    return x+(float(np.interp(x,centers,offsets)) if good else fit['offset_sec'])
                start,end=warp(a),warp(b); lyric=ust_kana(l)
                if song=='炎' and re.search(r'(?:^| )・',l): lyric='っ'
                if l in ['-','ー','+'] and previous:
                    phones,_=parse_token(previous,DICTIONARY);lyric=VOWEL_KANA.get(phones[-1],'')
                if lyric: previous=lyric
                voice,median,matched=note_metrics(start,end,t,pitch)
                flag=''
                if start<0 or end>duration or end<=start: flag='outside_recording'
                elif fit['score']<.45 or not good: flag='alignment_review'
                elif end-start>=.16 and voice<.25: flag='low_voicing_review'
                elif end-start>=.16 and voice>.55 and matched is not None and matched<.4: flag='pitch_review'
                rows.append({'source_ust':str(path),'note_index':i,'recording_start_sec':round(start,4),'recording_end_sec':round(end,4),'duration_sec':round(end-start,4),'midi_note':t,'ust_lyric':l,'phonetic_lyric':lyric,'voiced_fraction':voice,'median_pitch_difference_semitones':median,'pitch_match_fraction':matched,'review_flag':flag})
            score_reports.append({'path':str(path),'note_count':len(notes),'fit':fit,'local_windows':windows})
        rows.sort(key=lambda r:r['recording_start_sec'])
        lyric_changes=correct_lyrics(song,rows)
        end_changes=adjust_phrase_ends(rows,pitch,rms,duration)
        for a,b in zip(rows,rows[1:]):
            if a['recording_end_sec']>b['recording_start_sec']+.015:
                a['review_flag']=b['review_flag']='overlap_review'
        write_csv(PREP/'notes'/f'{song}.csv',rows)
        # Export a neutral 120 BPM score whose seconds match the original WAV.
        export_notes=[]
        for r in rows:
            a,b=r['recording_start_sec'],r['recording_end_sec']
            if a<0 or b<=a or b>duration or not r['phonetic_lyric']: continue
            export_notes.append({'position':round(a*960),'duration':max(1,round(b*960)-round(a*960)),'tone':r['midi_note'],'lyric':r['phonetic_lyric'],'pitch':{'data':[]},'vibrato':{'length':0}})
        ustx={'name':song+' 録音照合用','ustx_version':'0.6','resolution':480,'tempos':[{'position':0,'bpm':120.0}],'time_signatures':[{'bar_position':0,'beat_per_bar':4,'beat_unit':4}],'tracks':[{'track_name':song,'track_color':'Blue','singer':'','phonemizer':'OpenUtau.Plugin.Builtin.JapaneseCVPhonemizer'}],'voice_parts':[{'name':song,'track_no':0,'position':0,'notes':export_notes}],'wave_parts':[]}
        (PREP/'修正版USTX'/f'{song}_録音基準.ustx').write_text(yaml.safe_dump(ustx,allow_unicode=True,sort_keys=False))
        clips=make_clips(rows,duration)
        for ci,c in enumerate(clips,1):
            item={'song':song,'wav':str(wav),'candidate_id':f'{song_index:02d}_{ci:03d}',**c};all_clips.append(item)
            with sf.SoundFile(wav) as f:
                f.seek(round(c['start_sec']*f.samplerate));x=f.read(round(c['duration_sec']*f.samplerate),dtype='float32',always_2d=True)
                x=x.mean(axis=1);sf.write(PREP/'確認用WAV'/f"{item['candidate_id']}.wav",x,f.samplerate,subtype='PCM_24')
        reports.append({'song':song,'wav':str(wav),'audio_audit':audit(wav),'scores':score_reports,'note_count':len(rows),'candidate_count':len(clips),'source_sha256':source_hashes,'phrase_end_changes':end_changes,'lyric_changes':lyric_changes,'missing_lyrics':[r['ust_lyric'] for r in rows if not r['phonetic_lyric']]})
        (PREP/'alignment_summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    write_csv(PREP/'フレーズ候補.csv',all_clips)
    print('TOTAL',len(all_clips),flush=True)

if __name__=='__main__': main()
