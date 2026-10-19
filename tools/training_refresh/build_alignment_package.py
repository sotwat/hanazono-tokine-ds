#!/usr/bin/env python3
"""Create recording-relative note tables and conservative lyric review flags."""

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import jaconv
import numpy as np
import pykakasi
from rapidfuzz import fuzz

from ust_wav_alignment import OUTPUT, ROOT, USTS, HOP, load_notes


KAKASI = pykakasi.kakasi()
ALTERNATE = USTS / 'staple stable' / 'ivyleaf33' / 'Staple Stable Main.ust'


def path(song, basename):
    found = list((USTS / song).rglob(basename))
    if len(found) > 1:
        found = [item for item in found if '間奏' not in item.name]
    if len(found) != 1:
        raise ValueError((song, basename, found))
    return found[0]


SELECTED = {
    'Deep in Abyss': [(path('Deep in Abyss', 'Deep in Abyss レグパート.ust'), .38)],
    'Sincerely_take1': [(path('Sincerely', 'Sincerely  _Main.ust'), 8.08)],
    'Sincerely_take2': [(path('Sincerely', 'Sincerely  _Main.ust'), 8.10)],
    'ninelie': [(path('ninelie', 'aimerpart.ust'), 13.38, 0, 53),
                (path('ninelie', 'aimerpart.ust'), 155.7, 54, 100)],
    'staple stable': [(ALTERNATE, 14.84)],
    'いかないで': [(path('いかないで', 'いかないで.ust'), 22.56)],
    'それがあなたの幸せとしても': [(path('それがあなたの幸せとしても', '*単独音.ust'), -3.18)],
    'ただ君に晴れ': [(path('ただ君に晴れ', 'すっぴんmain.ust'), 1.66)],
    'フォニイ': [(path('フォニイ', 'main.ust'), -4.12)],
    'フォニイ_消える消える': [(path('フォニイ', 'kiere.ust'), -4.12)],
    'ミッドナイト・リフレクション_take1': [(path('ミッドナイト・リフレクション', 'sm44924021_ミッドナイトリフレクション.ust'), .78)],
    'ミッドナイト・リフレクション_take2': [(path('ミッドナイト・リフレクション', 'sm44924021_ミッドナイトリフレクション.ust'), .78)],
    '千本桜': [(path('千本桜', '千本桜（テト）.ust'), 34.22)],
    '君の知らない物語': [(path('君の知らない物語', 'main.ust.ust'), 9.38)],
    '君の記憶': [(path('君の記憶', 'main1.ust'), 1.54)],
    '地球最後の告白を': [
        (path('地球最後の告白を', '最初 ベタ打ち.ust'), .01),
        (path('地球最後の告白を', '1番 ベタ打ち.ust'), 31.08),
        (path('地球最後の告白を', '2番 ベタ打ち.ust'), 114.5),
        (path('地球最後の告白を', 'ラスト ベタ打ち.ust'), 190.3),
    ],
    '夜に駆ける': [(path('夜に駆ける', 'main.ust'), -.06)],
    '少女レイ': [(OUTPUT / '修正版USTX' / '少女レイ_DS_録音照合.ustx', 15.94)],
}


def kana(text):
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(part['hira'] for part in KAKASI.convert(text))
    text = jaconv.kata2hira(text)
    text = re.sub(r'[^ぁ-ゖa-z]', '', text.lower())
    return text


def ust_kana(text):
    text = unicodedata.normalize('NFKC', text).strip()
    if not text or text.upper() == 'R':
        return ''
    if re.search(r'[ぁ-ゖァ-ヺ]', text):
        # In VCV aliases such as "o い", the roman vowel is the preceding
        # context; the kana is the syllable sung on this note.
        # UST vowels, flags, and aliases after the syllable are not lyrics.
        text = re.search(r'[ぁ-ゖァ-ヺーっん]+', text).group()
    elif text.lower() in {'a', 'i', 'u', 'e', 'o'}:
        # Only a bare vowel is a sung vowel. "a R" denotes a rest, while
        # "a t_C5" is a consonant onset for the following syllable.
        return {'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お'}[text.lower()]
    elif re.fullmatch(r'[a-zA-Z]+', text):
        text = jaconv.alphabet2kana(text.lower())
    else:
        return ''
    return kana(text)


def note_metrics(start, end, tone, pitch):
    times = np.arange(start + (end-start)*.2, start + (end-start)*.8, HOP)
    if not len(times):
        times = np.array([(start+end)/2])
    frames = np.rint(times/HOP).astype(int)
    valid = (frames>=0)&(frames<len(pitch))
    observed = pitch[np.clip(frames,0,len(pitch)-1)]
    voiced = valid & (observed>0)
    if not voiced.any():
        return 0.,None,None
    folded = (observed[voiced]-tone+6)%12-6
    return round(float(voiced.mean()),3),round(float(np.median(folded)),2),round(float((abs(folded)<1.5).mean()),3)


def main():
    transcripts = defaultdict(list)
    for line in (OUTPUT/'transcripts.jsonl').read_text().splitlines():
        item=json.loads(line)
        transcripts[Path(item['wav']).stem].append(item)
    note_dir=OUTPUT/'notes'
    note_dir.mkdir(exist_ok=True)
    summary=[]
    lyric_review=[]
    for wav_stem,parts in SELECTED.items():
        report=json.loads((OUTPUT/f'{wav_stem}.json').read_text())
        pitch=np.load(OUTPUT/'features'/f'{wav_stem}.npz')['pitch']
        rows=[]
        for part in parts:
            source,offset=part[:2]
            lower,upper=part[2:] if len(part)>2 else (-float('inf'),float('inf'))
            notes,_=load_notes(source)
            for index,(start,end,tone,lyric) in enumerate(notes,1):
                if not lower<=start<upper:
                    continue
                a,b=start+offset,end+offset
                voiced,median,matched=note_metrics(a,b,tone,pitch)
                flag=''
                if a<0 or b>report['duration_sec']:
                    flag='outside_recording'
                elif b-a>=.16 and voiced<.25:
                    flag='low_voicing_review'
                elif b-a>=.16 and voiced>.55 and matched is not None and matched<.4:
                    flag='pitch_review'
                rows.append({'source_ust':str(source),'note_index':index,'recording_start_sec':round(a,3),
                    'recording_end_sec':round(b,3),'duration_sec':round(b-a,3),'midi_note':tone,
                    'ust_lyric':lyric,'phonetic_lyric':ust_kana(lyric),'voiced_fraction':voiced,
                    'median_pitch_difference_semitones':median,'pitch_match_fraction':matched,'review_flag':flag})
        rows.sort(key=lambda x:(x['recording_start_sec'],x['source_ust']))
        with (note_dir/f'{wav_stem}.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        summary.append({'wav':report['wav'],'duration_sec':report['duration_sec'],
                        'sources':[{'ust':str(x[0]),'offset_sec':x[1],
                                    'ust_time_range_sec':list(x[2:]) if len(x)>2 else None} for x in parts],
                        'note_count':len(rows),'aligned_start_sec':min(r['recording_start_sec'] for r in rows),
                        'aligned_end_sec':max(r['recording_end_sec'] for r in rows),
                        'low_voicing_count':sum(r['review_flag']=='low_voicing_review' for r in rows),
                        'pitch_review_count':sum(r['review_flag']=='pitch_review' for r in rows)})
        for item in transcripts[wav_stem]:
            start=item['start'];stop=start+item['duration']
            relevant=[r for r in rows if r['recording_start_sec']>=start+1 and r['recording_end_sec']<=stop-1]
            source=''.join(r['phonetic_lyric'] for r in relevant)
            recognized=kana(item['text'])
            similarity=round(fuzz.partial_ratio(source,recognized),1) if source and recognized else None
            lyric_review.append({'wav':Path(report['wav']).name,'chunk_start_sec':start,
                'chunk_end_sec':round(stop,2),'ust_kana':source,'asr_kana':recognized,
                'asr_text':item['text'],'similarity_heuristic':similarity,
                'review_flag': 'review' if similarity is not None and similarity<55 and len(source)>10 else ''})
    (OUTPUT/'selected_alignment.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (OUTPUT/'lyric_review.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=lyric_review[0].keys());writer.writeheader();writer.writerows(lyric_review)
    print('recordings',len(summary),'notes',sum(x['note_count'] for x in summary),
          'lyric_review_flags',sum(bool(x['review_flag']) for x in lyric_review))


if __name__=='__main__':main()
