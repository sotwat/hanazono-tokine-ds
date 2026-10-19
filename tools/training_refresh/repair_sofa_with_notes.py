#!/usr/bin/env python3
"""Use nearby UST note anchors to repair obviously collapsed SOFA vowels."""

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf


BASE = Path('/Users/aaa/Downloads/ustストック/DS_ustx/照合結果/学習準備')
VOWELS = {'a','i','u','e','o','N'}
SILENCE = {'AP','SP'}


def read_csv(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=BASE)
    parser.add_argument('--source',default='SOFA候補_AKM')
    args=parser.parse_args()
    root=args.root
    source=root/args.source
    output=root/'音符拘束補正'
    output.mkdir(exist_ok=True)
    candidates={x['candidate_id']:x for x in read_csv(root/'フレーズ候補.csv')}
    inputs={x['candidate_id']:x for x in read_csv(root/'SOFA入力一覧.csv')}
    results=read_csv(source/'transcriptions/transcriptions.csv')
    confidence={x['name']:float(x['confidence']) for x in read_csv(source/'confidence/confidence.csv')}
    revised=[]
    audit=[]
    for item in results:
        name=item['name']
        score=candidates[name]
        anchors=json.loads(inputs[name]['phone_anchors_json'])
        phones=item['ph_seq'].split()
        durations=[float(x) for x in item['ph_dur'].split()]
        spoken=[i for i,p in enumerate(phones) if p not in SILENCE]
        if len(spoken)!=len(anchors):
            continue
        clip_start=float(score['start_sec'])
        changed=0
        unresolved=0
        for anchor_index, phone_index in enumerate(spoken):
            phone=phones[phone_index]
            if phone not in VOWELS or durations[phone_index]>=.02:
                continue
            current_start=sum(durations[:phone_index])
            current_end=current_start+durations[phone_index]
            note_start=anchors[anchor_index]['start']-clip_start
            note_end=anchors[anchor_index]['end']-clip_start
            if abs(current_start-note_start)>.4 or note_end-note_start<.09:
                unresolved+=1
                continue
            wanted=max(.04,min(note_end-.08,current_start+.12))
            if phone_index+1<len(phones):
                following=phones[phone_index+1]
                keep=.03 if following not in VOWELS else .04
                move=min(max(0,wanted-current_end),max(0,durations[phone_index+1]-keep))
                if move>0:
                    durations[phone_index]+=move
                    durations[phone_index+1]-=move
                    changed+=1
            if durations[phone_index]<.02:
                unresolved+=1
        short=sum(p in VOWELS and d<.02 for p,d in zip(phones,durations))
        excessive=[]
        for ai,pi in enumerate(spoken):
            if durations[pi]>2 and anchors[ai]['end']-anchors[ai]['start']<1.5:
                excessive.append(pi)
        total=sum(durations)
        expected=sf.info(root/'確認用WAV'/f'{name}.wav').duration
        valid=abs(total-expected)<.03 and min(durations)>0 and not short and not excessive
        known_lyric_issue=(score['song']=='ninelie' and float(score['start_sec'])<38 and float(score['end_sec'])>36)
        known_consonant_issue=(score['song']=='少女レイ' and float(score['start_sec'])<48 and float(score['end_sec'])>46)
        vocal_flag='low_voicing_review' in score['review_flags']
        status=('candidate_pass' if valid and confidence[name]>=.55 and not vocal_flag
                and not known_lyric_issue and not known_consonant_issue else 'review')
        revised.append({'name':name,'ph_seq':' '.join(phones),
                        'ph_dur':' '.join(f'{x:.8f}' for x in durations)})
        audit.append({'candidate_id':name,'song':score['song'],'status':status,
                      'confidence':round(confidence[name],3),'vowels_repaired':changed,
                      'short_vowels_remaining':short,'unresolved_anchor':unresolved,
                      'long_phones_without_note_support':len(excessive),
                      'review_flags':score['review_flags']})
    for filename,data in [('transcriptions.csv',revised),('品質監査.csv',audit)]:
        with (output/filename).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({'inferred':len(results),'candidate_pass':sum(x['status']=='candidate_pass' for x in audit),
                      'repaired_vowels':sum(x['vowels_repaired'] for x in audit),
                      'remaining_short_vowels':sum(x['short_vowels_remaining'] for x in audit)},ensure_ascii=False))


if __name__=='__main__':
    main()
