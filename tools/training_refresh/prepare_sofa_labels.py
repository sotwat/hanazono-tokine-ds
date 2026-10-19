#!/usr/bin/env python3
"""Prepare provisional SOFA phoneme inputs from the verified clip index."""

import csv
import json
import argparse
from collections import Counter
from pathlib import Path


ROOT = Path('/Users/aaa/Downloads/ustストック/DS_ustx/照合結果/学習準備')
DICTIONARY = Path('/Users/aaa/Documents/DiffSinger/花園トキネAI制作/音源/花園トキネAI/tokine-dictionary.txt')


def mapping():
    result = {}
    for line in DICTIONARY.read_text().splitlines():
        if '\t' in line:
            syllable, phones = line.split('\t', 1)
            result[syllable] = phones.split()
    return result


def parse_token(token, dictionary):
    phones = []
    normalized = False
    while token:
        if token.startswith('っ'):
            phones.append('cl')
            token = token[1:]
            continue
        matches = [x for x in dictionary if token.startswith(x)]
        if matches:
            key = max(matches, key=len)
            phones.extend(dictionary[key])
            token = token[len(key):]
            continue
        if token[0] in 'ぁぃぅぇぉ':
            phones.extend(dictionary[{'ぁ':'あ','ぃ':'い','ぅ':'う','ぇ':'え','ぉ':'お'}[token[0]]])
            token = token[1:]
            normalized = True
            continue
        raise ValueError(token)
    return phones, normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    out = root/'SOFA候補'
    out.mkdir(exist_ok=True)
    dictionary = mapping()
    rows = list(csv.DictReader((root/'フレーズ候補.csv').open()))
    manifest = []
    for row in rows:
        sequence = []
        anchors = []
        changes = []
        error = ''
        previous = None
        collapsed_vowels = 0
        for event in json.loads(row['note_events_json']):
            token = event['lyric']
            try:
                phones, normalized = parse_token(token, dictionary)
                if (previous and len(phones)==1
                        and phones[0]==previous['phones'][-1]
                        and phones[0] in {'a','i','u','e','o','N'}
                        and event['start']-previous['end'] <= .03):
                    collapsed_vowels += 1
                    anchors[-1]['end'] = event['end']
                else:
                    sequence.extend(phones)
                    anchors.extend({'start':event['start'],'end':event['end'],
                                    'lyric':token} for _ in phones)
                if normalized:
                    changes.append(token)
                previous = {'phones':phones, 'end':event['end']}
            except ValueError as exc:
                error = f'{token}: {exc}'
                break
        item = {'candidate_id': row['candidate_id'], 'song': row['song'],
                'audio_duration_sec': row['duration_sec'], 'phoneme_count': len(sequence),
                'phone_anchors_json': json.dumps(anchors,ensure_ascii=False),
                'collapsed_adjacent_vowels':collapsed_vowels,
                'normalized_small_vowels': ','.join(changes), 'review_flags': row['review_flags'],
                'error': error}
        if not error:
            wav = root/'確認用WAV'/f"{row['candidate_id']}.wav"
            link = out/wav.name
            if not link.exists():
                link.symlink_to(wav)
            (out/f"{row['candidate_id']}.lab").write_text(' '.join(sequence)+'\n')
        manifest.append(item)
    with (root/'SOFA入力一覧.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    summary = {'candidates':len(rows), 'prepared':sum(not x['error'] for x in manifest),
               'excluded':sum(bool(x['error']) for x in manifest),
               'collapsed_adjacent_vowels':sum(x['collapsed_adjacent_vowels'] for x in manifest),
               'small_vowel_normalization_clips':sum(bool(x['normalized_small_vowels']) for x in manifest),
               'errors': dict(Counter(x['error'] for x in manifest if x['error']))}
    (root/'SOFA入力概要.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
