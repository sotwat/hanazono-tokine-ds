#!/usr/bin/env python3
"""Audit selected recordings and propose annotation clips without altering sources."""

import csv
import json
import math
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path('/Users/aaa/Downloads/ustストック')
ALIGN = ROOT / 'DS_ustx' / '照合結果'
OUT = ALIGN / '学習準備'


def selected(item):
    name = Path(item['wav']).stem
    return name not in {'Sincerely_take1', 'ミッドナイト・リフレクション_take1', 'フォニイ_消える消える'}


def audit(path):
    info = sf.info(path)
    peak = np.zeros(info.channels)
    square = np.zeros(info.channels)
    cross = 0.0
    near_clip = 0
    samples = 0
    quiet = 0
    for block in sf.blocks(path, blocksize=65536, dtype='float32', always_2d=True):
        a = block.astype(np.float64)
        peak = np.maximum(peak, np.max(np.abs(a), axis=0))
        square += np.sum(a*a, axis=0)
        if info.channels == 2:
            cross += float(np.sum(a[:, 0]*a[:, 1]))
        near_clip += int(np.sum(np.max(np.abs(a), axis=1) >= .999))
        quiet += int(np.sum(np.max(np.abs(a), axis=1) < .001))
        samples += len(a)
    rms = np.sqrt(square/samples)
    corr = cross/math.sqrt(square[0]*square[1]) if info.channels == 2 and min(square) else None
    return {
        'sample_rate': info.samplerate, 'channels': info.channels, 'subtype': info.subtype,
        'duration_sec': round(info.duration, 3), 'peak_dbfs': round(20*np.log10(max(peak)+1e-12), 2),
        'rms_dbfs': round(20*np.log10(max(rms)+1e-12), 2),
        'stereo_correlation': round(corr, 5) if corr is not None else '',
        'channel_rms_difference_db': round(abs(20*np.log10((rms[0]+1e-12)/(rms[1]+1e-12))), 3) if info.channels == 2 else '',
        'near_clip_samples': near_clip, 'near_silence_fraction': round(quiet/samples, 3),
    }


def note_rows(stem):
    with (ALIGN/'notes'/f'{stem}.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def proposals(rows, duration):
    # Musical rests are natural cut points. Preserve all notes within each clip.
    notes = sorted((float(r['recording_start_sec']), float(r['recording_end_sec']), r)
                   for r in rows if r['phonetic_lyric'] and r['ust_lyric'].upper() != 'R')
    groups = []
    current = []
    for note in notes:
        if current and note[0] - current[-1][1] >= .32:
            groups.append(current)
            current = []
        current.append(note)
    if current:
        groups.append(current)
    result = []
    for group in groups:
        part = []
        for note in group:
            if part and note[1] - part[0][0] > 14.0:
                result.append(part)
                part = []
            part.append(note)
        if part:
            result.append(part)
    merged = []
    for group in result:
        if merged and group[-1][1]-merged[-1][0][0] <= 14.5 and group[0][0]-merged[-1][-1][1] <= 1.5 and (merged[-1][-1][1]-merged[-1][0][0] < 5.0 or group[-1][1]-group[0][0] < 5.0):
            merged[-1].extend(group)
        else:
            merged.append(group)
    clips = []
    for group in merged:
        start = max(0, group[0][0]-.25)
        end = min(duration, group[-1][1]+.2)
        if end-start < 2.0:
            continue
        flags = sorted(set(n[2]['review_flag'] for n in group if n[2]['review_flag']))
        clips.append({
            'start_sec': round(start, 3), 'end_sec': round(end, 3),
            'duration_sec': round(end-start, 3), 'note_count': len(group),
            'ust_kana_preview': ''.join(n[2]['phonetic_lyric'] for n in group)[:70],
            'draft_syllables': ' '.join(n[2]['phonetic_lyric'] for n in group),
            'note_events_json': json.dumps([{'start':round(n[0],3),'end':round(n[1],3),
                                             'lyric':n[2]['phonetic_lyric'],
                                             'midi':int(n[2]['midi_note'])} for n in group], ensure_ascii=False),
            'review_flags': ','.join(flags),
            'needs_boundary_review': True,
        })
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true', help='write mono review clips and draft lyrics')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    align = json.loads((ALIGN/'selected_alignment.json').read_text())
    summaries = []
    clips = []
    for item in align:
        if not selected(item):
            continue
        path = Path(item['wav'])
        stem = path.stem
        stats = audit(path)
        rows = note_rows(stem)
        proposed = proposals(rows, stats['duration_sec'])
        summaries.append({'song':stem, 'wav':str(path), **stats,
                          'aligned_note_count':len(rows), 'clip_candidates':len(proposed),
                          'candidate_minutes':round(sum(x['duration_sec'] for x in proposed)/60, 2)})
        for i, clip in enumerate(proposed, 1):
            clips.append({'song':stem, 'wav':str(path), 'candidate_id':f'{len(summaries):02d}_{i:03d}', **clip})
    assert len(summaries) == 15
    for filename, data in [('録音監査.csv', summaries), ('フレーズ候補.csv', clips)]:
        with (OUT/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    if args.export:
        wav_out = OUT/'確認用WAV'
        lyric_out = OUT/'仮歌詞'
        wav_out.mkdir(exist_ok=True)
        lyric_out.mkdir(exist_ok=True)
        for item in clips:
            with sf.SoundFile(item['wav']) as source:
                source.seek(round(float(item['start_sec'])*source.samplerate))
                length = round((float(item['end_sec'])-float(item['start_sec']))*source.samplerate)
                audio = source.read(length, dtype='float32', always_2d=True)[:, 0]
                sf.write(wav_out/f"{item['candidate_id']}.wav", audio, source.samplerate, subtype='PCM_24')
            (lyric_out/f"{item['candidate_id']}.txt").write_text(item['draft_syllables']+'\n')
    print(json.dumps({'songs':len(summaries), 'raw_minutes':round(sum(x['duration_sec'] for x in summaries)/60, 2),
                      'candidates':len(clips), 'candidate_minutes':round(sum(x['duration_sec'] for x in clips)/60, 2),
                      'near_clip_songs':[x['song'] for x in summaries if x['near_clip_samples']],
                      'stereo_correlation_range':[min(x['stereo_correlation'] for x in summaries), max(x['stereo_correlation'] for x in summaries)]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
