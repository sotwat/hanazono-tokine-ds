#!/usr/bin/env python3
"""Compare UST/USTX note contours with local recordings without changing sources."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import parselmouth
import yaml


ROOT = Path('/Users/aaa/Downloads/ustストック')
WAVS = ROOT / 'DS_wav'
USTS = ROOT / 'DS_ustx' / '対応UST'
OUTPUT = ROOT / 'DS_ustx' / '照合結果'
HOP = 0.02


def norm(name: str) -> str:
    return unicodedata.normalize('NFC', name)


def parse_ust(path: Path):
    text = path.read_bytes().decode('cp932', errors='replace').replace('\r\n', '\n')
    settings = re.search(r'(?ms)^\[#SETTING\]\n(.*?)(?=^\[#)', text)
    tempo = re.search(r'(?m)^Tempo=([\d.]+)', settings.group(1) if settings else '')
    bpm = float(tempo.group(1)) if tempo else 120.0
    blocks = re.findall(r'(?ms)^\[#\d+\]\n(.*?)(?=^\[#|\Z)', text)
    tick = 0
    notes = []
    for block in blocks:
        length = re.search(r'(?m)^Length=(\d+)', block)
        if not length:
            continue
        duration = int(length.group(1))
        lyric = re.search(r'(?m)^Lyric=(.*)', block)
        tone = re.search(r'(?m)^NoteNum=(\d+)', block)
        word = lyric.group(1).strip() if lyric else ''
        if word and word.upper() != 'R' and tone:
            notes.append((tick / 8 / bpm, (tick + duration) / 8 / bpm, int(tone.group(1)), word))
        tick += duration
    return notes, tick / 8 / bpm


def parse_ustx(path: Path):
    data = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    resolution = data.get('resolution', 480)
    tempos = data.get('tempos') or [{'position': 0, 'bpm': data.get('bpm', 120)}]
    tempos = sorted(tempos, key=lambda item: item['position'])

    def seconds(tick: int) -> float:
        value = 0.0
        for i, tempo in enumerate(tempos):
            start = tempo['position']
            if tick <= start:
                break
            stop = min(tick, tempos[i + 1]['position']) if i + 1 < len(tempos) else tick
            value += (stop - start) / resolution * 60 / tempo['bpm']
            if tick <= stop:
                break
        return value

    notes = []
    end = 0
    for part in data.get('voice_parts', []):
        offset = part.get('position', 0)
        for note in part.get('notes', []):
            start = offset + note['position']
            stop = start + note['duration']
            notes.append((seconds(start), seconds(stop), note['tone'], str(note['lyric'])))
            end = max(end, stop)
    return notes, seconds(end)


def load_notes(path: Path):
    return parse_ustx(path) if path.suffix.lower() == '.ustx' else parse_ust(path)


def audio_features(path: Path, cache: Path):
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        data = np.load(cache)
        return data['pitch'], data['rms'], float(data['duration'])
    raw = subprocess.check_output([
        'ffmpeg', '-nostdin', '-loglevel', 'error', '-i', str(path),
        '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'
    ])
    audio = np.frombuffer(raw, dtype='<f4')
    duration = len(audio) / 16000
    pitch_obj = parselmouth.Sound(audio, 16000).to_pitch(
        time_step=HOP, pitch_floor=80, pitch_ceiling=900
    )
    hz = pitch_obj.selected_array['frequency'].astype('float32')
    pitch = np.zeros_like(hz)
    voiced = hz > 0
    pitch[voiced] = 69 + 12 * np.log2(hz[voiced] / 440)
    frame_len = int(16000 * HOP)
    rms = np.sqrt(np.mean(audio[:len(audio) // frame_len * frame_len].reshape(-1, frame_len) ** 2, axis=1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, pitch=pitch, rms=rms, duration=duration)
    return pitch, rms, duration


def align(notes, pitch, audio_duration):
    if len(notes) < 8:
        return None
    starts = np.array([note[0] for note in notes], dtype=float)
    ends = np.array([note[1] for note in notes], dtype=float)
    tones = np.array([note[2] for note in notes], dtype=float)
    durations = ends - starts
    eligible = durations >= 0.09
    if eligible.sum() < 8:
        return None
    starts, durations, tones = starts[eligible], durations[eligible], tones[eligible]
    times = np.concatenate([starts + durations * fraction for fraction in (0.3, 0.5, 0.7)])
    target = np.tile(tones, 3)
    score_end = max(note[1] for note in notes)
    offsets = np.arange(-min(score_end, 30), min(audio_duration, 100) + 0.001, 0.1)
    frame = np.rint((times[None, :] + offsets[:, None]) / HOP).astype(int)
    valid = (frame >= 0) & (frame < len(pitch))
    observed = pitch[np.clip(frame, 0, len(pitch) - 1)]
    voiced = valid & (observed > 0)
    difference = observed - target[None, :]
    folded = (difference + 6) % 12 - 6
    match = np.exp(-0.5 * (folded / 0.7) ** 2) * voiced
    scores = match.mean(axis=1)
    best = int(np.argmax(scores))
    local_offsets = np.arange(offsets[best] - 0.2, offsets[best] + 0.201, 0.02)
    local_best = None
    for offset in local_offsets:
        index = np.rint((times + offset) / HOP).astype(int)
        okay = (index >= 0) & (index < len(pitch))
        obs = pitch[np.clip(index, 0, len(pitch) - 1)]
        voiced_here = okay & (obs > 0)
        diff = obs - target
        fold = (diff + 6) % 12 - 6
        this_score = float((np.exp(-0.5 * (fold / 0.7) ** 2) * voiced_here).mean())
        if local_best is None or this_score > local_best[0]:
            close = voiced_here & (np.abs(fold) < 1.5)
            transpose = float(np.median(diff[close])) if close.sum() else None
            local_best = (this_score, float(offset), float(voiced_here.mean()), transpose, int(close.sum()))
    return {
        'score': round(local_best[0], 4),
        'offset_sec': round(local_best[1], 2),
        'voiced_fraction': round(local_best[2], 4),
        'median_pitch_difference_semitones': round(local_best[3], 2) if local_best[3] is not None else None,
        'matched_samples': local_best[4],
        'total_samples': len(times),
        'note_count': len(notes),
        'score_end_sec': round(score_end, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--song', help='Limit to one normalized song name')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wavs = sorted(WAVS.glob('*.wav'))
    for wav in wavs:
        name = norm(wav.stem)
        song = re.sub(r'_take\d+$|_消える消える$', '', name)
        if song == '君の記憶':
            pass
        if args.song and args.song != song:
            continue
        folder = USTS / song
        paths = list(folder.rglob('*.ust')) + list(folder.rglob('*.ustx')) if folder.exists() else []
        if song == '少女レイ':
            paths.append(ROOT / 'DS_ustx' / '少女レイ_DS_グリッド修正.ustx')
        pitch, rms, duration = audio_features(wav, OUTPUT / 'features' / f'{wav.stem}.npz')
        candidates = []
        for path in paths:
            notes, score_duration = load_notes(path)
            result = align(notes, pitch, duration)
            if result:
                result['ust'] = str(path)
                result['score_duration_sec'] = round(score_duration, 2)
                candidates.append(result)
        candidates.sort(key=lambda item: item['score'], reverse=True)
        report = {
            'wav': str(wav), 'song': song, 'duration_sec': round(duration, 2),
            'pitch_voiced_fraction': round(float((pitch > 0).mean()), 4),
            'rms_median': round(float(np.median(rms)), 5),
            'candidates': candidates,
        }
        (OUTPUT / f'{wav.stem}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        top = candidates[0] if candidates else None
        print(f'{wav.name}\t{top["score"] if top else "NO_UST"}\t{top["offset_sec"] if top else ""}\t{Path(top["ust"]).name if top else ""}', flush=True)


if __name__ == '__main__':
    main()
