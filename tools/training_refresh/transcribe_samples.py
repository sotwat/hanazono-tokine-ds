#!/usr/bin/env python3
"""Transcribe selected recording excerpts for manual lyric discrepancy review."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def audio_excerpt(path, start, duration):
    raw = subprocess.check_output(['ffmpeg', '-nostdin', '-loglevel', 'error', '-ss', str(start),
        '-t', str(duration), '-i', str(path), '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'])
    return np.frombuffer(raw, '<f4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', type=Path)
    parser.add_argument('start', type=float)
    parser.add_argument('duration', type=float)
    args = parser.parse_args()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(device).eval()
    audio = audio_excerpt(args.wav, args.start, args.duration)
    features = processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(device)
    with torch.inference_mode():
        ids = model.generate(features, language='ja', task='transcribe', max_new_tokens=160)
    print(json.dumps({'wav': args.wav.name, 'start': args.start, 'duration': args.duration,
                      'device': device, 'text': processor.batch_decode(ids, skip_special_tokens=True)[0]}, ensure_ascii=False))


def batch():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ust_wav_alignment import OUTPUT
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(device).eval()
    wanted = {'Sincerely_take2', 'ミッドナイト・リフレクション_take2'}
    skipped = {'Sincerely_take1', 'ミッドナイト・リフレクション_take1'}
    reports = sorted(OUTPUT.glob('*.json'))
    out = OUTPUT / 'transcripts.jsonl'
    completed = set()
    if out.exists():
        for line in out.read_text().splitlines():
            data = json.loads(line)
            completed.add((data['wav'], data['start']))
    for report_path in reports:
        if report_path.stem in skipped:
            continue
        report = json.loads(report_path.read_text())
        wav = Path(report['wav'])
        for start in range(0, int(report['duration_sec']), 25):
            if (wav.name, start) in completed:
                continue
            duration = min(27, report['duration_sec'] - start)
            if duration < 3:
                continue
            audio = audio_excerpt(wav, start, duration)
            features = processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(device)
            with torch.inference_mode():
                ids = model.generate(features, language='ja', task='transcribe', max_new_tokens=128)
            result = {'wav': wav.name, 'start': start, 'duration': round(duration, 2),
                      'text': processor.batch_decode(ids, skip_special_tokens=True)[0]}
            with out.open('a') as file:
                file.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(wav.name, start, result['text'][:60], flush=True)


if __name__ == '__main__':
    if len(__import__('sys').argv) == 2 and __import__('sys').argv[1] == '--batch':
        batch()
    else:
        main()
