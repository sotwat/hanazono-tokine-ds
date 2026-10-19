#!/usr/bin/env python3
"""Summarize local UST-to-recording pitch alignment in ten-second windows."""

import csv
import json
from pathlib import Path

import numpy as np

from ust_wav_alignment import HOP, OUTPUT, load_notes


def window_score(notes, pitch, start, stop, base_offset):
    selected = [n for n in notes if n[0] >= start and n[1] <= stop and n[1] - n[0] >= 0.09]
    if len(selected) < 8:
        return None
    starts = np.array([n[0] for n in selected])
    durations = np.array([n[1] - n[0] for n in selected])
    tones = np.array([n[2] for n in selected])
    times = np.concatenate([starts + durations * f for f in (0.3, 0.5, 0.7)])
    target = np.tile(tones, 3)
    best = None
    for offset in np.arange(base_offset - 2.0, base_offset + 2.001, 0.02):
        index = np.rint((times + offset) / HOP).astype(int)
        valid = (index >= 0) & (index < len(pitch))
        observed = pitch[np.clip(index, 0, len(pitch) - 1)]
        voiced = valid & (observed > 0)
        folded = (observed - target + 6) % 12 - 6
        score = float((np.exp(-0.5 * (folded / 0.7) ** 2) * voiced).mean())
        if best is None or score > best[0]:
            best = score, offset, float(voiced.mean())
    return {'ust_start': round(start, 2), 'ust_end': round(stop, 2),
            'note_count': len(selected), 'score': round(best[0], 3),
            'offset_sec': round(best[1], 2), 'voiced': round(best[2], 3)}


def main():
    rows = []
    for report_path in sorted(OUTPUT.glob('*.json')):
        report = json.loads(report_path.read_text())
        pitch = np.load(OUTPUT / 'features' / f'{report_path.stem}.npz')['pitch']
        for candidate in report['candidates']:
            if candidate['note_count'] < 25 or candidate['score'] < 0.55:
                continue
            notes, _ = load_notes(Path(candidate['ust']))
            note_start = min(n[0] for n in notes)
            note_end = max(n[1] for n in notes)
            for start in np.arange(np.floor(note_start / 10) * 10, note_end, 10):
                result = window_score(notes, pitch, start, start + 10, candidate['offset_sec'])
                if result:
                    result.update(wav=report_path.stem, ust=Path(candidate['ust']).name,
                                  global_offset=candidate['offset_sec'])
                    rows.append(result)
    path = OUTPUT / 'local_alignment.csv'
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['wav', 'ust', 'ust_start', 'ust_end',
                            'note_count', 'score', 'offset_sec', 'global_offset', 'voiced'])
        writer.writeheader()
        writer.writerows(rows)
    print(path, len(rows))


if __name__ == '__main__':
    main()
