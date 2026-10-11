"""Export timed note estimates to an untuned UST 2.0 file."""
import argparse
import json
from pathlib import Path


def export(notes, output, tempo, name):
    if tempo <= 0 or not notes:
        raise ValueError('A positive tempo and at least one note are required')
    rate = tempo * 480 / 60
    ordered = sorted(notes, key=lambda n: n['start'])
    events = []
    cursor = 0
    trimmed = 0
    for i, note in enumerate(ordered):
        start = max(cursor, round(note['start'] * rate))
        end = round(note['end'] * rate)
        if i + 1 < len(ordered):
            next_start = round(ordered[i + 1]['start'] * rate)
            if end > next_start:
                end = next_start
                trimmed += 1
        end = max(start + 1, end)
        tone = int(note['tone'])
        lyric = str(note['lyric'])
        if not 0 <= tone <= 127 or any(c in lyric for c in '\r\n'):
            raise ValueError('Invalid MIDI pitch or lyric')
        if start > cursor:
            events.append((start - cursor, 'R', 60))
        events.append((end - start, lyric, tone))
        cursor = end
    lines = ['[#VERSION]', 'UST Version2.0', 'Charset=UTF-8', '[#SETTING]',
             f'Tempo={tempo:g}', 'Tracks=1', f'ProjectName={name}',
             'VoiceDir=', 'OutFile=', 'CacheDir=', 'Mode2=True']
    for i, (length, lyric, tone) in enumerate(events):
        lines.extend([f'[#{i:04d}]', f'Length={length}', f'Lyric={lyric}',
                      f'NoteNum={tone}', 'Intensity=100', 'Modulation=0'])
    lines.append('[#TRACKEND]')
    output.write_bytes(('\r\n'.join(lines) + '\r\n').encode('utf-8'))
    return dict(notes=len(ordered), events=len(events), ticks=cursor,
                duration_seconds=cursor / rate, overlaps_trimmed=trimmed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tempo', type=float, required=True)
    parser.add_argument('--name', default='Transcription draft')
    args = parser.parse_args()
    print(json.dumps(export(json.loads(args.input.read_text()), args.output,
                            args.tempo, args.name), ensure_ascii=False))
