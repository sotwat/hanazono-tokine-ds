"""Build a recording-aligned, untuned score from phoneme alignment and F0."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
import yaml
from export_note_estimates import export


def pitch_segments(times, f0, start, end):
    use = (times >= start + .025) & (times < end - .02) & (f0 > 0)
    t = times[use]
    p = 69 + 12 * np.log2(f0[use] / 440)
    if len(p) < 3:
        return [], None
    p = median_filter(p, size=min(7, len(p) // 2 * 2 + 1), mode='nearest')
    states = np.arange(max(36, int(np.floor(np.percentile(p, 5))) - 1),
                       min(96, int(np.ceil(np.percentile(p, 95))) + 1) + 1)
    costs = np.minimum(abs(p[:, None] - states[None, :]), 2.)
    back = np.zeros((len(p), len(states)), dtype=int)
    value = costs[0].copy()
    penalty = 5.0
    for i in range(1, len(p)):
        transitions = value[:, None] + penalty * (states[:, None] != states[None, :])
        back[i] = transitions.argmin(axis=0)
        value = costs[i] + transitions[back[i], np.arange(len(states))]
    path = np.zeros(len(p), dtype=int)
    path[-1] = value.argmin()
    for i in range(len(p) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    labels = states[path]
    # Remove brief excursions rather than turning every vibrato cycle into a note.
    for _ in range(3):
        edges = np.r_[0, np.where(np.diff(labels) != 0)[0] + 1, len(labels)]
        changed = False
        for j, k in zip(edges[:-1], edges[1:]):
            if t[k - 1] - t[j] + .01 < .12 and len(edges) > 2:
                neighbours = []
                if j: neighbours.append(labels[j - 1])
                if k < len(labels): neighbours.append(labels[k])
                replacement = min(neighbours, key=lambda n: np.mean((p[j:k] - n) ** 2))
                labels[j:k] = replacement
                changed = True
        if not changed: break
    edges = np.r_[0, np.where(np.diff(labels) != 0)[0] + 1, len(labels)]
    result = []
    fitted = labels.copy()
    for j, k in zip(edges[:-1], edges[1:]):
        s = start if j == 0 else float((t[j - 1] + t[j]) / 2)
        e = end if k == len(labels) else float((t[k - 1] + t[k]) / 2)
        tone = int(round(np.median(p[j:k])))
        fitted[j:k] = tone
        result.append((s, e, tone))
    return result, float(np.median(abs(p - fitted)) * 100)


def build(rows, pitch, dictionary):
    times, f0 = pitch['t'], pitch['f0']
    notes, issues, phones = [], [], []
    previous_tone = None
    for row in rows:
        words = row['aligned_words']
        intervals = row['intervals']
        phone_cursor = 0
        nuclei = []
        for word, (s, e) in zip(words, intervals):
            count = len(dictionary[word])
            word_phones = row['phones_aligned'][phone_cursor:phone_cursor + count]
            word_intervals = row['phone_intervals'][phone_cursor:phone_cursor + count]
            if word_phones != dictionary[word]:
                raise ValueError(f'Phone mapping mismatch in phrase {row["index"]}')
            phones.extend(zip(word_phones, word_intervals))
            nucleus = next((span[0] for ph, span in zip(word_phones, word_intervals)
                            if ph in {'a', 'i', 'u', 'e', 'o', 'N'}), s)
            nuclei.append(float(nucleus))
            phone_cursor += count
        if phone_cursor != len(row['phones_aligned']):
            raise ValueError('Unassigned aligned phones')
        for i, (word, (s, e)) in enumerate(zip(words, intervals)):
            start = nuclei[i]
            end = float(e)
            if i + 1 < len(words) and intervals[i + 1][0] - e < .16:
                end = nuclei[i + 1]
            if end - start < .04:
                issues.append(dict(phrase=row['index'], syllable=i, time=start,
                                   reason='very_short_alignment', lyric=word))
            if end <= start:
                end = start + .005
            segments, error = pitch_segments(times, f0, start, float(e))
            if segments:
                a, _, tone = segments[-1]
                segments[-1] = (a, end, tone)
            if not segments:
                near = (times >= s - .12) & (times < e + .12) & (f0 > 0)
                nearby = 69 + 12 * np.log2(f0[near] / 440)
                tone = int(round(np.median(nearby))) if len(nearby) else previous_tone
                if tone is None:
                    raise ValueError('No pitch evidence at the beginning of the recording')
                segments = [(start, end, tone)]
                if word != 'っ':
                    issues.append(dict(phrase=row['index'], syllable=i, time=start,
                                       reason='no_stable_vowel_pitch', lyric=word))
            if error is not None and error > 60:
                issues.append(dict(phrase=row['index'], syllable=i, time=start,
                                   reason='pitch_between_semitones', lyric=word, cents=error))
            for j, (a, b, tone) in enumerate(segments):
                notes.append(dict(start=a, end=b, tone=tone, lyric=word if j == 0 else '+',
                                  phrase=row['index'], syllable=i,
                                  pitch_error_median_cent=error))
                previous_tone = tone
    notes.sort(key=lambda n: n['start'])
    for a, b in zip(notes, notes[1:]):
        if a['end'] > b['start']:
            a['end'] = b['start']
        if a['end'] <= a['start']:
            raise ValueError('Overlapping or collapsed note anchors')
    return notes, issues, phones


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('alignment', type=Path)
    p.add_argument('pitch', type=Path)
    p.add_argument('dictionary', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--tempo', required=True, type=float)
    p.add_argument('--name', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dictionary = {e['grapheme']: e['phonemes'] for e in yaml.safe_load(args.dictionary.read_text())['entries']}
    notes, issues, phones = build(json.loads(args.alignment.read_text()), np.load(args.pitch), dictionary)
    (args.output / 'notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    (args.output / 'review.json').write_text(json.dumps(issues, ensure_ascii=False, indent=2))
    stats = export(notes, args.output / (args.name + '.ust'), args.tempo, args.name)
    rate = args.tempo * 480 / 60
    ustx_notes = [dict(position=round(n['start'] * rate),
                       duration=max(1, round(n['end'] * rate) - round(n['start'] * rate)),
                       tone=n['tone'], lyric=n['lyric'], pitch=dict(data=[]), vibrato=dict(length=0))
                  for n in notes]
    project = dict(name=args.name, ustx_version='0.6', resolution=480,
                   tempos=[dict(position=0, bpm=args.tempo)],
                   time_signatures=[dict(bar_position=0, beat_per_bar=4, beat_unit=4)],
                   tracks=[dict(track_name=args.name, track_color='Blue', singer='',
                                phonemizer='OpenUtau.Plugin.Builtin.JapaneseCVPhonemizer')],
                   voice_parts=[dict(name=args.name, track_no=0, position=0, notes=ustx_notes)],
                   wave_parts=[])
    (args.output / (args.name + '.ustx')).write_text(yaml.safe_dump(project, allow_unicode=True, sort_keys=False))
    (args.output / 'phoneme_intervals.json').write_text(json.dumps(
        [dict(phoneme=ph, start=span[0], end=span[1]) for ph, span in phones], indent=2))
    stats.update(review_items=len(issues), syllables=sum(len(r['aligned_words']) for r in json.loads(args.alignment.read_text())))
    (args.output / 'validation.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))
