"""Regression checks for all packaged consonants at phrase starts."""
import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper


def check(bank, candidate, output):
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.log_severity_level = 3
    root = bank / 'dsdur'
    linguistic = ort.InferenceSession(str(root/'tokine_song_variance_v1.linguistic.onnx'), opts)
    old = ort.InferenceSession(str(root/'tokine_song_variance_v1.dur.onnx'), opts)
    new = ort.InferenceSession(str(candidate), opts)
    vowels = [2, 3, 9, 14, 20, 28]
    consonants = [i for i in range(4, 32) if i not in vowels]
    maximum = next(numpy_helper.to_array(i) for i in onnx.load(candidate).graph.initializer if i.name == 'lt_maximum')
    rows = []
    for consonant in consonants:
        for vowel in vowels:
            for length in [1, 3, 12, 39, 43, 86, 172, 258, 860]:
                for midi in [42, 58, 77]:
                    tokens = np.array([[1, consonant, vowel, 16, 3]], np.int64)
                    feed = dict(tokens=tokens, word_div=np.array([[2, 2, 1]], np.int64),
                                word_dur=np.array([[43, length, length]], np.int64))
                    e, m = linguistic.run(None, feed)
                    ins = dict(encoder_out=e, x_masks=m, ph_midi=np.full(tokens.shape, midi, np.int64))
                    before = old.run(None, ins)[0]; after = new.run(None, ins)[0]
                    assert np.isfinite(after).all() and (after >= 0).all()
                    assert np.array_equal(before[:, 2:], after[:, 2:])
                    assert before[0, 0] == after[0, 0]
                    assert after[0, 1] <= min(before[0, 1], 16, maximum[consonant]) + 1e-5
                    rows.append(dict(consonant=consonant, vowel=vowel, frames=length,
                                     midi=midi, before=float(before[0, 1]), after=float(after[0, 1])))
    # Consonant clusters share a total preutterance budget, while initial vowels
    # and the leading SP retain the previous model's output exactly.
    for cluster in [[16, 30], [7, 16], [24, 16, 30], []]:
        tokens = np.array([[1]+cluster+[3]], np.int64)
        e, m = linguistic.run(None, dict(tokens=tokens,
            word_div=np.array([[1+len(cluster), 1]], np.int64),
            word_dur=np.array([[43, 258]], np.int64)))
        feed = dict(encoder_out=e, x_masks=m, ph_midi=np.full(tokens.shape, 58, np.int64))
        a = old.run(None, feed)[0]; b = new.run(None, feed)[0]
        assert b[0, 1:1+len(cluster)].sum() <= 16.001
        assert a[0, 0] == b[0, 0] and a[0, -1] == b[0, -1]
        if not cluster: assert np.array_equal(a, b)
    output.write_text(json.dumps(dict(cases=len(rows), cluster_cases=4, rows=rows), indent=2))
    print(f'{len(rows)} phrase-onset cases and 4 cluster/vowel cases passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('bank', type=Path)
    p.add_argument('candidate', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); check(a.bank, a.candidate, a.output)
