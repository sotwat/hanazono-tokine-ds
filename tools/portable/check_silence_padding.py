"""Check silence gating without changing voiced samples or increasing peaks."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('filter', type=Path)
args = parser.parse_args()
options = ort.SessionOptions()
options.intra_op_num_threads = 1
session = ort.InferenceSession(str(args.filter), options)
rng = np.random.default_rng(12)
cases = 0
for frames in [1, 2, 3, 4, 8, 16, 40, 128]:
    for flag in [-1, 0, 1, 2, 3]:
        for boundary in [1, frames // 2, frames]:
            count = frames + 8
            wave = rng.normal(size=(1, count * 512)).astype(np.float32)
            control = np.zeros((1, count, 128), np.float32)
            control[:, :, 0] = -1
            control[:, boundary:boundary + frames, 0] = flag
            control[:, boundary:boundary + frames, 1] = boundary
            control[:, boundary:boundary + frames, 2] = boundary + frames
            result = session.run(None, {'wave': wave, 'control': control})[0]
            assert np.isfinite(result).all()
            assert np.abs(result).max() <= np.abs(wave).max()
            assert np.array_equal(wave[:, :boundary * 512], result[:, :boundary * 512])
            assert np.array_equal(wave[:, (boundary + frames) * 512:], result[:, (boundary + frames) * 512:])
            if flag == -1:
                assert np.array_equal(wave, result)
            if flag == 0:
                assert np.count_nonzero(result[:, boundary * 512:(boundary + frames) * 512]) == 0
            cases += 1
print(json.dumps({'cases': cases, 'finite': True, 'exact_voiced_bypass': True, 'bounded_peak': True}))
