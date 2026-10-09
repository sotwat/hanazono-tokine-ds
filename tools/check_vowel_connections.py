"""Compare sustain wrappers using identical raw spectra, without diffusion noise."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h


def wrapper(path, prefix):
    model = onnx.load(path)
    needed = {prefix + 'mel'}
    selected = []
    boundaries = {'tokens', 'durations', 'f0', 'source_f0', prefix + 'su_raw'}
    for node in reversed(model.graph.node):
        if any(v in needed and v not in boundaries for v in node.output):
            selected.append(node)
            needed.update(node.input)
    nodes = list(reversed(selected))
    inputs = [h.make_tensor_value_info('tokens', T.INT64, [1, 'N']),
              h.make_tensor_value_info('durations', T.INT64, [1, 'N']),
              h.make_tensor_value_info('f0', T.FLOAT, [1, 'F']),
              h.make_tensor_value_info(prefix + 'su_raw', T.FLOAT, [1, 'F', 128])]
    for node in nodes:
        for i, value in enumerate(node.input):
            if value == 'source_f0':
                node.input[i] = 'f0'
    graph = h.make_graph(nodes, 'sustain-only', inputs,
        [h.make_tensor_value_info(prefix + 'mel', T.FLOAT, [1, 'F', 128])],
        [v for v in model.graph.initializer if any(v.name in n.input for n in nodes)])
    result = h.make_model(graph, opset_imports=model.opset_import)
    result.ir_version = model.ir_version
    onnx.checker.check_model(result)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(result.SerializeToString(), opts, providers=['CPUExecutionProvider'])


def check(baseline, candidate, output):
    rows = []
    vowels = [3, 9, 14, 20, 28]
    cases = [(a, b, 207, 275) for a in vowels for b in vowels]
    cases += [(9, 3, 30, 275), (9, 3, 207, 30), (8, 3, 90, 90),
              (9, 8, 90, 90), (9, 1, 90, 90), (24, 3, 90, 90)]
    for prefix in ['original_', 'donor_']:
        before, after = wrapper(baseline, prefix), wrapper(candidate, prefix)
        for a, b, da, db in cases:
            durs = np.array([[12, da, db, 16]], np.int64)
            frames, boundary = int(durs.sum()), 12 + da
            raw = np.random.default_rng(72).normal(-4, 1, (1, frames, 128)).astype(np.float32)
            feed = dict(tokens=np.array([[1, a, b, 1]], np.int64), durations=durs,
                        f0=np.full((1, frames), 440, np.float32))
            feed[prefix + 'su_raw'] = raw
            x, y = before.run(None, feed)[0], after.run(None, feed)[0]
            allowed = np.zeros(frames, bool)
            pair = a in vowels and b in vowels
            if pair and da >= 40:
                allowed[boundary-6:boundary] = True
                if db >= 40:
                    allowed[boundary:boundary+10] = True
                    assert np.array_equal(y[:, boundary], raw[:, boundary+12])
            assert np.array_equal(x[:, ~allowed], y[:, ~allowed])
            assert np.isfinite(y).all()
            assert bool(np.any(x != y)) == bool(allowed.any())
            rows.append(dict(branch=prefix, tokens=[a,b], durations=[da,db],
                             unaffected_exact=True, finite=True))
    output.write_text(json.dumps(rows, indent=2))
    print(f'{len(rows)} cases passed; only eligible boundary windows changed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    check(args.baseline, args.candidate, args.output)
