"""Check affricate expansion, frame locality, and preservation of learned tensors."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def check(source, candidate, output, include_ch=False):
    opts = ort.SessionOptions(); opts.log_severity_level = 3; opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(candidate.parent/'maps.onnx'), opts)
    cases = 0
    for phone in range(1, 32):
        for duration in (0, 1, 2, 3, 4, 8, 16, 32):
            for following in (1, 3, 9, 14, 20, 28):
                tokens = np.array([[1, phone, following, 1]], np.int64)
                durations = np.array([[8, duration, 10, 8]], np.int64)
                f0 = np.full((1, int(durations.sum())), 440, np.float32)
                t, d, mask = session.run(None, dict(tokens=tokens, durations=durations, f0=f0))
                active = phone in ((6,27) if include_ch else (27,)) and duration >= 2 and following != 1
                assert d.sum() == durations.sum()
                if active:
                    assert t.tolist() == [[1, 26, 25 if phone == 6 else 24, following, 1]]
                    assert d[0, 1]+d[0, 2] == duration and np.all(d[0, 1:3] > 0)
                    assert np.all(mask[0, 8:8+duration] == 1)
                    assert np.count_nonzero(mask) == duration
                else:
                    assert np.array_equal(t, tokens) and np.array_equal(d, durations)
                    assert not np.any(mask)
                cases += 1
    before, after = onnx.load(source), onnx.load(candidate)
    weights = {v.name: v.SerializeToString() for v in after.graph.initializer}
    assert all(weights[v.name] == v.SerializeToString() for v in before.graph.initializer)
    # Test the actual final selection with fixed spectra, avoiding independent diffusion draws.
    selected = [n for n in after.graph.node if any(v in ('ta_original_mel','ta_donor_mel') for v in n.output)]
    h = onnx.helper; T = onnx.TensorProto
    inputs = [h.make_tensor_value_info('ta_ts_bool', T.BOOL, [1,'F',1])]
    for prefix in ('original_', 'donor_'):
        for name in ('ta_neural_'+prefix+'su_raw', 'tc_'+prefix+'mel'):
            inputs.append(h.make_tensor_value_info(name, T.FLOAT, [1,'F',128]))
    model = h.make_model(h.make_graph(selected, 'ts-locality', inputs,
        [h.make_tensor_value_info('ta_'+p+'mel', T.FLOAT, [1,'F',128]) for p in ('original_', 'donor_')]),
        opset_imports=after.opset_import, ir_version=after.ir_version)
    sel = ort.InferenceSession(model.SerializeToString(), opts)
    rng = np.random.default_rng(421)
    for active in (False, True):
        mask = np.zeros((1, 80, 1), bool)
        if active: mask[:, 8:24] = True
        feed = {'ta_ts_bool':mask}
        for inp in inputs[1:]: feed[inp.name] = rng.normal(-4,1,(1,80,128)).astype(np.float32)
        for prefix, result in zip(('original_', 'donor_'), sel.run(None, feed)):
            assert np.array_equal(result[:, ~mask[0,:,0]], feed['tc_'+prefix+'mel'][:, ~mask[0,:,0]])
            assert np.isfinite(result).all()
    report = dict(map_cases=cases, phone_boundaries_preserved=True,
        learned_tensors_byte_identical=True, non_target_mel_bit_identical=True)
    output.write_text(json.dumps(report, indent=2)); print(report)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--include-ch',action='store_true');a=p.parse_args();check(a.source,a.candidate,a.output,a.include_ch)
