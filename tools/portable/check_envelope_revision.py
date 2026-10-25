from pathlib import Path
import sys, json, numpy as np, torch, onnx, onnxruntime as ort
from envelope_revision import Restore
from short_vowel_resynthesis import RestoreVowel
from vowel_balance import VowelBalance
import argparse
p = argparse.ArgumentParser()
p.add_argument('artifacts', type=Path)
p.add_argument('bank', type=Path)
args = p.parse_args()
r = args.artifacts
cases = 0
for duration in [1, 6, 19, 39, 40, 51, 80, 200]:
    for eligible in [0.0, 1.0]:
        for gate in [0.0, 0.35, 1.0]:
            d = torch.tensor([[8, duration, 8]])
            f = int(d.sum())
            short = torch.randn(1, f, 128)
            back = torch.arange(f)[None].float()
            e = torch.zeros(1, f, 1)
            e[:, 8:8 + duration] = eligible
            g = torch.full((1, f, 1), gate)
            w = torch.ones(1, f, 1) * gate
            w[:, 8:8 + min(duration, 20)] = 0
            baseline = torch.randn(1, f, 128)
            a = RestoreVowel()(short, back, e, w, baseline)
            b = Restore()(short, back, e, w, g, baseline, d)
            assert torch.isfinite(b).all()
            mask = torch.zeros(f, dtype=torch.bool)
            mask[8:8 + min(duration, 20)] = True
            assert torch.equal(a[:, ~mask], b[:, ~mask])
            assert b.shape == a.shape
            if not eligible or not gate:
                assert torch.equal(a, b)
            cases += 1
sr = 44100
f = 160
t = torch.arange(f * 512) / sr
w = (0.06 * torch.sin(2 * torch.pi * 220 * t))[None]
c = torch.zeros(f, 128)
c[:, 3] = 220
c[:, 6] = 1
for (st, en, amp) in [(0, 40, 0.04), (40, 80, 0.07), (80, 120, 0.3), (120, 160, 0.06)]:
    c[st:en, 1] = st
    c[st:en, 2] = en
    w[:, st * 512:en * 512] *= amp / 0.06
out = VowelBalance(0.75, 0.9, 6.0, 12.0)(w, c)
assert torch.isfinite(out).all()
for (st, en) in [(0, 40), (40, 80), (80, 120), (120, 160)]:
    a = w[0, (st + 3) * 512:(en - 3) * 512]
    b = out[0, (st + 3) * 512:(en - 3) * 512]
    mask = a.abs() > 0.001
    ratio = b[mask] / a[mask]
    assert float(ratio.max() - ratio.min()) < 1e-05
bank = args.bank
retained = {}
for (old, new) in [(bank / 'tokine_ds_clarity_hybrid.tokine.onnx', r / 'final-build/acoustic.onnx'), (next((bank / 'dsvocoder').glob('*.onnx')), r / 'final-build/vocoder.onnx')]:
    a = onnx.load(old)
    b = onnx.load(new)
    onnx.checker.check_model(b)
    weights = {x.name: x.SerializeToString() for x in b.graph.initializer}
    assert all((weights[x.name] == x.SerializeToString() for x in a.graph.initializer))
    retained[new.name] = len(a.graph.initializer)
for name in ['omoi', 'negatta', 'tokiakasou']:
    a = np.load(r / 'whole' / (name + '-acoustic.npz'))
    b = np.load(r / 'whole' / (name + '-trial.npz'))
    assert np.array_equal(a['mel'][2], b['mel'][2])
    assert a['mel'].shape == b['mel'].shape
result = dict(scope_cases=cases, existing_initializers_retained=retained, control_f0_and_boundaries_identical=True, constant_core_gain_no_added_harmonics=True, perceptual_confirmation='not automated')
(r / 'verification.json').write_text(json.dumps(result, indent=2))
print(result)
