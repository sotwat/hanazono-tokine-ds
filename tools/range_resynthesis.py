"""Experimental DS donor + WORLD pitch resynthesis, independent of VOCALOID.

Input NPZ contains tokens, durations, f0 (or dbg_ prefixed equivalents).
The donor uses A3..C5. High-register blending ramps from D5 to F5.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
import onnxruntime as ort
import pyworld as pw
import soundfile as sf
import yaml

SR = 44100
HOP = 512
PERIOD = HOP / SR * 1000

def hz(midi):
    return 440 * 2 ** ((midi - 69) / 12)

def session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(model, opts, providers=['CPUExecutionProvider'])

def raw_vocoder(bank):
    model = onnx.load(next((bank / 'dsvocoder').glob('*.onnx')))
    names = {v for n in model.graph.node for v in n.output}
    name = 'he_raw' if 'he_raw' in names else 'waveform'
    del model.graph.output[:]
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 'samples']))
    return session(model.SerializeToString()), name

def register_weight(f0):
    f0 = np.asarray(f0, dtype=float)
    midi = 69 + 12*np.log2(np.maximum(f0, 1)/440)
    u = np.clip((midi-74)/3, 0, 1)
    high = np.where(f0 >= hz(77)-.001, 1.0, u*u*(3-2*u))
    low = ((f0 > 0) & (f0 <= hz(54)+.001)).astype(float)
    return np.where(f0 > 0, np.maximum(high, low), 0)

def edge_weight(frame_weight, samples, fade_ms=20):
    frame_weight = np.asarray(frame_weight, dtype=float)
    expanded = np.interp(np.arange(samples)/HOP, np.arange(len(frame_weight)), frame_weight)
    active = np.repeat(frame_weight > 1e-8, HOP)[:samples].astype(float)
    if len(active) < samples:
        active = np.pad(active, (0, samples-len(active)))
    length = round(SR * fade_ms / 1000)
    forward = np.convolve(active, np.ones(length)/length, mode='full')[:samples]
    backward = np.convolve(active[::-1], np.ones(length)/length, mode='full')[:samples][::-1]
    return np.clip(expanded * active * np.minimum(forward, backward), 0, 1)

def blend_waveforms(baseline, world, weight):
    mixed = np.sqrt(1-weight)*baseline + np.sqrt(weight)*world
    return np.where(weight <= 0, baseline, np.where(weight >= 1, world, mixed))

def render(bank, inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((bank / 'dsconfig.yaml').read_text())
    if config.get('sample_rate') != SR or config.get('hop_size') != HOP:
        raise ValueError('This prototype requires 44100 Hz / 512-hop Tokine models')
    ac = session(str(bank / config['acoustic']))
    voc, out_name = raw_vocoder(bank)
    target = np.asarray(inputs['f0'], np.float32)
    assert target.ndim == 2 and target.shape[0] == 1
    assert np.isfinite(target).all() and (target >= 0).all()
    assert inputs['durations'].sum() == target.shape[1]
    if target.max() >= SR/4:
        raise ValueError('Target pitch exceeds this renderer\'s safety limit (not a quality guarantee)')
    source = np.where(target > 0, np.clip(target, hz(57), hz(72)), 0).astype(np.float32)
    def synth(f0):
        feed = dict(tokens=inputs['tokens'].astype(np.int64), durations=inputs['durations'].astype(np.int64), f0=f0,
                    depth=np.array(.1, np.float32), steps=np.array(20, np.int64))
        mel = ac.run(['mel'], feed)[0]
        return voc.run([out_name], {'mel':mel, 'f0':f0})[0].ravel().astype(np.float64)
    baseline = synth(target)
    donor = synth(source)
    # WORLD retains the donor spectral envelope and aperiodicity while changing F0.
    times = np.arange(source.shape[1], dtype=np.float64)*HOP/SR
    source64 = np.ascontiguousarray(source[0], dtype=np.float64)
    spectral = pw.cheaptrick(donor, source64, times, SR, fft_size=2048)
    ap = pw.d4c(donor, source64, times, SR, fft_size=2048)
    world = pw.synthesize(np.ascontiguousarray(target[0], dtype=np.float64), spectral, ap, SR, PERIOD)
    world = np.pad(world, (0, max(0, len(baseline)-len(world))))[:len(baseline)]
    weight = edge_weight(register_weight(target[0]), len(baseline))
    hybrid = blend_waveforms(baseline, world, weight)
    assert np.array_equal(hybrid[weight == 0], baseline[weight == 0])
    for x in [baseline, donor, world, hybrid]:
        assert np.isfinite(x).all()
    for name, data in [('original',baseline),('donor',donor),('world',world),('hybrid',hybrid)]:
        sf.write(output/(name+'.wav'), data, SR, subtype='FLOAT')
    sf.write(output/'compare_original_then_hybrid.wav',np.concatenate([baseline,np.zeros(SR//2),hybrid]),SR,subtype='FLOAT')
    np.savez(output/'parameters.npz', target_f0=target, source_f0=source, weight=weight,
             spectral=spectral, aperiodicity=ap)
    report = {'sample_rate':SR,'samples':len(baseline),'donor_midi_range':[57,72],
              'high_blend_start_midi':74,'high_blend_full_midi':77,'target_low_midi':54,'unprocessed_samples_identical':True,
              'replacement_samples':int(np.count_nonzero(weight)),
              'peaks':{k:float(abs(x).max()) for k,x in [('original',baseline),('hybrid',hybrid)]},
              'processing':'WORLD spectral envelope and aperiodicity retained; target F0 substituted; no EQ or gain',
              'status':'experimental offline renderer; not a VOCALOID implementation or installed OpenUtau backend'}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('bank',type=Path);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();data=np.load(a.inputs)
    inputs={k:data[k if k in data else 'dbg_'+k] for k in ['tokens','durations','f0']}
    print(json.dumps(render(a.bank, inputs, a.output),indent=2))
