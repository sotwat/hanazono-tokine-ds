"""Add weak parallel saturation only between F5 and the existing F#5 gate."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
m = onnx.load(a.source)
outputs = {v for n in m.graph.node for v in n.output}
assert {'he_raw', 'he_f0', 'he_high_bool', 'waveform'} <= outputs
assert not any(v.startswith('fs_') for v in outputs)
for node in m.graph.node:
    for i, v in enumerate(node.output):
        if v == 'waveform':
            node.output[i] = 'fs_previous'
def const(name, value):
    m.graph.initializer.append(N.from_array(np.asarray(value, dtype=np.float32), name))
def node(op, inputs, output, **kwargs):
    m.graph.node.append(H.make_node(op, inputs, [output], **kwargs))
const('fs_threshold', 440 * 2 ** ((77 - 69) / 12))
const('fs_drive', 2)
const('fs_mix', .25)
node('GreaterOrEqual', ['he_f0', 'fs_threshold'], 'fs_above')
node('Not', ['he_high_bool'], 'fs_below')
node('And', ['fs_above', 'fs_below'], 'fs_active')
node('Cast', ['fs_active'], 'fs_mask', to=T.FLOAT)
node('Unsqueeze', ['fs_mask', 'he_axis'], 'fs_mask3')
node('Conv', ['fs_mask3', 'he_fadekernel'], 'fs_past', pads=[440, 0])
node('Conv', ['fs_mask3', 'he_fadekernel'], 'fs_future', pads=[0, 440])
node('Min', ['fs_past', 'fs_future'], 'fs_fade3')
node('Squeeze', ['fs_fade3', 'he_axis'], 'fs_fade')
node('Mul', ['fs_fade', 'fs_mask'], 'fs_weight')
node('Mul', ['he_raw', 'fs_drive'], 'fs_driven')
node('Tanh', ['fs_driven'], 'fs_tanh')
node('Div', ['fs_tanh', 'fs_drive'], 'fs_saturated')
node('Sub', ['fs_saturated', 'he_raw'], 'fs_delta')
node('Mul', ['fs_delta', 'fs_mix'], 'fs_parallel')
node('Mul', ['fs_parallel', 'fs_weight'], 'fs_wet')
node('Add', ['fs_previous', 'fs_wet'], 'fs_candidate')
node('Where', ['fs_active', 'fs_candidate', 'fs_previous'], 'waveform')
onnx.checker.check_model(m)
onnx.save(m, a.output)
