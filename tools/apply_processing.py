"""Apply v1.1.1 output processing to the original v1.1.0 vocoder."""
from pathlib import Path
import argparse
import onnx
p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('output');a=p.parse_args()
m=onnx.load(a.original)
for node in m.graph.node:
    for i,name in enumerate(node.output):
        if name=='waveform':node.output[i]='he_raw'
    for i,name in enumerate(node.input):
        if name=='waveform':node.input[i]='he_raw'
patch=onnx.load(Path(__file__).with_name('output-processing.onnx'))
m.graph.node.extend(patch.graph.node);m.graph.initializer.extend(patch.graph.initializer)
onnx.checker.check_model(m);onnx.save(m,a.output)
