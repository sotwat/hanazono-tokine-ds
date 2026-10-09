"""Remove only the F5 EQ branch from the patched vocoder."""
import argparse
import onnx
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
m=onnx.load(a.source)
keep=[n for n in m.graph.node if not any(x.startswith('f5_') and x!='f5_previous' for x in n.output) and 'waveform' not in n.output]
assert any('f5_previous' in n.output for n in keep)
for n in keep:
 for i,v in enumerate(n.output):
  if v=='f5_previous':n.output[i]='waveform'
m.graph.ClearField('node');m.graph.node.extend(keep)
init=[i for i in m.graph.initializer if not i.name.startswith('f5_')];m.graph.ClearField('initializer');m.graph.initializer.extend(init)
onnx.checker.check_model(m);onnx.save(m,a.output)
