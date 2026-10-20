"""Use stable-interior short resynthesis without retaining fixed-spectrum sustain."""
import argparse
from pathlib import Path

import onnx
import torch
from build_pair import append_model
from short_vowel_resynthesis import TimeMaps
from ts_affricate import dependencies


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    model = onnx.load(source)
    if any('original_ivg_old_wet' in n.output for n in model.graph.node):
        raise ValueError('Exclusive sustain routing is already installed')
    if not any('tc_eligible' in n.output for n in model.graph.node):
        raise ValueError('Short-vowel resynthesis must already be installed')
    layer = output/'maps.onnx'
    args = (torch.tensor([[1,3,16,14,1]]), torch.tensor([[8,87,6,20,8]]), torch.full((1,129),293.66))
    torch.onnx.export(TimeMaps(interior_guard=True), args, layer,
        input_names=['tokens','durations','f0'],output_names=['short_durations','short_f0','back','eligible'],
        opset_version=18,dynamo=False,dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'f0':{1:'F'},
            'short_durations':{1:'N'},'short_f0':{1:'C'},'back':{1:'F'},'eligible':{1:'F'}})
    nodes=[n for n in model.graph.node if not any(o.startswith('tm_') or
        o in ('tc_durations','tc_f0','tc_back','tc_eligible') for o in n.output)]
    constants=list(model.graph.initializer)
    # The restore pass already applies ss_weight. Keeping the fixed-spectrum
    # wet path underneath it leaves w*(1-w) of the rejected fixed spectrum.
    for prefix in ('original_', 'donor_'):
        for n in nodes:
            if prefix+'su_wet' in n.output:
                n.output[0] = prefix+'ivg_old_wet'
                break
        else:
            raise ValueError('Missing sustain wet path: '+prefix)
        nodes.extend([
            onnx.helper.make_node('Greater', ['tc_eligible', prefix+'ss_zero'], [prefix+'ivg_active']),
            onnx.helper.make_node('Where', [prefix+'ivg_active', prefix+'ss_zero', prefix+'ivg_old_wet'], [prefix+'su_wet']),
        ])
    append_model(nodes,constants,layer,'tm_',dict(tokens='tokens',durations='durations',f0='f0',
        short_durations='tc_durations',short_f0='tc_f0',back='tc_back',eligible='tc_eligible'))
    known={v.name for v in model.graph.input}|{v.name for v in constants}|{''};ordered=[]
    while nodes:
        ready=[n for n in nodes if dependencies(n)<=known]
        if not ready:raise ValueError('Unresolved graph dependency')
        for n in ready:ordered.append(n);known.update(n.output);nodes.remove(n)
    del model.graph.node[:];model.graph.node.extend(ordered)
    del model.graph.initializer[:];model.graph.initializer.extend(constants)
    onnx.checker.check_model(model);onnx.save(model,output/'acoustic.onnx')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();build(a.source,a.output)
