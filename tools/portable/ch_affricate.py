"""Extend the installed affricate pass to ch without changing external timing."""
import argparse
from pathlib import Path

import onnx
import torch
from build_pair import append_model
from ts_affricate import AffricateMaps, dependencies


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    model = onnx.load(source)
    if not any('ta_tokens' in n.output for n in model.graph.node):
        raise ValueError('The prior ts affricate pass is required')
    args = (torch.tensor([[1, 6, 14, 27, 28, 1]]),
            torch.tensor([[8, 7, 20, 7, 20, 8]]), torch.full((1, 70), 440.))
    layer = output/'maps.onnx'
    torch.onnx.export(AffricateMaps(include_ch=True), args, layer,
        input_names=['tokens', 'durations', 'f0'],
        output_names=['new_tokens', 'new_durations', 'ts_mask'],
        opset_version=18, dynamo=False,
        dynamic_axes={k: {1: n} for k, n in [('tokens','N'), ('durations','N'), ('f0','F'),
            ('new_durations','C'), ('new_tokens','C'), ('ts_mask','F')]})
    # Replace only the expansion map, retaining the existing neural pass and selection.
    nodes = [n for n in model.graph.node if not any(
        o.startswith('ta_/') or o.startswith('ta_onnx::') or
        o in ('ta_tokens', 'ta_durations', 'ta_ts_mask') for o in n.output)]
    constants = list(model.graph.initializer)
    append_model(nodes, constants, layer, 'ta_', dict(tokens='tokens', durations='durations', f0='f0',
        new_durations='ta_durations', new_tokens='ta_tokens', ts_mask='ta_ts_mask'))
    known = {v.name for v in model.graph.input} | {v.name for v in constants} | {''}
    ordered = []
    while nodes:
        ready = [n for n in nodes if dependencies(n) <= known]
        if not ready:
            raise ValueError('Unresolved graph dependencies')
        for node in ready:
            ordered.append(node); known.update(node.output); nodes.remove(node)
    del model.graph.node[:]; model.graph.node.extend(ordered)
    del model.graph.initializer[:]; model.graph.initializer.extend(constants)
    onnx.checker.check_model(model)
    onnx.save(model, output/'acoustic.onnx')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();build(a.source,a.output)
