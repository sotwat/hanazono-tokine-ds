"""Resynthesize ts as a short learned t followed by s, within its original span."""
import argparse
import copy
from pathlib import Path

import onnx
import torch
from onnx import helper as h
from build_pair import append_model
from short_vowel_resynthesis import rename_graph


def dependencies(node):
    result = set(node.input)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            local = {v.name for v in attr.g.input} | {v.name for v in attr.g.initializer} | {v for n in attr.g.node for v in n.output}
            result.update(set().union(*(dependencies(n) for n in attr.g.node))-local)
    return result


class AffricateMaps(torch.nn.Module):
    def forward(self, tokens, durations, f0):
        following = torch.cat((tokens[:, 1:], torch.zeros_like(tokens[:, :1])), dim=1)
        vowel = (following == 3) | (following == 9) | (following == 14) | (following == 20) | (following == 28)
        target = (tokens == 27) & (durations >= 2) & vowel
        stop = torch.minimum(durations // 3, torch.full_like(durations, 3)).clamp(min=1)
        first = torch.where(target, torch.full_like(tokens, 26), tokens)
        second = torch.full_like(tokens, 24)
        first_d = torch.where(target, stop, durations)
        second_d = torch.where(target, durations-stop, torch.zeros_like(durations))
        keep = torch.stack((torch.ones_like(target), target), dim=2).flatten()
        new_tokens = torch.stack((first, second), dim=2).flatten()[keep].unsqueeze(0)
        new_durations = torch.stack((first_d, second_d), dim=2).flatten()[keep].unsqueeze(0)
        starts = torch.cumsum(durations, dim=1)-durations
        t = torch.arange(f0.shape[1], device=f0.device)[None, None, :]
        inside = (t >= starts[:, :, None]) & (t < (starts+durations)[:, :, None])
        mask = (inside*target[:, :, None]).sum(dim=1).float().unsqueeze(2)
        return new_tokens, new_durations, mask


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    m = onnx.load(source)
    if any(n.name.startswith('ta_') for n in m.graph.node):
        raise ValueError('ts articulation is already installed')
    args = (torch.tensor([[1, 27, 28, 3, 1]]), torch.tensor([[8, 16, 20, 53, 8]]), torch.full((1, 105), 233.))
    layer = output/'maps.onnx'
    torch.onnx.export(AffricateMaps(), args, layer,
        input_names=['tokens', 'durations', 'f0'],
        output_names=['new_tokens', 'new_durations', 'ts_mask'],
        opset_version=18, dynamo=False,
        dynamic_axes={k: {1: n} for k, n in [('tokens','N'), ('durations','N'), ('f0','F'),
            ('new_durations','C'), ('new_tokens','C'), ('ts_mask','F')]})
    nodes, constants = [], list(m.graph.initializer)
    append_model(nodes, constants, layer, 'ta_', dict(tokens='tokens', durations='durations', f0='f0',
        new_durations='ta_durations', new_tokens='ta_tokens', ts_mask='ta_ts_mask'))
    needed = {'original_su_raw', 'donor_su_raw'}
    selected = []
    boundary = {v.name for v in m.graph.input}
    for node in reversed(m.graph.node):
        if any(o in needed and o not in boundary for o in node.output):
            selected.append(copy.deepcopy(node)); needed.update(dependencies(node))
    neural = copy.deepcopy(m.graph)
    del neural.node[:]; neural.node.extend(reversed(selected))
    fixed = {v.name: v.name for v in m.graph.initializer}
    fixed.update(tokens='ta_tokens', durations='ta_durations', f0='f0', depth='depth', steps='steps')
    rename_graph(neural, 'ta_neural_', fixed)
    nodes.extend(neural.node)
    nodes.append(h.make_node('Cast', ['ta_ts_mask'], ['ta_ts_bool'], to=onnx.TensorProto.BOOL))
    for node in m.graph.node:
        if 'paired_mel' in node.output:
            for prefix in ('original_', 'donor_'):
                nodes.append(h.make_node('Where', ['ta_ts_bool', 'ta_neural_'+prefix+'su_raw', 'tc_'+prefix+'mel'],
                    ['ta_'+prefix+'mel']))
                for i, name in enumerate(node.input):
                    if name == 'tc_'+prefix+'mel': node.input[i] = 'ta_'+prefix+'mel'
        nodes.append(node)
    # Preserve topological order after adding dependencies on the existing weights.
    known = {v.name for v in m.graph.input} | {v.name for v in constants} | {''}
    ordered = []
    while nodes:
        ready = [n for n in nodes if dependencies(n) <= known]
        if not ready:
            missing = sorted(set().union(*(dependencies(n) for n in nodes))-known-{v for n in nodes for v in n.output})
            raise ValueError('Unresolved graph dependencies: '+repr(missing))
        for node in ready:
            ordered.append(node); known.update(node.output); nodes.remove(node)
    del m.graph.node[:]; m.graph.node.extend(ordered)
    del m.graph.initializer[:]; m.graph.initializer.extend(constants)
    onnx.checker.check_model(m)
    onnx.save(m, output/'acoustic.onnx')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); build(a.source, a.output)
