"""Suppress cycle-incoherent residue only where acoustic sustain replacement is active."""
import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper as h, numpy_helper as nh
from build_pair import append_model


class SustainCycles(torch.nn.Module):
    def forward(self, wave, f0, control):
        pos = torch.arange(wave.shape[1], device=wave.device).unsqueeze(0)
        index = (pos // 512).clamp(max=f0.shape[1]-1)
        following = (index+1).clamp(max=f0.shape[1]-1)
        alpha = (pos % 512)/512.
        freq = f0.gather(1, index)*(1-alpha)+f0.gather(1, following)*alpha
        phase = torch.cumsum(freq/44100., dim=1)
        weights = control[:, :, 6]
        weight = weights.gather(1, index)*(1-alpha)+weights.gather(1, following)*alpha
        start = control[:, :, 1].gather(1, index)*512
        end = control[:, :, 2].gather(1, index)*512
        weight = (weight*4).clamp(0, 1)
        weight = weight*weight*(3-2*weight)
        weight = torch.where(freq >= 60, weight, torch.zeros_like(weight))
        total, count = wave.clone(), torch.ones_like(wave)
        radius = torch.ceil(44100/freq.clamp(min=60)*6).long()
        for shift in (-4., -3., -2., -1., 1., 2., 3., 4.):
            target = phase+shift
            lo = (pos-radius).clamp(min=0)
            hi = (pos+radius).clamp(max=wave.shape[1]-1)
            for _ in range(14):
                mid = (lo+hi)//2
                lower = phase.gather(1, mid)<target
                lo = torch.where(lower, mid+1, lo)
                hi = torch.where(lower, hi, mid)
            right = lo.clamp(max=wave.shape[1]-1)
            left = (right-1).clamp(min=0)
            p0, p1 = phase.gather(1, left), phase.gather(1, right)
            frac = ((target-p0)/(p1-p0).clamp(min=1e-6)).clamp(0, 1)
            value = wave.gather(1, left)*(1-frac)+wave.gather(1, right)*frac
            valid = ((left >= start) & (right < end) &
                     (target >= p0) & (target <= p1)).to(wave.dtype)
            total = total+value*valid
            count = count+valid
        # The zero-weight branch is bit-identical, including consonants and silence.
        return torch.where(weight > 0, wave*(1-weight)+(total/count)*weight, wave)


def build(acoustic: Path, vocoder: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(acoustic)
    if any(i.name == 'sc_pads' for i in model.graph.initializer):
        raise ValueError('Sustain periodicity already installed')
    model.graph.initializer.append(nh.from_array(np.array([0, 0, 6, 0, 0, 121], np.int64), 'sc_pads'))
    nodes = []
    patched = False
    for node in model.graph.node:
        if 'mel' in node.output:
            if list(node.input) != ['paired_mel', 'he_control']:
                raise ValueError('Expected portable acoustic control plane')
            nodes.extend([
                h.make_node('Pad', ['original_ss_weight', 'sc_pads', 'he_zero'], ['sc_extra']),
                h.make_node('Add', ['he_control', 'sc_extra'], ['sc_control']),
            ])
            node.input[1] = 'sc_control'
            patched = True
        nodes.append(node)
    if not patched:
        raise ValueError('Missing portable acoustic output')
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, output/'acoustic.onnx')

    layer = output/'sustain-cycles.onnx'
    frames = 32
    torch.onnx.export(SustainCycles(),
        (torch.randn(1, frames*512), torch.full((1, frames), 233.), torch.zeros(1, frames, 128)),
        layer, input_names=['wave', 'f0', 'control'], output_names=['clean_wave'],
        opset_version=18, dynamo=False,
        dynamic_axes={'wave': {1: 'samples'}, 'f0': {1: 'frames'},
                      'control': {1: 'frames'}, 'clean_wave': {1: 'samples'}})
    model = onnx.load(vocoder)
    nodes, initializers = list(model.graph.node), list(model.graph.initializer)
    for node in nodes:
        if 'waveform' in node.output:
            node.output[list(node.output).index('waveform')] = 'sc_before'
    initializers.append(nh.from_array(np.array([2], np.int64), 'sc_control_index'))
    nodes.append(h.make_node('Gather', ['mel', 'sc_control_index'], ['sc_control'], axis=0))
    append_model(nodes, initializers, layer, 'sc_filter_',
                 {'wave': 'sc_before', 'f0': 'f0', 'control': 'sc_control', 'clean_wave': 'waveform'})
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    onnx.save(model, output/'vocoder.onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('acoustic', type=Path)
    parser.add_argument('vocoder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.acoustic, args.vocoder, args.output)
