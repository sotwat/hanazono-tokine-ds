"""Experimental short-vowel neural resynthesis for failing sustained vowels."""
import argparse
import copy
from pathlib import Path

import onnx
import torch
from onnx import helper as h
from build_pair import append_model


class TimeMaps(torch.nn.Module):
    def __init__(self, interior_guard=False):
        super().__init__()
        self.interior_guard = interior_guard

    def forward(self, tokens, durations, f0):
        starts = torch.cumsum(durations, dim=1)-durations
        ends = starts+durations
        t = torch.arange(f0.shape[1], device=f0.device)[None, None, :]
        inside = (t >= starts[:, :, None]) & (t < ends[:, :, None])
        high = torch.where(inside, f0[:, None, :], torch.zeros_like(f0[:, None, :])).amax(dim=2)
        low = torch.where(inside, f0[:, None, :], torch.full_like(f0[:, None, :], 1e6)).amin(dim=2)
        vowel = (tokens == 3) | (tokens == 9) | (tokens == 14) | (tokens == 20) | (tokens == 28)
        if self.interior_guard:
            edge = torch.maximum(durations // 5, torch.full_like(durations, 6))
            core = inside & (t >= (starts+edge)[:, :, None]) & (t < (ends-edge)[:, :, None])
            core_high = torch.where(core, f0[:, None, :], torch.zeros_like(f0[:, None, :])).amax(dim=2)
            core_low = torch.where(core, f0[:, None, :], torch.full_like(f0[:, None, :], 1e6)).amin(dim=2)
            stable = core_high < core_low*1.0293
        else:
            stable = high < low*1.0293
        eligible = vowel & (durations >= 40) & (low > 60) & stable
        compressed = torch.where(eligible, torch.full_like(durations, 24), durations)
        cs = torch.cumsum(compressed, dim=1)-compressed
        ce = cs+compressed
        ct = torch.arange(compressed.sum(), device=f0.device)[None, None, :]
        ci = (ct >= cs[:, :, None]) & (ct < ce[:, :, None])
        # Endpoint-preserving maps do not move consonant or vowel boundaries.
        forward = starts[:, :, None]+(ct-cs[:, :, None])*(durations-1).clamp(min=0)[:, :, None]/(compressed-1).clamp(min=1)[:, :, None]
        forward = (forward*ci).sum(dim=1).float().clamp(min=0, max=f0.shape[1]-1)
        left = forward.long()
        right = (left+1).clamp(max=f0.shape[1]-1)
        alpha = forward-left
        short_f0 = f0.gather(1, left)*(1-alpha)+f0.gather(1, right)*alpha
        back = cs[:, :, None]+(t-starts[:, :, None])*(compressed-1).clamp(min=0)[:, :, None]/(durations-1).clamp(min=1)[:, :, None]
        back = (back*inside).sum(dim=1).float()
        mask = (inside*eligible[:, :, None]).sum(dim=1).float().unsqueeze(2)
        return compressed, short_f0, back, mask


class RestoreVowel(torch.nn.Module):
    def forward(self, short, back, eligible, weight, baseline):
        left = back.long().clamp(min=0, max=short.shape[1]-1)
        right = (left+1).clamp(max=short.shape[1]-1)
        a = (back-left).unsqueeze(2)
        li = left.unsqueeze(2).expand(-1, -1, 128)
        ri = right.unsqueeze(2).expand(-1, -1, 128)
        restored = short.gather(1, li)*(1-a)+short.gather(1, ri)*a
        w = (eligible*weight).clamp(0, 1)
        return torch.where(w > 0, baseline*(1-w)+restored*w, baseline)


def rename_graph(graph, prefix, fixed):
    def name(s):
        return fixed.get(s, prefix+s) if s else s
    for value in list(graph.input)+list(graph.output)+list(graph.value_info)+list(graph.initializer):
        value.name = name(value.name)
    for node in graph.node:
        node.name = prefix+node.name if node.name else ''
        for field in (node.input, node.output):
            for i, s in enumerate(field):
                field[i] = name(s)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                rename_graph(attr.g, prefix, fixed)


def build(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    m = onnx.load(source)
    if any('tc_original_su_raw' in n.output for n in m.graph.node):
        raise ValueError('Already patched')
    raw_end = max(i for i, n in enumerate(m.graph.node) if any(s in n.output for s in ('original_su_raw', 'donor_su_raw')))+1
    neural = copy.deepcopy(m.graph)
    del neural.node[raw_end:]
    fixed = {x.name: x.name for x in m.graph.initializer}
    fixed.update(tokens='tokens', durations='tc_durations', f0='tc_f0', depth='depth', steps='steps')
    rename_graph(neural, 'tc_', fixed)
    args = (torch.tensor([[1, 3, 16, 14, 1]]), torch.tensor([[8, 53, 7, 20, 8]]), torch.full((1, 96), 233.))
    maps = output/'maps.onnx'
    torch.onnx.export(TimeMaps(), args, maps, input_names=['tokens', 'durations', 'f0'], output_names=['short_durations', 'short_f0', 'back', 'eligible'], opset_version=18, dynamo=False, dynamic_axes={'tokens': {1:'N'}, 'durations':{1:'N'}, 'f0':{1:'F'}, 'short_durations':{1:'N'}, 'short_f0':{1:'C'}, 'back':{1:'F'}, 'eligible':{1:'F'}})
    restore = output/'restore.onnx'
    torch.onnx.export(RestoreVowel(), (torch.randn(1, 64, 128), torch.linspace(0, 63, 96)[None], torch.ones(1, 96, 1), torch.ones(1, 96, 1), torch.randn(1, 96, 128)), restore, input_names=['short','back','eligible','weight','baseline'], output_names=['restored'], opset_version=18,dynamo=False,dynamic_axes={'short':{1:'C'},'back':{1:'F'},'eligible':{1:'F'},'weight':{1:'F'},'baseline':{1:'F'},'restored':{1:'F'}})
    nodes, initializers = [], list(m.graph.initializer)
    append_model(nodes, initializers, maps, 'tm_', dict(tokens='tokens', durations='durations', f0='f0',short_durations='tc_durations',short_f0='tc_f0',back='tc_back',eligible='tc_eligible'))
    nodes.extend(neural.node)
    for n in m.graph.node:
        if 'paired_mel' in n.output:
            for p in ('original_', 'donor_'):
                append_model(nodes, initializers, restore, 'tc_restore_'+p, dict(short='tc_'+p+'su_raw',back='tc_back',eligible='tc_eligible',weight=p+'ss_weight',baseline=p+'mel',restored='tc_'+p+'mel'))
            for j, s in enumerate(n.input):
                if s in ('original_mel','donor_mel'):
                    n.input[j] = 'tc_'+s
        nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    del m.graph.initializer[:]
    m.graph.initializer.extend(initializers)
    onnx.checker.check_model(m)
    onnx.save(m, output/'acoustic.onnx')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    build(a.source, a.output)
