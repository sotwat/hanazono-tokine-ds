"""Trial bounded spectral contrast for r before u, within the r span."""
import argparse
from pathlib import Path

import onnx
import torch
from build_pair import append_model


class RuArticulation(torch.nn.Module):
    def forward(self, tokens, durations, mel):
        following = torch.cat((tokens[:, 1:], torch.zeros_like(tokens[:, :1])), 1)
        next_d = torch.cat((durations[:, 1:], torch.zeros_like(durations[:, :1])), 1)
        eligible = (tokens == 22) & (following == 28) & (durations >= 2) & (next_d >= 3)
        starts = durations.cumsum(1) - durations
        ends = starts + durations
        time = torch.arange(mel.shape[1], device=mel.device)[None, None, :]
        inside = eligible[:, :, None] & (time >= starts[:, :, None]) & (time < ends[:, :, None])
        # Reference only the following vowel, without altering its samples.
        anchor = (ends + torch.minimum(next_d // 2, torch.full_like(next_d, 3))).clamp(0, mel.shape[1]-1)
        index = (anchor[:, :, None] * inside).sum(1)
        reference = mel.gather(1, index.unsqueeze(2).expand(-1, -1, 128))
        contrast = mel - reference
        contrast = contrast - contrast.mean(2, keepdim=True)
        offset = (time - starts[:, :, None] + .5) / durations.clamp(min=1)[:, :, None]
        weight = ((torch.sin(offset.clamp(0, 1) * torch.pi).square()) * inside).sum(1).unsqueeze(2)
        adjusted = mel + (.65 * contrast).clamp(-.65, .65) * weight
        # Retain each frame's total linear mel amplitude; this is not gain boost.
        adjusted = adjusted + torch.logsumexp(mel, 2, keepdim=True) - torch.logsumexp(adjusted, 2, keepdim=True)
        return torch.where(inside.any(1).unsqueeze(2), adjusted, mel)


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    layer = output / 'ru.onnx'
    torch.onnx.export(RuArticulation(),
        (torch.tensor([[1, 22, 28, 1]]), torch.tensor([[8, 4, 16, 8]]), torch.randn(1, 36, 128)),
        layer, input_names=['tokens', 'durations', 'mel'], output_names=['articulated'],
        opset_version=18, dynamo=False,
        dynamic_axes={'tokens': {1: 'N'}, 'durations': {1: 'N'},
                      'mel': {1: 'F'}, 'articulated': {1: 'F'}})
    model = onnx.load(source)
    nodes = []
    for node in model.graph.node:
        if 'paired_mel' in node.output:
            for branch in ('original', 'donor'):
                old, new = 'svl_' + branch + '_joined', 'ru_' + branch + '_articulated'
                if old not in node.input:
                    raise ValueError('Expected production short vowel routing')
                append_model(nodes, model.graph.initializer, layer, 'ru_' + branch + '_',
                             dict(tokens='tokens', durations='durations', mel=old, articulated=new))
                for i, name in enumerate(node.input):
                    if name == old:
                        node.input[i] = new
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, output / 'acoustic.onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
