"""Trial: reduce WORLD noise excitation inside high voiced vowels."""
import argparse
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from build_pair import append_model


class HighVowelAP(torch.nn.Module):
    def forward(self, ap, f0, control):
        vowel = (control[:, :, 6] > .5).to(ap.dtype)
        edge = F.avg_pool1d(F.pad(vowel.unsqueeze(1), (2, 2)), 5, stride=1)[:, 0]
        midi = 69 + 12 * torch.log2(f0.clamp(min=1) / 440)
        register = ((midi - 74) / 3).clamp(0, 1)
        register = register.square() * (3 - 2 * register)
        weight = vowel * edge * register * (f0 > 0).to(ap.dtype)
        frequency = torch.arange(ap.shape[2], device=ap.device) * (22050 / 1024)
        band = ((10000 - frequency) / 4000).clamp(0, 1)
        factor = 1 - .45 * weight.unsqueeze(2) * band
        return torch.where(weight.unsqueeze(2) > 0, ap * factor, ap)


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    layer = output / 'high-vowel-ap.onnx'
    torch.onnx.export(HighVowelAP(),
        (torch.full((1, 32, 1025), .5), torch.full((1, 32), 784.), torch.zeros(1, 32, 128)),
        layer, input_names=['ap', 'f0', 'control'], output_names=['clean_ap'],
        opset_version=18, dynamo=False,
        dynamic_axes={'ap': {1: 'frames'}, 'f0': {1: 'frames'},
                      'control': {1: 'frames'}, 'clean_ap': {1: 'frames'}})
    model = onnx.load(source)
    nodes = list(model.graph.node)
    target = next(i for i, node in enumerate(nodes) if 'world_clamp_13' in node.output)
    if nodes[target].input[0] != 'he_ap_clean':
        raise ValueError('Expected production nasal AP routing')
    extra = [onnx.helper.make_node('Unsqueeze', ['he_nasal_control', 'rp_axis0'], ['hva_control'])]
    append_model(extra, model.graph.initializer, layer, 'hva_',
                 dict(ap='he_ap_clean', f0='f0', control='hva_control', clean_ap='hva_clean_ap'))
    nodes[target].input[0] = 'hva_clean_ap'
    del model.graph.node[:]
    model.graph.node.extend(nodes[:target] + extra + nodes[target:])
    onnx.checker.check_model(model)
    onnx.save(model, output / 'vocoder.onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
