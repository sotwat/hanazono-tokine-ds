"""Retain a donor frame instead of averaging its pitch-dependent mel harmonics."""
import argparse
from pathlib import Path

import onnx


def update(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    for prefix in ('original_', 'donor_'):
        target = prefix + 'ps_safe_mean'
        node = next(n for n in model.graph.node if target in n.output)
        if node.op_type != 'Where' or list(node.input) != [
            prefix + 'ps_hold_stable', prefix + 'su_hold_mean', prefix + 'ps_center_mel'
        ]:
            raise ValueError('Expected pitch-safe averaged sustain donor: ' + prefix)
        node.output[0] = prefix + 'ph_averaged'
        extra = []
        for part, source_name in (('average', node.output[0]),
                                  ('center', prefix + 'ps_center_mel')):
            extra.append(onnx.helper.make_node(
                'ReduceLogSumExp', [source_name, prefix + 'su_axis2'],
                [prefix + 'ph_' + part + '_energy'], keepdims=1))
        extra.extend([
            onnx.helper.make_node('Sub',
                [prefix + 'ph_average_energy', prefix + 'ph_center_energy'],
                [prefix + 'ph_offset']),
            onnx.helper.make_node('Add',
                [prefix + 'ps_center_mel', prefix + 'ph_offset'], [target]),
        ])
        nodes = []
        for existing in model.graph.node:
            nodes.append(existing)
            if prefix + 'ph_averaged' in existing.output:
                nodes.extend(extra)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    update(args.source, args.output)
