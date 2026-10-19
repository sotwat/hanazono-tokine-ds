"""Decide sustain repair per phone, before its raw output collapses."""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    for prefix in ['original_', 'donor_']:
        s = prefix + 'ss_'
        if any(x.name == s + 'tail_margin' for x in model.graph.initializer):
            raise ValueError('Anticipatory gate already installed')
        model.graph.initializer.append(nh.from_array(np.array(8., np.float32), s + 'tail_margin'))
        nodes = []
        patched = False
        for node in model.graph.node:
            if s + 'gate' in node.output:
                node.output[0] = s + 'frame_gate'
            if s + 'weight' in node.output:
                def add(op, inputs, output, **attrs):
                    nodes.append(h.make_node(op, inputs, [s + output], **attrs))
                add('Transpose', [s + 'frame_gate'], 'gate_row', perm=[0, 2, 1])
                add('Sub', [prefix + 'su_dur', s + 'tail_margin'], 'interior_end')
                add('Less', [prefix + 'su_offset', s + 'interior_end'], 'interior')
                add('Cast', [s + 'interior'], 'interior_float', to=onnx.TensorProto.FLOAT)
                add('Mul', [s + 'interior_float', prefix + 'su_activefloat'], 'valid')
                add('Mul', [s + 'valid', s + 'gate_row'], 'valid_gate')
                add('ReduceMax', [s + 'valid_gate', s + 'axis2'], 'phone_gate', keepdims=1)
                add('Transpose', [prefix + 'su_activefloat'], 'phone_column', perm=[0, 2, 1])
                add('MatMul', [s + 'phone_column', s + 'phone_gate'], 'gate')
                patched = True
            nodes.append(node)
        if not patched:
            raise ValueError('Expected selective sustain weight: ' + prefix)
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
