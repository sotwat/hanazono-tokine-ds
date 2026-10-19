"""Complete the held-spectrum mix when a phone has a substantial energy deficit."""
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
        gate = prefix + 'ss_phone_gate'
        strength = prefix + 'ss_gate_strength'
        if any(v.name == strength for v in model.graph.initializer):
            raise ValueError('Complete sustain repair already installed')
        model.graph.initializer.append(nh.from_array(np.array(4., np.float32), strength))
        nodes = []
        patched = False
        for node in model.graph.node:
            if gate in node.output:
                node.output[0] = gate + '_partial'
                nodes.extend([
                    node,
                    h.make_node('Mul', [gate + '_partial', strength], [gate + '_scaled']),
                    h.make_node('Clip', [gate + '_scaled', prefix + 'ss_zero', prefix + 'ss_one'], [gate]),
                ])
                patched = True
            else:
                nodes.append(node)
        if not patched:
            raise ValueError('Expected anticipatory phone gate: ' + prefix)
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
