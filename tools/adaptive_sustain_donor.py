"""Search the interior of each short sustained phone for its donor spectrum."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    prefixes = [v.name[:-len('su_donor_centers')] for v in model.graph.initializer
                if v.name.endswith('su_donor_centers')]
    if not prefixes or any(v.name.endswith('su_adaptive_fractions') for v in model.graph.initializer):
        raise ValueError('Expected fixed-window sustain donor selection')
    for prefix in prefixes:
        s = lambda key: prefix + 'su_' + key
        for key, value in [('adaptive_fractions', np.array([.25,.35,.45,.55,.65,.75], np.float32)),
                           ('adaptive_offsets', np.array([-2,0,2], np.float32))]:
            model.graph.initializer.append(nh.from_array(value, s(key)))
        nodes = []
        for node in model.graph.node:
            if s('donor_indices_unclipped') in node.output:
                nodes.extend([
                    h.make_node('Mul', [s('dur'), s('adaptive_fractions')], [s('adaptive_centers')]),
                    h.make_node('Unsqueeze', [s('adaptive_centers'), s('donor_axis3')], [s('adaptive_centers4')]),
                    h.make_node('Add', [s('adaptive_centers4'), s('adaptive_offsets')], [s('adaptive_local')]),
                    h.make_node('Cast', [s('donor_starts')], [s('adaptive_starts')], to=onnx.TensorProto.FLOAT),
                    h.make_node('Add', [s('adaptive_starts'), s('adaptive_local')], [s('adaptive_global')]),
                    h.make_node('Cast', [s('adaptive_global')], [s('donor_indices_unclipped')], to=onnx.TensorProto.INT64),
                ])
                continue
            if s('donor_center') in node.output:
                node.op_type = 'GatherElements'
                del node.input[:]
                node.input.extend([s('adaptive_centers'), s('donor_best')])
                del node.attribute[:]
                node.attribute.append(h.make_attribute('axis', 2))
            nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    update(args.source, args.output)
