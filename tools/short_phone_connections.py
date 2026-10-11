"""Apply bounded consonant joins after short vowels as well as sustained vowels."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    if output.exists():
        raise FileExistsError(output)
    m = onnx.load(source)
    nodes = list(m.graph.node)
    for prefix in ['original_', 'donor_']:
        g = lambda k: prefix + 'lt_' + k
        s = lambda k: prefix + 'su_' + k
        minimum = next(i for i in m.graph.initializer if i.name == g('min'))
        minimum.CopyFrom(nh.from_array(np.array(4., np.float32), g('min')))
        m.graph.initializer.append(nh.from_array(np.array(.25, np.float32), g('previous_fraction')))
        for node in nodes:
            for i, name in enumerate(node.input):
                if name == g('left'):
                    node.input[i] = g('bounded_left')
        nodes.extend([
            h.make_node('Mul', [s('vv_prev_dur'), g('previous_fraction')], [g('previous_budget0')]),
            h.make_node('Floor', [g('previous_budget0')], [g('previous_budget')]),
            h.make_node('Min', [g('left'), g('previous_budget')], [g('bounded_left')]),
        ])
    def dependencies(node):
        result = set(node.input)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                graph = attr.g
                local = {v.name for v in graph.input} | {v.name for v in graph.initializer}
                local.update(x for n in graph.node for x in n.output)
                result.update(set().union(*(dependencies(n) for n in graph.node)) - local)
        return result
    known = {v.name for v in m.graph.input} | {v.name for v in m.graph.initializer} | {''}
    ordered = []
    while nodes:
        ready = [n for n in nodes if dependencies(n) <= known]
        if not ready:
            raise ValueError('Unresolved graph dependencies')
        for node in ready:
            ordered.append(node)
            known.update(node.output)
            nodes.remove(node)
    del m.graph.node[:]
    m.graph.node.extend(ordered)
    onnx.checker.check_model(m)
    onnx.save(m, output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    update(a.source, a.output)
