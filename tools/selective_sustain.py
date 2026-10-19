"""Use held spectra only where the raw vowel loses substantial energy."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    for prefix in ['original_', 'donor_']:
        stem = prefix + 'ss_'
        def const(key, value, dtype=np.float32):
            model.graph.initializer.append(nh.from_array(np.array(value, dtype), stem + key))
        def node(op, inputs, key, **kwargs):
            return h.make_node(op, inputs, [stem + key], **kwargs)
        for key, value in [('zero', 0.), ('one', 1.), ('floor', 1e-10), ('begin', .5), ('width', .25), ('two', 2.), ('three', 3.)]:
            const(key, value)
        const('axis2', [2], np.int64)
        nodes = []
        patched = False
        for existing in model.graph.node:
            if prefix + 'su_wet' in existing.output:
                assert list(existing.input) == [prefix+'su_difference', prefix+'su_weight3']
                nodes.extend([node('Greater', [prefix+'su_weight3', stem+'zero'], 'active'), node('Where', [stem+'active', prefix+'su_loop', prefix+'su_raw'], 'held_reference')])
                for label, tensor in [('raw', prefix+'su_raw'), ('held', stem+'held_reference')]:
                    nodes.extend([
                        node('Exp', [tensor], label+'_amplitude'),
                        node('ReduceMean', [stem+label+'_amplitude', stem+'axis2'], label+'_energy', keepdims=1),
                        node('Transpose', [stem+label+'_energy'], label+'_channels', perm=[0, 2, 1]),
                        node('AveragePool', [stem+label+'_channels'], label+'_smooth', kernel_shape=[9], pads=[4, 4], strides=[1], count_include_pad=0),
                        node('Transpose', [stem+label+'_smooth'], label+'_envelope', perm=[0, 2, 1]),
                    ])
                nodes.extend([
                    node('Max', [stem+'held_envelope', stem+'floor'], 'reference'),
                    node('Div', [stem+'raw_envelope', stem+'reference'], 'ratio'),
                    node('Sub', [stem+'begin', stem+'ratio'], 'deficit'),
                    node('Div', [stem+'deficit', stem+'width'], 'linear0'),
                    node('Clip', [stem+'linear0', stem+'zero', stem+'one'], 'linear'),
                    node('Mul', [stem+'linear', stem+'linear'], 'square'),
                    node('Mul', [stem+'linear', stem+'two'], 'twice'),
                    node('Sub', [stem+'three', stem+'twice'], 'factor'),
                    node('Mul', [stem+'square', stem+'factor'], 'gate'),
                    node('Mul', [prefix+'su_weight3', stem+'gate'], 'weight'),
                ])
                existing.input[1] = stem+'weight'
                patched = True
            nodes.append(existing)
        if not patched:
            raise ValueError('Expected sustain mix: '+prefix)
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
