"""Bound pre-vowel consonants after phrase breaks in the packaged duration model."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, destination):
    model = onnx.load(source)
    if any(node.name == 'onset_timing_output' for node in model.graph.node):
        raise ValueError('Phrase onset timing is already installed')
    required = {'lt_group', 'lt_consonant', 'lt_maximum2', 'lt_onef', 'lt_epsilon', 'lt_axis1'}
    available = {name for node in model.graph.node for name in node.output}
    available.update(value.name for value in model.graph.initializer)
    if not required <= available:
        raise ValueError('Requires the long_note_timing duration model')
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == 'ph_dur_pred':
                node.output[i] = 'onset_original'
    model.graph.initializer.extend([
        nh.from_array(np.array(0, np.float32), 'onset_zero'),
        nh.from_array(np.array(16, np.float32), 'onset_budget'),
    ])
    nodes = []
    def n(op, inputs, output, **kwargs):
        nodes.append(h.make_node(op, inputs, [output], **kwargs))
    # OpenUtau consumes initial consonant predictions as absolute frame counts;
    # unlike later groups, it does not normalize them to the group duration.
    n('Equal', ['lt_group', 'onset_zero'], 'onset_first_group')
    n('Greater', ['lt_consonant', 'onset_zero'], 'onset_is_consonant')
    n('And', ['onset_first_group', 'onset_is_consonant'], 'onset_target')
    n('Min', ['onset_original', 'lt_maximum2'], 'onset_bounded')
    n('Where', ['onset_target', 'onset_bounded', 'onset_zero'], 'onset_leads')
    n('ReduceSum', ['onset_leads', 'lt_axis1'], 'onset_total', keepdims=1)
    n('Max', ['onset_total', 'lt_epsilon'], 'onset_safe')
    n('Div', ['onset_budget', 'onset_safe'], 'onset_scale0')
    n('Min', ['onset_scale0', 'lt_onef'], 'onset_scale')
    n('Mul', ['onset_bounded', 'onset_scale'], 'onset_final')
    nodes.append(h.make_node('Where', ['onset_target', 'onset_final', 'onset_original'],
                             ['ph_dur_pred'], name='onset_timing_output'))
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    update(args.source, args.destination)
