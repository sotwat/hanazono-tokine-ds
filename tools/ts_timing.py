"""Limit ts anticipation while preserving subsequent vowel anchors."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, destination, cap=7):
    model = onnx.load(source)
    if any(x.name == 'tt_cap' for x in model.graph.initializer):
        raise ValueError('ts timing already installed')
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == 'ph_dur_pred':
                node.output[i] = 'tt_original'
    for name, value, dtype in [('cap', cap, np.float32), ('zero', 0, np.float32),
                              ('ts', 27, np.int64)]:
        model.graph.initializer.append(nh.from_array(np.array(value, dtype), 'tt_'+name))
    nodes = []
    def n(op, inputs, output, **kwargs):
        nodes.append(h.make_node(op, inputs, ['tt_'+output], **kwargs))
    def group_sum(source, name):
        n('Unsqueeze', [source, 'lt_axis2'], name+'3')
        n('MatMul', ['lt_membership', 'tt_'+name+'3'], name+'sum3')
        n('Squeeze', ['tt_'+name+'sum3', 'lt_axis2'], name)
    group_sum('tt_original', 'total')
    n('Max', ['lt_length', 'lt_epsilon'], 'length_safe')
    n('Div', ['tt_total', 'tt_length_safe'], 'normalization')
    n('Mul', ['tt_cap', 'tt_normalization'], 'raw_cap')
    # The first group is consumed as absolute frames; later groups are scaled.
    n('Where', ['onset_first_group', 'tt_cap', 'tt_raw_cap'], 'bound')
    n('Min', ['tt_original', 'tt_bound'], 'bounded')
    n('Equal', ['lt_tokeni', 'tt_ts'], 'is_ts')
    n('Greater', ['lt_length', 'tt_zero'], 'positive_length')
    n('And', ['tt_is_ts', 'tt_positive_length'], 'target')
    n('Where', ['tt_target', 'tt_bounded', 'tt_original'], 'limited')
    n('Sub', ['tt_original', 'tt_limited'], 'saved')
    group_sum('tt_saved', 'saved_group')
    n('Mul', ['tt_original', 'lt_vowel'], 'vowels')
    group_sum('tt_vowels', 'vowel_total')
    n('Max', ['tt_vowel_total', 'lt_epsilon'], 'vowel_safe')
    n('Div', ['tt_vowels', 'tt_vowel_safe'], 'vowel_ratio')
    n('Mul', ['tt_saved_group', 'tt_vowel_ratio'], 'vowel_extra')
    # Keep initial SP unchanged; all real vowel-to-vowel groups keep their sum.
    n('Where', ['onset_first_group', 'tt_zero', 'tt_vowel_extra'], 'extra')
    nodes.append(h.make_node('Add', ['tt_limited', 'tt_extra'], ['ph_dur_pred']))
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path); a = p.parse_args()
    update(a.source, a.destination)
