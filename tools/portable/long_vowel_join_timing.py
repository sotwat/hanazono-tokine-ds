"""Finish sustained-vowel onset morphs at the incoming vowel boundary."""
import argparse
from pathlib import Path
import onnx
from onnx import helper as h
from ts_affricate import dependencies


def build(source, output):
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    nodes = list(model.graph.node)
    if any('original_gv_join_shift' in n.output for n in nodes):
        raise ValueError('Vowel join timing is already installed')
    for prefix in ('original_', 'donor_'):
        g = prefix+'gv_'
        elapsed = next(n for n in nodes if g+'elapsed' in n.output)
        after = next(n for n in nodes if g+'after' in n.output)
        assert elapsed.op_type == 'Add' and after.op_type == 'Greater'
        elapsed.output[0] = g+'elapsed_before_join'
        after.input[1] = g+'join_leftneg'
        nodes.extend([
            h.make_node('Constant', [], [g+'join_min'], value=h.make_tensor('join_min', 1, [], [40.])),
            h.make_node('GreaterOrEqual', [prefix+'su_vv_prev_dur',g+'join_min'], [g+'join_long']),
            h.make_node('Where', [g+'join_long',g+'two',prefix+'su_zero'], [g+'join_shift']),
            h.make_node('Add', [g+'elapsed_before_join',g+'join_shift'], [g+'elapsed']),
            h.make_node('Sub', [g+'leftneg',g+'join_shift'], [g+'join_leftneg']),
        ])
    known = {v.name for v in model.graph.input}|{v.name for v in model.graph.initializer}|{''}
    ordered = []
    while nodes:
        ready = [n for n in nodes if dependencies(n)<=known]
        if not ready:
            raise ValueError('Unresolved graph dependency')
        for n in ready:
            ordered.append(n);known.update(n.output);nodes.remove(n)
    del model.graph.node[:];model.graph.node.extend(ordered)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();build(a.source,a.output)
