"""Keep connected syllables sustained while retaining the softer phrase ending."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    model = onnx.load(source)
    if output.exists():
        raise FileExistsError(output)
    prefixes = [v.name[:-len('su_endfade')] for v in model.graph.initializer if v.name.endswith('su_endfade')]
    if not prefixes:
        raise ValueError('Expected the sustain release wrapper')
    for prefix in prefixes:
        name = lambda value: prefix + 'su_' + value
        if any(v.name == name('connected_fade') for v in model.graph.initializer):
            raise ValueError('Contextual release already applied')
        constants = [('next_start', [1], np.int64), ('next_end', [2147483647], np.int64),
                     ('next_axis', [1], np.int64), ('next_pads', [0,0,0,1], np.int64),
                     ('silence_id', 1, np.int64), ('connected_fade', 6., np.float32)]
        for key, value, dtype in constants:
            model.graph.initializer.append(nh.from_array(np.array(value,dtype),name(key)))
        nodes = []
        for node in model.graph.node:
            if name('release') in node.output:
                nodes.extend([
                    h.make_node('Slice',['tokens',name('next_start'),name('next_end'),name('next_axis')],[name('next_slice')]),
                    h.make_node('Pad',[name('next_slice'),name('next_pads'),name('silence_id')],[name('next_tokens')]),
                    h.make_node('Equal',[name('next_tokens'),name('silence_id')],[name('ends_phrase')]),
                    h.make_node('Unsqueeze',[name('ends_phrase'),name('axis2')],[name('ends_phrase3')]),
                    h.make_node('Where',[name('ends_phrase3'),name('endfade'),name('connected_fade')],[name('context_fade')]),
                ])
                node.input[1] = name('context_fade')
            nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    update(args.source,args.output)
