"""Lengthen the sustain-to-original tail crossfade without altering its onset."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper


def update(source, output, frames=12):
    model = onnx.load(source)
    fades = [value for value in model.graph.initializer if value.name.endswith('su_endfade')]
    if not fades:
        raise ValueError('Expected a sustain-coverage model')
    if output.exists():
        raise FileExistsError(output)
    for value in fades:
        value.CopyFrom(numpy_helper.from_array(np.array(frames, dtype=np.float32), value.name))
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    update(args.source, args.output)
