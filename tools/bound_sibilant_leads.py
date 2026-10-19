"""Limit sustained-note s/sh anticipation in the duration wrapper."""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def update(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    maximum = next(x for x in model.graph.initializer if x.name == 'lt_maximum')
    values = numpy_helper.to_array(maximum).copy()
    if values.shape != (32,) or not np.array_equal(values[[24, 25]], [16, 16]):
        raise ValueError('Expected the existing 32-token duration wrapper with s/sh cap 16')
    # This cap is used only by the existing >=40-frame, non-silence groups.
    values[[24, 25]] = 9
    maximum.CopyFrom(numpy_helper.from_array(values, maximum.name))
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    update(args.source, args.output)
