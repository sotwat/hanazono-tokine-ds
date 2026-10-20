"""Reduce only the n anticipation ceiling by one native frame."""
import argparse
from pathlib import Path
import onnx
from onnx import numpy_helper as nh


def update(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    model = onnx.load(source)
    caps = next(v for v in model.graph.initializer if v.name == 'ct_caps')
    values = nh.to_array(caps).copy()
    if values.shape != (32,) or values[19] != 6:
        raise ValueError('Expected the six-frame n ceiling')
    values[19] = 5
    caps.CopyFrom(nh.from_array(values, 'ct_caps'))
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    update(a.source, a.destination)
