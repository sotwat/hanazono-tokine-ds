"""Check register, phoneme and frequency scope of the trial AP layer."""
import argparse
import numpy as np
import onnxruntime as ort


def check(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options)
    rng = np.random.default_rng(24)
    count = 0
    for frames in (5, 16, 73):
        for midi in (0, 50, 73, 74, 75, 76, 77, 79, 90):
            for active in (False, True):
                ap = rng.uniform(.001, .999, (1, frames, 1025)).astype('float32')
                hz = 440 * 2**((midi-69)/12) if midi else 0
                f0 = np.full((1, frames), hz, np.float32)
                control = np.zeros((1, frames, 128), np.float32)
                if active:
                    control[:, 1:-1, 6] = 1
                result = session.run(None, dict(ap=ap, f0=f0, control=control))[0]
                assert np.isfinite(result).all()
                assert (result <= ap).all() and np.min(result/ap) >= .54999
                assert np.array_equal(result[control[:, :, 6] == 0], ap[control[:, :, 6] == 0])
                assert np.array_equal(result[:, :, 465:], ap[:, :, 465:])
                if midi <= 74 or not active:
                    assert np.array_equal(result, ap)
                elif frames > 5:
                    assert np.max(abs(result-ap)) > .01
                count += 1
    print(f'{count} scope cases passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('layer')
    check(parser.parse_args().layer)
