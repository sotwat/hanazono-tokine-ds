"""Check exact bypass and frame energy retention for the ru trial."""
import argparse
import numpy as np
import onnxruntime as ort


def check(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options)
    rng = np.random.default_rng(83)
    count = 0
    for phone in range(32):
        for following in range(32):
            rd, ud = (2, 3, 5, 9)[following % 4], (2, 3, 12)[phone % 3]
            durations = np.array([[8, rd, ud, 8]], np.int64)
            tokens = np.array([[1, phone, following, 1]], np.int64)
            mel = rng.normal(-4, 1, (1, int(durations.sum()), 128)).astype('float32')
            result = session.run(None, dict(tokens=tokens, durations=durations, mel=mel))[0]
            active = phone == 22 and following == 28 and ud >= 3
            mask = np.zeros(mel.shape[1], bool)
            if active:
                mask[8:8+rd] = True
            assert np.isfinite(result).all()
            assert np.array_equal(result[:, ~mask], mel[:, ~mask])
            energy = lambda x: np.logaddexp.reduce(x, axis=2)
            assert np.allclose(energy(result), energy(mel), atol=3e-6)
            if active:
                assert np.max(abs(result-mel)) > .05
            count += 1
    print(f'{count} token-pair scope and energy cases passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('layer')
    check(parser.parse_args().layer)
