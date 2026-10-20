"""Verify time maps, unchanged boundaries and full interior replacement."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from stable_vowel_resynthesis import EdgeTimeMaps, RestoreEdges


def check(directory):
    session = ort.InferenceSession(str(directory / 'maps.onnx'), providers=['CPUExecutionProvider'])
    count = 0
    for token in range(32):
        for duration in (1,19,20,24,39,40,84,160):
            for mode in ('flat','glide','edge'):
                tokens = torch.tensor([[1,token,1]])
                durations = torch.tensor([[7,duration,9]])
                f0 = torch.full((1,duration+16),293.66)
                if mode == 'glide':
                    f0[0,7:7+duration] = torch.linspace(250,350,duration)
                if mode == 'edge' and duration >= 20:
                    f0[0,7:11] = torch.linspace(260,293.66,4)
                expected = EdgeTimeMaps()(tokens,durations,f0)
                actual = session.run(None,dict(tokens=tokens.numpy(),durations=durations.numpy(),f0=f0.numpy()))
                for a,b in zip(expected,actual):
                    np.testing.assert_allclose(a.numpy(),b,rtol=1e-5,atol=2e-5)
                compressed,short_f0,back,mask = expected
                assert bool((back[:,1:] >= back[:,:-1]).all())
                assert short_f0.shape[1] == compressed.sum()
                if mask.max():
                    np.testing.assert_allclose(back[0,7:11],np.arange(7,11),atol=1e-5)
                    end = 7+int(compressed[0,1])
                    np.testing.assert_allclose(back[0,7+duration-4:7+duration],np.arange(end-4,end),atol=1e-5)
                else:
                    np.testing.assert_array_equal(back.numpy(),np.arange(duration+16)[None])
                count += 1
    tokens = torch.tensor([[1,3,9,1]])
    durations = torch.tensor([[0,84,0,8]])
    compressed,_,back,mask = EdgeTimeMaps()(tokens,durations,torch.full((1,92),293.66))
    baseline = torch.zeros(1,92,128)
    short = torch.ones(1,int(compressed.sum()),128)
    expected = RestoreEdges()(short,back,mask,baseline,durations)
    restore = ort.InferenceSession(str(directory/'restore.onnx'),providers=['CPUExecutionProvider'])
    actual = restore.run(None,dict(short=short.numpy(),back=back.numpy(),eligible=mask.numpy(),baseline=baseline.numpy(),durations=durations.numpy()))[0]
    np.testing.assert_allclose(expected.numpy(),actual,atol=1e-6)
    assert np.array_equal(actual[:,84:],baseline[:,84:].numpy())
    assert actual[0,0,0] == actual[0,83,0] == 0
    assert (actual[0,4:78] == 1).all()
    assert np.isfinite(actual).all()
    report = dict(map_cases=count,zero_duration=True,unchanged_boundaries=True,full_interior=True)
    (directory/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    check(parser.parse_args().directory)
