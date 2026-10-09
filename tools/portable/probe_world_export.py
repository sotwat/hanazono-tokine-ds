"""Probe standard ONNX export of diffsptk WORLD; not a release builder."""
import argparse
from pathlib import Path
import torch, diffsptk, numpy as np
import inspect, importlib

def lower_bound(x, query):
    lo = torch.zeros_like(query, dtype=torch.int64)
    hi = torch.full_like(lo, x.shape[-1])
    for _ in range(32):
        mid = (lo + hi) // 2
        value = x.gather(-1, mid.clamp(max=x.shape[-1] - 1))
        right = (mid < x.shape[-1]) & (value < query)
        lo = torch.where(right, mid + 1, lo)
        hi = torch.where(right, hi, mid)
    return lo

# Fixed padding covers the bounded A3..C5 donor F0 without data-dependent shapes.
common = importlib.import_module("diffsptk.third_party.world.common")
interp_namespace = dict(vars(common), lower_bound=lower_bound)
exec(inspect.getsource(common.interp1).replace("torch.searchsorted(x, xq, right=False)", "lower_bound(x, xq)"), interp_namespace)
for module_name in ["diffsptk.modules.pitch_spec", "diffsptk.modules.world_synth"]:
    setattr(importlib.import_module(module_name), "interp1", interp_namespace["interp1"])
namespace = dict(vars(common))
source = inspect.getsource(common.linear_smoothing).replace("max_boundary = int(torch.amax(boundary))", "max_boundary = 512")
exec(source, namespace)
for module_name in ["diffsptk.modules.ap", "diffsptk.modules.pitch_spec"]:
    setattr(importlib.import_module(module_name), "linear_smoothing", namespace["linear_smoothing"])

# Input shape equality is guaranteed by the wrapper and shared dynamic dimensions.
ws = importlib.import_module("diffsptk.modules.world_synth")
ws_namespace = dict(vars(ws))
import textwrap, re
ws_source = textwrap.dedent(inspect.getsource(ws.WorldSynthesis.forward))
ws_source = ws_source.replace("torch.exp(-1j * self.ramp[:D] * coefficient.unsqueeze(-1))", "torch.complex(torch.cos(self.ramp[:D] * coefficient.unsqueeze(-1)), -torch.sin(self.ramp[:D] * coefficient.unsqueeze(-1)))")
ws_source = re.sub(r"if len\(set\(\[.*?\]\)\) != 1:", "if False:", ws_source)
for value in ["periodic_response", "aperiodic_response"]:
    ws_source = ws_source.replace(f"torch.fft.fftshift({value}, dim=-1)", f"torch.cat([{value}[..., self.fft_length//2:], {value}[..., :self.fft_length//2]], dim=-1)")
ws_source = ws_source.replace("pulse_locations = time_axis[pulse_locations_index]", "torch._check(pulse_locations_index[0].shape[0] >= 2)\n    pulse_locations = time_axis[pulse_locations_index]")
exec(ws_source, ws_namespace)
ws.WorldSynthesis.forward = ws_namespace["forward"]

class World(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = diffsptk.Aperiodicity(512, 44100, 2048, algorithm='d4c', out_format='a')
        self.sp = diffsptk.PitchAdaptiveSpectralAnalysis(512, 44100, 2048, algorithm='cheap-trick', out_format='power')
        self.synth = diffsptk.WorldSynthesis(512,44100,2048)
    def forward(self, waveform, source_f0, target_f0):
        ap=self.ap(waveform,source_f0)
        sp=self.sp(waveform,source_f0)
        return self.synth(target_f0,ap,sp)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    torch.set_num_threads(4)
    n=64
    t=torch.arange(n*512)/44100
    x=(.2*torch.sin(2*torch.pi*440*t)).unsqueeze(0)
    f=torch.full((1,n),440.)
    model=World().eval()
    with torch.no_grad():print('eager',model(x,f,f*1.5).shape,flush=True)
    torch.onnx.export(model,(x,f,f*1.5),str(a.output),opset_version=18,
        input_names=['waveform','source_f0','target_f0'],output_names=['waveform_out'],
        dynamo=True,external_data=False,optimize=False,
        dynamic_shapes=({1:512*torch.export.Dim('frames',min=16)},{1:torch.export.Dim('frames',min=16)},{1:torch.export.Dim('frames',min=16)}))
