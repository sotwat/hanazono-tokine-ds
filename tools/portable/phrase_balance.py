"""Apply a bounded scalar gain per rendered phrase, measured on voiced frames."""
import argparse
from pathlib import Path
import onnx
import torch
from build_pair import append_model

class PhraseBalance(torch.nn.Module):
    def forward(self, wave, mask):
        index = (torch.arange(wave.shape[1], device=wave.device) // 512).clamp(max=mask.shape[1]-1)
        weight = mask[:, index]
        count = weight.sum(dim=1, keepdim=True)
        rms = ((wave.square()*weight).sum(dim=1, keepdim=True)/count.clamp(min=1)+1e-10).sqrt()
        gain = (0.10/rms).pow(0.7).clamp(min=0.7079458, max=2.0)
        gain = torch.where((rms > 0.004) & (count >= 1024), gain, torch.ones_like(gain))
        headroom = 0.95/wave.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        gain = torch.minimum(gain, headroom)
        return wave*gain

def build(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = output.parent/'phrase-balance-layer.onnx'
    torch.onnx.export(PhraseBalance(), (torch.randn(1,32*512)*.05,torch.ones(1,32)), layer,
        input_names=['wave','mask'], output_names=['balanced'], opset_version=18,dynamo=False,
        dynamic_axes={'wave':{1:'samples'},'mask':{1:'frames'},'balanced':{1:'samples'}})
    model = onnx.load(source)
    assert not any(n.startswith('pb_') for n in [x.name for x in model.graph.initializer]), 'already patched'
    assert any('he_mask' in n.output for n in model.graph.node)
    for node in model.graph.node:
        for field in (node.input,node.output):
            for i,name in enumerate(field):
                if name == 'waveform': field[i] = 'pb_input_wave'
    for value in model.graph.value_info:
        if value.name=='waveform':value.name='pb_input_wave'
    nodes=[]
    append_model(nodes,model.graph.initializer,layer,'pb_',{'wave':'pb_input_wave','mask':'he_mask','balanced':'waveform'})
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model,output)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();build(args.source,args.output)
