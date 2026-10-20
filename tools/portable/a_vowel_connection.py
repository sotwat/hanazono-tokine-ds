"""Uninstalled trial: retain a moving spectrum across long vowel-to-/a/ joins."""
import argparse
from pathlib import Path
import onnx
import torch
from build_pair import append_model
from ts_affricate import dependencies


class Connection(torch.nn.Module):
    def forward(self, tokens, durations, base, existing):
        starts = durations.cumsum(1)-durations
        previous = torch.cat((tokens[:, :1]*0, tokens[:, :-1]), 1)
        pd = torch.cat((durations[:, :1]*0, durations[:, :-1]), 1)
        target = (tokens == 3) & ((previous == 9) | (previous == 14) | (previous == 20) | (previous == 28)) & (pd >= 40) & (durations >= 6)
        t = torch.arange(base.shape[1])[None, None, :]
        off = t-starts[:, :, None]
        end = torch.minimum(durations.float()*.5, torch.full_like(durations, 24).float())
        active = target[:, :, None] & (off > -6) & (off < end[:, :, None])
        def smooth(x):
            x=x.clamp(0, 1)
            return x*x*(3-2*x)
        advance = 2*smooth((off+6).float()/6)*(1-smooth(off.float()/6))
        shift = (advance*active).sum(1)
        pos = (t[:, 0].float()+shift).clamp(0, base.shape[1]-1)
        left=pos.long();right=(left+1).clamp(max=base.shape[1]-1)
        alpha=(pos-left).unsqueeze(2)
        dynamic=base.gather(1,left.unsqueeze(2).expand(-1,-1,128))*(1-alpha)+base.gather(1,right.unsqueeze(2).expand(-1,-1,128))*alpha
        mask=active.any(1).unsqueeze(2)
        return torch.where(mask, dynamic, existing)


def build(source, output):
    output.mkdir(parents=True,exist_ok=False)
    model=onnx.load(source)
    layer=output/'connection.onnx'
    torch.onnx.export(Connection(), (torch.tensor([[1,14,3,16,3,1]]),torch.tensor([[8,57,52,6,34,8]]),torch.randn(1,165,128),torch.randn(1,165,128)),layer,
        input_names=['tokens','durations','base','existing'],output_names=['joined'],opset_version=18,dynamo=False,
        dynamic_axes={'tokens':{1:'N'},'durations':{1:'N'},'base':{1:'F'},'existing':{1:'F'},'joined':{1:'F'}})
    nodes=list(model.graph.node);constants=list(model.graph.initializer)
    for prefix in ('original_','donor_'):
        node=next(n for n in nodes if list(n.input)==[prefix+'gv_dry',prefix+'gv_mix'])
        destination=node.output[0]
        assert node.op_type=='Add'
        node.output[0]=prefix+'ac_previous_mel'
        append_model(nodes,constants,layer,prefix+'ac_',dict(tokens='tokens',durations='durations',base=prefix+'gv_base',existing=prefix+'ac_previous_mel',joined=destination))
    known={v.name for v in model.graph.input}|{v.name for v in constants}|{''};ordered=[]
    while nodes:
        ready=[n for n in nodes if dependencies(n)<=known]
        if not ready:raise ValueError('Unresolved dependencies')
        for n in ready:ordered.append(n);known.update(n.output);nodes.remove(n)
    del model.graph.node[:];model.graph.node.extend(ordered)
    del model.graph.initializer[:];model.graph.initializer.extend(constants)
    onnx.checker.check_model(model);onnx.save(model,output/'acoustic.onnx')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();build(a.source,a.output)
