from pathlib import Path
import sys,torch,onnx
from vowel_balance import VowelBalance
from build_pair import append_model
import argparse
p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);a=p.parse_args();b=a.bank;r=a.output;r.mkdir(parents=True,exist_ok=False);layer=r/'balance-layer.onnx'
torch.onnx.export(VowelBalance(.25,1.,6.,12.,continuous_vowels=True),(torch.randn(1,64*512)*.05,torch.zeros(64,128)),layer,input_names=['wave','control'],output_names=['balanced'],opset_version=18,dynamo=False,dynamic_axes={'wave':{1:'S'},'control':{0:'F'},'balanced':{1:'S'}})
m=onnx.load(next((b/'dsvocoder').glob('*.onnx')));old=list(m.graph.node);indices=[i for i,n in enumerate(old) if any(o.startswith('evb_') or o=='vb_balanced' for o in n.output)];assert indices==list(range(min(indices),max(indices)+1));nodes=[];append_model(nodes,m.graph.initializer,layer,'evb_',dict(wave='pb_input_wave',control='rp_control',balanced='vb_balanced'));nodes=old[:min(indices)]+nodes+old[max(indices)+1:];del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,r/'vocoder.onnx')
