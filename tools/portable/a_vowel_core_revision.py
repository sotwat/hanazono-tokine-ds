"""Trial a longer learned /a/ core while preserving the accepted mel level contour."""
import argparse,copy
from pathlib import Path
import torch,onnx
from a_vowel_rearticulation import build as base_build
from short_vowel_resynthesis import rename_graph
from ts_affricate import dependencies
from build_pair import append_model

class Contour(torch.nn.Module):
    def forward(self,reference,candidate):
        offset=torch.logsumexp(reference,dim=2,keepdim=True)-torch.logsumexp(candidate,dim=2,keepdim=True)
        changed=(reference!=candidate).any(dim=2,keepdim=True)
        return torch.where(changed,candidate+offset,reference)

def build(source,output):
    output.mkdir(parents=True,exist_ok=False)
    base_build(source,output/'reference')
    base_build(source,output/'longer',24,6,15)
    ref=onnx.load(output/'reference/acoustic.onnx');alt=onnx.load(output/'longer/acoustic.onnx')
    boundary={v.name for v in ref.graph.input}|{'ta_original_mel','ta_donor_mel'}
    needed={'ar_original_mel','ar_donor_mel'};selected=[]
    for n in reversed(alt.graph.node):
        if (set(n.output)&needed)-boundary:
            selected.append(copy.deepcopy(n));needed.update(dependencies(n))
    g=copy.deepcopy(alt.graph);del g.node[:];g.node.extend(reversed(selected))
    fixed={name:name for name in boundary}
    # Only the trained tensors are shared; exported context/join constants are nodes.
    fixed.update({v.name:v.name for v in alt.graph.initializer})
    rename_graph(g,'acr_',fixed)
    layer=output/'contour.onnx'
    torch.onnx.export(Contour(),(torch.randn(1,80,128),torch.randn(1,80,128)),layer,input_names=['reference','candidate'],output_names=['result'],dynamic_axes={k:{1:'F'} for k in ['reference','candidate','result']},opset_version=18,dynamo=False)
    nodes=list(ref.graph.node);constants=list(ref.graph.initializer);nodes.extend(g.node)
    for branch in ['original','donor']:
        append_model(nodes,constants,layer,'acr_contour_'+branch+'_',dict(reference='ar_'+branch+'_mel',candidate='acr_ar_'+branch+'_mel',result='acr_'+branch+'_result'))
    for n in nodes:
        if 'paired_mel' in n.output:
            for i,name in enumerate(n.input):
                if name in ['ar_original_mel','ar_donor_mel']:n.input[i]='acr_'+name[3:-4]+'_result'
    known={v.name for v in ref.graph.input}|{v.name for v in constants}|{''};ordered=[]
    while nodes:
        ready=[n for n in nodes if dependencies(n)<=known]
        if not ready:raise ValueError('unresolved dependencies')
        for n in ready:ordered.append(n);known.update(n.output);nodes.remove(n)
    del ref.graph.node[:];ref.graph.node.extend(ordered);del ref.graph.initializer[:];ref.graph.initializer.extend(constants)
    onnx.checker.check_model(ref);onnx.save(ref,output/'acoustic.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.source,a.output)
