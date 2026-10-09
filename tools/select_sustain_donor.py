"""Select a stronger early vowel window for short sustained phonemes."""
from pathlib import Path
import argparse
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def update(source, output):
    m=onnx.load(source)
    if output.exists():raise FileExistsError(output)
    prefixes=[i.name[:-len('su_medium_long')] for i in m.graph.initializer if i.name.endswith('su_medium_long')]
    for prefix in prefixes:
        name=lambda s:prefix+'su_'+s
        centers=np.array([10,14,18,22,26,30],np.int64)
        constants=[('donor_offsets',centers[:,None]+np.array([-2,0,2])),('donor_centers',centers.astype(np.float32)),('donor_axis3',np.array([3],np.int64)),('donor_axes',np.array([3,4],np.int64)),('donor_radius',np.array(2,np.float32))]
        for k,v in constants:m.graph.initializer.append(nh.from_array(v,name(k)))
        nodes=[]
        for node in m.graph.node:
            if name('elapsed') in node.output:
                nodes.extend([
                    h.make_node('Unsqueeze',[name('starts3'),name('donor_axis3')],[name('donor_starts')]),
                    h.make_node('Add',[name('donor_starts'),name('donor_offsets')],[name('donor_indices_unclipped')]),
                    h.make_node('Sub',[name('frames'),name('onei')],[name('donor_last')]),
                    h.make_node('Clip',[name('donor_indices_unclipped'),name('zeroi'),name('donor_last')],[name('donor_indices')]),
                    h.make_node('Squeeze',[name('raw'),name('axis0')],[name('donor_raw2')]),
                    h.make_node('Gather',[name('donor_raw2'),name('donor_indices')],[name('donor_mels')],axis=0),
                    h.make_node('Exp',[name('donor_mels')],[name('donor_amplitudes')]),
                    h.make_node('ReduceMean',[name('donor_amplitudes'),name('donor_axes')],[name('donor_scores')],keepdims=0),
                    h.make_node('ArgMax',[name('donor_scores')],[name('donor_best')],axis=2,keepdims=1),
                    h.make_node('Gather',[name('donor_centers'),name('donor_best')],[name('donor_center')],axis=0),
                    h.make_node('Where',[name('old_length'),name('effective_center'),name('donor_center')],[name('selected_center')]),
                    h.make_node('Where',[name('old_length'),name('effective_radius'),name('donor_radius')],[name('selected_radius')]),
                ])
            if name('local') in node.output:node.input[1]=name('selected_center')
            if name('swing') in node.output:node.input[1]=name('selected_radius')
            nodes.append(node)
        del m.graph.node[:];m.graph.node.extend(nodes)
    onnx.checker.check_model(m);output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.source,a.output)
