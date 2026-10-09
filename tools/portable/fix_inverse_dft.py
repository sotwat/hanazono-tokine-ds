"""Expand Hermitian inputs before inverse DFT (opset 18)."""
import sys
import onnx
import numpy as np
from onnx import helper as h, numpy_helper as nh

def fix(model):
    nodes=[]
    for node in model.graph.node:
        attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
        if node.op_type=='DFT' and attrs.get('inverse')==1 and attrs.get('onesided')==1:
            prefix=node.name+'_hermitian'
            constants={'start':[-2],'end':[0],'axis':[attrs['axis']],'step':[-1], 'conjugate':[1.,-1.]}
            for key,value in constants.items():
                model.graph.initializer.append(nh.from_array(np.array(value,dtype=np.float32 if key=='conjugate' else np.int64),prefix+'_'+key))
            nodes.append(h.make_node('Slice',[node.input[0]]+[prefix+'_'+k for k in ['start','end','axis','step']],[prefix+'_reverse']))
            nodes.append(h.make_node('Mul',[prefix+'_reverse',prefix+'_conjugate'],[prefix+'_conj']))
            nodes.append(h.make_node('Concat',[node.input[0],prefix+'_conj'],[prefix+'_full'],axis=attrs['axis']))
            node.input[0]=prefix+'_full'
            for attr in node.attribute:
                if attr.name=='onesided':attr.i=0
        nodes.append(node)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    return model

if __name__=='__main__':
    m=fix(onnx.load(sys.argv[1]));onnx.checker.check_model(m);onnx.save(m,sys.argv[2])
