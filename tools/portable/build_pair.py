"""Build paired acoustic/vocoder ONNXs for hosts preserving the mel batch axis."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,compose,version_converter
F=onnx.TensorProto.FLOAT

def base(path):
 m=onnx.load(path)
 version=max(x.version for x in m.opset_import if x.domain=='')
 del m.opset_import[:];m.opset_import.append(h.make_opsetid('',version))
 if version<18:m=version_converter.convert_version(m,18)
 return m

def append_model(nodes,initializers,path,prefix,mapping,output=None):
 m=base(path)
 if output:
  del m.graph.output[:];m.graph.output.append(h.make_tensor_value_info(output,F,[1,'samples']))
  # Remove postprocessing that is no longer consumed.
  required={output};keep=[]
  for node in reversed(m.graph.node):
   if required.intersection(node.output):keep.append(node);required.update(node.input)
  del m.graph.node[:];m.graph.node.extend(reversed(keep))
 def prefix_graph(graph):
  for value in list(graph.input)+list(graph.output)+list(graph.value_info)+list(graph.initializer):
   value.name=prefix+value.name
  for node in graph.node:
   node.name=prefix+node.name if node.name else ''
   for field in [node.input,node.output]:
    for i,name in enumerate(field):
     if name:field[i]=prefix+name
   for attr in node.attribute:
    if attr.type==onnx.AttributeProto.GRAPH:prefix_graph(attr.g)
 prefix_graph(m.graph)
 rename={prefix+k:v for k,v in mapping.items()}
 def rename_graph(graph):
  for node in graph.node:
   for field in [node.input,node.output]:
    for i,name in enumerate(field):field[i]=rename.get(name,name)
   for attr in node.attribute:
    if attr.type==onnx.AttributeProto.GRAPH:rename_graph(attr.g)
 rename_graph(m.graph)
 nodes.extend(m.graph.node);initializers.extend(m.graph.initializer)
 return m

def build(bank,world,out):
 out.mkdir(parents=True,exist_ok=True)
 ac=bank/'tokine_ds_clarity_hybrid.tokine.onnx';voc=next((bank/'dsvocoder').glob('*.onnx'))
 def constants():return [nh.from_array(np.array(v,np.float32),k) for k,v in [('zero',0),('donor_min',220),('donor_max',523.2511306)]]
 def donor():return [h.make_node('Clip',['f0','donor_min','donor_max'],['clipped']),h.make_node('Greater',['f0','zero'],['voiced']),h.make_node('Where',['voiced','clipped','zero'],['source_f0'])]
 nodes=donor();initializers=constants()
 mapping={k:k for k in ['tokens','durations','f0','depth','steps']}
 append_model(nodes,initializers,ac,'original_',dict(mapping,mel='original_mel'))
 append_model(nodes,initializers,ac,'donor_',dict(mapping,f0='source_f0',mel='donor_mel'))
 nodes.append(h.make_node('Concat',['original_mel','donor_mel'],['mel'],axis=0))
 inputs=base(ac).graph.input
 m=h.make_model(h.make_graph(nodes,'Tokine dual acoustic',inputs,[h.make_tensor_value_info('mel',F,[2,'frames',128])],initializers),opset_imports=[h.make_opsetid('',18)],ir_version=10)
 onnx.checker.check_model(m);onnx.save(m,out/'acoustic.onnx')
 nodes=donor();initializers=constants()
 for idx,name in [(0,'original_mel'),(1,'donor_mel')]:
  initializers.append(nh.from_array(np.array([idx],np.int64),name+'_index'))
  nodes.append(h.make_node('Gather',['mel',name+'_index'],[name],axis=0))
 append_model(nodes,initializers,voc,'original_',{'mel':'original_mel','f0':'f0','he_raw':'baseline'},'he_raw')
 append_model(nodes,initializers,voc,'donor_',{'mel':'donor_mel','f0':'source_f0','he_raw':'donor_wave'},'he_raw')
 append_model(nodes,initializers,world,'world_',{'waveform':'donor_wave','source_f0':'source_f0','target_f0':'f0','waveform_out':'world_wave'})
 append_model(nodes,initializers,out/'blend.onnx','blend_',{'baseline':'baseline','world':'world_wave','f0':'f0','waveform':'waveform'})
 m=h.make_model(h.make_graph(nodes,'Tokine high WORLD vocoder',[h.make_tensor_value_info('mel',F,[2,'frames',128]),h.make_tensor_value_info('f0',F,[1,'frames'])],[h.make_tensor_value_info('waveform',F,[1,'samples'])],initializers),opset_imports=[h.make_opsetid('',18)],ir_version=10)
 onnx.checker.check_model(m);onnx.save(m,out/'vocoder.onnx')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('world',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.bank,a.world,a.output)
