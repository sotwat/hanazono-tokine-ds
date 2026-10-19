"""Refresh both acoustic branches while preserving the accepted processing graph."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,onnx
from onnx import numpy_helper as nh

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def refresh(template,baseline,retrained,output):
 bank=onnx.load(template);old=onnx.load(baseline);new=onnx.load(retrained)
 # Tensor names alone are insufficient: both raw exports must have identical topology.
 def topology(m):
  g=onnx.ModelProto();g.CopyFrom(m)
  for i in g.graph.initializer:i.ClearField('raw_data');i.ClearField('float_data');i.ClearField('int32_data');i.ClearField('int64_data');i.ClearField('double_data')
  g.graph.name='';g.ClearField('metadata_props');g.ClearField('producer_name');g.ClearField('producer_version');g.ClearField('doc_string')
  return g.SerializeToString()
 assert topology(old)==topology(new),'Export topology changed; review the mapping before replacing any weights'
 a={x.name:x for x in old.graph.initializer};b={x.name:x for x in new.graph.initializer};c={x.name:x for x in bank.graph.initializer};count=0;changed=0
 for prefix in ['original_','donor_']:
  for k,v in a.items():
   name=prefix+k;assert name in c,name
   assert np.array_equal(nh.to_array(c[name]),nh.to_array(v)),f'Template differs from raw baseline: {name}'
   assert v.dims==b[k].dims
   value=nh.to_array(b[k]);changed+=not np.array_equal(nh.to_array(v),value)
   c[name].CopyFrom(nh.from_array(value,name));count+=1
 onnx.checker.check_model(bank);output.parent.mkdir(parents=True,exist_ok=True);onnx.save(bank,output)
 result=dict(template_sha256=digest(template),baseline_sha256=digest(baseline),retrained_sha256=digest(retrained),output_sha256=digest(output),raw_export_topology_identical=True,branch_tensors_replaced=count,changed_branch_tensors=changed,all_nonraw_processing_preserved=True)
 output.with_suffix('.refresh.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('template',type=Path);p.add_argument('baseline',type=Path);p.add_argument('retrained',type=Path);p.add_argument('output',type=Path);a=p.parse_args();refresh(a.template,a.baseline,a.retrained,a.output)
