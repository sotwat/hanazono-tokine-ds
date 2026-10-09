"""Broaden the existing sustain wrapper while retaining consonant attacks."""
import argparse
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh

def update(source,output):
 m=onnx.load(source)
 prefixes=[i.name[:-len('su_long')] for i in m.graph.initializer if i.name.endswith('su_long')]
 if not prefixes:raise ValueError('No existing sustain wrapper')
 if any(i.name.endswith('su_medium_long') for i in m.graph.initializer):
  raise ValueError('Coverage patch is already applied; use the original bank')
 for prefix in prefixes:
  def name(s):return prefix+'su_'+s
  values={'long':40.,'endfade':4.,'short_start':32.,'short_fade':12.,'short_center':22.,'short_radius':10.,'old_long':172.,'medium_long':60.,'brief_start':20.,'brief_fade':8.,'brief_center':14.,'brief_radius':6.}
  for key,value in values.items():
   found=next((i for i in m.graph.initializer if i.name==name(key)),None)
   tensor=nh.from_array(np.array(value,np.float32),name(key))
   if found is None:m.graph.initializer.append(tensor)
   else:found.CopyFrom(tensor)
  # Sonorants and sustained fricatives can be prolonged; stops/closure and silence cannot.
  ids=np.array([2,3,9,10,12,14,18,19,20,24,25,28,31],np.int64)
  next(i for i in m.graph.initializer if i.name==name('vowels')).CopyFrom(nh.from_array(ids,name('vowels')))
  nodes=[]
  for node in m.graph.node:
   if name('elapsed') in node.output:
    nodes.append(h.make_node('GreaterOrEqual',[name('dur'),name('old_long')],[name('old_length')]))
    nodes.append(h.make_node('GreaterOrEqual',[name('dur'),name('medium_long')],[name('medium_length')]))
    for key in ['start','fade','center','radius']:
     nodes.append(h.make_node('Where',[name('medium_length'),name('short_'+key),name('brief_'+key)],[name('shorter_'+key)]))
     nodes.append(h.make_node('Where',[name('old_length'),name(key),name('shorter_'+key)],[name('effective_'+key)]))
   for i,v in enumerate(node.input):
    for key in ['start','fade','center','radius']:
     if v==name(key):node.input[i]=name('effective_'+key)
   if name('release') in node.output:node.input[1]=name('endfade')
   nodes.append(node)
  del m.graph.node[:];m.graph.node.extend(nodes)
 onnx.checker.check_model(m);output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,output)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.source,a.output)
