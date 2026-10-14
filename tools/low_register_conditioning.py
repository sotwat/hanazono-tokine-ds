"""Condition low acoustic F0 while retaining target vocoder pitch and repair weak repeated vowels."""
from pathlib import Path
import onnx,numpy as np,argparse
from onnx import helper as h,numpy_helper as nh
parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();m=onnx.load(args.source)
if any(i.name.startswith('lr_') for i in m.graph.initializer):raise ValueError('Low register conditioning already applied')
if args.output.exists():raise FileExistsError(args.output)
# Limit changed pitch conditioning to the original acoustic network. All output F0 remains original.
for n in m.graph.node:
 if any('/fs2/' in x for x in n.output):
  for i,x in enumerate(n.input):
   if x=='rp_normalized':n.input[i]='lr_conditioned'
for k,v in [('lr_floor',220.),('lr_guard',246.94165),('lr_zero',0.)]:m.graph.initializer.append(nh.from_array(np.array(v,np.float32),k))
nodes=[h.make_node('Max',['rp_normalized','lr_floor'],['lr_floorf0']),h.make_node('Greater',['rp_normalized','lr_zero'],['lr_voiced']),h.make_node('Where',['lr_voiced','lr_floorf0','lr_zero'],['lr_conditioned'])]
# Existing same-vowel donor guard misses two-semitone notes with small pitch curves.
pre='original_'
s=lambda x:pre+'su_'+x;g=lambda x:pre+'gv_'+x
for k,v in [('short_fraction',.15),('low_near_ratio',2**(4/12))]:m.graph.initializer.append(nh.from_array(np.array(v,np.float32),g(k)))
nodes.extend([h.make_node('Mul',[s('dur'),g('short_fraction')],[g('short_edge0')]),h.make_node('Clip',[g('short_edge0'),s('one'),g('two')],[g('short_edge')]),h.make_node('LessOrEqual',[g('midpitch'),'lr_floor'],[g('low_note')]),h.make_node('Unsqueeze',[g('low_note'),s('axis2')],[g('low_note3')]),h.make_node('Where',[g('low_note3'),g('short_edge'),g('six')],[g('attack_edge')]),h.make_node('Where',[g('low_note3'),g('short_edge'),g('release')],[g('release_edge')]),h.make_node('LessOrEqual',[g('pitchmax'),'lr_guard'],[g('low_pair')]),h.make_node('LessOrEqual',[g('pitchratio'),g('low_near_ratio')],[g('low_near')]),h.make_node('And',[g('low_pair'),g('low_near')],[g('low_donor')]),h.make_node('Or',[g('near_original'),g('low_donor')],[g('near')])])
for n in m.graph.node:
 if g('near') in n.output:n.output[0]=g('near_original')
 if g('weak_attack') in n.output:n.input[1]=g('attack_edge')
 if g('weak_release') in n.output:n.input[1]=g('release_edge')
# Topological ordering includes implicit If captures.
def dep(n):
 z=set(n.input)
 for a in n.attribute:
  if a.type==onnx.AttributeProto.GRAPH:
   gg=a.g;local={v.name for v in gg.input}|{v.name for v in gg.initializer}|{o for q in gg.node for o in q.output};z.update(set().union(*(dep(q) for q in gg.node))-local)
 return z
pending=list(m.graph.node)+nodes;out=[];known={v.name for v in m.graph.input}|{v.name for v in m.graph.initializer}|{''}
while pending:
 ready=[n for n in pending if dep(n)<=known]
 if not ready:raise RuntimeError('unresolved')
 for n in ready:out.append(n);known.update(n.output);pending.remove(n)
del m.graph.node[:];m.graph.node.extend(out);onnx.checker.check_model(m);args.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,args.output)
