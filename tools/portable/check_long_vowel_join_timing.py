"""Check identical-input onset timing, scope and consonant/rest bypass."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h
from ts_affricate import dependencies


def fragment(path,prefix):
    model=onnx.load(path)
    outputs=[prefix+'mel',prefix+'gv_w',prefix+'gv_anchor']
    boundary={'tokens','durations','f0','source_f0',prefix+'su_raw'}
    needed=set(outputs);selected=[]
    for n in reversed(model.graph.node):
        if (set(n.output)&needed)-boundary:
            selected.append(n);needed.update(dependencies(n))
    nodes=list(reversed(selected))
    for n in nodes:
        for i,x in enumerate(n.input):
            if x=='source_f0':n.input[i]='f0'
    inputs=[h.make_tensor_value_info(k,7,[1,'N']) for k in ('tokens','durations')]
    inputs += [h.make_tensor_value_info('f0',1,[1,'F']),h.make_tensor_value_info(prefix+'su_raw',1,[1,'F',128])]
    graph=h.make_graph(nodes,'join timing',inputs,[h.make_tensor_value_info(k,1,[None,None,None]) for k in outputs],list(model.graph.initializer))
    m=h.make_model(graph,opset_imports=model.opset_import,ir_version=model.ir_version)
    opt=ort.SessionOptions();opt.log_severity_level=3;opt.intra_op_num_threads=2
    return ort.InferenceSession(m.SerializeToString(),opt)


def check(before,after,report):
    vowels=(3,9,14,20,28);cases=0;unchanged=0
    for prefix in ('original_','donor_'):
        old,new=fragment(before,prefix),fragment(after,prefix)
        conditions=[(a,b,da,db) for a in vowels for b in vowels for da in (4,39,40,87,250) for db in (1,5,6,8,90)]
        conditions += [(a,b,87,12) for a in vowels for b in range(32) if b not in vowels]
        conditions += [(a,b,12,87) for a in range(32) if a not in vowels for b in vowels]
        for a,b,da,db in conditions:
            ds=np.array([[8,da,db,8]],np.int64);frames=int(ds.sum());boundary=8+da
            raw=np.random.default_rng(20).normal(-4,.5,(1,frames,128)).astype(np.float32)
            feed={'tokens':np.array([[1,a,b,1]],np.int64),'durations':ds,'f0':np.full((1,frames),293.66,np.float32),prefix+'su_raw':raw}
            x,w0,_=old.run(None,feed);y,w1,anchor=new.run(None,feed)
            assert np.isfinite(y).all() and x.shape==y.shape
            eligible=a in vowels and b in vowels and da>=40 and db>=6
            if eligible:
                allowed=np.zeros(frames,bool);allowed[max(0,boundary-6):boundary+2]=True
                assert np.array_equal(x[:,~allowed],y[:,~allowed]),(prefix,a,b,da,db)
                # All incoming vowel shapes reach their full anchor at boundary.
                assert w1[0,2,boundary]==1 and w0[0,2,boundary]<1
                assert np.allclose(y[0,boundary],anchor[0,2],atol=1e-6)
            else:
                assert np.array_equal(x,y),(prefix,a,b,da,db);unchanged+=1
            cases+=1
    a=onnx.load(before);b=onnx.load(after);weights={v.name:v.SerializeToString() for v in b.graph.initializer}
    assert all(weights[v.name]==v.SerializeToString() for v in a.graph.initializer)
    result={'cases':cases,'unchanged_controls':unchanged,'incoming_vowel_complete_at_boundary':True,'original_consonant_and_rest_paths_bit_exact':True,'all_existing_initializers_byte_identical':True,'completion_advance_frames':2,'completion_advance_ms':2*512/44100*1000,'perceptual_quality':'requires listening'}
    report.write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();check(a.before,a.after,a.report)
