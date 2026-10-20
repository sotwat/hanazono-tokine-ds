"""Check stable interiors, ongoing pitch movement, and duration-map boundaries."""
import argparse,json
from pathlib import Path
from onnx import helper as h, numpy_helper as nh
import numpy as np
import onnx
import onnxruntime as ort


def check(source,folder,fixture):
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.log_severity_level=3
    session=ort.InferenceSession(str(folder/'maps.onnx'),opt);count=0
    for length in (1,23,39,40,87,250):
        for token in (3,9,14,20,28,16,1,2):
            for pattern in ('steady','attack','release','glide','vibrato','unvoiced'):
                d=np.array([[8,length,7]],np.int64);t=np.array([[1,token,16]],np.int64)
                f=np.full((1,int(d.sum())),293.66,np.float32);v=f[0,8:8+length];edge=max(length//5,6)
                if pattern=='attack':v[:min(edge,length)]=np.linspace(261.63,293.66,min(edge,length))
                if pattern=='release':v[-min(edge,length):]=np.linspace(293.66,329.63,min(edge,length))
                if pattern=='glide':v[:]=np.linspace(261.63,329.63,length)
                if pattern=='vibrato':v[:]=293.66*2**(np.sin(np.arange(length)*.8)*100/1200)
                if pattern=='unvoiced':v[:]=0
                dd,ff,back,eligible=session.run(None,dict(tokens=t,durations=d,f0=f))
                expected=token in (3,9,14,20,28) and length>=40 and pattern in ('steady','attack','release')
                assert dd[0,1]==(24 if expected else length)
                assert ff.shape==(1,int(dd.sum())) and back.shape==f.shape
                assert np.isfinite(ff).all() and np.isfinite(back).all()
                assert back[0,8]==8 and abs(back[0,8+length-1]-(8+dd[0,1]-1))<1e-4
                assert np.all(eligible[0,8:8+length]==expected)
                assert not eligible[0,:8].any() and not eligible[0,8+length:].any();count+=1
    z=np.load(fixture);out=session.run(None,dict(tokens=z['dbg_tokens'],durations=z['dbg_durations'],f0=z['dbg_f0']))
    starts=np.r_[0,np.cumsum(z['dbg_durations'][0])];assert out[0][0,15]==24 and np.all(out[3][0,starts[15]:starts[16]]==1)
    a=onnx.load(source);b=onnx.load(folder/'acoustic.onnx');weights={v.name:v.SerializeToString() for v in b.graph.initializer}
    assert all(weights[v.name]==v.SerializeToString() for v in a.graph.initializer)
    # Isolate the changed wet-routing graph with identical upstream tensors.
    wet_cases=0
    for prefix in ('original_', 'donor_'):
        outputs={prefix+'ivg_active',prefix+'su_wet'}
        nodes=[n for n in b.graph.node if outputs.intersection(n.output)]
        assert len(nodes)==2
        consumers=[n for n in b.graph.node if prefix+'ivg_old_wet' in n.input]
        assert len(consumers)==1 and consumers[0].op_type=='Where'
        graph=h.make_graph(nodes,'wet routing',[
            h.make_tensor_value_info('tc_eligible',1,[1,'F',1]),
            h.make_tensor_value_info(prefix+'ivg_old_wet',1,[1,'F',128])],
            [h.make_tensor_value_info(prefix+'su_wet',1,[1,'F',128])],
            [nh.from_array(np.array(0,np.float32),prefix+'ss_zero')])
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',18)],ir_version=10)
        wet_session=ort.InferenceSession(model.SerializeToString(),opt)
        raw=np.random.default_rng(20).normal(size=(1,90,128)).astype(np.float32)
        for enabled in (0,1):
            y=wet_session.run(None,{'tc_eligible':np.full((1,90,1),enabled,np.float32),prefix+'ivg_old_wet':raw})[0]
            assert np.array_equal(y,np.zeros_like(raw) if enabled else raw);wet_cases+=1
    report=dict(wet_routing_cases=wet_cases,eligible_fixed_spectrum_residue_zero=True,noneligible_wet_bit_exact=True,cases=count,actual_target_frames_before=87,actual_target_compressed_frames=24,
                actual_target_now_eligible=True,learned_tensors_byte_identical=True)
    (folder/'interior-checks.json').write_text(json.dumps(report,indent=2));print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('folder',type=Path);p.add_argument('fixture',type=Path)
    a=p.parse_args();check(a.source,a.folder,a.fixture)
