"""Check onset-only amplitude changes and exact bypass of all other samples."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h
from ts_affricate import dependencies


def check(directory):
    model=onnx.load(directory/'acoustic.onnx');needed={'aa_control'};selected=[]
    for n in reversed(model.graph.node):
        if needed.intersection(n.output):selected.append(n);needed.update(dependencies(n))
    graph=h.make_graph(list(reversed(selected)),'attack control',list(model.graph.input),[h.make_tensor_value_info('aa_control',1,[1,'F',128])],[i for i in model.graph.initializer if i.name in needed])
    fragment=h.make_model(graph,opset_imports=model.opset_import,ir_version=model.ir_version)
    opt=ort.SessionOptions();opt.log_severity_level=3;opt.intra_op_num_threads=2
    control=ort.InferenceSession(fragment.SerializeToString(),opt)
    count=0
    for previous in range(32):
        for current in range(32):
            d=np.array([[8,40,30,8]],np.int64);f=int(d.sum())
            c=control.run(None,dict(tokens=np.array([[1,previous,current,1]],np.int64),durations=d,f0=np.full((1,f),293.66,np.float32),depth=np.array(.1,np.float32),steps=np.array(20,np.int64)))[0][0]
            expected=np.zeros(f,np.float32)
            if previous in (3,9,14,20,28) and current==3:expected[48:78]=1
            # The first test phone can itself be an /a/ after silence: it must bypass.
            np.testing.assert_array_equal(c[:,7],expected)
            assert np.count_nonzero(c[:,8:])==0
            count+=1
    session=ort.InferenceSession(str(directory/'attack-layer.onnx'),opt)
    waveform_cases=0
    for frames in (8,12,24,80):
        for hz in (110.,440.,880.):
            for flag in (0.,1.):
                for amplitude in (.04,.4,.98):
                    f=frames+16;n=f*512;t=np.arange(n)/44100
                    x=(amplitude*np.sin(2*np.pi*hz*t)).astype(np.float32)[None]
                    c=np.zeros((f,128),np.float32);c[8:8+frames,1]=8;c[8:8+frames,2]=8+frames;c[8:8+frames,7]=flag
                    y=session.run(None,dict(wave=x,control=c))[0]
                    local=t-8*512/44100;duration=frames*512/44100
                    allowed=(local>0)&(local<min(.24,duration*.45))&(flag>0)&(duration>=.12)&(amplitude<.97)
                    assert np.array_equal(y[:,~allowed],x[:,~allowed])
                    assert np.isfinite(y).all() and np.max(abs(y))<=max(.970001,amplitude)
                    nonzero=abs(x)>1e-6;ratio=y[nonzero]/x[nonzero]
                    assert ratio.min()>=.999999 and ratio.max()<=10**(.2)+1e-5
                    waveform_cases+=1
    result=dict(control_cases=count,waveform_cases=waveform_cases,non_a_and_rest_bit_exact=True,unchanged_phase_and_duration=True,max_gain_db=4)
    (directory/'checks.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);check(p.parse_args().directory)
