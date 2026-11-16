"""Check gain bounds and unchanged waveform shape, including loud-input bypass."""
import tempfile
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H, TensorProto as T
import onnxruntime as ort
from output_gain import build

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    graph=H.make_graph([H.make_node('Identity',['input'],['waveform'])],'gain_test',
                      [H.make_tensor_value_info('input',T.FLOAT,[1,'samples'])],
                      [H.make_tensor_value_info('waveform',T.FLOAT,[1,'samples'])])
    source=H.make_model(graph,opset_imports=[H.make_opsetid('',18)])
    source.ir_version=10
    onnx.save(source,root/'base.onnx')
    build(root/'base.onnx',root/'gain.onnx')
    session=ort.InferenceSession(str(root/'gain.onnx'),providers=['CPUExecutionProvider'])
    for peak in [0,.1,.5,.6,.75,.9,1.2]:
        wave=np.array([[0,peak,-peak,peak*.37]],dtype=np.float32)
        result=session.run(None,{'input':wave})[0]
        gain=max(1.,min(10**(3/20),.75/max(peak,1e-8)))
        np.testing.assert_allclose(result,wave*gain,rtol=1e-6,atol=1e-7)
        assert np.max(abs(result))<=max(peak,.75)+1e-6
    print('7 silence/normal/headroom/loud-input cases passed')
