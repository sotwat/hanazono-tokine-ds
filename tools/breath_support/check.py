"""Check AP isolation, rest bypass, timing and native singing regression fixtures."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort,soundfile as sf
def seed(graph):
    for node in graph.node:
        if node.op_type.startswith('Random'):
            attrs=[a for a in node.attribute if a.name!='seed'];del node.attribute[:];node.attribute.extend(attrs)
            node.attribute.append(onnx.helper.make_attribute('seed',float(int(hashlib.sha256(node.output[0].encode()).hexdigest()[:6],16))))
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.GRAPH:seed(attr.g)


def options():
    opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1;opt.log_severity_level=3
    return opt


def session(path):
    m=onnx.load(path);seed(m.graph);return ort.InferenceSession(m.SerializeToString(),options())


def check(bank,build,out):
    out.mkdir(exist_ok=True,parents=True);report={}
    baseac=bank/'tokine_ds_clarity_hybrid.tokine.onnx';basevoc=next((bank/'dsvocoder').glob('*.onnx'))
    for name,phones,ds,hz in [('ni-tsu',[1,19,14,27,28,1],[12,6,40,7,37,12],600.),('vowel-connect',[1,17,14,3,17,3,24,20,1],[15,7,40,50,8,30,8,35,15],330.)]:
        dur=np.array([ds],np.int64)
        feed=dict(tokens=np.array([phones],np.int64),durations=dur,f0=np.full((1,sum(ds)),hz,np.float32),depth=np.array(.1,np.float32),steps=np.array(20,np.int64))
        x=session(baseac).run(None,feed)[0];y=session(build/'acoustic.onnx').run(None,feed)[0]
        assert np.array_equal(x,y),(name,'mel regression',np.max(abs(x-y)))
        f0=x[2,:,3][None].copy();a=session(basevoc).run(None,dict(mel=x,f0=f0))[0];b=session(build/'vocoder.onnx').run(None,dict(mel=y,f0=f0))[0]
        assert np.array_equal(a,b),(name,'wave regression',np.max(abs(a-b)))
        report[name]=dict(mel_bit_identical=True,wave_bit_identical=True,samples=a.size)
        print(name,report[name],flush=True)
    # New APs of several lengths and note pitches; the inhale itself is unpitched.
    for frames in [8,16,26,44,70]:
        for hz in [130.8128,440.,880.]:
            d=np.array([[20,frames,20]],np.int64);f=np.full((1,int(d.sum())),hz,np.float32)
            feed=dict(tokens=np.array([[1,32,1]],np.int64),durations=d,f0=f,depth=np.array(.1,np.float32),steps=np.array(20,np.int64))
            mel=session(build/'acoustic.onnx').run(None,feed)[0]
            assert np.all(mel[2,20:20+frames,3]==0);assert np.all(mel[2,20:20+frames,11]==1)
            wav=session(build/'vocoder.onnx').run(None,dict(mel=mel,f0=f))[0].reshape(-1)
            assert len(wav)==int(d.sum())*512 and np.isfinite(wav).all() and np.max(abs(wav))<1
            part=wav[20*512:(20+frames)*512];rms=float(np.sqrt(np.mean(part**2)))
            assert rms>.001,(frames,hz,rms)
            if hz==440:sf.write(out/f'breath-{frames}.wav',wav,44100,subtype='PCM_24')
            report[f'AP-{frames}-{hz}']=dict(samples=len(wav),rms_dbfs=float(20*np.log10(rms)),peak=float(abs(wav).max()))
            print(frames,hz,report[f'AP-{frames}-{hz}'],flush=True)
    (out/'checks.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('build',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.bank,a.build,a.output)
