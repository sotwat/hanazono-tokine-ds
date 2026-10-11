"""Check cycle coherence, noise attenuation and exact bypass outside attacks."""
import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort,soundfile as sf


def control(tokens,durations):
    out=np.zeros((1,int(sum(durations)),128),np.float32);start=0;prev=0
    for index,(t,d) in enumerate(zip(tokens,durations)):
        out[0,start:start+d,0]=float(prev==1 and t in [3,9,14,20,28])-float(t==1 and index+1<len(tokens) and tokens[index+1] in [3,9,14,20,28]);out[0,start:start+d,1]=start;out[0,start:start+d,2]=start+d;start+=d;prev=t
    return out


def check(directory,fixture,raw_vocoder):
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.log_severity_level=3
    s=ort.InferenceSession(str(directory/'filter.onnx'),opts);rows=[]
    for midi in [48,60,72,77,87]:
        n=40;f=np.full((1,n),440*2**((midi-69)/12),np.float32);phase=np.cumsum(np.repeat(f/44100,512,axis=1),axis=1);pure=np.sin(2*np.pi*phase).astype(np.float32)*.2;c=control([1,14,1],[8,24,8])
        y=s.run(None,dict(wave=pure,f0=f,control=c))[0];mask=np.zeros(pure.shape[1],bool);mask[8*512:8*512+3308]=True
        mask[:8*512]=True
        assert np.array_equal(pure[:,~mask],y[:,~mask]);err=float(np.max(abs(y[:,8*512+400:8*512+1800]-pure[:,8*512+400:8*512+1800])));assert err<.001
        noise=np.random.default_rng(93).normal(0,.04,pure.shape).astype(np.float32);noisy=pure+noise;clean=s.run(None,dict(wave=noisy,f0=f,control=c))[0];ix=slice(8*512+400,8*512+1800);gain=float(20*np.log10(np.sqrt(np.mean((clean[:,ix]-pure[:,ix])**2))/np.sqrt(np.mean(noise[:,ix]**2))));assert gain < -3
        rows.append(dict(midi=midi,noise_db=gain,pure_max_error=err))
    boundary_cases=0
    for previous in [1,3,8,14,22]:
        for current in [3,9,14,20,28,8,22]:
            for duration in [1,2,8,40,180]:
                durations=[8,8,duration,8];c=control([1,previous,current,1],durations);frames=sum(durations)
                f=(440*2**(.4*np.sin(np.arange(frames)/2))).astype(np.float32)[None]
                wave=np.random.default_rng(32).normal(0,.1,(1,frames*512)).astype(np.float32)
                y=s.run(None,dict(wave=wave,f0=f,control=c))[0];mask=np.repeat(c[0,:,0]!=0,512)
                assert np.array_equal(y[:,~mask],wave[:,~mask]);assert np.isfinite(y).all();boundary_cases+=1
    z=np.load(fixture);c=control(z['dbg_tokens'][0],z['dbg_durations'][0]);before=ort.InferenceSession(str(raw_vocoder),opts).run(None,dict(mel=z['mel'],f0=z['dbg_f0']))[0];after=s.run(None,dict(wave=before,f0=z['dbg_f0'],control=c))[0];assert np.isfinite(after).all();assert abs(after).max()<1
    full=ort.InferenceSession(str(directory/'vocoder.onnx'),opts).run(None,dict(mel=np.concatenate([z['mel'],c]),f0=z['dbg_f0']))[0]
    # Neural baseline is deterministic; WORLD noise outside the onset is independent.
    assert np.allclose(full[:,:16*512],after[:,:16*512],atol=1e-6)
    sf.write(directory/'Leia_い_周期ノイズ対策_前→後.wav',np.r_[before.ravel(),np.zeros(22050),after.ravel()],44100,subtype='PCM_24');np.savez(directory/'comparison.npz',before=before,after=after,control=c,f0=z['dbg_f0']);result=dict(boundary_cases=boundary_cases,synthetic=rows,peak=float(abs(after).max()),finite=True,integrated_onset_matches=True);(directory/'checks.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('fixture',type=Path);p.add_argument('raw_vocoder',type=Path);a=p.parse_args();check(a.directory,a.fixture,a.raw_vocoder)
