"""Check duration budgets and acoustic continuity over the supported phone set."""
import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort
from check_vowel_connections import wrapper


def check(base,candidate,output):
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.log_severity_level=3
    def sessions(bank):
        return [ort.InferenceSession(str(bank/'dsdur'/('tokine_song_variance_v1.'+name+'.onnx')),opts) for name in ['linguistic','dur']]
    oldl,oldd=sessions(base);newl,newd=sessions(candidate)
    timing=[];consonants=[x for x in range(4,32) if x not in [9,14,20,28]]
    for vowel in [2,3,9,14,20,28]:
        for consonant in consonants:
            for length in [0,1,2,3,12,30,39,40,43,59,86,172,258,860]:
                t=np.array([[1,8,vowel,consonant,vowel]],np.int64)
                feed=dict(tokens=t,word_div=np.array([[2,2,1]],np.int64),word_dur=np.array([[43,length,length]],np.int64))
                e,m=newl.run(None,feed);y=newd.run(None,dict(encoder_out=e,x_masks=m,ph_midi=np.full(t.shape,62,np.int64)))[0]
                assert np.isfinite(y).all() and (y>=0).all()
                if length<40:
                    e,m=oldl.run(None,feed);old=oldd.run(None,dict(encoder_out=e,x_masks=m,ph_midi=np.full(t.shape,62,np.int64)))[0]
                    assert np.allclose(y,old,rtol=0,atol=1e-5)
                else:
                    assert abs(float(y[0,2:4].sum())-length)<1e-3
                    assert y[0,3]<=16.001 and y[0,2]>=length*.599
                    assert abs(float(y[0,4])-length)<1e-3
                timing.append(dict(vowel=vowel,consonant=consonant,frames=length,lead=float(y[0,3])))
    acoustic=[]
    for prefix in ['original_','donor_']:
        session=wrapper(candidate/'tokine_ds_clarity_hybrid.tokine.onnx',prefix)
        for vowel in [3,9,14,20,28]:
            for consonant in consonants:
                for length in [40,60,172,258]:
                    d=np.array([[12,length,8,86,16]],np.int64);f=int(d.sum());raw=np.full((1,f,128),-4,np.float32)
                    raw[:,12:12+length]=np.linspace(-3,-8,length)[None,:,None]
                    raw[:,:12]=-10;raw[:,-16:]=-10
                    feed=dict(tokens=np.array([[1,vowel,consonant,vowel,1]],np.int64),durations=d,f0=np.full((1,f),440,np.float32));feed[prefix+'su_raw']=raw
                    y=session.run(None,feed)[0];assert np.isfinite(y).all()
                    tail=y[:,12+length-6:12+length-3]
                    assert np.exp(tail).mean()>.001
                    assert np.array_equal(y[:,12+length+3:12+length+8],raw[:,12+length+3:12+length+8])
                    acoustic.append(dict(branch=prefix,vowel=vowel,consonant=consonant,frames=length,tail=float(np.exp(tail).mean())))
    edges=0
    for prefix in ['original_','donor_']:
        session=wrapper(candidate/'tokine_ds_clarity_hybrid.tokine.onnx',prefix)
        for consonant in [4,8,22,24]:
            for size in [0,1]:
                d=np.array([[12,90,size,86,16]],np.int64);f=int(d.sum())
                feed=dict(tokens=np.array([[1,3,consonant,3,1]],np.int64),durations=d,f0=np.full((1,f),440,np.float32))
                feed[prefix+'su_raw']=np.random.default_rng(19).normal(-4,1,(1,f,128)).astype(np.float32)
                y=session.run(None,feed)[0];assert np.isfinite(y).all() and y.shape==(1,f,128);edges+=1
    output.write_text(json.dumps(dict(duration_cases=len(timing),acoustic_cases=len(acoustic),edge_cases=edges,duration=timing,acoustic=acoustic),indent=2))
    print(len(timing),'duration cases;',len(acoustic),'acoustic cases;',edges,'edge cases passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.base,a.candidate,a.output)
