"""Check anticipation ceilings and anchor-preserving duration redistribution."""
import argparse,json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from consonant_timing import CAPS


def check(bank,candidate,output):
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.log_severity_level=3
    root=bank/'dsdur'
    lm=ort.InferenceSession(str(root/'tokine_song_variance_v1.linguistic.onnx'),opt)
    old=ort.InferenceSession(str(root/'tokine_song_variance_v1.dur.onnx'),opt)
    new=ort.InferenceSession(str(candidate),opt)
    count=0;max_errors=[]
    for phone in list(CAPS)+[7]:
        for vowel in (2,3,9,14,20,28):
            for length in (0,1,3,8,20,39,40,43,86,258,860):
                for midi in (42,63,79):
                    for initial in (False,True):
                        t=np.array([[1,phone,vowel,phone,vowel]],np.int64)
                        div=np.array([[2,2,1]] if initial else [[1,3,1]],np.int64)
                        e,m=lm.run(None,dict(tokens=t,word_div=div,word_dur=np.array([[43,length,length]],np.int64)))
                        feed=dict(encoder_out=e,x_masks=m,ph_midi=np.full(t.shape,midi,np.int64))
                        a=old.run(None,feed)[0];b=new.run(None,feed)[0]
                        assert np.isfinite(b).all() and (b>=0).all()
                        assert a[0,0]==b[0,0] and a[0,4]==b[0,4]
                        if phone not in CAPS or (length==0 and not initial): assert np.array_equal(a,b)
                        elif initial:
                            assert b[0,1]<=CAPS[phone]+1e-5
                            assert abs(a[0,2:4].sum()-b[0,2:4].sum())<1e-3
                            assert b[0,3]/max(b[0,2:4].sum(),1e-8)*length<=CAPS[phone]+1e-4
                        else:
                            assert abs(a[0,1:4].sum()-b[0,1:4].sum())<1e-3
                            for i in (1,3):assert b[0,i]/max(b[0,1:4].sum(),1e-8)*length<=CAPS[phone]+1e-4
                            assert (b[0,1]+b[0,3])/max(b[0,1:4].sum(),1e-8)*length<=9.0001
                        count+=1
    clusters=0
    for cluster in ([16,30],[24,16,30],[12,30],[27,30],[]):
        for length in (0,1,8,43,258):
            for initial in (False,True):
                t=np.array([[1,3]+list(cluster)+[3]],np.int64)
                div=np.array([[2+len(cluster),1]] if initial else [[1,1+len(cluster),1]],np.int64)
                wd=np.array([[43,length]] if initial else [[43,length,length]],np.int64)
                e,m=lm.run(None,dict(tokens=t,word_div=div,word_dur=wd))
                feed=dict(encoder_out=e,x_masks=m,ph_midi=np.full(t.shape,63,np.int64))
                a=old.run(None,feed)[0];b=new.run(None,feed)[0]
                assert np.isfinite(b).all()
                if initial: assert b[0,2:-1].sum()<=9.0001
                else:
                    assert abs(a[0,1:-1].sum()-b[0,1:-1].sum())<1e-3
                    assert b[0,2:-1].sum()/max(b[0,1:-1].sum(),1e-8)*length<=9.0001
                clusters+=1
    output.write_text(json.dumps(dict(cases=count,cluster_cases=clusters,caps=CAPS,
        group_sums_preserved=True,initial_SP_unchanged=True,closure_unchanged=True),indent=2))
    print(count,'duration cases and',clusters,'cluster cases passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.bank,a.candidate,a.output)
