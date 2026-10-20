"""Exercise real duration predictions across ts contexts and note lengths."""
import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def check(bank, candidate, output):
    opt = ort.SessionOptions(); opt.intra_op_num_threads = 2; opt.log_severity_level = 3
    root = bank/'dsdur'
    lm = ort.InferenceSession(str(root/'tokine_song_variance_v1.linguistic.onnx'), opt)
    old = ort.InferenceSession(str(root/'tokine_song_variance_v1.dur.onnx'), opt)
    new = ort.InferenceSession(str(candidate), opt)
    count = 0
    for phone in (27, 24, 16, 26):
        for vowel in (2,3,9,14,20,28):
            for length in (1,3,8,20,39,40,43,86,258):
                for midi in (42,63,79):
                    for initial in (False,True):
                        t = np.array([[1,phone,vowel,phone,vowel]],np.int64)
                        div = np.array([[2,2,1]] if initial else [[1,3,1]],np.int64)
                        d = np.array([[43,length,length]],np.int64)
                        e,m = lm.run(None,dict(tokens=t,word_div=div,word_dur=d))
                        feed=dict(encoder_out=e,x_masks=m,ph_midi=np.full(t.shape,midi,np.int64))
                        a=old.run(None,feed)[0]; b=new.run(None,feed)[0]
                        assert np.isfinite(b).all() and (b>=0).all()
                        if phone != 27: assert np.array_equal(a,b)
                        if initial:
                            assert b[0,1] <= 7.00001 or phone!=27
                            assert a[0,0] == b[0,0] and a[0,4]==b[0,4]
                            assert abs(a[0,2:4].sum()-b[0,2:4].sum())<1e-4
                            assert b[0,3]/max(b[0,2:4].sum(),1e-8)*length <= 7.00001 or phone!=27
                        else:
                            assert abs(a[0,1:4].sum()-b[0,1:4].sum())<1e-4
                            if phone==27:
                                for i in (1,3): assert b[0,i]/max(b[0,1:4].sum(),1e-8)*length <= 7.00001
                        count += 1
    output.write_text(json.dumps(dict(cases=count,finite=True,non_ts_bit_exact=True,
        normalized_ts_max_frames=7,noninitial_group_sums_preserved=True),indent=2))
    print(count, 'passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check(a.bank,a.candidate,a.output)
