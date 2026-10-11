"""Compare attack processing with identical OpenUtau raw tensors and pitch."""
import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort,soundfile as sf
from check_vowel_connections import wrapper


def render(before,after,vocoder,fixture,output):
    z=np.load(fixture);opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.log_severity_level=3
    v=ort.InferenceSession(str(vocoder),opts);waves=[];mels=[];rows=[];output.mkdir(parents=True,exist_ok=True)
    for tag,model in [('before',before),('after',after)]:
        arr=[]
        for pr in ['original_','donor_']:
            raw=z['dbg_'+pr+'su_raw'];f=z['dbg_f0'] if pr=='original_' else np.clip(z['dbg_f0'],220,523.2511306)
            mel=wrapper(model,pr).run(None,dict(tokens=z['dbg_tokens'],durations=z['dbg_durations'],f0=f,**{pr+'su_raw':raw}))[0];arr.append(mel)
        y=v.run(None,dict(mel=np.concatenate(arr),f0=z['dbg_f0']))[0].ravel();assert np.isfinite(y).all();assert abs(y).max()<1
        sf.write(output/f'Leia_い開始_{tag}.wav',y,44100,subtype='PCM_24');waves.append(y);mels.append(arr[0]);np.savez(output/f'mel_{tag}.npz',original=arr[0],donor=arr[1]);rows.append(dict(tag=tag,peak=float(abs(y).max()),finite=True))
    a,b=mels;changed=np.where(np.max(abs(a-b),axis=(0,2))>0)[0]
    report=dict(changed_frames=changed.tolist(),waves=rows,first_pair_flux_before=float(np.mean(abs(a[:,8]-a[:,9]))),first_pair_flux_after=float(np.mean(abs(b[:,8]-b[:,9]))),max_energy_relative_error=float(np.max(abs(np.mean(np.exp(a),axis=2)/np.mean(np.exp(b),axis=2)-1))),onset_wave_changed_samples=int(np.count_nonzero(waves[0][8*512:11*512]!=waves[1][8*512:11*512])))
    sf.write(output/'Leia_い開始_修正前→修正後.wav',np.r_[waves[0],np.zeros(22050),waves[1]],44100,subtype='PCM_24');(output/'verification.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['before','after','vocoder','fixture','output']:p.add_argument(name,type=Path)
    a=p.parse_args();render(a.before,a.after,a.vocoder,a.fixture,a.output)
