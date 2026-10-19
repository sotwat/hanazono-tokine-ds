from pathlib import Path
import sys,subprocess,numpy as np,pyworld as pw,json
sys.path.insert(0,str(Path(__file__).parent));import prepare
p=prepare.mod.PREP/'features'
for song in prepare.mod.SONGS:
 wav=next(x for x in (prepare.mod.SRC/'DS_wav').glob('*.wav') if prepare.mod.nfc(x.stem)==song)
 out=p/f'{song}_world.npz'
 if out.exists():continue
 x=np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(wav),'-ac','1','-ar','16000','-f','f64le','-']),'<f8').copy()
 f,t=pw.dio(x,16000,f0_floor=65,f0_ceil=1500,frame_period=20);f=pw.stonemask(x,f,t,16000)
 midi=np.where(f>0,69+12*np.log2(np.maximum(f,1e-6)/440),0)
 np.savez(out,t=t,pitch=midi,f0=f);print(song,flush=True)
