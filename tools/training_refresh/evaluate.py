import argparse,csv,json,sys,subprocess,shutil
from pathlib import Path
import numpy as np,soundfile as sf,parselmouth,torch
sys.path.insert(0,str(Path(__file__).parent))
from compare_singing_render import compare
B=Path('/Users/aaa/Documents/DiffSinger');R=Path(__file__).parent;E=R/'evaluation';EXP='tokine_ds_refresh_20261019'
def prep():
 E.mkdir(exist_ok=True)
 for name in ['heldout','low_register']:
  for ext in ['.ds','_original.wav']:shutil.copy2(B/'artifacts/tokine-ds-added3/evaluation'/f'{name}{ext}',E/f'{name}{ext}')
 rows=[r for r in csv.DictReader((R/'raw/tokine/transcriptions.csv').open()) if r['name'].startswith('validation_') and int(r['name'].split('_')[1])>=40]
 seen=set();segments=[];audio=[];offset=0
 for r in rows:
  song=r['name'].split('_')[1]
  if song in seen:continue
  seen.add(song);x,sr=sf.read(R/'raw/tokine/wavs'/f"{r['name']}.wav");p=parselmouth.Sound(x,sr).to_pitch_ac(time_step=.01,pitch_floor=65,pitch_ceiling=1100);hz=p.selected_array['frequency'];v=hz>0;assert v.any();f=np.interp(np.arange(int(np.ceil(len(x)/sr/.01)))*.01,p.xs()[v],hz[v]);segments.append(dict(offset=round(offset,6),ph_seq=r['ph_seq'],ph_dur=r['ph_dur'],f0_timestep=.01,f0_seq=' '.join(f'{h:.2f}' for h in f),lang='ja'));audio.extend([x,np.zeros(round(sr*.5))]);offset+=len(x)/sr+.5
 (E/'new_heldout.ds').write_text(json.dumps(segments,ensure_ascii=False));sf.write(E/'new_heldout_original.wav',np.concatenate(audio),44100,subtype='PCM_24')
def run(exp,step,tag):
 out=E/tag;out.mkdir(exist_ok=True)
 for name in ['heldout','low_register','new_heldout']:
  target=out/f'{name}.wav'
  if not target.exists():
   with (out/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'scripts/infer.py','acoustic',str(E/f'{name}.ds'),'--exp',exp,'--ckpt',str(step),'--spk','tokine','--seed','42','--steps','20','--depth','.1','--out',str(out),'--title',name],cwd=B,stdout=f,stderr=subprocess.STDOUT,check=True)
  x,sr=sf.read(target);assert np.isfinite(x).all() and np.std(x)>.001
 metrics={name:compare(E/f'{name}_original.wav',out/f'{name}.wav') for name in ['heldout','low_register','new_heldout']};(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(tag,{k:v['gain_aligned_log_mel_mae'] for k,v in metrics.items()},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--exp',default=EXP);p.add_argument('--step',type=int);p.add_argument('--tag');a=p.parse_args()
 if a.prepare:prep()
 else:run(a.exp,a.step,a.tag or f'step{a.step}')
