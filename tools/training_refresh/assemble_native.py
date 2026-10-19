"""Package actual OpenUtau phrase-cache output as a matched listening pair.

No resynthesis, normalization, or pitch/timing edits are performed here.
The WAV payload of each rendered phrase is preserved bit for bit.
"""
import array, datetime, hashlib, json, math, wave
from pathlib import Path
R=Path(__file__).parent
cache=Path.home()/'Library/Caches/OpenUtau'
def read(p):
    with wave.open(str(p),'rb') as w:
        assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,44100)
        return w.readframes(w.getnframes())
new=json.loads((R/'native-verification.json').read_text())['cache_outputs']
newpaths=sorted([cache/x['file'] for x in new],key=lambda p:p.stat().st_mtime)
old=[]
for p in cache.glob('ds-*.wav'):
    t=datetime.datetime.fromtimestamp(p.stat().st_mtime)
    if t.date()==datetime.date(2026,10,19) and (21,25)<=(t.hour,t.minute)<=(21,30):old.append(p)
# The initial unsupported lyric fixture was rerendered after its phonetic hint fix.
old=[p for p in old if p.name!='ds-65337a11bd70fbb6-depth0.10-steps20.wav']
assert len(old)==len(newpaths)==16
old_by_length={len(read(p)):p for p in old};assert len(old_by_length)==16
pieces={'従来版':[],'更新版':[]};rows=[];cursor=0
for i,np in enumerate(newpaths):
    nb=read(np);op=old_by_length[len(nb)];ob=read(op)
    assert ob!=nb
    row={'index':i+1,'start_sec':cursor,'seconds':len(nb)/2/44100}
    for tag,p,b in [('従来版',op,ob),('更新版',np,nb)]:
        a=array.array('h',b);peak=max(abs(v) for v in a)/32768
        assert peak<1 and peak>1e-4
        row[tag]={'cache':p.name,'sha256':hashlib.sha256(b).hexdigest(),'peak':peak,'rms':math.sqrt(sum(v*v for v in a)/len(a))/32768}
        pieces[tag].extend([b,b'\0\0'*22050])
    cursor+=row['seconds']+.5;rows.append(row)
for tag,ps in pieces.items():
    out=R/f'OpenUtau比較_{tag}.wav'
    with wave.open(str(out),'wb') as w:
        w.setparams((1,2,44100,0,'NONE','not compressed'));w.writeframes(b''.join(ps))
    # Prove each segment in the deliverable equals its native cache payload.
    b=read(out);offset=0
    for payload in ps:assert b[offset:offset+len(payload)]==payload;offset+=len(payload)
report={'method':'actual OpenUtau AU native phrase WAV payloads concatenated, 0.5s silence between phrases; not a full-score menu export','previous_fixture':'native-comparison.ustx (corrected phonetic hint)','updated_fixture':'native-updated.ustx','phrases':16,'seconds':cursor,'rows':rows}
(R/'native-pair-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'phrases':16,'seconds':cursor,'previous_peak':max(x['従来版']['peak'] for x in rows),'updated_peak':max(x['更新版']['peak'] for x in rows)},ensure_ascii=False))
