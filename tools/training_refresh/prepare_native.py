from pathlib import Path
import csv,json,copy,yaml
R=Path(__file__).parent;P=R/'preparation';base=yaml.safe_load(Path('/Users/aaa/Documents/DiffSinger/artifacts/tokine-phrase-balance-20261015/comparison.ustx').read_text());template=copy.deepcopy(base['voice_parts'][0]['notes'][0]);base.update(name='花園トキネDS 再学習比較',bpm=120,tempos=[dict(position=0,bpm=120)],voice_parts=[],wave_parts=[]);base['tracks'][0]['singer']='花園トキネDS v1.2.4-rc.1';base['tracks'][0]['track_name']='再学習比較'
manifest=json.loads((R/'dataset_manifest.json').read_text());cands={r['candidate_id']:r for r in csv.DictReader((P/'フレーズ候補.csv').open())};position=960;seen=set();index=[]
for item in manifest:
 if not item['validation'] or item['origin']!='20261019_corrected_or_added' or item['song'] in seen:continue
 seen.add(item['song']);r=cands[item['source_candidate_id']];notes=[];start=float(r['start_sec']);end=0
 for e in json.loads(r['note_events_json']):
  n=copy.deepcopy(template);n.update(position=round((e['start']-start)*960),duration=max(1,round((e['end']-e['start'])*960)),tone=e['midi'],lyric=('い [i cl]' if e['lyric']=='いっ' else e['lyric']),phoneme_expressions=[],phoneme_overrides=[]);n['pitch']={'data':[{'x':-40,'y':0,'shape':'io'},{'x':40,'y':0,'shape':'io'}],'snap_first':True};n['vibrato']['length']=0;notes.append(n);end=max(end,n['position']+n['duration'])
 for n,nextn in zip(notes,notes[1:]):n['duration']=min(n['duration'],nextn['position']-n['position'])
 base['voice_parts'].append(dict(duration=end+240,name=item['song'],comment='',track_no=0,position=position,notes=notes,curves=[],masked_curves=[]));index.append(dict(song=item['song'],start_sec=position/960,end_sec=(position+end)/960));position+=end+960
(R/'native-comparison.ustx').write_text(yaml.safe_dump(base,allow_unicode=True,sort_keys=False));(R/'native-sections.json').write_text(json.dumps(index,ensure_ascii=False,indent=2));print('phrases',len(index),'seconds',position/960)
