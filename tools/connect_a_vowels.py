"""Create a separate USTX trial joining contiguous a vowels as melisma."""
from pathlib import Path
import argparse,yaml
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
if a.source.resolve()==a.output.resolve():raise ValueError('Use a separate output file')
d=yaml.safe_load(a.source.read_text());count=0
for part in d.get('voice_parts',[]):
    previous=None
    for note in part['notes']:
        if previous and previous['lyric'] in ('あ','+') and note['lyric']=='あ' and previous['position']+previous['duration']==note['position']:
            note['lyric']='+';count+=1
        previous=note
if a.output.exists():raise FileExistsError(a.output)
a.output.write_text(yaml.safe_dump(d,allow_unicode=True,sort_keys=False));print(count)
