import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import prepare_tokine_added7 as mod
import prepare_tokine_added_songs as older
root=Path(__file__).parent
mod.OUT=root;mod.PREP=root/'preparation'
mod.SONGS={
'サムライハート':['サムライハート/サムライハート_main1.ust'],
'I beg you':['I beg you ust/I beg you main ベタ打ち.ust'],
'メルト':['メルト002.ust'],
'花になって':['花になって/花になって.vsqx'],
'Lemon':['Lemon UST by Bread/main.ust','Lemon UST by Bread/main2.ust'],
'ray_原曲キー':['Bump of Chicken Ray UST/Ray main.ust'],
'ray_キー+5':['Bump of Chicken Ray UST/Ray main.ust'],
'ないない':['nainai/main.ust'],
'ミカヅキ':['ミカヅキ UST/*/*.ust'],
'怪物':['怪物 ust/怪物.ust'],
'季節は次々死んでいく':['季節は次々死んでいくUST/*メイン.ust'],
'明け星':['明け星ustセット/明け星メイン-ほぼベタ打ち.ust']}
original_parse=mod.parse
song_now=None
ray_calls=0
original_resolve=mod.resolve
# The same melody is recorded in two real keys; neither recording is augmented audio.
def resolve(pattern):
 global song_now,ray_calls
 song_now=next(s for s,pats in mod.SONGS.items() if pattern in pats)
 if 'Ray main' in pattern:
  ray_calls+=1;song_now='ray_キー+5' if ray_calls==2 else 'ray_原曲キー'
 matches=[p for p in mod.SRC.rglob('*') if p.is_file() and __import__('fnmatch').fnmatch(mod.nfc(p.relative_to(mod.SRC)),pattern)]
 if not matches:raise FileNotFoundError(pattern)
 return sorted(matches)
def parse(path):
 if path.suffix=='.vsqx':return older.read_score('花になって')[1]
 notes=original_parse(path)
 shift=5 if song_now=='ray_キー+5' else 0
 return [(a,b,t+shift,l) for a,b,t,l in notes]
mod.resolve=resolve;mod.parse=parse
source=Path(mod.__file__).read_text()
# Checkpoint each completed song so interruption does not erase completed output.
source=source[source.index('def main():'):source.index("if __name__=='__main__':")]
source=source.replace("enumerate(SONGS.items(),19)","enumerate(SONGS.items(),40)")
source=source.replace("        reports.append(","        reports.append(")
source=source.replace("    write_csv(PREP/'フレーズ候補.csv',all_clips)","    write_csv(PREP/'フレーズ候補.csv',all_clips)")
exec(source,mod.__dict__)
if __name__=='__main__':mod.main()
