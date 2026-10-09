"""Assemble the approved WORLD/sustain voicebank without private trial material."""
import argparse,json,shutil,hashlib,zipfile
from pathlib import Path
import onnx,yaml

def strip(message):
 if hasattr(message,'metadata_props'):message.ClearField('metadata_props')
 if hasattr(message,'doc_string'):message.ClearField('doc_string')
 for field,value in message.ListFields():
  if field.type==field.TYPE_MESSAGE:
   if field.is_repeated:
    for child in value:strip(child)
   else:strip(value)

def package(bank,build,destination,version):
 name=f'花園トキネDS v{version}';out=destination/name;out.mkdir(parents=True,exist_ok=False)
 for p in bank.iterdir():
  if p.name in ['dsdur','dspitch','dsvocoder']:shutil.copytree(p,out/p.name,dirs_exist_ok=True)
  elif p.suffix in ['.yaml','.txt','.png'] or p.name in ['TERMS.md','THIRD_PARTY.md'] or p.name.endswith(('.phonemes.json','.languages.json')):shutil.copy2(p,out/p.name)
 for src,dst in [(build/'acoustic.onnx',out/'tokine_ds_clarity_hybrid.tokine.onnx'),(build/'vocoder.onnx',next((out/'dsvocoder').glob('*.onnx')))]:
  model=onnx.load(src);strip(model);onnx.checker.check_model(model);onnx.save(model,dst)
 c=yaml.safe_load((out/'character.yaml').read_text());c['name']=name;(out/'character.yaml').write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
 (out/'character.txt').write_text(f'name={name}\nauthor=花園トキネ\nweb=https://github.com/sotwat/hanazono-tokine-ds\n')
 shutil.copytree(Path(__file__).parent/'licenses',out/'licenses',dirs_exist_ok=True)
 third_party=(out/'THIRD_PARTY.md')
 if '## WORLD処理の追加' not in third_party.read_text():
  with third_party.open('a') as f:f.write('\n## WORLD処理の追加\n\nCopyright 2022 SPTK Working Group. diffsptk 4.0.0 (Apache-2.0) のWORLD実装をONNXへ変換し、固定パディング・二分探索・FFT変換を変更しています。WORLD: Copyright (c) 2010 M. Morise (BSD-3-Clause)。ライセンス全文はlicenses内。第三者コードとその派生部分の権利は各ライセンスに従います。\n')
 (out/'README.md').write_text(f'''# 花園トキネDS v{version}

高音のWORLD再合成とロングトーンの持続処理を音源内に組み込んだ配布版です。

## 導入

OpenUtauの「ツール」→「音源をインストール」から、このZIPを選んでください。歌手名は「{name}」です。Pythonや追加プラグインのインストールは不要です。

## 変更点

- D5から高音処理を段階的に適用し、F5以上で全面適用します。判定は連続した音程に従います。
- D5以下は従来のDiffSinger合成です。
- 0.46秒以上の母音、鼻音、一部の摩擦音を持続処理の対象にします。停止子音は繰り返しません。
- 短い持続音では、母音前半から強い区間を選んで持続させます。
- 次の音につながる場合とフレーズ末尾で終端処理を分け、伸びと滑らかな終わり方を両立するよう調整しました。
- 以前の高音EQ・音量加算・サチュレーションは重ねません。

## 対応環境

macOS Apple SiliconのOpenUtau v0.1.572.1で検証。音源に同梱した音響モデルとボコーダーをセットで使ってください。再生ソフトの改造は不要です。Windows・Linux・他のDiffSingerエディターは未検証です。

この版は音響モデルからボコーダーへ2組のmelを渡します。この受け渡しに対応しないソフトでは動きません。従来版より合成時間とメモリー使用量が増えます。

制作者が試聴し、配布を承認したrc.4の音源を正式版にしています。次の子音が早く始まって母音が短くなる場合は、譜面の音素位置の調整も必要です。個別の曲の譜面調整は音源には含まれません。

利用条件はTERMS.md、第三者素材の表記はTHIRD_PARTY.mdをご確認ください。
''')
 files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='version.json'}
 (out/'version.json').write_text(json.dumps(dict(version=version,status='release',high_world=True,low_world=False,sustain_min_frames=40,sustain_release_frames=12,sustain_connected_release_frames=6,sustain_donor_selection='strongest-early-window',files=files),ensure_ascii=False,indent=2))
 zip_path=destination/f'HanazonoTokineDS-v{version}.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(destination))
 with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
 digest=hashlib.sha256(zip_path.read_bytes()).hexdigest();(destination/'SHA256SUMS.txt').write_text(f'{digest}  {zip_path.name}\n');print(zip_path,digest)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('build',type=Path);p.add_argument('destination',type=Path);p.add_argument('--version',required=True);a=p.parse_args();package(a.bank,a.build,a.destination,a.version)
