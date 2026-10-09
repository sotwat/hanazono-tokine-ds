"""Assemble a self-contained release candidate; exclude compiler paths and trials."""
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
 name=f'花園トキネDS v{version}';out=destination/name;out.mkdir(parents=True,exist_ok=True)
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

高音のWORLD再合成を音源内に組み込んだ配布候補版です。

## 導入

OpenUtauの「ツール」→「音源をインストール」から、このZIPを選んでください。歌手名は「{name}」です。Pythonや追加プラグインのインストールは不要です。

## 変更点

- D5から高音処理を段階的に適用し、F5以上で全面適用します。判定は連続した音程に従います。
- D5以下は従来のDiffSinger合成です。
- 0.46秒以上の母音、鼻音、一部の摩擦音を持続処理の対象にします。停止子音は繰り返しません。
- 以前の高音EQ・音量加算・サチュレーションは重ねません。

## 対応環境

macOS Apple SiliconのOpenUtau v0.1.572.1で検証。音源に同梱した音響モデルとボコーダーをセットで使ってください。再生ソフトの改造は不要です。Windows・Linux・他のDiffSingerエディターは未検証です。

この版は音響モデルからボコーダーへ2組のmelを渡します。この受け渡しに対応しないソフトでは動きません。従来版より合成時間とメモリー使用量が増えます。

以前の試聴版とはWORLDの実装が異なり、音色の同等性は本人の確認待ちです。そのため正式版を置き換えず、配布候補版として公開しています。

利用条件はTERMS.md、第三者素材の表記はTHIRD_PARTY.mdをご確認ください。
''')
 files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='version.json'}
 (out/'version.json').write_text(json.dumps(dict(version=version,status='release-candidate',high_world=True,low_world=False,sustain_min_frames=40,sustain_release_frames=4,files=files),ensure_ascii=False,indent=2))
 zip_path=destination/f'HanazonoTokineDS-v{version}.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(destination))
 with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
 digest=hashlib.sha256(zip_path.read_bytes()).hexdigest();(destination/'SHA256SUMS.txt').write_text(f'{digest}  {zip_path.name}\n');print(zip_path,digest)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('build',type=Path);p.add_argument('destination',type=Path);p.add_argument('--version',required=True);a=p.parse_args();package(a.bank,a.build,a.destination,a.version)
