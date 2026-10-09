"""Package an approved model set using the previous public distribution layout."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
import onnx
import onnxruntime as ort
import yaml


def build(base, approved, output, version, readme):
    output.mkdir(parents=True, exist_ok=False)
    bank=output / ('花園トキネDS v'+version)
    shutil.copytree(base,bank)
    models=sorted(approved.rglob('*.onnx'))
    assert len(models)==8
    checks=[]
    opts=ort.SessionOptions();opts.log_severity_level=3;opts.intra_op_num_threads=2
    for source in models:
        relative=source.relative_to(approved);target=bank/relative
        assert target.is_file(),relative
        shutil.copy2(source,target)
        assert target.read_bytes()==source.read_bytes()
        onnx.checker.check_model(str(target))
        ort.InferenceSession(str(target),opts,providers=['CPUExecutionProvider'])
        checks.append(str(relative))
    name='花園トキネDS v'+version
    meta=yaml.safe_load((bank/'character.yaml').read_text());meta['name']=name
    (bank/'character.yaml').write_text(yaml.safe_dump(meta,allow_unicode=True,sort_keys=False))
    (bank/'character.txt').write_text('name='+name+'\nauthor=花園トキネ\n')
    (bank/'README.md').write_text(readme.read_text())
    manifest=json.loads((bank/'version.json').read_text())
    manifest.update(version=version,status='stable',approved_model=json.loads((approved/'LOCAL_REVISION.json').read_text())['version'],vowel_morph_frames_before=8,vowel_morph_frames_after=8)
    if version == '1.2.2':
        manifest.update(sustain_timbre='constant-five-point-mean',
                        fricative_morph_frames_before=8,fricative_morph_frames_after=2,
                        fricative_morph_phones=['f','h','s','sh','z'])
    manifest['files']={str(f.relative_to(bank)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sorted(bank.rglob('*')) if f.is_file() and f.name!='version.json'}
    (bank/'version.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    archive=output/('HanazonoTokineDS-v'+version+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(bank.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(output))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,digest in manifest['files'].items():
            assert hashlib.sha256(z.read(bank.name+'/'+rel)).hexdigest()==digest
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (output/'SHA256SUMS.txt').write_text(digest+'  '+archive.name+'\n')
    (output/'validation.json').write_text(json.dumps(dict(models_equal_approved=checks,onnx_check=True,ort_load=True,zip_crc=True,manifest_hashes=True,sha256=digest,size=archive.stat().st_size),indent=2))
    print(digest,archive)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ['base','approved','output']:p.add_argument(arg,type=Path)
    p.add_argument('version');p.add_argument('readme',type=Path)
    a=p.parse_args();build(a.base,a.approved,a.output,a.version,a.readme)
