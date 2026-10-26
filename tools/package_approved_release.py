"""Package byte-identical installed models with release-only documentation."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output, version, readme):
    output.mkdir(parents=True, exist_ok=False)
    bank = output / f'花園トキネDS v{version}'
    bank.mkdir()
    source_manifest = json.loads((source / 'version.json').read_text())
    for relative, expected in source_manifest['files'].items():
        src = source / relative
        assert sha(src) == expected, relative
        if relative in {'LOCAL_REVISION.json', 'TRAINING_REFRESH.json', 'README.md'}:
            continue
        target = bank / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    name = bank.name
    meta = yaml.safe_load((bank / 'character.yaml').read_text())
    meta['name'] = name
    (bank / 'character.yaml').write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    (bank / 'character.txt').write_text(f'name={name}\nauthor=花園トキネ\nweb=https://github.com/sotwat/hanazono-tokine-ds\n')
    (bank / 'README.md').write_text(readme.read_text())
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 2
    models = {}
    for model in sorted(bank.rglob('*.onnx')):
        rel = model.relative_to(bank)
        assert sha(model) == sha(source / rel)
        onnx.checker.check_model(str(model))
        session = ort.InferenceSession(str(model), opts, providers=['CPUExecutionProvider'])
        models[str(rel)] = sha(model)
        del session
    assert len(models) == 8
    for config in bank.rglob('*.yaml'):
        text = config.read_text()
        assert '/Users/' not in text
        data = yaml.safe_load(text)
        if config.name == 'dsconfig.yaml':
            for key in ['acoustic', 'linguistic', 'dur', 'pitch', 'phonemes', 'languages']:
                if key in data:
                    assert (config.parent / data[key]).is_file(), (config, key)
    files = {str(p.relative_to(bank)): sha(p) for p in sorted(bank.rglob('*')) if p.is_file()}
    manifest = {'version': version, 'status': 'stable', 'source_revision': '2026-10-26-ni-tsu-stability', 'files': files}
    (bank / 'version.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    archive = output / f'HanazonoTokineDS-v{version}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bank.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel, digest in files.items():
            assert hashlib.sha256(z.read(f'{name}/{rel}')).hexdigest() == digest
        assert not any('..' in Path(n).parts or n.startswith('/') for n in z.namelist())
    digest = sha(archive)
    (output / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n')
    report = {'version': version, 'models_equal_installed': models, 'onnx_check': True, 'ort_load': True,
              'zip_crc': True, 'manifest_hashes': True, 'sha256': digest, 'size': archive.stat().st_size}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('version')
    parser.add_argument('readme', type=Path)
    args = parser.parse_args()
    build(args.source, args.output, args.version, args.readme)
