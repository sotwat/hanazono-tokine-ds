"""Align kana dictionary aliases with OpenUtau DIFFS JA romanization."""
import argparse
import json
from pathlib import Path
import yaml

# DIFFS JA romanizes lyrics before querying the singer dictionary.
ROMAN_ALIASES = {
    'ye': 'いぇ', 'wi': 'うぃ', 'we': 'うぇ', 'kye': 'きぇ',
    'she': 'しぇ', 'je': 'じぇ', 'swi': 'すぃ', 'zwi': 'ずぃ',
    'che': 'ちぇ', 'tsa': 'つぁ', 'tsi': 'つぃ', 'tse': 'つぇ',
    'tso': 'つぉ', 'ti': 'てぃ', 'di': 'でぃ', 'tu': 'とぅ',
    'du': 'どぅ', 'hye': 'ひぇ', 'bye': 'びぇ', 'rye': 'りぇ',
    'tyu': 'てゅ', 'dyu': 'でゅ',
    'kwa': 'くぁ', 'kwi': 'くぃ', 'kwe': 'くぇ', 'kwo': 'くぉ',
    'gwa': 'ぐぁ', 'gwi': 'ぐぃ', 'gwe': 'ぐぇ', 'gwo': 'ぐぉ',
    'pwa': 'ぷぁ', 'pwi': 'ぷぃ', 'pwe': 'ぷぇ', 'pwo': 'ぷぉ',
    'gye': 'ぎぇ', 'nye': 'にぇ', 'mye': 'みぇ', 'pye': 'ぴぇ',
}
EXTRA_KANA = {
    'ぎぇ': ['g', 'y', 'e'], 'にぇ': ['n', 'y', 'e'],
    'みぇ': ['m', 'y', 'e'], 'ぴぇ': ['p', 'y', 'e'],
    'てゅ': ['t', 'y', 'u'], 'でゅ': ['d', 'y', 'u'],
    'ふゃ': ['f', 'y', 'a'], 'ふゅ': ['f', 'y', 'u'], 'ふょ': ['f', 'y', 'o'],
    **{kana: [c, 'w', v] for c, row in [('k', 'く'), ('g', 'ぐ'), ('p', 'ぷ')]
       for kana, v in [(row+'ぁ', 'a'), (row+'ぃ', 'i'), (row+'ぇ', 'e'), (row+'ぉ', 'o')]},
}


def katakana(text):
    return ''.join(chr(ord(c)+0x60) if '\u3041' <= c <= '\u3096' else c for c in text)


def patch_dictionary(path):
    text = path.read_text()
    data = yaml.safe_load(text)
    entries = {e['grapheme']: e['phonemes'] for e in data['entries']}
    known = {s['symbol'] for s in data['symbols']}
    additions = {}
    def add(key, phones):
        if not set(phones) <= known:
            raise ValueError(f'Unsupported symbols for {key}: {phones}')
        if key in entries:
            if entries[key] != phones:
                raise ValueError(f'Conflicting existing alias: {key}')
        else:
            entries[key] = phones
            additions[key] = phones
    for key, phones in EXTRA_KANA.items():
        add(key, phones)
    for key, kana in ROMAN_ALIASES.items():
        add(key, entries[kana])
    for key, phones in list(entries.items()):
        if any('\u3041' <= c <= '\u3096' for c in key):
            add(katakana(key), phones)
    if additions:
        suffix = yaml.safe_dump([{'grapheme': k, 'phonemes': list(v)} for k,v in additions.items()],
                                allow_unicode=True, sort_keys=False)
        path.write_text(text.rstrip()+'\n'+suffix)
    return additions


def update(bank):
    result = {}
    for folder in [bank, bank/'dsdur', bank/'dspitch']:
        path = folder/'dsdict-ja.yaml'
        if path.exists():
            result[str(path.relative_to(bank))] = patch_dictionary(path)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('bank', type=Path)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    result = update(a.bank)
    if a.report:
        a.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print({k: len(v) for k,v in result.items()})
