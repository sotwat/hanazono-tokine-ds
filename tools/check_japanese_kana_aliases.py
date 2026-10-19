"""Check the DIFFS JA lyric spellings observed in the installed renderer."""
import argparse
import json
from pathlib import Path
import yaml
from japanese_kana_aliases import ROMAN_ALIASES, EXTRA_KANA


def check(bank, profile):
    rows = json.loads(profile.read_text())
    for folder in [bank, bank/'dsdur', bank/'dspitch']:
        d = yaml.safe_load((folder/'dsdict-ja.yaml').read_text())
        es = {e['grapheme']: e['phonemes'] for e in d['entries']}
        symbols = {s['symbol'] for s in d['symbols']}
        assert len(es) == len(d['entries']), 'Duplicate graphemes'
        for row in rows:
            roman = row['roman']
            phones = es.get(roman, es.get(roman.lower()))
            if phones is None and roman in symbols:
                phones = [roman]
            assert phones and set(phones) <= symbols, row
            # Existing wo deliberately maps を to o; do not change that alias.
            if row['lyric'] not in ['うぉ', 'ウォ']:
                assert phones == es[row['lyric']], row
        for roman, kana in ROMAN_ALIASES.items():
            assert es[roman] == es[kana], roman
        for kana, phones in EXTRA_KANA.items():
            assert es[kana] == phones, kana
    return dict(variants=len(rows), dictionaries=3, passed=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('bank', type=Path)
    p.add_argument('--profile', type=Path, default=Path(__file__).with_name('japanese_kana_romanization.json'))
    a = p.parse_args()
    print(json.dumps(check(a.bank, a.profile)))
