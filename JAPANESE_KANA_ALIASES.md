# Extended Japanese kana aliases

OpenUtau's DIFFS JA phonemizer romanizes kana before querying `dsdict-ja.yaml`. Having a kana entry alone therefore does not guarantee recognition: for example, すぃ becomes `swi`, でぃ becomes `di`, and どぅ becomes `du`.

`tools/japanese_kana_aliases.py` adds the missing roman aliases and corresponding katakana spellings to the root, duration and pitch dictionaries. It also supplies common additional combinations using existing model symbols. This is a dictionary repair, not retraining or a change to synthesis effects.

```sh
# Back up the bank before applying; refresh its file-hash manifest afterwards.
python tools/japanese_kana_aliases.py /path/to/bank --report aliases.json
python tools/check_japanese_kana_aliases.py /path/to/bank
```

The script refuses conflicting aliases and unknown model symbols, preserves existing entries, and is idempotent. The static romanization profile was obtained by invoking the installed OpenUtau AU `DiffSingerJapanesePhonemizer.Romanize` method. It covers 278 hiragana/katakana spellings, including the existing kana vocabulary and added consonant combinations. This is a bounded supported set, not a promise to interpret arbitrary kana strings or lyric annotations.

Native validation reproduced 66 errors on the initial 262-spelling fixture before the patch. The same fixture had zero errors afterwards. The expanded 278-spelling fixture also had zero phoneme errors. Actual OpenUtau renderer synthesis was checked on 16 selected syllables, including すぃ, でぃ, どぅ, いぇ, てぃ, とぅ, ずぃ, しぇ, じぇ, ちぇ, つぁ, つぃ, つぇ, つぉ, でゅ and ふゅ. This verifies conversion and synthesis, not native-quality pronunciation of every untrained combination.

The existing `wo` alias remains unchanged so を continues to map to `o`. As a pre-existing consequence of romanization, うぉ also resolves through that alias. The patch does not introduce a separate /v/ phoneme absent from this model. Non-kana annotations such as `い息` are outside this repair; they are not silently converted to ordinary vowels or silence.
