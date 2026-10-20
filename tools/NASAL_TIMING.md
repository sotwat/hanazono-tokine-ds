# Small n timing adjustment

The user heard the n in バックアップ「はじまりの」 slightly early. The duration
wrapper previously capped n (token 19) at six native frames. `nasal_timing.py`
changes that ceiling to five frames, leaving all graph nodes and every other
initializer unchanged. Run it after `consonant_timing.py` on the six-frame bank.

```sh
python tools/nasal_timing.py BEFORE.dur.onnx AFTER.dur.onnx
```

The existing redistribution lengthens the preceding vowel by the saved amount,
so the following vowel anchor and phrase duration are retained. It is a ceiling,
not a fixed n length: shorter predicted n phones keep their duration. This does
not target syllabic N/ん, ny, m, or other consonants. Acoustic sustain, morph,
pitch processing and vocoder are unchanged.

Validation used `check_consonant_timing.check` with `CAPS[19] = 5`: 9504 duration
and 50 cluster cases passed. Nodes were byte-identical; the only changed value
was `ct_caps[19]`, six to five. The actual native renderer moved n from
2261.803294 to 2273.608838 ms (11.805544 ms later), kept /o/ at 2333.331 ms and
preserved 159232 samples. Output was finite and unclipped. This is timing and
integrity verification, not listening approval. Before/after audio and evidence
are local in `DiffSinger/artifacts/tokine-n-timing-20261020`.
