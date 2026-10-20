# Quiet vowel balance

`vowel_balance.py` appends a vowel mask in unused control channel 6 of the paired acoustic model and a bounded gain stage before the vocoder's existing phrase balance and peak guard. Existing trained tensors, pitch, duration, sustain and boundary processing remain unchanged. Install the acoustic and vocoder together.

The stage measures the central 60% RMS of each a/e/i/o/u phone and uses the median vowel level in the rendered phrase as its reference. Vowels at least six frames long and above RMS 0.008 qualify. At least two qualifying vowels are needed. For a deficit greater than 1 dB, it compensates half the excess, capped at +1.5 dB. Gain is constant inside each phone, with 20 ms cosine transitions at its edges. This is not a time-varying sustain envelope or an attempt to amplify vanished tails. Silence, consonants and N receive no direct gain; the final phrase balance may slightly change the scalar gain of the whole phrase.

This conservative limiter reduces quiet outliers; it does not make every vowel equally loud or establish that every audible imbalance is a synthesis fault. Extreme near-silence is excluded to avoid amplifying noise. High/low pitches use the same bounded stage.

Build from the previous pair:

```sh
python tools/portable/vowel_balance.py OLD_ACOUSTIC OLD_VOCODER OUTPUT_DIRECTORY
python tools/portable/check_vowel_balance.py OUTPUT_DIRECTORY/vowel-balance-layer.onnx
```

2026-10-20: 504 dynamic length/level/pitch/bypass checks passed (65.406–1760 Hz), including exact consonant/rest bypass and endpoint equality. Four native OpenUtau phrases from バックアップ retain identical phones, timing and sample counts, with finite, unclipped outputs. Fixed-mel comparisons avoid diffusion randomness: the reported B3「こ」in「このおもい」rises about 0.57 dB after phrase normalization; the following「の・お」change by about −0.03 dB. Existing initializer bytes are preserved in both models. No score edit or retraining is involved. Listening approval remains pending; these numerical checks do not prove absence of all audible artifacts.

Evidence: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-ko-level-20261020`.
