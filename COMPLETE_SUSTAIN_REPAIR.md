# Partial sustain repair and roughness

The sustained B-flat3 vowel in the opening of Hana no Tou kept a substantial
fraction of the decaying raw spectrum. The anticipatory gate reached only
0.574 in the original branch and 0.277 in the donor branch. A steady requested
pitch therefore still produced a changing spectrum during the hold.

`tools/complete_sustain_repair.py` maps the existing per-phone gate through
`clip(4 * gate, 0, 1)`. Healthy phones with zero deficit remain bypassed.
Substantial deficits reach a full repair, while the existing temporal
crossfade, pitch compatibility guard, duration eligibility and release
remain in force. This is a shared rule, without song or lyric matching.
Learned tensors, F0, timing, gain and EQ are unchanged.

Using identical native raw spectra and F0, the steady interior's short-window
RMS coefficient of variation fell from 0.122 to 0.0114. Correlation at the
requested pitch period rose from 0.916 to 0.964. These measurements show
greater stability in this fixture; they do not establish universal perceptual
quality or replace listening approval.
