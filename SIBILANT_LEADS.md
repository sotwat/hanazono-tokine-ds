# Sustained vowels before s/sh

The duration wrapper previously allowed s/sh to occupy up to 16 frames of
the preceding sustained note. At 44.1 kHz / hop 512, this is about 186 ms
before velocity scaling and host alignment. It could make the preceding
vowel end prematurely.

`tools/bound_sibilant_leads.py` reduces that limit to 9 frames (about
104 ms). Existing group eligibility (at least 40 frames, no silence),
consonant budget, velocity and vowel alignment are retained. No acoustic
weights, sustain spectra, pitch curves, gain or score edits are required.

On the local ray phrase at BPM 132, B3, the 660-tick 「が」 followed by a
60-tick 「し」 had a 176.14 ms sh lead. Native OpenUtau rendering with the
new duration model gives 106.06 ms; the preceding a grows from 448.86 to
518.94 ms. The i vowel anchor stays at exactly the same time. The
2000–2050 ms window in the test excerpt changes from RMS 0.01391 to
0.08010 without an output gain change. Listening approval remains separate.

The update changes only entries 24 and 25 of the wrapper's `lt_maximum`
initializer. Learned duration parameters and all other initializers and
nodes are preserved. It applies by phoneme class, without song or lyric
conditions. Short groups and other consonant classes retain their rules.
