# Sustained syllable coverage (2026-10-10)

`extend_sustain_coverage.py SOURCE OUTPUT` updates the existing acoustic sustain wrapper, including both branches of the portable WORLD bank. It rejects an already-updated graph. Vocoders and learned weights are unchanged.

The previous 172-frame (~2 s) minimum excluded many long notes in Tell Your World: the inspected score contains 0.6 s `だ` and 1 s `の`. These syllables already contain supported vowel phonemes; the duration gate, rather than the kana spelling, excluded them. The old 26-frame release blend also returned to the decayed raw mel during the last ~0.3 s.

New eligibility: 40 frames (~0.46 s) or longer. Supported tokens are a/e/i/o/u/N/m/n/f/h/s/sh/z. Stop attacks, closure, silence and short phonemes retain their original mel. Japanese CV syllables sustain their vowel after the consonant; repeated stop bursts would change pronunciation.

| Phoneme duration | Reused local frames | Start | Crossfade |
|---|---|---|---|
| 40–59 frames | 8–20 | 20 | 8 |
| 60–171 frames | 12–32 | 32 | 12 |
| 172+ frames | 34–74 | 74 | 26 |

The final blend to the original release is 4 frames (~46 ms). No loudness floor or gain is added. Earlier smooth cosine traversal is retained.

Verification: `check_sustain_coverage.py BANK REPORT.json` covers 48-frame CV vowels, N and s, unchanged 30-frame vowels, unchanged stop tokens, and a high-register case. Non-target mel is bit-identical to the raw branch within each run, and all audio is finite. The 70/90/155/260-frame da/no tests retained their attacks. For 90 frames, da tail RMS changed from 0.0519 to 0.1793 (new early 0.1816); no from 0.0720 to 0.1448 (new early 0.1317). These separate stochastic renders demonstrate sustained energy, not identical waveforms outside the changed region.

OpenUtau AU rendered two excerpts of the user's score with v1.1.3; output files are private and not included in this repository. The user's original score was saved and reopened with v1.1.3 selected. The local WORLD candidate is v1.2.0-rc.2. Public release v1.2.0-rc.1 is unchanged. Listening approval and exhaustive pronunciation testing remain outside these numerical checks.
