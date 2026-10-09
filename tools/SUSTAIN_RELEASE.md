# Sustain release revision

The previous four-frame return to the original mel could end a held syllable abruptly. `soften_sustain_release.py` changes only `su_endfade` (including both branches of the portable acoustic model) to twelve frames, approximately 139 ms at 44.1 kHz / 512 hop.

Local verification used identical raw mel for both wrapper variants: da at 70/90/260 frames and no at 90 frames. Mel outside the last twelve frames is bit-identical. Maximum adjacent-frame RMS decrease in the tail decreased by 13–38%; the middle of the note retained its level. This is a numerical check, not listener approval.

A reported real-song case also had the next fricative starting approximately 180 ms before its note. Changing the note velocity did not move that boundary with this renderer. A targeted phoneme offset of +144 ticks at 150 BPM (120 ms) extended the preceding vowel. OpenUtau rendered the corrected phrase; a previously unvoiced interval became periodic. This score correction is local to the reported phrase and is not a global phoneme rule. Private scores and audio are excluded from this repository.

The owner reported that the preceding sustain demonstration did not improve this syllable. Do not treat earlier synthetic tests as evidence that the real phrase had improved. The twelve-frame revision is installed locally for comparison, not yet published as a new voicebank release.
