# Sustain release revision

The previous four-frame return to the original mel could end a held syllable abruptly. `soften_sustain_release.py` changes only `su_endfade` (including both branches of the portable acoustic model) to twelve frames, approximately 139 ms at 44.1 kHz / 512 hop.

Local verification used identical raw mel for both wrapper variants: da at 70/90/260 frames and no at 90 frames. Mel outside the last twelve frames is bit-identical. Maximum adjacent-frame RMS decrease in the tail decreased by 13–38%; the middle of the note retained its level. This is a numerical check, not listener approval.

A reported real-song case also had the next fricative starting approximately 180 ms before its note. Changing the note velocity did not move that boundary with this renderer. A targeted phoneme offset of +144 ticks at 150 BPM (120 ms) extended the preceding vowel. OpenUtau rendered the corrected phrase; a previously unvoiced interval became periodic. This score correction is local to the reported phrase and is not a global phoneme rule. Private scores and audio are excluded from this repository.

The owner reported that the preceding sustain demonstration did not improve this syllable. Do not treat earlier synthetic tests as evidence that the real phrase had improved. The twelve-frame revision is installed locally for comparison, not yet published as a new voicebank release.

## Connected-syllable strength (local rc.4)

Owner feedback: duration improved in rc.3, but the sustained voice sounded weak. Preserve that accepted duration and separately evaluate strength.

`contextual_sustain_release.py` uses a six-frame return for connected phonemes and retains twelve frames before SP/AP or sequence end. The first native comparison gained only 0.9 dB in the reported tail. A fixed shift of donor positions worsened the result and was rejected.

`select_sustain_donor.py` chooses the strongest of six early windows (centers 10/14/18/22/26/30, three samples at ±2 frames) by mean linear mel amplitude for eligible phonemes shorter than 172 frames. It oscillates within ±2 frames of that center. Longer phonemes retain the previous donor mapping. This selects existing acoustic content; it does not apply a gain floor or amplify the output.

The actual phrase was rendered in OpenUtau AU. In the reported vowel, RMS changed +0.11 dB at 1.2–1.4 s, +1.21 dB at 1.45–1.6 s, and +5.59 dB at 1.6–1.7 s compared with rc.3. Outputs have identical length, finite samples, and no clipping. Separate renders include stochastic variation: these figures describe the produced comparisons, not exact isolated causal estimates. The existing coverage check passes for short vowels, sustained consonants, stops, and high pitch; out-of-sustain mel remains identical to raw mel. Naturalness and perceived strength await owner listening. Local rc.4 is selected in the user's score; distribution ZIP remains rc.2.
