# Sustain entry without a late recovery dip

The frame-local energy gate could enable held spectra only after a vowel
had already collapsed. On the ray B3 vowel, the corrected sh anticipation
made the vowel eligible for sustain repair; a short amplitude dip and
recovery appeared at the entry into that repair.

`tools/anticipate_sustain.py` makes the energy decision once per eligible
phone. It tests the existing gate inside the phone's active sustain region,
excluding the final eight frames (about 93 ms), and uses the maximum need
for repair there. The existing sustain crossfade then starts before the
collapse. Phoneme membership prevents a following phone from supplying the
decision. Duration eligibility, pitch compatibility, donor selection and
the existing releases remain in force.

With identical native raw spectra and F0, the four-frame RMS around frame
213 in the ray fixture changes from 0.0441 to 0.0716; neighboring windows
are about 0.07. The later held portion stays about 0.075. This verifies
removal of that dip, not perceptual approval of every sustained sound.
No gain, EQ, score, learned tensor, duration or F0 change is included.
