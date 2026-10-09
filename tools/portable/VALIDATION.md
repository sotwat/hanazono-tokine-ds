# v1.2.0-rc.2 verification (2026-10-10)

- macOS Apple Silicon; ONNX Runtime CPU 1.23.2: all eight packaged models load.
- Standard ONNX domains only; IR 10 / opset 18 for the modified pair.
- Dynamic WORLD lengths 32, 64 and 137 frames: expected length, finite output.
- Same-run baseline comparison at MIDI 60 and 74, plus unvoiced F0: bit-identical output. These synthetic single-token cases do not establish low-register pronunciation quality.
- MIDI 77 and 81, 64 frames: finite output, median detected F0 errors 2.27 and 3.38 cents; peaks 0.422 and 0.444.
- MIDI 77, 1723 frames (~20 seconds): finite output, detected median F0 error 3.17 cents; a numerical/length test, not subjective sustained-vowel approval.
- Blending with mixed voiced/unvoiced F0 and both waveforms distinct: maximum error versus the approved NumPy blend 2.62e-7; bypass samples bit-identical.
- Packaged pair in `/Applications/OpenUtau.app`, version 0.1.572.1: singer recognized; 11-note test phrase rendered after retiring the old tensor/wave caches. Native cache 44.1 kHz, 148,992 samples, finite, peak 0.530762. UI showed playback reaching the end without a render error. Application binaries were not changed.
- ZIP CRC valid; compiler-local user paths removed; no recordings, song scores, private test inputs or lyrics included.
- Earlier pyworld vs eager diffsptk parameter comparison on one donor: median absolute log spectral-envelope error 0.0751 and median absolute aperiodicity difference 0.00412. This is not a perceptual equivalence claim.
- The acoustic wrapper sustains 40-frame vowels and N/m/n/f/h/s/sh/z, with duration-adaptive source windows. 48-frame /a/ and /o/, N and /s/ change only eligible mel frames; 30-frame /a/ and stop /d/ remain bit-identical to raw mel. Paired vocoder remains finite, including high pitch. Actual UST phrase excerpts are verified in the local sustain-coverage record, not included here.

Unverified: Windows/Linux, other editors, GPU providers, fresh ZIP installation through the wizard, owner listening approval of the portable implementation and new sustain update. Distribution is a prerelease, not a replacement of the stable release.
