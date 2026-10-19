# Sustained donor harmonics

`tools/preserve_sustain_harmonics.py SOURCE OUTPUT` updates the pitch-safe sustain wrapper in both acoustic branches. It keeps the selected center frame's spectral shape instead of averaging five log-mel frames, and matches its total linear mel energy to the previous donor. Log-sum-exp avoids energy underflow. The learned tensors, donor selection, F0 guards, timing, crossfades and sustain eligibility are unchanged.

The previous full sustain mix improved steadiness, but the user reported audible distortion. Averaging nearby mel spectra can smear pitch-dependent harmonic structure even when their F0s pass the stability threshold. This revision addresses that mechanism; it is not evidence that every audible artifact is fixed.

On the same native raw spectra and F0, a diagnostic comparison at the reported A#3 vowel reduced off-harmonic spectral energy from 2.56% to 1.71% (100–8000 Hz, 4096-sample Hann window, harmonic bands ±35 Hz). Steady RMS changed from 0.0827 to 0.0852. These are local diagnostics, not perceptual quality scores. The native renderer retained phoneme timing and 130560 output samples. The existing 192-condition sustain checks passed. Listening acceptance is pending.

Only tooling and this description belong in Git. Private audio, scores, fixtures and model weights remain local. Keep a backup of the previous acoustic model and metadata before deployment.
