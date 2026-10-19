# Rejected experiment: periodicity during replaced sustain

Rejected on 2026-10-20 after the user reported a worse, distinctly different timbre at the sustained vowel. Both installed models were restored to their preceding versions. The off-harmonic-energy reduction did not establish acceptable voice quality; cycle averaging can remove natural cycle variation and change timbre. Retained only as an experiment record; do not use in distribution or reinstall as a fix. The original distortion remains unresolved.

Experimental reproduction only:

```sh
python tools/portable/sustain_periodicity.py acoustic.onnx vocoder.onnx output
python tools/portable/check_sustain_periodicity.py output/sustain-cycles.onnx checks.json
```

The acoustic model retains both existing mel planes and adds its original-branch sustain weight to control channel 6. The vocoder uses integrated target F0 to average nine equal-phase waveform cycles, within the same phoneme. Only samples where sustain replacement is active can change. The interpolated mask is strengthened by four and passed through smoothstep, so partial spectral replacement does not leave most of the irregular waveform residue untouched. It does not amplify decayed audio or extend note durations. Learned tensors, existing effects and timing remain unchanged.

This addresses residual cycle-incoherent noise after holding a mel spectrum. A held spectrum can still produce irregular waveform cycles through the neural vocoder. Removing mel-frame averaging alone did not resolve the user's reported distortion.

Validation includes exact zero-weight bypass, phone-boundary isolation, preservation of synthetic harmonic signals and vibrato at 110/233/440/880 Hz, and noise reduction. With identical native raw spectra and F0 at the reported sustained vowel, off-harmonic energy changed from 1.709% to 0.0336% (100–8000 Hz, 4096-sample Hann window, harmonic bands ±35 Hz). RMS changed from 0.08525 to 0.08300. This diagnostic does not establish perceptual acceptance. Native full-context rendering retains phoneme timing and 130560 samples.

The acoustic and vocoder should be installed together. If the new vocoder receives the previous control plane, channel 6 is zero and the added filter bypasses. Preserve backups of both models and update package manifests. Audio, model weights and private score fixtures do not belong in the source repository.
