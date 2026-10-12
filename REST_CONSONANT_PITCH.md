# Unvoiced consonant F0 after a rest

The user reported a rasp on Leia's phrase-initial ki after accepting the SP silence gate. Native diagnostic input, retaining the preceding phrase and the original pitch curves, shows k at frames 192–201. Its first two frames carry MIDI 87 (the previous D#6), then MIDI 81, then MIDI 72, before the intended onset bend. The i starts at MIDI 75.786 and returns to C5. Such inherited F0 can change both acoustic conditioning and the neural/WORLD blend during the consonant. This is a measured conditioning defect, not proof of the entire perceptual cause.

`tools/portable/rest_consonant_pitch.py` conditions consecutive unvoiced consonants following SP on the following vowel's onset F0. It covers ch/cl/f/h/hy/k/ky/p/s/sh/t/ts, requires the previous non-unvoiced token to be SP and the following token to be a vowel, and retains original F0 on vowels, voiced consonants, connected consonants and silence. No lyric matching, note editing or vocal pitch flattening is used.

The acoustic model receives corrected F0 before both original/donor branches. Control channel 3 carries the same F0 to the paired vocoder, keeping neural and WORLD synthesis consistent. The preceding silence gate remains active. Install acoustic/vocoder together. This is a local listening trial; other hosts are unverified.

```sh
python tools/portable/rest_consonant_pitch.py silence-acoustic.onnx silence-vocoder.onnx output
python tools/portable/check_rest_consonant_pitch.py output/acoustic.onnx
```

1,200 independent cases cover 12 consonants, 5 vowels, 5 durations, connected/after-rest contexts, and clusters. They verify exact untouched F0 and the next-vowel reference. Native tensors/audio stay in `DiffSinger/artifacts/tokine-ki-onset`, not the Git repository.

Native verification: the 291-frame diagnostic retains the full preceding phrase. k frames 192–201 now all receive 651.1657 Hz (MIDI 75.786), matching i onset. The acoustic control channel 3 equals the debug F0 exactly. Native waveform is finite, 148,992 samples at 44.1 kHz, peak 0.680115. Before/after audio uses independent diffusion/WORLD draws; differences outside the target do not prove a processing regression or perceptual improvement. The local bank was updated as a listening trial, the original score is retained and the public release is unchanged.
