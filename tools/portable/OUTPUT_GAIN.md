# Output gain (2026-11-16)

The installed Tokine DS bank measured -23.10/-22.07/-20.12 LUFS for identical
G3/D4/A4 phoneme-duration/F0 inputs. Installed Lewisia measured
-24.45/-21.06/-14.93 and Adachi Rei DS -15.66/-14.01/-14.77 LUFS.
Each case is 3.646 seconds, 20 diffusion steps, model-default depth and neutral
gender/velocity. These are controlled short ONNX bank renders, not a full-song
listening ranking or a claim about all voicebanks.

Apply a maximum +3 dB to each complete vocoder output. One scalar applies to the
entire phrase: max(1, min(10^(3/20), 0.75 / max(abs(waveform)))).
This preserves phase, timing and the within-phrase envelope. It does not use
sample clipping, compression or EQ. The bound limits added gain; an already loud
input is unchanged, so it does not repair pre-existing clipping. Track mixing
and user boosts can still exceed full scale.

Build with:
    python tools/portable/output_gain.py ORIGINAL_VOCODER OUTPUT_VOCODER
    python tools/portable/check_output_gain.py

Mac baseline: source commit 8a26430; installed acoustic SHA256
a22fc80f3bfb403c220a669ecc53819f5ef27f9b22699dd546a52aaeb42c8568,
vocoder f1b9b528a8529ae1ad6f8ae2ab4a8801ea18baf91ccd0e7b09beaa4de62cfd0a.
Output vocoder: 6abd0635c687fbb781798d3538bdbe6ac3eb8aaa940eb0980ef55648321ed1ec.
In each installed bank replace only that local vocoder, change dsconfig.yaml
vocoder from pc_nsf_hifigan_44.1k_hop512_128bin_2025.02 to dsvocoder, append
LOCAL_REVISION.output_gain, and refresh version.json hashes for changed files.
The explicit local vocoder path avoids a shared dependency and changes the
render configuration for cache invalidation. Other singers are unchanged.

Two current-acoustic low/high native-input fixtures produced exactly the
original waveform times the scalar (max error 0), with unchanged sample count.
Peaks changed .32035→.45250 and .33878→.47854. OpenUtau AU reloaded the bank,
rendered a short check score with no new render errors and played to 5.986 s.
Original model/metadata snapshots and machine-local measurements are preserved
in DiffSinger/artifacts/tokine-loudness-20261116. Rebuild from those originals,
not from an already patched model. The public release ZIP is unchanged.
