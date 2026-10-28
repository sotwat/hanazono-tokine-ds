# Recording-derived AP breaths

Local addition after public v1.3.0. The public v1.3.0 archive does not contain this feature.

## Current local revision: accepted 14 takes

The initial 51-candidate output was rejected by the owner as noise. The owner then explicitly selected takes 6, 9, 13, 14, 15, 16, 18, 24, 27, 28, 34, 35, 43 and 45. Only these 14 original clips (4.145 seconds) supply the replacement breath decoder. Unaccepted clips and the old 51-candidate decoder weights are not used for this training.

`train_selected.py` learns each accepted clip's mel trajectory, conditioned on its identity, with a 25→128→128→128 MLP. The former time/duration-only regressor averaged different breaths; a rerun on 12 accepted training clips and 2 held-out clips was barely better than its constant baseline (.63014 vs .63124), so that rerun was not installed. The identity-conditioned model uses all 14 accepted clips, with 278 training frames and 70 held-out interpolation frames. Its selected checkpoint is step2400, normalized interpolation MSE .015217 vs constant baseline 1.036301. These are within-recording reconstruction diagnostics, not unseen-recording validation or proof of listening quality. The owner has not yet approved the new synthesized audio.

At inference an AP selects the accepted identity nearest its requested duration. Its normalized time trajectory is stretched to the note length; conditioning uses the recorded duration to avoid out-of-distribution duration features. Identity does not change inside one AP. The model predicts mel and uses the existing neural vocoder; it does not paste recorded waveform samples. The existing AP32 mask/F0-zero/output routing is retained.

Reproduce this revision with `python tools/breath_support/train_selected.py SELECTED_DATASET NEW_TRAINING_DIR`, then use the build/check commands below with its decoder.pt and a clean approved v1.3.0 bank. `build.py` supports both checkpoint formats. Keep the paired models and refresh the installed manifest together. Two no-AP fixtures remain bit identical in mel and waveform; 15 duration/pitch cases pass. Native OpenUtau AP/ni/tsu renders report no errors and unchanged ni/tsu phoneme metadata. The Mac installation uses local revision `2026-10-28-selected14-breath`; the public ZIP remains unchanged. The following section records the initial, superseded candidate experiment.

## Initial candidate experiment (superseded)

The singer's existing recordings supply the breaths. FoxBreatheLabeler first classifies only existing aligned SP/AP pause intervals. Threshold .4 finds complete events; require mean probability >=.60, peak >=.8, duration .12–1.1 seconds, RMS >=-55 dBFS and no near clipping. These gates reduce consonant contamination, but classifier labels are not human listening annotations. The selected set has 51 excerpts from 22 songs, 12.208 seconds. Original audio is untouched. New labels/clips and checkpoints stay in the private artifact directory.

Train a small duration-conditioned 11→96→96→128 mel decoder on 42 clips, reserving 9 clips from four other songs for validation. Per-recording gain is normalized toward -33 dBFS (gain capped at 15), keeping within-clip envelope and spectrum. The selected checkpoint is step300 out of 1800 steps: validation normalized MSE .68891, constant training-mean baseline .78931. Selection uses validation; these are spectral diagnostics, not a proof of perceptual quality. Seed2741, AdamW lr.002, weight_decay.01. This is a dedicated breath decoder learned from recordings, not a full singing-model retrain or a copied sample playback.

AP becomes acoustic token32; the existing singing graph sees its previous SP token1, then only AP frames receive predicted breath mel. Pitch/duration model token dictionaries retain their existing AP1. Acoustic control column11 identifies the new sound, column3 becomes zero during it to force unvoiced generation. The existing neural vocoder's raw output supplies the breath; a five-ms edge fade and AP-only output selection bypass singing sustain/level/silence corrections during the inhale. No-AP inputs preserve the original graph's values exactly. BREATHINESS remains disabled; this feature inserts an inhale, not a breathiness curve or automatic placement of breaths.

In OpenUtau using DIFFS JA/DIFFSINGER, enter `AP` as a note's lyric (phonetic hint `AP` also works). Set its length to the desired breath, typically .2–.5 seconds. It has no intended note pitch. Ordinary rests stay SP and are not automatically converted. Add the AP note in an existing gap rather than shifting the score's other notes. `breath` is an optional dictionary alias. Existing scores are not edited.

## Reproduce

Use the existing DiffSinger Python environment with torch, numpy, soundfile, ONNX/ORT, YAML and the same mel extractor. Fetch the detector from https://github.com/autumn-DL/FoxBreatheLabeler and its official 0.2 release (`20ms.zip`). Inspect/pin the upstream source and checkpoint before running. Detector is only used for offline annotation and is not bundled in the voicebank.

```
python tools/breath_support/extract.py OUTPUT/dataset REFRESH_DATA DETECTOR_DIR
python tools/breath_support/train.py OUTPUT/dataset OUTPUT/training
python tools/breath_support/build.py APPROVED_BANK OUTPUT/training/decoder.pt OUTPUT/build
python tools/breath_support/check.py APPROVED_BANK OUTPUT/build OUTPUT/check
```

The refresh dataset requires `dataset_manifest.json` and `raw/tokine/{transcriptions.csv,wavs}`. DETECTOR_DIR contains FoxBreatheLabeler/ and 20ms/config.yaml + model.ckpt. `train.py` uses the owner's existing DiffSinger mel extractor path. Build artifacts must be combined with the original bank files, AP token32 in the acoustic phoneme JSON, AP/breath entries in dsdict-ja.yaml and refreshed checksums. The paired ONNX files must be installed together.

Validation: two no-AP acoustic + waveform fixtures are bit identical with fixed random seeds; 15 AP cases cover five lengths and three note pitches, with finite/unclipped output, exact sample count and unvoiced AP. Native OpenUtau renders 200/300/500-ms AP notes; errors empty. Previous ni/tsu fixture phoneme metadata remains identical. The detector recognizes the synthesized short breaths, but the 500-ms example has a lower average probability; no claim of human-verified naturalness is made. Private artifacts include the raw detector scores and comparison WAV for listening.
