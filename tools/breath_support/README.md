# Recording-derived AP breaths

Local addition after public v1.3.0. The public v1.3.0 archive does not contain this feature.

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
