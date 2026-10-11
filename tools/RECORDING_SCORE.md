# Recording-aligned score draft

These utilities build an untuned UST/USTX from an isolated singer recording,
supplied kana phrases, SOFA phoneme alignment, and RMVPE F0. They do not infer
lyrics from scratch or certify a draft as clean training data.

Required local dependencies: NumPy, SciPy, SoundFile, PyYAML, PyTorch,
Torchaudio, the SOFA source/model, and a compatible RMVPE source/model.
Use trusted model files: the SOFA loader needs its serialized hyperparameters.
The RMVPE loader expects a tensor state dictionary and checks all used weights.
Neither model weights nor recordings are included here.

A private phrase JSON is a list of objects with `index`, `start`, `end` (seconds),
`text`, and hiragana `kana`. Choose non-overlapping windows that cover the sung
phrases. Verify the lyric readings against the actual performance. Automatic
speech recognition can suggest windows but is not a reliable lyric reference.

```sh
python tools/analyze_recording_pitch.py recording.wav rmvpe.pt rmvpe_source_dir pitch.npz
python tools/align_recording_phrases.py recording.wav phrases.json dsdict-ja.yaml sofa.ckpt sofa_source_dir alignment.json
python tools/build_recording_score.py alignment.json pitch.npz dsdict-ja.yaml output --tempo 193 --name recording_draft
```

The tempo must be supplied from a checked score or other reliable reference.
Note positions retain measured seconds; they are not snapped to a beat grid.
A Japanese long-vowel mark extends the previous syllable rather than creating
another forced vowel. Note anchors follow vowel onsets. Pitch estimation excludes
the following consonant, even when the score connects adjacent note anchors.
A change penalty and minimum stable segment duration suppress brief pitch
excursions. Continuations use `+`; neither pitch curves nor vibrato are added.

Outputs include the UST, USTX, timed notes, phoneme intervals, validation metadata,
and review items. Review very short alignments and missing stable pitch before
using their phrases for training. Preserve the continuous F0 separately: integer
UST pitches cannot represent every performed pitch deviation.
