# Sustained upper-mid closed vowels

2026-10-26: user accepted the final processed trial for Backup measure 51 i/ts/u.

Extends the existing WORLD synthesis path to i/u lasting at least 20 frames (232 ms), fading in with smoothstep from MIDI 67 to 69. Existing higher-pitch routing remains. Control column 8 carries the eligibility mask; original/donor mel and control columns 0–7 remain identical. Uses the existing boundary smoothing, loudness balance and peak guard. No score, F0, phoneme duration, learned weight or independent-a contour changes. ts itself is unchanged; the following u uses the alternative path. This is a bounded rendering correction, not retraining or a claim that every distortion is resolved.

Validation: 1536 scope and Torch/ONNX cases passed. Native OpenUtau rendered successfully with identical phoneme metadata and finite nonclipping audio. Fixed-seed omoi, negatta, tokiakasou output waves are byte-identical to baseline. User rejected duration-only trials; accepted the WORLD diagnostic and then the final normal-processing trial. The earlier a-connection edits were not the cause: user retracted that hypothesis, and sustained i/ts/u acoustic frames matched the older model.

Reproduction: closed_vowel_source.py BANK NEW_OUTPUT_DIRECTORY, then check_closed_vowel_source.py NEW_OUTPUT_DIRECTORY. Apply acoustic.onnx and vocoder.onnx together to the ca2c9765/d61a1bd0 baseline pair. Artifact: /Users/aaa/Documents/DiffSinger/artifacts/tokine-itsu-stability-20261026.

Waveform display was hidden by the piano-roll zoom detail threshold (64/480 px/tick); a slight timeline zoom restored waveform and phoneme labels in the native UI.
