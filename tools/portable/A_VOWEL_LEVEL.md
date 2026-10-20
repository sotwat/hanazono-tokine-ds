# Independent /a/ level trim

2026-10-21: the standalone /a/ in the current Track9 phrase was slightly louder than its neighbours. Apply −0.5 dB after the final vocoder normalization, with 20 ms raised-cosine edges. The predicate matches the existing independent /a/ resynthesis: immediately after e/i/o/u lasting at least 40 frames, /a/ at least 12 frames. No lyric or song matching. Channel 7 carries the eligibility flag; this builder rejects the earlier experimental attack patch that used that channel.

Build with `a_vowel_level.py ACOUSTIC VOCODER OUTPUT`. Both generated models must be installed together. Trained mel paths, F0, phoneme boundaries and score remain unchanged. All samples outside eligible phones are identical in fixed-seed comparisons; within the plateau the waveform is multiplied by 10^(-0.5/20), retaining spectral shape.

Local evidence: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-a-level-fine-20261021/verification.json`. Actual DIFFS JA excerpt, 187904 samples, target frames 129–181, measured −0.50000012 dB; outside samples exactly equal, finite and unclipped. Installed in both local banks, manifests verified and old DS cache retired. Public releases are unchanged. Perceived balance still requires the user's listening evaluation.
