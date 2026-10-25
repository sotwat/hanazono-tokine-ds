# Independent /a/ level trim

2026-10-21: the standalone /a/ in the current Track9 phrase was slightly louder than its neighbours. Apply −0.5 dB after the final vocoder normalization, with 20 ms raised-cosine edges. The predicate matches the existing independent /a/ resynthesis: immediately after e/i/o/u lasting at least 40 frames, /a/ at least 12 frames. No lyric or song matching. Channel 7 carries the eligibility flag; this builder rejects the earlier experimental attack patch that used that channel.

Build with `a_vowel_level.py ACOUSTIC VOCODER OUTPUT`. Both generated models must be installed together. Trained mel paths, F0, phoneme boundaries and score remain unchanged. All samples outside eligible phones are identical in fixed-seed comparisons; within the plateau the waveform is multiplied by 10^(-0.5/20), retaining spectral shape.

Local evidence: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-a-level-fine-20261021/verification.json`. Actual DIFFS JA excerpt, 187904 samples, target frames 129–181, measured −0.50000012 dB; outside samples exactly equal, finite and unclipped. Installed in both local banks, manifests verified and old DS cache retired. Public releases are unchanged. Perceived balance still requires the user's listening evaluation.

### 2026-10-26 独立「あ」の音量を再調整

本人が「き→あ」の「あ」がまだ強いと指摘。接続直後のRMSが直前/i/末尾より約2dB大きいことを比較音声で確認。共通の独立/a/最終波形trimを-0.5から-1.25dB（追加-0.75dB）へ調整。対象条件・20ms端fade・接続補間・音色・音素境界は維持。これは残る音量差の調整であり、学習済み音量偏り自体の根治とは記載しない。

artifact `/Users/aaa/Documents/DiffSinger/artifacts/tokine-a-level-20261026`。固定seedで対象外sample一致、omoi/negatta全wave一致、tokiakasou対象中央値-0.75000024dB。native OpenUtau errorsなし・全phone metadata一致。両bank vocoder `331d1329ffb1a97520132c4524a2a1ec17ba4428195a2ea392ff89e0b7753d6d`、acoustic e709466e維持、manifest36項目ずつ一致。旧vocoder/metadata/cache220件退避。AU再起動し元バックアップ180BPM/Track9 DIFFS JAへ復帰確認。Release未更新。本人の今回の聴感承認は未取得。比較 `tokiakasou-comparison.wav` は前→0.5秒無音→後。
