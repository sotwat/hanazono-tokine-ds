# Continuous vowel gain

Build: `python tools/portable/retune_vowel_balance.py BANK NEW_OUTPUT_DIR`. Input must contain the existing evb balance block. Install the exported vocoder; acoustic is unchanged.

### 2026-10-26 音高に伴う音量差と母音接続の補正連続性

本人が「あ」がまだ強いと報告し、「と」「き」が弱い、低い音の音量が小さい可能性を指摘。固定seed比較の実測では補正前の低め/to ki/は-21.74/-23.69dB、ka/soは-15.63/-12.95dB。既存補正後も約2〜3dB差が残る。一方/a/全体はkiより強くなく、平均音量だけを原因と断定しない。音高だけを条件に一律boostせず、既存有声母音の測定型balanceをdeadband .75→.25dB、strength .9→1へ。さらに母音直接接続で補正gainが境界ごとに0dBへ戻る不連続を、隣接する補正gainの中間値へ20ms両端で移るよう修正。子音/休符のgainは従来通り。独立aの-1.25dB trimとacoustic9f8de792は維持。

`tokiakasou`持続中央RMS: to -20.74→-20.42、ki -21.28→-20.42、ka -19.12→-19.92、so -18.85→-19.92dB。対象4音の差は約2.43→.5dB。aは-22.49→-21.67（既存trim維持）。96条件の連続性/子音bypass/ONNX一致検査pass。3phraseの再合成errorsなし、全phone metadata一致、finite/nonclip、wave符号保存。これは再学習による根治や全曲聴感の保証ではない。今回の本人承認は未取得。

artifact `/Users/aaa/Documents/DiffSinger/artifacts/tokine-level-continuity-20261026`。両bank vocoder `d61a1bd0d45cb2541e53978539de84b6005231f08c9121a37605c08b27e5a071` 導入、manifest36件ずつ一致。旧vocoder/metadata/cache87件をinstalledへ退避。AU保存→終了→導入→再起動、元バックアップ180BPM/Track9 DIFFS JA復帰確認。Release未更新。比較tokiakasou-comparison.wavは前→0.5秒無音→後。sourceはvowel_balance.py、retune_vowel_balance.py、check_continuous_vowel_balance.py。既存skill/tool調査を再利用、新依存なし。
