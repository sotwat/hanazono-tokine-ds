# フレーズの音量差補正（ローカル試作、2026-10-15）

ホシアイ「昨日と今日の間を」付近で、周辺よりフレーズが弱くなる指摘に対応。
vocoderの最終出力に、レンダリングフレーズ全体で一定のgainを適用する。
有声音mask（既存control plane2/channel4、母音・N・m・n）だけでRMSを測定。
reference RMS=0.10、gain=(reference/RMS)^0.7、通常の範囲−3〜+6.02dB。
RMS<=0.004、対象1024sample未満は補正をbypass。peak<=0.95のため必要なら追加減衰する。

子音を含むフレーズ全体へ同じgainを掛け、発音内の強弱を保持する。
長音の減衰部分だけを持ち上げる処理、EQ、サチュレーションではない。
音響モデル・F0・音符・長さ・低域WORLD不採用は保持する。
OpenUtauが1フレーズとして扱う区間内の個々の弱い音は、この補正だけでは解消しない。
RMSは聴感音量そのものではない。本人の比較試聴は未確認。

## 再現

```sh
python tools/portable/phrase_balance.py BEFORE_VOCODER.onnx OUTPUT/vocoder.onnx
python tools/portable/check_phrase_balance.py OUTPUT/phrase-balance-layer.onnx
```

既に補正を組み込んだvocoderへ重複適用しない。acousticは既存のcontrol planeが必要。
比較では旧キャッシュを復元可能な場所へ移動し、OpenUtauを再起動する。
モデル・録音・キャッシュはリポジトリへ含めない。

## 検証

- silence/短区間/音量/peakの84条件に合格。
- 同じnative生成波形4区間へ新layerを適用し、有声音RMS差5.37→1.61dB。
- 各区間gain +0.47/+4.23/+3.12/+2.41dB、peak最大0.373。
- full vocoderも4区間で期待するscalar gainと一致、同尺・有限・peak<0.95。
- 独立再合成には神経vocoder/WORLDの乱数による小差がある。

ローカル詳細: DiffSinger/artifacts/tokine-phrase-balance-20261015/verification.json と full-graph-verification.json。
公開Releaseの変更を意味しない。
