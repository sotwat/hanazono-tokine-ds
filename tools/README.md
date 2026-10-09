# v1.1.1 出力補正の再現

Pythonとonnxを使用します。v1.1.0の `dsvocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx` に対し、次を実行します。

```sh
python apply_processing.py ORIGINAL.onnx OUTPUT.onnx
```

`output-processing.onnx` は追加ノードとEQ係数だけの保存形式で、単独推論モデルではありません。元の学習重みは含みません。処理はF5のEQ-only、F#5以上のEQ・+3 dB・弱いparallel tanh、E5以下のバイパスです。EQは音程追従の257tap FIR、切り替えは対象域内10ms。高音処理に既存の局所ピーク保護を使用します。

補正コードと係数は同梱ボコーダーの条件に従いCC BY-NC-SA 4.0。元の帰属・ライセンスは配布ZIPのdsvocoder/NOTICE.txtにあります。
