# 花園トキネDS

OpenUtauで使える日本語のDiffSinger歌声音源です。18曲分の本人歌唱をもとにした配布版です。

## 開発状況

| 項目 | 状況 |
| --- | --- |
| 18曲版の歌唱生成 | OpenUtauで確認済み |
| 15曲版との比較 | 制作者が改善を確認（2026-10-02） |
| 配布版 | v1.0.0／初回配布版 |
| Windows・Linuxでの動作 | 未検証 |

## ダウンロード

[最新版の配布ZIPはこちら](https://github.com/sotwat/hanazono-tokine-ds/releases/latest)

## 導入

1. GitHub ReleasesからZIPをダウンロードして展開します。
2. OpenUtauの「ツール」→「音源フォルダを開く」で開いたSingersフォルダへ、展開した「花園トキネDS」フォルダを入れます。同名の音源がある場合は先に別の場所へ退避してください。
3. OpenUtauを再起動し、歌手「花園トキネDS」を選びます。
4. フォネマイザを「DIFFS JA」、レンダラーを「DIFFSINGER」に設定します。

ボコーダーを同梱しているため、別途のボコーダー導入は不要です。OpenUtau AU v0.1.570.12で元の18曲モデルの歌唱を確認しています。配布構成では全8モデルの読み込み・設定参照・ZIP整合性を検査しました。配布ZIPを新規導入した環境での歌唱は未検証です。

## 歌わせ方と対応機能

歌詞はひらがなで入力します。小さい「っ」や「ん」も入力できます。まず音符の音程・長さと歌詞を整えてから再生してください。

音程カーブと音量カーブを使用できます。GENC、VELC、ENE、BREC、VOIC、TENCの専用制御モデルは含まれていません。推奨の比較設定は20 steps・depth 0.60です。

## 音源について

花園トキネの18曲分の歌唱と、NIT SONG070 F001を使った2話者学習から、花園トキネの声へ固定して書き出しています。追加3曲から採用した区間は約167秒です。PJSを追加した試作モデルは含みません。音長・音程モデルは15曲版と共通です。

発音の不明瞭さや、音域・歌詞による声質のばらつきが残る場合があります。18曲版と15曲版の比較は40音・4フレーズで行いました。

## 利用条件

非商用の歌唱作品への利用・公開は可能です。作品には「歌唱：花園トキネDS」と表記してください。花園トキネのモデル・辞書・アイコンの改変や再配布は許可していません。商用利用は事前相談が必要です。詳細は [利用規約](TERMS.md) を確認してください。第三者素材にはそれぞれのライセンスが適用されます。

同梱ボコーダーはOpenVPI CommunityによるPC-NSF-HiFiGANで、CC BY-NC-SA 4.0です。ボコーダーの再配布は非商用に限り、帰属表示とライセンス表示が必要です。詳しくは `dsvocoder/NOTICE.txt` と [公式配布ページ](https://openvpi.github.io/vocoders/) を参照してください。この音源の規約で第三者素材の条件を変更することはできません。

## クレジット

- 声・キャラクター：花園トキネ
- 補助学習データ：NIT SONG070 F001／HTS Working Group、Nagoya Institute of Technology（CC BY 3.0）。`NIT-SONG070-COPYING.txt` を同梱。
- ボコーダー：OpenVPI Community／PC-NSF-HiFiGAN 2025.02（CC BY-NC-SA 4.0）。重みは無改変。
- サムライハートの参照UST：968103
- ノーダウトの参照UST：いろあい
- staple stableの参照UST：ivyleaf33 (Vyx Le)。原配布の非商用条件を尊重します。
- 学習・書き出し基盤：[OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger)
- 対応アプリ：[OpenUtau](https://github.com/openutau/OpenUtau)

歌唱録音、歌詞、UST／VSQX、学習用データ、学習チェックポイント、比較譜面・音声は同梱していません。
