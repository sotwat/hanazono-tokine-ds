# 花園トキネDS

OpenUtau向けの日本語DiffSinger歌声音源です。修正済み録音への差し替えと追加素材を使った学習更新に加え、ロングトーンの歪み・途中の音量の膨らみ・母音接続・子音の発音を調整しました。2026年10月26日までにMacへ反映したモデルと同じ一式です。

[最新版の配布ZIP](https://github.com/sotwat/hanazono-tokine-ds/releases/latest) · [v1.3.0の変更点](https://github.com/sotwat/hanazono-tokine-ds/releases/tag/v1.3.0)

## ダウンロードと導入

`HanazonoTokineDS-v1.3.0.zip` をOpenUtauの「ツール」→「音源をインストール」から読み込んでください。再起動後、歌手「花園トキネDS v1.3.0」、フォネマイザ「DIFFS JA」、レンダラー「DIFFSINGER」を選びます。Pythonや追加プラグインは不要です。

旧版とは別の歌手名で登録されます。既存の曲で使う場合は、トラックの歌手を新版へ変更してください。他のUTAU音源を削除する必要はありません。

## 変更点

- 修正済みのサムライハート・I beg you・メルト・花になってと、新しく追加された録音素材を学習更新に使用しました。本人録音の選別後321区間のうち、294区間を学習、27区間を検証に使用しています。
- ロングトーンの持続処理と合成経路を更新し、機械的な歪みと途中の音量の膨らみを抑えました。
- 母音同士の接続、独立した「あ」の立ち上がり、音域・音素による音量差を調整しました。
- 「た」「つ」「ち」「る」などの子音と、子音が早く始まりすぎる問題を調整しました。拗音・外来音の辞書も拡充しています。
- 高めの「に」と「つ」の母音について、試聴で採用した雑音・スペクトルの揺れを抑える処理を含みます。「つ」の子音処理は維持しています。

音程カーブと音量カーブを使用できます。推奨比較設定は20 steps・depth 0.10です。専用のGENC・VELC・ENE・BREC・VOIC・TENC制御モデルは含みません。

## 確認範囲と制限

macOS Apple Silicon / OpenUtau AU v0.1.576.0で、修正版の再合成・再生・波形表示を確認しています。配布時に全8 ONNXの構造とCPU読み込み、モデルと設定の一致、ZIPとSHA-256を検査しています。Windows・Linux・他のエディターは未検証です。

同梱の音響モデル・長さモデル・ボコーダー・辞書・設定を一式で使用してください。音響モデルとボコーダーの間で複数のmelと制御情報を渡す構成のため、対応しないソフトでは動作しません。従来版より合成時間とメモリー使用量が増える場合があります。

試聴で確認した箇所を改善した版です。すべての曲・音域・歌詞の音質や、完全に一定の音量を保証するものではありません。今回の持続音・接続の調整は音源内の合成処理によるもので、再学習だけで問題を解消したという意味ではありません。

## 利用条件とクレジット

非商用の歌唱作品への利用・公開が可能です。作品に「歌唱：花園トキネDS」と表記してください。花園トキネのモデル・辞書・アイコンの改変や再配布は許可していません。商用利用は事前相談が必要です。詳細はTERMS.mdをご確認ください。

声・キャラクターは花園トキネ。補助学習データのNIT SONG070 F001はHTS Working Group / Nagoya Institute of Technology（CC BY 3.0）、ボコーダーのPC-NSF-HiFiGAN 2025.02はOpenVPI Community（CC BY-NC-SA 4.0）によるものです。WORLD・diffsptk由来の処理も含みます。第三者素材にはそれぞれのライセンスが適用されます。THIRD_PARTY.md、dsvocoder/NOTICE.txt、licensesをご確認ください。

歌唱録音・歌詞・UST/VSQX・学習データ・チェックポイント・比較音声は同梱していません。

## クレジット

- 声・キャラクター：花園トキネ
- 補助学習データ：NIT SONG070 F001／HTS Working Group、Nagoya Institute of Technology（CC BY 3.0）。`NIT-SONG070-COPYING.txt` を同梱。
- ボコーダー：OpenVPI Community／PC-NSF-HiFiGAN 2025.02（CC BY-NC-SA 4.0）。元の学習重みは維持し、出力に音域別補正を追加。
- サムライハートの参照UST：968103
- ノーダウトの参照UST：いろあい
- staple stableの参照UST：ivyleaf33 (Vyx Le)。原配布の非商用条件を尊重します。
- Tell Your Worldの参照UST：GowitheBeat
- 学習・書き出し基盤：[OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger)
- 対応アプリ：[OpenUtau](https://github.com/openutau/OpenUtau)
