# Recording refresh — 2026-10-19

Local training candidate: **花園トキネDS v1.3.0-rc.1**. The public Release is unchanged.

## Inputs and selection

The owner confirmed that Samurai Heart, I beg you, Melt and Hana ni Natte were replaced with corrected recordings. Their 72 old clips were removed. Eight recordings were added: Lemon, ray (original key), ray (+5), Nainai, Mikazuki, Kaibutsu, Kisetsu wa Tsugitsugi Shindeiku and Akeboshi. The +5 take is an actual recording.

509 phrase candidates yielded 109 accepted clips (385.535 seconds). Acceptance requires ASR lyric support >=85, SOFA confidence >=0.65, positive phoneme durations, matching phone sequences, note-boundary agreement and independent Parselmouth/WORLD pitch agreement. SOFA must use `--g2p Phoneme`; Dictionary mode incorrectly drops consonants from phoneme input. Only main1 is used for Samurai Heart; its main2 is backing/ad-lib material. Both Lemon parts cover consecutive lead sections.

212 older clips remain after removing the four replaced titles and seven training clips whose padding overlaps validation. Tokine: 321 selected clips, 294 training / 27 validation, 21.560 minutes total / 18.918 training minutes. 33 recordings are represented; 千本桜 is validation-only in this refresh. All twelve October 19 recordings have training clips. The existing NIT070 material contributes 249 training clips; no additional speaker was introduced. ANIMA's provisional score is not used.

Ray's original-key validation uses score note indices 139–149. None occurs in the accepted +5 training material. There is no independent +5 validation clip. The added-material acoustic evaluation contains eleven recording excerpts. Historical source hashes are unavailable for 145 legacy rows; these rows were not represented as historically hash-verified. Originals and source scores are unchanged.

## Training and model selection

Start from `tokine_ds_multispeaker_nit070/model_ckpt_steps_4000.ckpt`, before the four defective recordings were learned. Do not initialize from the current 25-recording model. Existing speaker embeddings are frozen. The first run trains all other acoustic parameters for 1500 updates. A second run freezes the entire FS2 and trains 600 updates. Batch size <=4 / <=2200 frames, gradient accumulation 2, learning rate 3e-5.

Neither full update preserved the older acoustic diagnostics sufficiently. The selected candidate linearly regularizes step500 toward the clean initial checkpoint: 35% trained parameters / 65% clean prior. Unchanged tensors, including speaker embeddings, are copied exactly. No parameter from a checkpoint trained on the defective four recordings is included.

Gain-aligned log-mel MAE, fixed seed42 / depth0.1 / 20 inference steps; lower is better:

| Candidate | Older normal | Older low | Added/replaced |
|---|---:|---:|---:|
| Previous installed acoustic | .646384 | .581455 | .920122 |
| Clean initial | .641485 | .584157 | .942739 |
| All acoustic step500 | .734030 | .625458 | .778295 |
| All acoustic step1000 | .792792 | .682804 | .724183 |
| All acoustic step1500 | .755717 | .639884 | .745414 |
| Frozen FS2 step300 | .743965 | .633162 | .778067 |
| Frozen FS2 step600 | .726340 | .614232 | .776899 |
| 50% update | .672116 | .582829 | .849214 |
| Selected 35% update | .659300 | .577926 | .875110 |

The selected candidate stays within a 3% regression limit on both older tests: normal +2.0%, low -0.6%, added/replaced -4.9%. These are acoustic diagnostics, not listening scores. Perceptual improvement across every song is unverified.

## Processing and comparisons

`refresh_acoustic.py` requires identical raw export topology and exact correspondence between the template's two raw acoustic branches and the baseline raw export. It replaces 364 branch tensors (266 changed), preserving all other graph nodes and constants. Keep the paired, previously accepted vocoder, duration/pitch models, dictionaries, WORLD high-range processing, sustain/connections, rest boundary control and low-register/phrase-volume processing. Low-range WORLD remains disabled.

`render_bank.py` compares both complete processed acoustic/vocoder pairs with identical inputs: two cached native OpenUtau low/high excerpts, followed by eleven added-material excerpts using recording-derived fixed F0 and phoneme lengths. The combined audio is an acoustic comparison, not an exported song performance with automatic score timing. No gain matching is applied to the output files. Native UI rendering is checked separately.

## Reproduction in the owner's private workspace

Recordings, third-party scores, labels, checkpoints and ONNX exports are intentionally excluded from Git. These scripts depend on the existing private preparation artifacts and local SOFA/DiffSinger installations. Paths refer to the owner's machine; copy this directory to an ignored artifact workspace before running it. Do not run preparation inside the Git checkout.

```sh
cp -R tools/training_refresh /path/to/private/tokine-refresh-work
cd /path/to/private/tokine-refresh-work
# Review local source/score paths in prepare.py first.
python prepare.py
python transcribe.py
python pitch_check.py
# Run SOFA from its installation, using the absolute preparation/SOFA候補 path:
# python infer.py --ckpt ckpt/tokine-ja/step.100000.ckpt \
#   --folder /absolute/work/preparation/SOFA候補 --g2p Phoneme \
#   --ap_detector None --out_formats transcriptions,lab --save_confidence
python repair_sofa_with_notes.py --root /absolute/work/preparation --source SOFA候補
python build_dataset.py
```

Run DiffSinger's binarization against the generated config, then from the DiffSinger root:

```sh
python scripts/train.py --config /absolute/work/config.yaml \
  --exp_name tokine_ds_refresh_20261019 --reset
python scripts/train.py --config /absolute/work/config-frozen-20261019.yaml \
  --exp_name tokine_ds_refresh_20261019_frozen --reset
python /absolute/work/regularize_checkpoint.py \
  checkpoints/tokine_ds_multispeaker_nit070/model_ckpt_steps_4000.ckpt \
  checkpoints/tokine_ds_refresh_20261019/model_ckpt_steps_500.ckpt \
  checkpoints/tokine_ds_refresh_20261019_balanced/model_ckpt_steps_500.ckpt --weight .35
python scripts/export.py acoustic --exp tokine_ds_refresh_20261019_balanced \
  --ckpt 500 --freeze_spk tokine --out /absolute/work/export-balanced
```

The checked configuration snapshots, selection metrics and graph-preservation hashes are retained alongside this document. The acoustic replacement script also asserts the baseline graph and template tensors before writing. Do not package a raw export by itself: it omits the accepted processing graph and paired vocoder.

Native verification: the AU application displayed v1.3.0-rc.1 and completed rendering the eleven-part fixture. Sixteen new cache waveforms were finite (peak <=0.5382), with no ERR entries in the checked interval. Unrelated external singer references stall startup; they were isolated temporarily for this check and restored afterward. Permanent removal is awaiting the owner's decision. The current AU session has the updated singer loaded.
