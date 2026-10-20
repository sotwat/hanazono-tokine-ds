# Connected vowel lead

`TokineVowelLead` extends OpenUtau's Japanese DiffSinger phonemizer. For contiguous note groups, an incoming single `a/e/i/o/u` begins up to 20 ms earlier when the previous group's last phoneme is also a vowel. The lead is limited to 25% of that preceding vowel's duration. Conversion uses the host tempo map and rounds to ticks.

Gaps, phrase starts, consonant onsets and incoming vowels after `N` retain the base timing. Notes, pitch curves and trained models are not changed. Host phoneme overrides still apply after phonemization.

## Build and install

Build against the same OpenUtau Core assembly as the target host:

```sh
dotnet build TokineVowelLead.csproj -c Release \
  -p:OpenUtauDir='/Applications/OpenUtau AU.app/Contents/MacOS' \
  -o ./bin/plugin
```

Quit OpenUtau, copy only `HanazonoTokine.VowelLead.dll` into its user `Plugins` directory, and restart. On this Mac the directory is `~/Library/OpenUtau/Plugins`.

Choose **TOKINE JA** on the singer track. A voicebank may set `default_phonemizer: HanazonoTokine.TokineVowelLead` in `character.yaml`; existing scores with an explicit DIFFS JA track selection must be changed separately. This is a host plugin, not a portable ONNX voicebank feature. The current build targets .NET 10 and requires a compatible OpenUtau host; compatibility with other hosts is not established.

## Verification, 2026-10-20

Native OpenUtau phonemization at BPM 180: 84 control parts produced 119 render phrases, with 30 eligible onsets moved by 20.1389 ms (tick rounding), 164 other onsets unchanged, and no phoneme errors. Controls included all five vowels, same-vowel joins, a preceding CV, `N`, incoming CV, and gaps. Phrase ends stayed unchanged.

The actual current score excerpt was rendered through OpenUtau's DiffSinger renderer before and after the change. Four vowel onsets moved by 20.1389 ms; the output was finite and unclipped. The installed app loaded TOKINE JA on both current tracks and displayed the earlier vowel boundary. These checks establish timing and rendering behavior, not universal listening quality.

Local evidence: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-vowel-lead-20261020/` (`timing-check.json`, `actual-verification.json`, `installation.json`, native runners, inputs and before/after audio). No public release package was updated.
