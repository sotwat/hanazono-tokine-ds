using System;
using OpenUtau.Api;
using OpenUtau.Core.DiffSinger;

namespace HanazonoTokine {
    [Phonemizer("Tokine Japanese Phonemizer", "TOKINE JA", language: "JA")]
    public class TokineVowelLead : DiffSingerJapanesePhonemizer {
        static bool Vowel(string symbol) => symbol is "a" or "e" or "i" or "o" or "u";

        protected override void ProcessPart(Note[][] phrase) {
            base.ProcessPart(phrase);
            for (int i = 1; i < phrase.Length; i++) {
                var current = phrase[i][0];
                var previous = phrase[i - 1];
                var previousEnd = previous[^1].position + previous[^1].duration;
                if (previousEnd != current.position ||
                    !partResult.TryGetValue(current.position, out var phones) ||
                    phones.Count != 1 || !Vowel(phones[0].Item1) ||
                    !partResult.TryGetValue(previous[0].position, out var before) ||
                    before.Count == 0 || !Vowel(before[^1].Item1)) continue;
                var onset = timeAxis.TickPosToMsPos(current.position + phones[0].Item2);
                var tailStart = timeAxis.TickPosToMsPos(previous[0].position + before[^1].Item2);
                // Leave most of a short preceding vowel intact; never cross its onset.
                var lead = LeadMilliseconds(onset - tailStart);
                var shifted = timeAxis.MsPosToTickPos(onset - lead) - current.position;
                phones[0] = Tuple.Create(phones[0].Item1, shifted);
            }
        }

        public static double LeadMilliseconds(double previousTailMs) =>
            Math.Min(20.0, Math.Max(0.0, previousTailMs * 0.25));
    }
}
