# Phrase onset timing

`phrase_onset_timing.py SOURCE_DURATION OUTPUT_DURATION` extends a duration graph already processed by `long_note_timing.py`. It bounds phrase-initial consonants using the existing consonant limits and a 16-frame cluster limit. SP, vowel predictions and later groups are unchanged. The acoustic model and trained weights are unchanged. Natural pre-vowel lead remains intentional.

Run `check_phrase_onsets.py ORIGINAL_BANK CANDIDATE_DURATION REPORT_JSON` before installation. The original bank must contain the duration model before this patch. It checks 3,888 model conditions plus four cluster/vowel cases.

The portable graph does not know the rest between score notes. For very short gaps, the accompanying OpenUtau host patch also bounds the lead by the rest plus at most 30% / 70ms of the preceding note. That host change is maintained in `sotwat/openutau-logic-au`, with a 54,000-condition real-phonemizer test. Ordinary connected notes are excluded from the host rest limiter.

This is a general timing rule, without song, lyric or score-position exceptions. Timing regression checks do not guarantee perceptual quality for every possible score.
