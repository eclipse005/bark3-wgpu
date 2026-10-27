//! Subtitle segmentation: words -> timed segments.
//!
//! The decoder emits word spans on a fixed 80 ms frame grid (see
//! [`crate::transcript`]). Subtitle consumers want segments: a phrase that is
//! shown for a few seconds. Grouping is a display rule, not an acoustic one —
//! no re-alignment, no interpolation, every segment edge is a word edge the
//! model already produced.
//!
//! Break rules, in order of precedence:
//!
//! * a silence longer than [`Limits::gap_seconds`] ends the segment;
//! * a sentence-final word (`.`/`!`/`?`/`…`, ignoring a trailing quote) ends it
//!   once the segment carries [`SENTENCE_MIN_CHARS`] characters;
//! * a clause mark (`,`/`;`/`:`/`—`) ends it once it is over
//!   [`CLAUSE_MIN_CHARS`] characters — the fallback that keeps a long sentence
//!   from being cut mid-phrase;
//! * a segment never exceeds [`Limits::max_seconds`] or [`Limits::max_chars`].
//!
//! Short segments are padded to [`MIN_SECONDS`] of screen time without ever
//! running into the next segment or past the end of the audio.

use serde::Serialize;

use crate::transcript::Word;

/// Minimum screen time for a segment, so a two-word segment does not flash.
pub const MIN_SECONDS: f32 = 1.2;

/// A sentence break needs at least this much text, otherwise `No. Thanks.`
/// would become two one-word segments.
const SENTENCE_MIN_CHARS: usize = 24;

/// A clause break only applies once the segment is already long.
const CLAUSE_MIN_CHARS: usize = 60;

/// Segmentation policy. [`Default`] is what the CLI uses and what the
/// regression baseline pins.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Limits {
    /// Silence between words that closes a segment instead of joining across it.
    pub gap_seconds: f32,
    /// Soft cap on segment text length in characters.
    pub max_chars: usize,
    /// Hard cap on segment duration in seconds.
    pub max_seconds: f32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            gap_seconds: 1.0,
            max_chars: 84,
            max_seconds: 7.0,
        }
    }
}

/// One subtitle event: what to show, and when.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Segment {
    /// Start in seconds from the start of the caller's audio.
    pub start: f32,
    /// End in seconds; never past the next segment or past the audio.
    pub end: f32,
    /// Segment text: the words joined with single spaces, no line breaks.
    pub text: String,
}

/// Groups words into segments under `limits`.
///
/// `words` must be in reading order with non-decreasing start times, which is
/// what [`crate::transcript::TranscribeResult`] guarantees. `duration` is the
/// audio length and only caps the padded segment ends.
pub fn words_to_segments(words: &[Word], duration: f32, limits: &Limits) -> Vec<Segment> {
    let groups = group_indices(words, limits);
    let mut segments = Vec::with_capacity(groups.len());
    for (i, &(first, last)) in groups.iter().enumerate() {
        let start = words[first].start;
        let spoken_end = words[last].end;
        // Screen time may extend past the last token, but never into the next
        // segment, past the audio, or backwards over the tokens themselves.
        let ceiling = groups
            .get(i + 1)
            .map_or(f32::INFINITY, |&(next, _)| words[next].start)
            .min(duration.max(spoken_end));
        let end = (start + MIN_SECONDS)
            .max(spoken_end)
            .min(ceiling)
            .max(spoken_end);
        let text = words[first..=last]
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        segments.push(Segment { start, end, text });
    }
    segments
}

/// Inclusive index ranges of the words in each segment.
fn group_indices(words: &[Word], limits: &Limits) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut chars = 0;
    for (i, word) in words.iter().enumerate() {
        let len = word.text.chars().count();
        if i == start {
            chars = len;
            continue;
        }
        let prev = &words[i - 1];
        let candidate = chars + 1 + len;
        let span = prev.end - words[start].start;
        if ends_segment(prev, word, span, candidate, limits) {
            groups.push((start, i - 1));
            start = i;
            chars = len;
        } else {
            chars = candidate;
        }
    }
    if start < words.len() {
        groups.push((start, words.len() - 1));
    }
    groups
}

fn ends_segment(prev: &Word, next: &Word, span: f32, chars: usize, limits: &Limits) -> bool {
    next.start - prev.end > limits.gap_seconds
        || span >= limits.max_seconds
        || chars > limits.max_chars
        || (ends_sentence(&prev.text) && chars >= SENTENCE_MIN_CHARS)
        || (ends_clause(&prev.text) && chars > CLAUSE_MIN_CHARS)
}

/// Last character of the text, skipping closing quotes and brackets so
/// `strategy."` still counts as a sentence end.
fn last_meaningful(text: &str) -> Option<char> {
    text.chars()
        .rev()
        .find(|c| !matches!(c, '"' | '\'' | '”' | '’' | ')' | ']' | '}'))
}

fn ends_sentence(text: &str) -> bool {
    matches!(last_meaningful(text), Some('.' | '!' | '?' | '…'))
}

fn ends_clause(text: &str) -> bool {
    matches!(last_meaningful(text), Some(',' | ';' | ':' | '—'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f32, end: f32) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
        }
    }

    fn segment(words: &[Word], duration: f32) -> Vec<Segment> {
        words_to_segments(words, duration, &Limits::default())
    }

    #[test]
    fn silence_ends_a_segment() {
        let words = [word("Hello", 1.0, 1.4), word("there", 2.6, 3.0)];
        let segments = segment(&words, 5.0);
        assert_eq!(segments.len(), 2);
        // Padded to the minimum screen time, which still ends before the next
        // segment starts.
        assert_eq!(segments[0].end, 1.0 + MIN_SECONDS);
        assert_eq!(segments[1].start, 2.6);
    }

    #[test]
    fn sentence_end_breaks_only_with_enough_text() {
        let short = [word("Yes.", 1.0, 1.4), word("Right", 1.5, 1.9)];
        assert_eq!(segment(&short, 4.0).len(), 1);

        let long = [
            word("this", 1.0, 1.2),
            word("is", 1.2, 1.3),
            word("the", 1.3, 1.4),
            word("complete", 1.4, 1.9),
            word("answer.", 1.9, 2.2),
            word("Next", 2.3, 2.7),
        ];
        assert_eq!(segment(&long, 4.0).len(), 2);
    }

    #[test]
    fn sentence_end_ignores_a_trailing_quote() {
        let words = [
            word("he", 1.0, 1.2),
            word("shouted", 1.2, 1.4),
            word("it", 1.4, 1.6),
            word("very", 1.6, 1.8),
            word("loudly.\"", 1.8, 2.2),
            word("Next", 2.3, 2.7),
        ];
        assert_eq!(segment(&words, 4.0).len(), 2);
    }

    #[test]
    fn clause_break_only_applies_to_long_segments() {
        let short = [
            word("Well", 1.0, 1.2),
            word("then,", 1.2, 1.5),
            word("fine", 1.5, 1.8),
        ];
        assert_eq!(segment(&short, 3.0).len(), 1);
    }

    #[test]
    fn segment_length_is_capped_by_characters() {
        let words: Vec<Word> = (0..12)
            .map(|i| word("abcdefghij", i as f32, i as f32 + 0.5))
            .collect();
        let segments = segment(&words, 20.0);
        assert!(
            segments.iter().all(|s| s.text.chars().count() <= 84),
            "{:?}",
            segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>()
        );
        assert!(segments.len() > 1);
    }

    #[test]
    fn segment_length_is_capped_by_time() {
        let words: Vec<Word> = (0..40)
            .map(|i| word("word", i as f32 * 0.4, i as f32 * 0.4 + 0.4))
            .collect();
        let segments = segment(&words, 20.0);
        assert!(
            segments
                .iter()
                .all(|s| s.end - s.start <= Limits::default().max_seconds + 0.4)
        );
    }

    #[test]
    fn plain_words_share_one_segment() {
        let words: Vec<Word> = (0..8)
            .map(|i| word("alpha", i as f32 * 0.4, i as f32 * 0.4 + 0.35))
            .collect();
        let segments = segment(&words, 5.0);
        assert_eq!(segments.len(), 1, "{segments:?}");
        assert_eq!(
            segments[0].text,
            "alpha alpha alpha alpha alpha alpha alpha alpha"
        );
    }

    #[test]
    fn short_segments_get_minimum_screen_time() {
        let words = [word("Hi", 1.0, 1.08)];
        let segments = segment(&words, 10.0);
        assert_eq!(segments[0].end, 1.0 + MIN_SECONDS);
    }

    #[test]
    fn a_segment_never_runs_past_the_audio() {
        let words = [word("Bye", 9.6, 9.9)];
        let segments = segment(&words, 10.0);
        assert_eq!(segments[0].end, 10.0);
    }

    #[test]
    fn every_word_lands_in_exactly_one_segment() {
        let words: Vec<Word> = (0..50)
            .map(|i| word("w", i as f32 * 0.3, i as f32 * 0.3 + 0.25))
            .collect();
        let segments = segment(&words, 20.0);
        let joined = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined.split(' ').count(), words.len());
        assert!(segments.windows(2).all(|pair| pair[0].end <= pair[1].start));
    }

    #[test]
    fn custom_limits_are_honoured() {
        let words: Vec<Word> = (0..6)
            .map(|i| word("word", i as f32, i as f32 + 0.5))
            .collect();
        let tight = Limits {
            gap_seconds: 1.0,
            max_chars: 10,
            max_seconds: 7.0,
        };
        let segments = words_to_segments(&words, 5.0, &tight);
        assert!(segments.iter().all(|s| s.text.chars().count() <= 10));
        assert!(segments.len() >= 3);
    }

    #[test]
    fn empty_input_has_no_segments() {
        assert!(segment(&[], 1.0).is_empty());
    }
}
