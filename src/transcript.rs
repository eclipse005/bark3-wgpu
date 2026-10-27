//! Transcript assembly: decoder tokens -> text, words, segments, and JSON.
//!
//! The decoder emits SentencePiece pieces on a fixed 80 ms frame grid. A piece
//! that starts with [`SPACE_MARKER`] (`▁`, the metaspace marker) opens a new
//! word; every other piece continues the current one. Two views come out of
//! that single stream:
//!
//! * [`TranscribeResult::text`] — what the reference tokenizer prints, i.e.
//!   `batch_decode(skip_special_tokens=True)`. This stays byte-identical so the
//!   regression gate compares the model, not a normalisation of it.
//! * [`TranscribeResult::words`] — the same pieces grouped into words with
//!   start/end times, and [`TranscribeResult::segments`] — those words grouped
//!   into subtitle segments (see [`crate::segment`]).
//!
//! Grouping rules for words:
//!
//! * `▁` + text opens a word.
//! * A piece without `▁` always continues the current word, so `▁Hel` + `lo`
//!   is one word and `▁don` + `'t` is `don't`.
//! * Punctuation attaches to the word before it, even when the piece carries a
//!   space marker (`▁-` is a hyphen glued to the previous word, not a word of
//!   its own). A bare `▁` is a word break with no content.
//! * Special tokens are dropped, exactly like `skip_special_tokens=True`.

use std::fmt;
use std::sync::OnceLock;

use serde::Serialize;

use crate::cfg;
use crate::segment::{self, Limits, Segment};
use crate::tdt::Token;

/// The metaspace word marker SentencePiece uses in place of a space.
pub const SPACE_MARKER: char = '▁';

/// Seconds per encoder frame: the conv front end subsamples the 10 ms mel
/// frames by 8.
pub const FRAME_SECONDS: f32 = 0.08;

/// Silence the reference pipeline prepends before feature extraction. Decode
/// frames are indexed from the start of that padded signal, so timestamps move
/// back by this much to land on the caller's audio.
pub const LEAD_PAD_SECONDS: f32 = 0.25;

/// One recognised word with the span of audio it covers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Word {
    /// Word text with punctuation attached and no surrounding spaces.
    pub text: String,
    /// Start of the first token, in seconds from the start of the audio.
    pub start: f32,
    /// End of the last token (its frame plus its TDT duration), in seconds.
    pub end: f32,
}

/// A finished transcription: the reference text, the word timeline, the
/// subtitle segments, and the raw decode for callers that need ids or frames.
#[derive(Debug, Clone, Default)]
pub struct TranscribeResult {
    /// Reference rendering of the token stream (byte-exact gate target).
    pub text: String,
    /// Word-level view with timestamps, in reading order.
    pub words: Vec<Word>,
    /// Subtitle segments built from `words`.
    pub segments: Vec<Segment>,
    /// Raw decoder output, including frames and durations.
    pub tokens: Vec<Token>,
    /// Length of the transcribed audio in seconds (without the lead pad).
    pub duration: f32,
}

impl TranscribeResult {
    /// Assembles every view from the decoder output.
    pub fn new(tokens: Vec<Token>, vocab: &[String], duration: f32, limits: &Limits) -> Self {
        let text = normalize_spacing(&Self::text_from_tokens(&tokens, vocab));
        let words = Self::words_from_tokens(&tokens, vocab, duration);
        let mut segments = segment::words_to_segments(&words, duration, limits);
        for segment in &mut segments {
            segment.text = normalize_spacing(&segment.text);
        }
        Self {
            text,
            words,
            segments,
            tokens,
            duration,
        }
    }

    /// The reference transcript: pieces concatenated with `▁` as a space,
    /// specials dropped, one leading space trimmed.
    pub fn text_from_tokens(tokens: &[Token], vocab: &[String]) -> String {
        let mut text = String::new();
        for token in tokens {
            let piece = piece(vocab, token.id);
            if is_special(token.id, piece) {
                continue;
            }
            text.push_str(&piece.replace(SPACE_MARKER, " "));
        }
        text.trim_start().to_string()
    }

    /// Groups pieces into words and converts their frames to seconds.
    pub fn words_from_tokens(tokens: &[Token], vocab: &[String], duration: f32) -> Vec<Word> {
        let mut words = Vec::new();
        let mut current = String::new();
        let (mut start_frame, mut end_frame) = (0u32, 0u32);

        for token in tokens {
            let piece = piece(vocab, token.id);
            if is_special(token.id, piece) {
                continue;
            }
            let (opens_word, body) = match piece.strip_prefix(SPACE_MARKER) {
                Some(body) => (true, body),
                None => (false, piece),
            };
            // A leading marker on punctuation must not start a word: `▁-` and
            // `▁'` belong to the word before them. A bare digit piece is its own
            // word though: the tokenizer writes `2022` as four unmarked pieces,
            // and gluing them onto the previous word would give that word the
            // number's time span.
            let digit_after_letters = is_digits(body)
                && !current.is_empty()
                && !current.ends_with(|c: char| c.is_ascii_digit());
            if body.is_empty() || (opens_word && !is_punctuation(body)) || digit_after_letters {
                flush(&mut words, &mut current, start_frame, end_frame, duration);
            }
            if body.is_empty() {
                continue;
            }
            if current.is_empty() {
                start_frame = token.frame;
                end_frame = 0;
            }
            current.push_str(body);
            end_frame = end_frame.max(token.frame.saturating_add(token.dur));
        }
        flush(&mut words, &mut current, start_frame, end_frame, duration);
        words
    }

    /// JSON: `{ text, duration, segments: [...], words: [...] }`.
    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct View<'a> {
            text: &'a str,
            duration: f32,
            segments: &'a [Segment],
            words: &'a [Word],
        }
        serde_json::to_string(&View {
            text: &self.text,
            duration: self.duration,
            segments: &self.segments,
            words: &self.words,
        })
        .expect("transcript JSON contains no non-finite floats")
    }
}

impl fmt::Display for TranscribeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// Concatenate vocab pieces, skipping special tokens — the reference
/// `batch_decode(skip_special_tokens=True)` rendering, before the spacing pass.
pub fn tokens_to_text(tokens: &[Token], vocab: &[String]) -> String {
    normalize_spacing(&TranscribeResult::text_from_tokens(tokens, vocab))
}

/// Puts spaces where the tokenizer leaves them out.
///
/// The vocabulary stores digits as bare single-character pieces (ids 234-243:
/// `0`..`9`) and marks a word start with `▁` only on word pieces. SentencePiece
/// decode concatenates pieces as they are, so a year arrives as
/// `thirteenth,2022` and a spoken number can arrive as `March13`. The content is
/// right; only the spacing is missing, and no consumer wants the glued form.
///
/// This pass only ever adds or removes spaces — it never touches a character:
///
/// * a space before `, . ; : ! ? ) ] }` and closing quotes goes away;
/// * a space is added after those marks when a letter or digit follows, except
///   where the mark is inside a number (`1,000`, `3.5`, `12:30`);
/// * a space is added between a run of two or more letters and a following
///   digit (`March13` -> `March 13`), which leaves `H1N1` and `3D` alone.
#[must_use]
pub fn normalize_spacing(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len() + 8);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] == ' ' {
                j += 1;
            }
            let next = chars.get(j).copied();
            let drop = matches!(
                next,
                Some(',' | '.' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '\'' | '"' | '”')
            ) || out.is_empty();
            if !drop && !out.ends_with(' ') {
                out.push(' ');
            }
            i = j;
            continue;
        }
        out.push(c);
        i += 1;
        if matches!(c, ',' | ';' | ':' | '!' | '?') || c == '.' {
            let next = chars.get(i).copied();
            let prev = out.chars().nth_back(1);
            if let Some(next) = next {
                let digits_after = chars[i..].iter().take_while(|d| d.is_ascii_digit()).count();
                // `1,000` is one number (exactly three digits in the group);
                // `13,2022` is a date and wants the space. A colon or a point
                // between two digits stays as it is (`12:30`, `3.5`).
                let in_number = prev.is_some_and(|p| p.is_ascii_digit())
                    && next.is_ascii_digit()
                    && match c {
                        ',' => digits_after == 3,
                        _ => true,
                    };
                if next.is_alphanumeric() && !in_number {
                    out.push(' ');
                }
            }
        }
    }
    letters_before_digits(&out)
}

/// `March13` -> `March 13`, but `H1N1` stays as it is.
fn letters_before_digits(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 4);
    let mut letters = 0usize;
    for (i, &c) in chars.iter().enumerate() {
        if c.is_alphabetic() {
            letters += 1;
        } else if c.is_ascii_digit() {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            if !prev_digit && letters >= 2 {
                out.push(' ');
            }
            letters = 0;
        } else {
            letters = 0;
        }
        out.push(c);
    }
    out
}

/// The model's SentencePiece vocabulary, indexed by token id.
///
/// Bundled with the crate: this is the vocabulary of the checkpoint the code
/// targets, so there is no file to find at run time.
pub fn vocab() -> &'static [String] {
    static VOCAB: OnceLock<Vec<String>> = OnceLock::new();
    VOCAB.get_or_init(|| {
        serde_json::from_str(include_str!("../data/vocab.json"))
            .expect("bundled vocab.json is a JSON array of strings")
    })
}

fn flush(
    words: &mut Vec<Word>,
    current: &mut String,
    start_frame: u32,
    end_frame: u32,
    duration: f32,
) {
    let text = current.trim();
    if !text.is_empty() {
        let start = round_to_ms(seconds_at(start_frame), duration);
        // A zero-duration token still covers the frame it was emitted on, so a
        // word never collapses to a zero-length span.
        let end = round_to_ms(seconds_at(end_frame.max(start_frame + 1)), duration);
        words.push(Word {
            text: text.to_string(),
            start,
            end: end.max(start),
        });
    }
    current.clear();
}

/// Where a decode frame lands on the caller's timeline, before rounding.
fn seconds_at(frame: u32) -> f64 {
    frame as f64 * FRAME_SECONDS as f64 - LEAD_PAD_SECONDS as f64
}

/// Rounds to milliseconds so the API, the JSON and the printed table agree.
fn round_to_ms(seconds: f64, duration: f32) -> f32 {
    let clamped = seconds.clamp(0.0, duration.max(0.0) as f64);
    ((clamped * 1000.0).round() / 1000.0) as f32
}

fn piece(vocab: &[String], id: u32) -> &str {
    vocab.get(id as usize).map_or("", String::as_str)
}

/// Whether a token is model syntax rather than text. Ids cover blank/pad/eos/
/// unk; the string test catches the `<|pnc|>`-style control tokens the model
/// can emit, which the reference tokenizer also treats as special.
fn is_special(id: u32, piece: &str) -> bool {
    matches!(id, cfg::BLANK | cfg::PAD | cfg::EOS | cfg::UNK)
        || (piece.starts_with("<|") && piece.ends_with("|>"))
}

/// True when the piece is one or more ASCII digits and nothing else.
fn is_digits(piece: &str) -> bool {
    !piece.is_empty() && piece.chars().all(|c| c.is_ascii_digit())
}

/// True when the piece has no letters or digits, i.e. it is punctuation and
/// belongs to the word it follows.
fn is_punctuation(piece: &str) -> bool {
    !piece.is_empty() && !piece.chars().any(char::is_alphanumeric)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids 0, 2 and 3 are `<unk>`/`<pad>`/`<eos>` in the model config, so this
    /// table exercises the skip rules without a full 8193-entry vocabulary.
    const VOCAB: &[&str] = &[
        "<unk>",   // 0 UNK
        "<|pnc|>", // 1 control token, dropped by the string test
        "<pad>",   // 2 PAD
        "<eos>",   // 3 EOS
        "▁Hello",  // 4
        "▁there",  // 5
        ",",       // 6
        "▁-",      // 7
        "▁",       // 8 bare marker
        "▁well",   // 9
        "▁known",  // 10
        "'s",      // 11
        "▁Hel",    // 12
        "lo",      // 13
    ];

    fn vocab() -> Vec<String> {
        VOCAB.iter().map(|s| s.to_string()).collect()
    }

    fn token(id: u32, frame: u32, dur: u32) -> Token {
        Token { id, frame, dur }
    }

    fn texts(words: &[Word]) -> Vec<&str> {
        words.iter().map(|w| w.text.as_str()).collect()
    }

    #[test]
    fn merges_subword_pieces_into_words() {
        let tokens = [token(12, 10, 2), token(13, 12, 2), token(4, 14, 2)];
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        assert_eq!(texts(&words), ["Hello", "Hello"]);
        assert_eq!(words[0].start, 0.55);
        assert_eq!(words[0].end, 0.87);
        assert_eq!(words[1].start, 0.87);
    }

    #[test]
    fn punctuation_attaches_to_the_previous_word() {
        let tokens = [token(4, 10, 2), token(6, 12, 1), token(5, 14, 2)];
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        assert_eq!(texts(&words), ["Hello,", "there"]);
        assert_eq!(words[0].end, 0.79);
    }

    #[test]
    fn marked_punctuation_does_not_open_a_word() {
        let tokens = [token(9, 10, 2), token(7, 12, 1), token(10, 14, 2)];
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        assert_eq!(texts(&words), ["well-", "known"]);
    }

    #[test]
    fn apostrophe_continuation_stays_in_the_word() {
        let tokens = [token(4, 10, 2), token(11, 12, 2)];
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        assert_eq!(texts(&words), ["Hello's"]);
    }

    #[test]
    fn specials_are_dropped_from_both_renderings() {
        let tokens = [
            token(0, 8, 1),
            token(2, 9, 1),
            token(1, 9, 1),
            token(4, 10, 2),
            token(3, 20, 0),
        ];
        let text = TranscribeResult::text_from_tokens(&tokens, &vocab());
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        assert_eq!(text, "Hello");
        assert_eq!(texts(&words), ["Hello"]);
    }

    #[test]
    fn bare_marker_only_breaks_the_word() {
        let tokens = [token(4, 10, 2), token(8, 12, 0), token(5, 14, 2)];
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        assert_eq!(texts(&words), ["Hello", "there"]);
    }

    #[test]
    fn zero_duration_token_still_spans_one_frame() {
        let tokens = [token(4, 10, 0)];
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        assert_eq!(words[0].start, 0.55);
        assert_eq!(words[0].end, 0.63);
    }

    #[test]
    fn timestamps_are_clamped_to_the_audio() {
        // Frame 1 sits inside the 0.25 s lead pad; frame 200 is past the end.
        let tokens = [token(4, 1, 1), token(5, 200, 4)];
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 1.0);
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[1].start, 1.0);
        assert_eq!(words[1].end, 1.0);
    }

    #[test]
    fn text_is_words_joined_for_plain_sentences() {
        let tokens = [
            token(12, 10, 2),
            token(13, 12, 2),
            token(4, 14, 2),
            token(6, 16, 1),
        ];
        let text = TranscribeResult::text_from_tokens(&tokens, &vocab());
        let words = TranscribeResult::words_from_tokens(&tokens, &vocab(), 4.0);
        let joined = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(text, joined);
    }

    #[test]
    fn spacing_pass_joins_numbers_to_their_words() {
        // What the tokenizer gives: bare digit pieces, bare commas.
        assert_eq!(normalize_spacing("thirteenth,2022."), "thirteenth, 2022.");
        assert_eq!(
            normalize_spacing("March13,2022 , a man"),
            "March 13, 2022, a man"
        );
        assert_eq!(
            normalize_spacing("costs 1,000 dollars"),
            "costs 1,000 dollars"
        );
        assert_eq!(normalize_spacing("at 12:30 pm"), "at 12:30 pm");
        assert_eq!(normalize_spacing("H1N1 and 3D"), "H1N1 and 3D");
        assert_eq!(normalize_spacing("a 3.5 second clip"), "a 3.5 second clip");
        assert_eq!(normalize_spacing("Hello, there."), "Hello, there.");
        // Content is never touched, only spaces.
        let odd = "Hello  world .";
        assert_eq!(normalize_spacing(odd), "Hello world.");
    }

    #[test]
    fn json_carries_text_duration_segments_and_words() {
        let tokens = [token(4, 10, 2)];
        let result = TranscribeResult::new(tokens.to_vec(), &vocab(), 4.0, &Limits::default());
        let value: serde_json::Value = serde_json::from_str(&result.to_json()).unwrap();
        assert_eq!(value["text"], "Hello");
        assert_eq!(value["duration"], 4.0);
        assert_eq!(value["words"][0]["text"], "Hello");
        assert_eq!(value["words"][0]["start"], 0.55);
        assert_eq!(value["words"][0]["end"], 0.71);
        assert_eq!(value["segments"][0]["text"], "Hello");
        assert_eq!(value["segments"][0]["start"], 0.55);
    }

    #[test]
    fn segments_partition_the_words() {
        let tokens = [
            token(4, 10, 2),
            token(5, 14, 2),
            token(6, 18, 1),
            token(9, 60, 2),
        ];
        let result = TranscribeResult::new(tokens.to_vec(), &vocab(), 6.0, &Limits::default());
        let joined = result
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined, result.text);
        assert!(result.segments.windows(2).all(|p| p[0].end <= p[1].start));
    }
}
