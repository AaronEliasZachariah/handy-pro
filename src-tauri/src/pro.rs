//! handy-pro: the "Pro" app-aware post-processing layer.
//!
//! Given the foreground [`AppContext`], resolve a profile via the user's rules, compose the
//! system prompt (shared base cleanup + the profile instruction + a vocabulary hint), and
//! apply a conservative vocabulary fixup to the model's output. All of this is additive and
//! only engages when `pro_app_aware_enabled` is set; otherwise upstream behavior is untouched.

use crate::app_context::AppContext;
use crate::settings::{AppSettings, ProMatchType};
use log::debug;

/// The shared cleanup instruction every profile builds on. Mirrors Handy's default prompt
/// intent but is phrased for a chat "system" message (the transcript is the user message).
pub const BASE_CLEANUP: &str = "You are a dictation post-processor. Clean up the user's raw \
speech-to-text transcript: remove filler words (um, uh, \"like\"/\"you know\" used as filler), \
false starts, and spoken repetitions; fix obvious dictation and transcription errors; correct \
capitalization and punctuation, converting spoken punctuation to symbols where clearly intended. \
Output ONLY the cleaned text — no preamble, labels, explanations, or surrounding quotes. Preserve \
the speaker's meaning, intent, and original language.";

/// Resolve the active profile key for the given foreground app using the user's rules.
/// First enabled rule whose (process|title) substring matches wins; otherwise the default.
pub fn resolve_profile_key(settings: &AppSettings, ctx: Option<&AppContext>) -> String {
    if let Some(ctx) = ctx {
        let process = ctx.process_name.to_lowercase();
        let title = ctx.window_title.to_lowercase();
        for rule in &settings.pro_app_rules {
            if !rule.enabled {
                continue;
            }
            let pattern = rule.pattern.trim().to_lowercase();
            if pattern.is_empty() {
                continue;
            }
            let haystack = match rule.match_type {
                ProMatchType::Process => &process,
                ProMatchType::Title => &title,
            };
            if haystack.contains(&pattern) {
                debug!(
                    "Pro: matched rule '{}' ({:?} contains '{}') -> profile '{}'",
                    rule.id, rule.match_type, pattern, rule.profile_key
                );
                return rule.profile_key.clone();
            }
        }
    }
    settings.pro_default_profile.clone()
}

/// The (enabled) profile instruction for a key, if any.
fn profile_prompt<'a>(settings: &'a AppSettings, key: &str) -> Option<&'a str> {
    settings
        .pro_profiles
        .iter()
        .find(|p| p.key == key && p.enabled)
        .map(|p| p.prompt.as_str())
}

/// A one-line vocabulary hint for the system prompt, or None when the vocab is empty.
fn vocab_hint(settings: &AppSettings) -> Option<String> {
    let pairs: Vec<String> = settings
        .pro_vocabulary
        .iter()
        .filter(|e| !e.from.trim().is_empty() && !e.to.trim().is_empty())
        .map(|e| format!("\"{}\" -> \"{}\"", e.from.trim(), e.to.trim()))
        .collect();
    if pairs.is_empty() {
        None
    } else {
        Some(format!(
            "Domain vocabulary — if the transcript contains a near-miss of any term on the left, \
             correct it to the exact spelling on the right: {}.",
            pairs.join("; ")
        ))
    }
}

/// Compose the full system prompt for a profile: base cleanup + profile instruction + vocab.
pub fn build_pro_system_prompt(settings: &AppSettings, profile_key: &str) -> String {
    let mut parts = vec![BASE_CLEANUP.to_string()];
    if let Some(prompt) = profile_prompt(settings, profile_key) {
        if !prompt.trim().is_empty() {
            parts.push(prompt.trim().to_string());
        }
    }
    if let Some(hint) = vocab_hint(settings) {
        parts.push(hint);
    }
    parts.join("\n\n")
}

/// Conservative, whole-word, case-insensitive vocabulary fixup applied to model output.
pub fn apply_vocabulary(text: &str, settings: &AppSettings) -> String {
    let mut out = text.to_string();
    for entry in &settings.pro_vocabulary {
        let from = entry.from.trim();
        let to = entry.to.trim();
        if from.is_empty() || to.is_empty() {
            continue;
        }
        let pattern = format!(r"(?i)\b{}\b", regex::escape(from));
        if let Ok(re) = regex::Regex::new(&pattern) {
            // NoExpand: treat `to` literally so a `$` in a replacement isn't read as a group ref.
            out = re.replace_all(&out, regex::NoExpand(to)).into_owned();
        }
    }
    out
}
