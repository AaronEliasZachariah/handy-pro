//! handy-pro: Tauri commands for the app-aware post-processing layer (settings mutators,
//! foreground-app introspection, and the live-test entry point).

use crate::app_context::{AppContext, DetectedContext};
use crate::settings::{get_settings, write_settings, ProAppRule, ProProfile, ProVocabEntry};
use tauri::AppHandle;

#[tauri::command]
#[specta::specta]
pub fn set_pro_app_aware_enabled(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.pro_app_aware_enabled = enabled;
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_pro_profiles(app: AppHandle, profiles: Vec<ProProfile>) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.pro_profiles = profiles;
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_pro_app_rules(app: AppHandle, rules: Vec<ProAppRule>) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.pro_app_rules = rules;
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_pro_default_profile(app: AppHandle, profile_key: String) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.pro_default_profile = profile_key;
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_pro_vocabulary(app: AppHandle, vocabulary: Vec<ProVocabEntry>) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.pro_vocabulary = vocabulary;
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_pro_timeout_ms(app: AppHandle, timeout_ms: u64) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.pro_timeout_ms = timeout_ms;
    write_settings(&app, settings);
    Ok(())
}

/// The app + profile detected during the most recent dictation (for the live-test panel).
#[tauri::command]
#[specta::specta]
pub fn get_last_app_context() -> Result<Option<DetectedContext>, String> {
    Ok(crate::app_context::last_detected())
}

/// The app currently in the foreground right now (used for diagnostics / a "detect" button).
/// Note: when called from the settings window this returns handy itself.
#[tauri::command]
#[specta::specta]
pub fn get_foreground_app_context() -> Result<Option<AppContext>, String> {
    Ok(crate::app_context::foreground_app())
}

/// Run the Pro post-processor on pasted text for a chosen profile (the live-test panel).
/// Returns the cleaned text, or a human-readable error so the user can debug their setup.
#[tauri::command]
#[specta::specta]
pub async fn pro_test_post_process(
    app: AppHandle,
    raw_text: String,
    profile_key: String,
) -> Result<String, String> {
    let settings = get_settings(&app);
    crate::actions::run_pro_post_process(&settings, &raw_text, &profile_key).await
}
