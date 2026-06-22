#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{get_settings, AppSettings, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{change_tray_icon, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tauri::Manager;
use tauri::{AppHandle, Emitter};

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

async fn post_process_transcription(
    settings: &AppSettings,
    transcription: &str,
    pro_profile: Option<&str>,
) -> Option<String> {
    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    // handy-pro: when the app-aware layer is active, compose the prompt from the resolved
    // profile (base cleanup + profile instruction + vocabulary) instead of the single global
    // selected prompt. Otherwise behave exactly like upstream.
    let pro_active = settings.pro_app_aware_enabled && pro_profile.is_some();

    let system_prompt: String;
    let legacy_prompt: String;
    if pro_active {
        let profile_key = pro_profile.unwrap();
        system_prompt = crate::pro::build_pro_system_prompt(settings, profile_key);
        legacy_prompt = format!("{}\n\nTranscript:\n{}", system_prompt, transcription);
        debug!("Pro post-processing active for profile '{}'", profile_key);
    } else {
        let selected_prompt_id = match &settings.post_process_selected_prompt_id {
            Some(id) => id.clone(),
            None => {
                debug!("Post-processing skipped because no prompt is selected");
                return None;
            }
        };

        let prompt = match settings
            .post_process_prompts
            .iter()
            .find(|prompt| prompt.id == selected_prompt_id)
        {
            Some(prompt) => prompt.prompt.clone(),
            None => {
                debug!(
                    "Post-processing skipped because prompt '{}' was not found",
                    selected_prompt_id
                );
                return None;
            }
        };

        if prompt.trim().is_empty() {
            debug!("Post-processing skipped because the selected prompt is empty");
            return None;
        }

        system_prompt = build_system_prompt(&prompt);
        legacy_prompt = prompt.replace("${output}", transcription);
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Disable reasoning for providers where post-processing rarely benefits from it.
    // - custom: top-level reasoning_effort (works for local OpenAI-compat servers)
    // - openrouter: nested reasoning object; exclude:true also keeps reasoning text
    //   out of the response so it can't pollute structured-output JSON parsing
    let (reasoning_effort, reasoning) = match provider.id.as_str() {
        "custom" | "ollama" => (Some("none".to_string()), None),
        "openrouter" => (
            None,
            Some(crate::llm_client::ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
        ),
        _ => (None, None),
    };

    // handy-pro: apply the configured timeout to the whole post-process call so a slow or
    // hung model never stalls dictation — on timeout the caller pastes the raw transcript.
    let timeout_ms = settings.pro_timeout_ms;

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            Some(finalize(
                                strip_invisible_chars(&result),
                                settings,
                                pro_active,
                            ))
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        let call = crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt.clone()),
            Some(json_schema),
            reasoning_effort.clone(),
            reasoning.clone(),
        );

        match run_with_timeout(call, timeout_ms).await {
            Ok(Ok(Some(content))) => {
                let parsed = match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(json) => json
                        .get(TRANSCRIPTION_FIELD)
                        .and_then(|t| t.as_str())
                        .map(strip_invisible_chars)
                        .unwrap_or_else(|| {
                            error!("Structured output response missing 'transcription' field");
                            strip_invisible_chars(&content)
                        }),
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        strip_invisible_chars(&content)
                    }
                };
                return Some(finalize(parsed, settings, pro_active));
            }
            Ok(Ok(None)) => {
                error!("LLM API response has no content");
                return None;
            }
            Ok(Err(e)) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
            Err(_) => {
                warn!(
                    "Post-processing timed out after {}ms; using raw transcript",
                    timeout_ms
                );
                return None;
            }
        }
    }

    // Non-structured mode. For the Pro path, send the composed instruction as a SYSTEM message
    // and the transcript as the USER message — small local models (e.g. transcriber fine-tunes)
    // echo the instructions back if everything is crammed into one user turn. Upstream keeps its
    // single-user-message behavior (system = None) so its prompt templates are unaffected.
    let (user_content, system_msg) = if pro_active {
        (transcription.to_string(), Some(system_prompt.clone()))
    } else {
        (legacy_prompt, None)
    };
    debug!("Non-structured post-processing (pro={})", pro_active);

    let call = crate::llm_client::send_chat_completion_with_schema(
        &provider,
        api_key,
        &model,
        user_content,
        system_msg,
        None,
        reasoning_effort,
        reasoning,
    );

    match run_with_timeout(call, timeout_ms).await {
        Ok(Ok(Some(content))) => Some(finalize(
            strip_invisible_chars(&content),
            settings,
            pro_active,
        )),
        Ok(Ok(None)) => {
            error!("LLM API response has no content");
            None
        }
        Ok(Err(e)) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id, e
            );
            None
        }
        Err(_) => {
            warn!(
                "Post-processing timed out after {}ms; using raw transcript",
                timeout_ms
            );
            None
        }
    }
}

/// handy-pro: await a future with an optional timeout (`0` = no timeout).
async fn run_with_timeout<F, T>(fut: F, timeout_ms: u64) -> Result<T, tokio::time::error::Elapsed>
where
    F: std::future::Future<Output = T>,
{
    if timeout_ms == 0 {
        Ok(fut.await)
    } else {
        tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), fut).await
    }
}

/// handy-pro: apply the vocabulary fixup when the Pro layer produced the text.
fn finalize(text: String, settings: &AppSettings, pro_active: bool) -> String {
    if pro_active {
        crate::pro::apply_vocabulary(&text, settings)
    } else {
        text
    }
}

/// handy-pro: run the Pro post-processor for the live-test panel, surfacing real errors so the
/// user can debug their setup. Uses a generous fixed timeout since the first local model load
/// can take longer than the dictation-time budget.
pub(crate) async fn run_pro_post_process(
    settings: &AppSettings,
    text: &str,
    profile_key: &str,
) -> Result<String, String> {
    let provider = settings
        .active_post_process_provider()
        .cloned()
        .ok_or_else(|| "No post-processing provider is selected.".to_string())?;
    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();
    if model.trim().is_empty() {
        return Err(format!(
            "Provider '{}' has no model configured.",
            provider.label
        ));
    }
    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();
    let system_prompt = crate::pro::build_pro_system_prompt(settings, profile_key);
    let (reasoning_effort, reasoning) = match provider.id.as_str() {
        "custom" | "ollama" => (Some("none".to_string()), None),
        "openrouter" => (
            None,
            Some(crate::llm_client::ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
        ),
        _ => (None, None),
    };
    let timeout_ms = 30_000;

    let result = if provider.supports_structured_output {
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": { (TRANSCRIPTION_FIELD): { "type": "string" } },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });
        let call = crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key,
            &model,
            text.to_string(),
            Some(system_prompt),
            Some(json_schema),
            reasoning_effort,
            reasoning,
        );
        match run_with_timeout(call, timeout_ms).await {
            Ok(Ok(Some(content))) => match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(json) => json
                    .get(TRANSCRIPTION_FIELD)
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or(content),
                Err(_) => content,
            },
            Ok(Ok(None)) => return Err("The model returned an empty response.".to_string()),
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(format!("Timed out after {}ms.", timeout_ms)),
        }
    } else {
        // System message + user message (not one crammed user turn) so small local models
        // follow the instruction instead of echoing it.
        let call = crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key,
            &model,
            text.to_string(),
            Some(system_prompt),
            None,
            reasoning_effort,
            reasoning,
        );
        match run_with_timeout(call, timeout_ms).await {
            Ok(Ok(Some(content))) => content,
            Ok(Ok(None)) => return Err("The model returned an empty response.".to_string()),
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(format!("Timed out after {}ms.", timeout_ms)),
        }
    };

    Ok(crate::pro::apply_vocabulary(
        &strip_invisible_chars(&result),
        settings,
    ))
}

async fn maybe_convert_chinese_variant(
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    // Check if language is set to Simplified or Traditional Chinese
    let is_simplified = settings.selected_language == "zh-Hans";
    let is_traditional = settings.selected_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("selected_language is not Simplified or Traditional Chinese; skipping translation");
        return None;
    }

    debug!(
        "Starting Chinese translation using OpenCC for language: {}",
        settings.selected_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
    app_context: Option<crate::app_context::AppContext>,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;

    if let Some(converted_text) = maybe_convert_chinese_variant(&settings, transcription).await {
        final_text = converted_text;
    }

    if post_process {
        // handy-pro: resolve the app-aware profile (if the Pro layer is on) and record what was
        // detected so the settings/live-test panel can show it without re-dictating.
        let pro_profile = if settings.pro_app_aware_enabled {
            let key = crate::pro::resolve_profile_key(&settings, app_context.as_ref());
            let ctx = app_context.clone().unwrap_or_default();
            crate::app_context::set_last_detected(crate::app_context::DetectedContext {
                process_name: ctx.process_name,
                window_title: ctx.window_title,
                profile_key: key.clone(),
            });
            Some(key)
        } else {
            None
        };

        if let Some(processed_text) =
            post_process_transcription(&settings, &final_text, pro_profile.as_deref()).await
        {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;

            // Record the prompt used (for history). Pro mode uses the composed profile prompt.
            if let Some(profile_key) = &pro_profile {
                post_process_prompt =
                    Some(crate::pro::build_pro_system_prompt(&settings, profile_key));
            } else if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| &prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    } else if final_text != transcription {
        post_processed_text = Some(final_text.clone());
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Load ASR model and VAD model in parallel
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });

        let binding_id = binding_id.to_string();
        change_tray_icon(app, TrayIconState::Recording);
        show_recording_overlay(app);

        // Get the microphone mode to determine audio feedback timing
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<String> = None;
        if is_always_on {
            // Always-on mode: Play audio feedback immediately, then apply mute after sound finishes
            debug!("Always-on mode: Playing audio feedback immediately");
            let rm_clone = Arc::clone(&rm);
            let app_clone = app.clone();
            // The blocking helper exits immediately if audio feedback is disabled,
            // so we can always reuse this thread to ensure mute happens right after playback.
            std::thread::spawn(move || {
                play_feedback_sound_blocking(&app_clone, SoundType::Start);
                rm_clone.apply_mute();
            });

            if let Err(e) = rm.try_start_recording(&binding_id) {
                debug!("Recording failed: {}", e);
                recording_error = Some(e);
            }
        } else {
            // On-demand mode: Start recording first, then play audio feedback, then apply mute
            // This allows the microphone to be activated before playing the sound
            debug!("On-demand mode: Starting recording first, then audio feedback");
            let recording_start_time = Instant::now();
            match rm.try_start_recording(&binding_id) {
                Ok(()) => {
                    debug!("Recording started in {:?}", recording_start_time.elapsed());
                    // Small delay to ensure microphone stream is active
                    let app_clone = app.clone();
                    let rm_clone = Arc::clone(&rm);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        debug!("Handling delayed audio feedback/mute sequence");
                        // Helper handles disabled audio feedback by returning early, so we reuse it
                        // to keep mute sequencing consistent in every mode.
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                        rm_clone.apply_mute();
                    });
                }
                Err(e) => {
                    debug!("Failed to start recording: {}", e);
                    recording_error = Some(e);
                }
            }
        }

        if recording_error.is_none() {
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);
        } else {
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            utils::hide_recording_overlay(app);
            change_tray_icon(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        change_tray_icon(app, TrayIconState::Transcribing);
        show_transcribing_overlay(app);

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;

        // handy-pro: capture the foreground app now, while the user's target window still has
        // focus (the recording overlay is non-activating, so this reflects the real target app).
        let app_context = if post_process {
            crate::app_context::foreground_app()
        } else {
            None
        };

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id) {
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save
                    let transcription_time = Instant::now();
                    let transcription_result = tm.transcribe(samples);

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                transcription
                            );

                            if post_process {
                                show_processing_overlay(&ah);
                            }
                            let processed = process_transcription_output(
                                &ah,
                                &transcription,
                                post_process,
                                app_context.clone(),
                            )
                            .await;

                            // Save to history if WAV was saved
                            if wav_saved {
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription,
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            if processed.final_text.is_empty() {
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let final_text = processed.final_text;
                                ah.run_on_main_thread(move || {
                                    match utils::paste(final_text, ah_clone.clone()) {
                                        Ok(()) => debug!(
                                            "Text pasted successfully in {:?}",
                                            paste_time.elapsed()
                                        ),
                                        Err(e) => {
                                            error!("Failed to paste transcription: {}", e);
                                            let _ = ah_clone.emit("paste-error", ());
                                        }
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    change_tray_icon(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    change_tray_icon(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            debug!("Global Shortcut Transcription error: {}", err);
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            utils::hide_recording_overlay(&ah);
                            change_tray_icon(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                utils::hide_recording_overlay(&ah);
                change_tray_icon(&ah, TrayIconState::Idle);
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});
