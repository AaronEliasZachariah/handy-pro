//! handy-pro: foreground application detection (Windows).
//!
//! Captures the process name + window title of the app that currently has focus, so the
//! post-processor can pick an app-aware profile. On non-Windows targets this degrades to
//! `None` and the Pro layer falls back to the default profile.

use serde::{Deserialize, Serialize};
use specta::Type;
use std::sync::Mutex;

/// The foreground app at dictation time.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Type)]
pub struct AppContext {
    /// Executable file name without extension, lower-cased is used for matching
    /// (the raw, original-case name is kept here for display).
    pub process_name: String,
    pub window_title: String,
}

/// What the routing resolved for the last dictation — surfaced to the live-test panel so
/// the user can see which app/profile was detected without re-dictating.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Type)]
pub struct DetectedContext {
    pub process_name: String,
    pub window_title: String,
    pub profile_key: String,
}

static LAST_DETECTED: Mutex<Option<DetectedContext>> = Mutex::new(None);

/// Record the most recently detected app + resolved profile (called from the pipeline).
pub fn set_last_detected(ctx: DetectedContext) {
    if let Ok(mut guard) = LAST_DETECTED.lock() {
        *guard = Some(ctx);
    }
}

/// Read the most recently detected app + resolved profile (for the live-test panel).
pub fn last_detected() -> Option<DetectedContext> {
    LAST_DETECTED.lock().ok().and_then(|g| g.clone())
}

#[cfg(target_os = "windows")]
pub fn foreground_app() -> Option<AppContext> {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::{CloseHandle, MAX_PATH};
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
    };

    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.0.is_null() {
            return None;
        }

        // Window title
        let mut window_title = String::new();
        let len = GetWindowTextLengthW(hwnd);
        if len > 0 {
            let mut buf = vec![0u16; (len + 1) as usize];
            let copied = GetWindowTextW(hwnd, &mut buf);
            if copied > 0 {
                window_title = String::from_utf16_lossy(&buf[..copied as usize]);
            }
        }

        // Process name via PID -> QueryFullProcessImageNameW
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, Some(&mut pid as *mut u32));
        let mut process_name = String::new();
        if pid != 0 {
            if let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) {
                let mut buf = vec![0u16; MAX_PATH as usize];
                let mut size = buf.len() as u32;
                let ok = QueryFullProcessImageNameW(
                    handle,
                    PROCESS_NAME_WIN32,
                    PWSTR(buf.as_mut_ptr()),
                    &mut size,
                );
                if ok.is_ok() && size > 0 {
                    let full = String::from_utf16_lossy(&buf[..size as usize]);
                    // Strip directory and extension: "...\Code.exe" -> "Code"
                    let file = full.rsplit(['\\', '/']).next().unwrap_or(&full).to_string();
                    process_name = file
                        .strip_suffix(".exe")
                        .or_else(|| file.strip_suffix(".EXE"))
                        .unwrap_or(&file)
                        .to_string();
                }
                let _ = CloseHandle(handle);
            }
        }

        if process_name.is_empty() && window_title.is_empty() {
            return None;
        }
        Some(AppContext {
            process_name,
            window_title,
        })
    }
}

#[cfg(not(target_os = "windows"))]
pub fn foreground_app() -> Option<AppContext> {
    None
}
