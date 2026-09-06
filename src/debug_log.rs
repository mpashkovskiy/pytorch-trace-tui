//! Process-global, file-based debug logger. Opt-in via `--debug`.
//!
//! When disabled (the default) every logging call is a cheap no-op and nothing
//! is created on disk. When enabled, a unique directory is created under the
//! system temp dir (`$TMPDIR/pytorch-trace-tui-<pid>-<nanos>/`) holding a single
//! human-readable `debug.log`. Entries are concise and `HH:MM:SS`-timestamped —
//! never raw JSON — so the log stays readable at a glance.
//!
//! Crucially, this logger NEVER writes to stdout/stderr, so MCP mode's stdout
//! stays strictly valid JSON-RPC. Diagnostics go only to the file.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// The global sink. `None` (either uninitialized or explicitly disabled) means
/// every log call is a no-op; `Some(Mutex<File>)` is the open `debug.log`.
static LOGGER: OnceLock<Option<Mutex<File>>> = OnceLock::new();

/// Opens (but does not install) a debug log. Returns `Ok(None)` when disabled —
/// creating nothing on disk — or `Ok(Some((path, file)))` with a freshly created
/// unique temp directory and an appendable `debug.log` inside it. Split out from
/// [`init`] so the enabled/disabled decision is testable without touching the
/// process-global `OnceLock`.
pub fn open_log(enabled: bool) -> Result<Option<(PathBuf, File)>> {
    if !enabled {
        return Ok(None);
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("pytorch-trace-tui-{}-{}", process::id(), nanos));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create debug log dir: {}", dir.display()))?;
    let path = dir.join("debug.log");
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open debug log: {}", path.display()))?;
    Ok(Some((path, file)))
}

/// Installs the process-global logger. Idempotent-safe: the first call wins (the
/// `OnceLock` ignores later writes). Returns the log file path when enabled so
/// the caller can surface it (e.g. on stderr in TUI mode), or `None` when
/// disabled. On the enabled path, writes an initial header + startup marker.
pub fn init(enabled: bool) -> Result<Option<PathBuf>> {
    let opened = open_log(enabled)?;
    let (path, slot) = match opened {
        Some((path, file)) => (Some(path), Some(Mutex::new(file))),
        None => (None, None),
    };
    let _ = LOGGER.set(slot);
    if let Some(p) = &path {
        log_line(&format!("pytorch-trace-tui debug log: {}", p.display()));
        log_line("startup");
    }
    Ok(path)
}

/// Splits epoch seconds into `(hour, minute, second)` in UTC. Shared with
/// `main.rs`'s date formatting so the time math lives in one place.
pub(crate) fn hms_parts(secs: u64) -> (u64, u64, u64) {
    (secs / 3600 % 24, secs / 60 % 60, secs % 60)
}

/// Current wall-clock time as `HH:MM:SS` (UTC), the prefix for every log line.
fn now_hms() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = hms_parts(secs);
    format!("{:02}:{:02}:{:02}", h, m, s)
}

/// Writes one timestamped line. No-op when the logger is disabled/uninitialized.
/// Flushes so a concurrently-reading process (e.g. an integration test) sees the
/// line promptly. Lock poisoning is treated as best-effort (logging never
/// panics the caller).
pub fn log_line(msg: &str) {
    write_raw(&format!("{} {}\n", now_hms(), msg));
}

/// Writes one timestamped, indented detail line (two spaces after the
/// timestamp), used for the per-argument breakdown under an `MCP request:` line.
pub fn log_indented(msg: &str) {
    write_raw(&format!("{}   {}\n", now_hms(), msg));
}

fn write_raw(line: &str) {
    let Some(Some(mutex)) = LOGGER.get() else {
        return;
    };
    if let Ok(mut file) = mutex.lock() {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
}

/// Human-readable byte size (e.g. `147.0 MB`). Kept here so both the CLI startup
/// path and the MCP trace-loading logs share one implementation.
pub fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    format!("{:.1} {}", size, UNITS[unit])
}

/// Counts data rows in a produced table string (CSV or TSV): total lines minus
/// the one header line. Never returns a negative count.
pub fn count_rows(table: &str) -> usize {
    table.lines().count().saturating_sub(1)
}

/// Summarizes a large tabular payload to a short `"<n> rows"` string so the log
/// records the shape of a result without echoing the (potentially huge) body.
pub fn summarize_rows(table: &str) -> String {
    format!("{} rows", count_rows(table))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S1: disabled path creates nothing and yields `None`.
    #[test]
    fn s1_open_log_false_is_noop() {
        let out = open_log(false).expect("disabled open is Ok");
        assert!(out.is_none(), "disabled logger opens no file");
    }

    /// S2: enabled path creates a real `debug.log` under the system temp dir and
    /// the file is writable/appendable.
    #[test]
    fn s2_open_log_true_creates_log_under_tempdir() {
        let (path, mut file) = open_log(true).expect("enabled open is Ok").expect("some file");
        assert!(path.exists(), "debug.log exists: {}", path.display());
        assert_eq!(
            path.file_name().and_then(|n| n.to_str()),
            Some("debug.log"),
            "file is named debug.log"
        );
        assert!(
            path.starts_with(std::env::temp_dir()),
            "log lives under temp dir: {}",
            path.display()
        );
        // Appendable.
        writeln!(file, "probe").expect("writable");
        let contents = std::fs::read_to_string(&path).expect("readable");
        assert!(contents.contains("probe"));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// S2: the timestamp prefix is exactly `HH:MM:SS ` for a written line.
    #[test]
    fn s2_timestamp_prefix_format() {
        let hms = now_hms();
        let bytes = hms.as_bytes();
        assert_eq!(hms.len(), 8, "HH:MM:SS is 8 chars: {hms:?}");
        assert!(bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit());
        assert_eq!(bytes[2], b':');
        assert!(bytes[3].is_ascii_digit() && bytes[4].is_ascii_digit());
        assert_eq!(bytes[5], b':');
        assert!(bytes[6].is_ascii_digit() && bytes[7].is_ascii_digit());
    }

    #[test]
    fn s2_hms_parts_math() {
        assert_eq!(hms_parts(0), (0, 0, 0));
        assert_eq!(hms_parts(3661), (1, 1, 1));
        assert_eq!(hms_parts(86399), (23, 59, 59));
        assert_eq!(hms_parts(86400), (0, 0, 0), "wraps at 24h");
    }

    /// S4: a large table is reduced to a short count string and the body never
    /// leaks into the summary.
    #[test]
    fn s4_large_table_summarized_to_counts() {
        let mut csv = String::from("idx,name,dur\n");
        for i in 0..10_000 {
            csv.push_str(&format!("{i},secret_kernel_name_{i},1.0\n"));
        }
        let summary = summarize_rows(&csv);
        assert_eq!(summary, "10000 rows");
        assert!(
            !summary.contains("secret_kernel_name"),
            "summary must not echo the body"
        );
        assert_eq!(count_rows("only_header\n"), 0, "header-only => 0 rows");
        assert_eq!(count_rows(""), 0, "empty => 0 rows");
    }

    #[test]
    fn human_size_scales() {
        assert_eq!(human_size(0), "0.0 B");
        assert_eq!(human_size(1536), "1.5 KB");
        assert_eq!(human_size(147 * 1024 * 1024), "147.0 MB");
    }
}
