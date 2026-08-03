mod align;
mod app;
mod clipboard;
mod trace;
mod ui;

use anyhow::{bail, Context, Result};
use app::App;
use clap::Parser;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, stdout, BufRead, Write};
use std::panic;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "pytorch-trace-tui",
    about = "Interactive TUI viewer for GPU kernels in PyTorch profiler traces",
    version
)]
struct Cli {
    /// Trace file path(s). Pass two or more to overlay and align them by
    /// gpu_user_annotation. If omitted, scans the current dir for *.pt.trace.json.gz
    traces: Vec<String>,
}

fn trace_label(path: &str, _index: usize) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.trim_end_matches(".gz").trim_end_matches(".json").trim_end_matches(".pt.trace"))
        .filter(|s| !s.is_empty())
        .unwrap_or("trace")
        .to_string()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let paths: Vec<String> = if cli.traces.is_empty() {
        match select_trace_interactively()? {
            Some(paths) => paths,
            None => return Ok(()),
        }
    } else {
        cli.traces
    };

    let mut labelled: Vec<(String, trace::Trace)> = Vec::new();
    let mut total_kernels = 0usize;
    let mut total_annotations = 0usize;
    for (index, path) in paths.iter().enumerate() {
        let trace = trace::parse_trace(path)
            .with_context(|| format!("Failed to load trace: {}", path))?;
        total_kernels += trace.kernels.len();
        total_annotations += trace.annotations.len();
        labelled.push((trace_label(path, index), trace));
    }

    if total_kernels == 0 {
        println!("No GPU kernels found in trace file(s): {}", paths.join(", "));
        return Ok(());
    }

    eprintln!(
        "Loaded {} GPU kernels ({} annotations) from {} trace(s). Starting TUI...",
        total_kernels,
        total_annotations,
        paths.len()
    );

    let app = App::new_multi(labelled);
    run_tui(app)
}

/// Parses a picker selection line into 1-based indices. Accepts space- and/or
/// comma-separated numbers (e.g. "1 3", "1,3", "2, 4 5"). Returns None for an
/// empty line or a quit request ("q"); Err on any non-numeric or out-of-range
/// token. Duplicates are preserved in the order given.
fn parse_selection(input: &str, count: usize) -> Result<Option<Vec<usize>>> {
    let input = input.trim().trim_matches(char::from(0));
    if input.is_empty() || input.eq_ignore_ascii_case("q") {
        return Ok(None);
    }
    let mut chosen = Vec::new();
    for tok in input.split([' ', ',']).filter(|t| !t.is_empty()) {
        let n: usize = tok
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid selection: {}", tok))?;
        if n < 1 || n > count {
            bail!("Selection out of range: {}", n);
        }
        chosen.push(n);
    }
    if chosen.is_empty() {
        return Ok(None);
    }
    Ok(Some(chosen))
}

fn select_trace_interactively() -> Result<Option<Vec<String>>> {
    let mut traces: Vec<(PathBuf, std::time::SystemTime, u64)> = std::fs::read_dir(".")
        .context("Failed to read current directory")?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".pt.trace.json.gz"))
                .unwrap_or(false)
        })
        .filter_map(|p| {
            let meta = std::fs::metadata(&p).ok()?;
            let mtime = meta.modified().ok()?;
            let size = meta.len();
            Some((p, mtime, size))
        })
        .collect();

    if traces.is_empty() {
        bail!("No trace file given and no *.pt.trace.json.gz found in current directory");
    }

    traces.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Select trace(s) to open:");
    for (i, (path, mtime, size)) in traces.iter().enumerate() {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let dt = format_mtime(*mtime);
        println!("  [{}] {}  {}  {}", i + 1, dt, human_size(*size), name);
    }
    print!(
        "Enter number(s) (e.g. 1 or '1 3' to overlay, 1-{}, or q to quit): ",
        traces.len()
    );
    stdout().flush().ok();

    let mut raw = Vec::new();
    io::stdin()
        .lock()
        .read_until(b'\n', &mut raw)
        .context("Failed to read selection")?;
    let decoded = String::from_utf8_lossy(&raw);

    match parse_selection(&decoded, traces.len())? {
        None => Ok(None),
        Some(choices) => Ok(Some(
            choices
                .into_iter()
                .map(|c| traces[c - 1].0.to_string_lossy().into_owned())
                .collect(),
        )),
    }
}

fn format_mtime(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    let mut y = 1970u32;
    let mut d = days;
    loop {
        let dy = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if d < dy { break; }
        d -= dy;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [31u64, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 1u32;
    let mut rem = d;
    for &md in &month_days {
        if rem < md { break; }
        rem -= md;
        mo += 1;
    }
    let day = rem + 1;
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, mo, day, h, m, s)
}

fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    format!("{:.1} {}", size, UNITS[unit])
}

fn run_tui(mut app: App) -> Result<()> {
    // Install a panic hook that restores the terminal before printing the panic
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        original_hook(info);
    }));

    setup_terminal()?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let result = event_loop(&mut terminal, &mut app);

    restore_terminal()?;
    result
}

fn setup_terminal() -> Result<()> {
    enable_raw_mode().context("Failed to enable raw mode")?;
    execute!(stdout(), EnterAlternateScreen, EnableMouseCapture)
        .context("Failed to enter alternate screen")?;
    Ok(())
}

fn restore_terminal() -> Result<()> {
    disable_raw_mode().context("Failed to disable raw mode")?;
    execute!(stdout(), LeaveAlternateScreen, DisableMouseCapture)
        .context("Failed to leave alternate screen")?;
    Ok(())
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    let mut clipboard = crate::clipboard::ClipboardManager::new();
    loop {
        let term_height = terminal.size()?.height as usize;
        let lane_rows = term_height.saturating_sub(1 + 12 + 3);
        app.ensure_active_lane_visible(lane_rows.max(1));

        terminal.draw(|frame| {
            ui::render(frame, app);
        })?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if app.sequence.is_some() {
                        match key.code {
                            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                                app.close_sequence();
                            }
                            KeyCode::Up => {
                                app.sequence_scroll_up(1);
                            }
                            KeyCode::Down => {
                                app.sequence_scroll_down(1, sequence_viewport(term_height, app));
                            }
                            KeyCode::PageUp => {
                                app.sequence_scroll_up(sequence_viewport(term_height, app));
                            }
                            KeyCode::PageDown => {
                                let vp = sequence_viewport(term_height, app);
                                app.sequence_scroll_down(vp, vp);
                            }
                            KeyCode::Char('y') | KeyCode::Char('Y') => {
                                if let Some(csv) = app.sequence_csv() {
                                    let mut out = stdout();
                                    match clipboard.copy(&csv, &mut out) {
                                        Ok(outcome) => {
                                            let mut msg = String::new();
                                            if outcome.via_native {
                                                msg.push_str("copied to clipboard");
                                            } else if outcome.via_osc52 {
                                                msg.push_str("sent via OSC52");
                                            } else {
                                                msg.push_str("clipboard unavailable");
                                            }
                                            msg.push_str(&format!(
                                                "; CSV also saved to {}",
                                                outcome.file_path.display()
                                            ));
                                            app.sequence_status = Some(msg);
                                        }
                                        Err(e) => {
                                            app.sequence_status =
                                                Some(format!("copy failed: {}", e));
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }
                    if app.search_active {
                        match key.code {
                            KeyCode::Esc => app.search_cancel(),
                            KeyCode::Enter => app.search_commit(),
                            KeyCode::Backspace => app.search_backspace(),
                            KeyCode::Char(c) => app.search_push(c),
                            _ => {}
                        }
                        continue;
                    }
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                        KeyCode::Char('/') => {
                            app.search_start();
                        }
                        KeyCode::Char('n') | KeyCode::Char('N') => {
                            app.start_sequence();
                        }
                        KeyCode::Char('g') | KeyCode::Char('G') => {
                            app.align_to_selected_kernel();
                        }
                        KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Left => {
                            app.prev_item();
                        }
                        KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Right => {
                            app.next_item();
                        }
                        KeyCode::Char('w') | KeyCode::Char('W') | KeyCode::Up => {
                            app.zoom_in();
                        }
                        KeyCode::Char('s') | KeyCode::Char('S') | KeyCode::Down => {
                            app.zoom_out();
                        }
                        KeyCode::Tab => {
                            app.next_lane();
                        }
                        KeyCode::BackTab => {
                            app.prev_lane();
                        }
                        _ => {}
                    }
                }
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }

    Ok(())
}

/// Number of scrollable kernel rows the sequence popup shows. Mirrors the UI's
/// geometry: popup is 70% of terminal height, inner drops 2 border rows, then
/// the header row and the footer block (blank + optional status + reps-line +
/// hint) are reserved.
fn sequence_viewport(term_height: usize, app: &App) -> usize {
    let popup_h = term_height * 70 / 100;
    let inner = popup_h.saturating_sub(2);
    let footer_reserved = 3 + usize::from(app.sequence_status.is_some());
    inner.saturating_sub(1 + footer_reserved).max(1)
}

#[cfg(test)]
mod tests {
    use super::parse_selection;

    #[test]
    fn single_number() {
        assert_eq!(parse_selection("2", 5).unwrap(), Some(vec![2]));
    }

    #[test]
    fn space_separated_multi() {
        assert_eq!(parse_selection("1 3", 5).unwrap(), Some(vec![1, 3]));
    }

    #[test]
    fn comma_separated_multi() {
        assert_eq!(parse_selection("2,4,5", 5).unwrap(), Some(vec![2, 4, 5]));
    }

    #[test]
    fn mixed_separators_and_spacing() {
        assert_eq!(parse_selection(" 2, 4 5 ", 5).unwrap(), Some(vec![2, 4, 5]));
    }

    #[test]
    fn quit_and_empty_return_none() {
        assert_eq!(parse_selection("q", 5).unwrap(), None);
        assert_eq!(parse_selection("Q", 5).unwrap(), None);
        assert_eq!(parse_selection("", 5).unwrap(), None);
        assert_eq!(parse_selection("   ", 5).unwrap(), None);
    }

    #[test]
    fn out_of_range_is_error() {
        assert!(parse_selection("0", 5).is_err());
        assert!(parse_selection("6", 5).is_err());
        assert!(parse_selection("1 9", 5).is_err());
    }

    #[test]
    fn non_numeric_is_error() {
        assert!(parse_selection("1 x", 5).is_err());
        assert!(parse_selection("abc", 5).is_err());
    }

    #[test]
    fn duplicates_preserved_in_order() {
        assert_eq!(parse_selection("3 1 3", 5).unwrap(), Some(vec![3, 1, 3]));
    }
}
