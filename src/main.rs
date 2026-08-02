mod app;
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
    /// Trace file path; if omitted, scans the current dir for *.pt.trace.json.gz
    trace: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let trace_path = match cli.trace {
        Some(path) => path,
        None => match select_trace_interactively()? {
            Some(path) => path,
            None => return Ok(()),
        },
    };

    let trace = trace::parse_trace(&trace_path)
        .with_context(|| format!("Failed to load trace: {}", trace_path))?;

    if trace.kernels.is_empty() {
        println!("No GPU kernels found in trace file: {}", trace_path);
        return Ok(());
    }

    let stream_count = {
        let mut s: Vec<u64> = trace.kernels.iter().map(|k| k.stream).collect();
        s.sort_unstable();
        s.dedup();
        s.len()
    };
    eprintln!(
        "Loaded {} GPU kernels ({} annotations) across {} stream(s). Starting TUI...",
        trace.kernels.len(),
        trace.annotations.len(),
        stream_count
    );

    let app = App::new(trace);
    run_tui(app)
}

fn select_trace_interactively() -> Result<Option<String>> {
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

    println!("Select a trace to open:");
    for (i, (path, mtime, size)) in traces.iter().enumerate() {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let dt = format_mtime(*mtime);
        println!("  [{}] {}  {}  {}", i + 1, dt, human_size(*size), name);
    }
    print!("Enter number (1-{}, or q to quit): ", traces.len());
    stdout().flush().ok();

    let mut raw = Vec::new();
    io::stdin()
        .lock()
        .read_until(b'\n', &mut raw)
        .context("Failed to read selection")?;
    let decoded = String::from_utf8_lossy(&raw);
    let input = decoded.trim().trim_matches(char::from(0));

    if input.eq_ignore_ascii_case("q") || input.is_empty() {
        return Ok(None);
    }

    let choice: usize = input
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid selection: {}", input))?;
    if choice < 1 || choice > traces.len() {
        bail!("Selection out of range: {}", choice);
    }

    Ok(Some(traces[choice - 1].0.to_string_lossy().into_owned()))
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
    loop {
        let term_height = terminal.size()?.height as usize;
        let lane_rows = term_height.saturating_sub(1 + 12 + 3);
        app.ensure_active_stream_visible(lane_rows.max(1));

        terminal.draw(|frame| {
            ui::render(frame, app);
        })?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
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
                        KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Left => {
                            app.prev();
                        }
                        KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Right => {
                            app.next();
                        }
                        KeyCode::Char('w') | KeyCode::Char('W') | KeyCode::Up => {
                            app.zoom_in();
                        }
                        KeyCode::Char('s') | KeyCode::Char('S') | KeyCode::Down => {
                            app.zoom_out();
                        }
                        KeyCode::Char('e') | KeyCode::Char('E') => {
                            app.toggle_focus();
                        }
                        KeyCode::Tab => {
                            app.next_stream();
                        }
                        KeyCode::BackTab => {
                            app.prev_stream();
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
