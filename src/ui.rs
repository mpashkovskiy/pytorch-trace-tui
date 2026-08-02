use crate::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};

const KERNEL_COLORS: &[Color] = &[
    Color::Rgb(59, 130, 246),
    Color::Rgb(16, 185, 129),
    Color::Rgb(245, 158, 11),
    Color::Rgb(239, 68, 68),
    Color::Rgb(168, 85, 247),
    Color::Rgb(20, 184, 166),
    Color::Rgb(249, 115, 22),
    Color::Rgb(236, 72, 153),
];

const MEMCPY_COLOR: Color = Color::Rgb(139, 92, 246);
const MEMSET_COLOR: Color = Color::Rgb(251, 191, 36);

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(10),
        ])
        .split(area);

    render_header(frame, chunks[0], app);
    render_lane(frame, chunks[1], app);
    render_info_panel(frame, chunks[2], app);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let line = Line::from(vec![
        Span::styled(" GPU Trace Viewer ", Style::new().fg(Color::Black).bg(Color::Cyan).bold()),
        Span::raw("  "),
        Span::styled("Stream: ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            format!("cuda:{}", app.active_stream()),
            Style::new().fg(Color::Yellow).bold(),
        ),
        Span::styled(
            format!(" ({}/{})", app.active_stream_idx + 1, app.streams.len()),
            Style::new().fg(Color::DarkGray),
        ),
        Span::raw("  "),
        Span::styled("Kernels: ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", app.filtered_indices.len()),
            Style::new().fg(Color::Green),
        ),
        Span::raw("  "),
        Span::styled("Sel: ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            if app.filtered_indices.is_empty() {
                "-/-".to_string()
            } else {
                format!("{}/{}", app.selected_kernel + 1, app.filtered_indices.len())
            },
            Style::new().fg(Color::Yellow),
        ),
        Span::raw("  "),
        Span::styled("Zoom: ", Style::new().fg(Color::DarkGray)),
        Span::styled(app.zoom_label(), Style::new().fg(Color::Magenta).bold()),
        Span::styled(
            "  [Tab/S-Tab] stream  [A/D] kernel  [W/S] zoom  [Q] quit",
            Style::new().fg(Color::DarkGray),
        ),
    ]);

    frame.render_widget(Paragraph::new(line).style(Style::new().bg(Color::Black)), area);
}

fn render_lane(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Blue))
        .title(format!(
            " GPU Streams ({}) — active cuda:{} ",
            app.streams.len(),
            app.active_stream()
        ))
        .title_style(Style::new().fg(Color::Cyan).bold());

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let total_width = inner.width as usize;
    let label_width = stream_label_width(app);
    if total_width <= label_width + 4 {
        return;
    }
    let lane_width = total_width - label_width;

    let (ts_start, ts_end) = app.global_visible_window();
    let time_span = (ts_end - ts_start).max(1.0);

    let mut lines: Vec<Line> = Vec::new();

    let visible_rows = inner.height as usize;
    let start = app.stream_view_offset.min(app.streams.len().saturating_sub(1));
    let end = (start + visible_rows).min(app.streams.len());

    for stream_idx in start..end {
        let stream_id = app.streams[stream_idx];
        let is_active = stream_idx == app.active_stream_idx;

        let label = format!("cuda:{}", stream_id);
        let label_padded = pad_to(label, label_width - 1);
        let label_style = if is_active {
            Style::new().fg(Color::Yellow).bold()
        } else {
            Style::new().fg(Color::DarkGray)
        };

        let mut spans: Vec<Span<'static>> = vec![
            Span::styled(label_padded, label_style),
            Span::styled("│", Style::new().fg(Color::DarkGray)),
        ];
        spans.extend(build_stream_lane(app, stream_idx, ts_start, time_span, lane_width));
        lines.push(Line::from(spans));
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

fn stream_label_width(app: &App) -> usize {
    let max = app
        .streams
        .iter()
        .map(|s| format!("cuda:{}", s).chars().count())
        .max()
        .unwrap_or(6);
    max + 2
}

fn build_stream_lane(
    app: &App,
    stream_idx: usize,
    ts_start: f64,
    time_span: f64,
    width: usize,
) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut cursor: usize = 0;

    let stream_id = app.streams[stream_idx];
    let is_active_stream = stream_idx == app.active_stream_idx;
    let kernel_indices = app.kernel_indices_for_stream(stream_id);

    let ts_end = ts_start + time_span;

    for (list_idx, &kernel_idx) in kernel_indices.iter().enumerate() {
        let k = &app.kernels[kernel_idx];
        let is_selected = is_active_stream && list_idx == app.selected_kernel;

        let Some((k_start_col, k_end_col)) =
            crate::app::kernel_columns(k.ts, k.end_ts(), ts_start, ts_end, width)
        else {
            continue;
        };

        if k_start_col > cursor {
            spans.push(Span::styled(
                " ".repeat(k_start_col - cursor),
                Style::new().bg(Color::Black),
            ));
        }

        let block_width = k_end_col.saturating_sub(k_start_col).max(1);
        let bg = kernel_color(k.cat.as_str(), list_idx);
        let label = pad_to(ellipsize(&k.name, block_width), block_width);

        let style = if is_selected {
            Style::new().fg(Color::White).bg(bg).bold()
        } else {
            Style::new().fg(Color::Black).bg(bg)
        };

        spans.push(Span::styled(label, style));
        cursor = k_end_col;
    }

    if cursor < width {
        spans.push(Span::styled(
            " ".repeat(width - cursor),
            Style::new().bg(Color::Black),
        ));
    }

    spans
}

fn render_info_panel(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Yellow))
        .title(" Kernel Info ")
        .title_style(Style::new().fg(Color::Yellow).bold());

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(k) = app.selected_event() else {
        frame.render_widget(
            Paragraph::new("No kernel selected").style(Style::new().fg(Color::DarkGray)),
            inner,
        );
        return;
    };

    let na = || "N/A".to_string();

    let lines: Vec<Line> = vec![
        Line::from(vec![
            kv_label("Name: "),
            kv_value(&k.name),
            Span::raw("  "),
            kv_label("Cat: "),
            kv_value(&k.cat),
        ]),
        Line::from(vec![
            kv_label("Stream: "),
            kv_value(&format!("cuda:{}", k.stream)),
            Span::raw("  "),
            kv_label("Device: "),
            kv_value(&format!("{}", k.device)),
        ]),
        Line::from(vec![
            kv_label("Start: "),
            kv_value(&format!("{:.3} μs", k.ts)),
            Span::raw("  "),
            kv_label("End: "),
            kv_value(&format!("{:.3} μs", k.end_ts())),
            Span::raw("  "),
            kv_label("Dur: "),
            kv_value(&format!("{:.3} μs", k.dur)),
        ]),
        Line::from(vec![
            kv_label("Grid: "),
            kv_value(&k.grid.clone().unwrap_or_else(na)),
            Span::raw("  "),
            kv_label("Block: "),
            kv_value(&k.block.clone().unwrap_or_else(na)),
        ]),
        Line::from(vec![
            kv_label("Shared Mem: "),
            kv_value(&k.shared_memory.map_or_else(na, |v| format!("{} B", v))),
            Span::raw("  "),
            kv_label("Regs/Thread: "),
            kv_value(&k.registers_per_thread.map_or_else(na, |v| v.to_string())),
        ]),
        Line::from(vec![
            kv_label("Correlation: "),
            kv_value(&k.correlation.map_or_else(na, |v| v.to_string())),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines), inner);
}

fn kernel_color(cat: &str, idx: usize) -> Color {
    match cat {
        "gpu_memcpy" => MEMCPY_COLOR,
        "gpu_memset" => MEMSET_COLOR,
        _ => KERNEL_COLORS[idx % KERNEL_COLORS.len()],
    }
}

fn ellipsize(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else if max_chars == 1 {
        ".".to_string()
    } else {
        let mut out: String = chars[..max_chars - 1].iter().collect();
        out.push('…');
        out
    }
}

fn pad_to(mut s: String, width: usize) -> String {
    let n = s.chars().count();
    if n < width {
        for _ in 0..(width - n) {
            s.push(' ');
        }
        s
    } else {
        ellipsize(&s, width)
    }
}

fn kv_label(s: &str) -> Span<'static> {
    Span::styled(s.to_string(), Style::new().fg(Color::DarkGray))
}

fn kv_value(s: &str) -> Span<'static> {
    Span::styled(s.to_string(), Style::new().fg(Color::White).bold())
}
