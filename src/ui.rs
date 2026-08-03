use crate::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
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

    if app.sequence.is_some() {
        render_sequence_popup(frame, area, app);
    }
}

fn centered_rect(area: Rect, pct_x: u16, pct_y: u16) -> Rect {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - pct_y) / 2),
            Constraint::Percentage(pct_y),
            Constraint::Percentage((100 - pct_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - pct_x) / 2),
            Constraint::Percentage(pct_x),
            Constraint::Percentage((100 - pct_x) / 2),
        ])
        .split(v[1])[1]
}

fn render_sequence_popup(frame: &mut Frame, area: Rect, app: &App) {
    let Some(seq) = app.sequence.as_ref() else {
        return;
    };
    let popup = centered_rect(area, 70, 70);
    frame.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Magenta))
        .title(" Kernel Sequence ")
        .title_style(Style::new().fg(Color::Magenta).bold());
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let mut lines: Vec<Line> = Vec::new();

    // Columns: idx (fixed), median (fixed, right-aligned), name (fills the rest).
    let idx_w = 5usize;
    let med_w = 12usize;
    let name_w = (inner.width as usize)
        .saturating_sub(idx_w + 2 + med_w)
        .max(8);

    let header = format!(
        "{:>idx$}  {:<name$}{:>med$}",
        "idx",
        "kernel name",
        "median",
        idx = idx_w,
        name = name_w,
        med = med_w,
    );
    lines.push(Line::from(Span::styled(
        header,
        Style::new().fg(Color::Cyan).bold(),
    )));

    // Reserve rows for the header plus the footer block (blank + status +
    // reps-line + hint); the remainder is the scrollable viewport for rows.
    let footer_reserved = 3 + usize::from(app.sequence_status.is_some());
    let viewport = (inner.height as usize)
        .saturating_sub(1 + footer_reserved)
        .max(1);
    let total = seq.rows.len();
    let scroll = seq.scroll.min(total.saturating_sub(viewport));
    let end = (scroll + viewport).min(total);

    for (i, (idx, name, _dur)) in seq.rows[scroll..end].iter().enumerate() {
        let row_idx = scroll + i;
        let med = seq
            .median
            .as_ref()
            .and_then(|m| m.get(row_idx))
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        let name_disp = ellipsize(name, name_w);
        let row = format!(
            "{:>idx$}  {:<name$}{:>med$}",
            idx,
            name_disp,
            format!("{:.2}", med),
            idx = idx_w,
            name = name_w,
            med = med_w,
        );
        lines.push(Line::from(Span::styled(row, Style::new().fg(Color::White))));
    }

    lines.push(Line::from(""));
    if let Some(status) = app.sequence_status.as_ref() {
        lines.push(Line::from(Span::styled(
            status.clone(),
            Style::new().fg(Color::Green),
        )));
    }
    lines.push(Line::from(Span::styled(
        format!("median across {} block(s)", seq.reps_found),
        Style::new().fg(Color::DarkGray),
    )));
    let scroll_hint = if total > viewport {
        format!("  [{}-{}/{} ↑↓ PgUp/PgDn]", scroll + 1, end, total)
    } else {
        String::new()
    };
    lines.push(Line::from(Span::styled(
        format!("[y] copy   [Esc/n] close{}", scroll_hint),
        Style::new().fg(Color::DarkGray),
    )));

    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let lane_kind = if app.active_lane_is_annotations() {
        "annotations"
    } else {
        "kernels"
    };
    let line = Line::from(vec![
        Span::styled(" GPU Trace Viewer ", Style::new().fg(Color::Black).bg(Color::Cyan).bold()),
        Span::raw("  "),
        Span::styled("Stream: ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            format!("cuda:{}", app.active_stream()),
            Style::new().fg(Color::Yellow).bold(),
        ),
        Span::raw("  "),
        Span::styled("Lane: ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            format!("{} ({}/{})", lane_kind, app.active_lane + 1, app.lanes.len()),
            Style::new().fg(Color::Cyan),
        ),
        Span::raw("  "),
        Span::styled("Sel: ", Style::new().fg(Color::DarkGray)),
        Span::styled(
            if app.active_lane_len() == 0 {
                "-/-".to_string()
            } else {
                format!("{}/{}", app.selected_item + 1, app.active_lane_len())
            },
            Style::new().fg(Color::Yellow),
        ),
        Span::raw("  "),
        Span::styled("Zoom: ", Style::new().fg(Color::DarkGray)),
        Span::styled(app.zoom_label(), Style::new().fg(Color::Magenta).bold()),
        Span::styled(
            "  [/] search  [Tab/S-Tab] lane  [A/D] item  [W/S] zoom  [Q] quit",
            Style::new().fg(Color::DarkGray),
        ),
    ]);

    if app.search_active {
        let prompt_style = if app.search_no_match {
            Style::new().fg(Color::Red).bold()
        } else {
            Style::new().fg(Color::Black).bg(Color::Yellow).bold()
        };
        let line = Line::from(vec![
            Span::styled(" /search: ", Style::new().fg(Color::Black).bg(Color::Yellow).bold()),
            Span::styled(format!("{}\u{2588}", app.search_query), prompt_style),
            Span::styled(
                if app.search_no_match { "  (no match)" } else { "  [Enter] keep  [Esc] cancel" },
                Style::new().fg(Color::DarkGray),
            ),
        ]);
        frame.render_widget(Paragraph::new(line).style(Style::new().bg(Color::Black)), area);
        return;
    }

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
    let start = app.lane_view_offset.min(app.lanes.len().saturating_sub(1));
    let end = (start + visible_rows).min(app.lanes.len());

    for lane_idx in start..end {
        let lane = &app.lanes[lane_idx];
        let is_active_lane = lane_idx == app.active_lane;

        let label = if lane.is_annotations() {
            String::new()
        } else {
            format!("cuda:{}", lane.stream_id())
        };
        let label_padded = pad_to(label, label_width - 1);
        let label_style = if is_active_lane {
            Style::new().fg(Color::Yellow).bold()
        } else {
            Style::new().fg(Color::DarkGray)
        };

        let mut spans: Vec<Span<'static>> = vec![
            Span::styled(label_padded, label_style),
            Span::styled("│", Style::new().fg(Color::DarkGray)),
        ];
        spans.extend(build_lane(app, lane_idx, ts_start, time_span, lane_width));
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

fn build_lane(
    app: &App,
    lane_idx: usize,
    ts_start: f64,
    time_span: f64,
    width: usize,
) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut cursor: usize = 0;

    let lane = &app.lanes[lane_idx];
    let is_active_lane = lane_idx == app.active_lane;
    let ts_end = ts_start + time_span;
    let ann_bg = Color::Rgb(90, 90, 110);

    for (pos, &item_idx) in lane.item_indices().iter().enumerate() {
        let (name, ts, end_ts, block_bg) = match lane {
            crate::app::Lane::Kernels { .. } => {
                let k = &app.kernels[item_idx];
                (
                    k.name.as_str(),
                    k.ts,
                    k.end_ts(),
                    kernel_color(k.cat.as_str(), pos),
                )
            }
            crate::app::Lane::Annotations { .. } => {
                let a = &app.annotations[item_idx];
                (a.name.as_str(), a.ts, a.end_ts(), ann_bg)
            }
        };

        let is_selected = is_active_lane && pos == app.selected_item;

        let Some((start_col, end_col)) =
            crate::app::kernel_columns(ts, end_ts, ts_start, ts_end, width)
        else {
            continue;
        };

        // Clamp to cursor so sub-column items don't each steal a full column and
        // push later items off the lane.
        let start_col = start_col.max(cursor);
        if start_col >= end_col || start_col >= width {
            continue;
        }

        if start_col > cursor {
            spans.push(Span::styled(
                " ".repeat(start_col - cursor),
                Style::new().bg(Color::Black),
            ));
        }

        let block_width = end_col.saturating_sub(start_col).max(1);
        let label = pad_to(ellipsize(name, block_width), block_width);

        let style = if is_selected {
            Style::new().fg(Color::Black).bg(Color::White)
        } else if lane.is_annotations() {
            Style::new().fg(Color::White).bg(block_bg)
        } else {
            Style::new().fg(Color::Black).bg(block_bg)
        };

        spans.push(Span::styled(label, style));
        cursor = end_col;
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
    let is_annotation = app.active_lane_is_annotations();
    let title = if is_annotation {
        " Annotation Info "
    } else {
        " Kernel Info "
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Yellow))
        .title(title)
        .title_style(Style::new().fg(Color::Yellow).bold());

    let inner = block.inner(area);
    frame.render_widget(block, area);

    use crate::app::SelectedTraceItem;
    let Some(item) = app.selected_trace_item() else {
        frame.render_widget(
            Paragraph::new("No item selected").style(Style::new().fg(Color::DarkGray)),
            inner,
        );
        return;
    };

    match item {
        SelectedTraceItem::Annotation(a) => {
            let lines: Vec<Line> = vec![
                Line::from(vec![kv_label("Name: "), kv_value(&a.name)]),
                Line::from(vec![
                    kv_label("Stream: "),
                    kv_value(&format!("cuda:{}", a.stream)),
                ]),
                Line::from(vec![
                    kv_label("Start: "),
                    kv_value(&format!("{:.3} μs", a.ts)),
                    Span::raw("  "),
                    kv_label("End: "),
                    kv_value(&format!("{:.3} μs", a.end_ts())),
                    Span::raw("  "),
                    kv_label("Dur: "),
                    kv_value(&format!("{:.3} μs", a.dur)),
                ]),
            ];
            frame.render_widget(Paragraph::new(lines), inner);
        }
        SelectedTraceItem::Kernel(k) => {
            let na = || "N/A".to_string();
            let lines: Vec<Line> = vec![
                Line::from(vec![kv_label("Name: "), kv_value(&k.name)]),
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
            frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
        }
    }
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
