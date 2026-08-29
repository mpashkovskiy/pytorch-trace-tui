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

const DIFF_ADDED: Color = Color::Rgb(34, 197, 94);
const DIFF_DELETED: Color = Color::Rgb(220, 38, 38);

pub fn render(frame: &mut Frame, app: &mut App) {
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
    let line = Line::from(vec![
        Span::styled(" GPU Trace Viewer ", Style::new().fg(Color::Black).bg(Color::Cyan).bold()),
        Span::styled(
            if app.traces.len() == 2 {
                "  [/] search  [Tab] lane  [A/D] item  [W/S] zoom  [E] export  [G] diff/normal  [Q] quit"
            } else {
                "  [/] search  [Tab] lane  [A/D] item  [W/S] zoom  [E] export  [Q] quit"
            },
            Style::new().fg(Color::DarkGray),
        ),
    ]);

    if let Some(status) = app.status.as_deref() {
        let line = Line::from(vec![
            Span::styled(" GPU Trace Viewer ", Style::new().fg(Color::Black).bg(Color::Cyan).bold()),
            Span::raw("  "),
            Span::styled(status.to_string(), Style::new().fg(Color::Black).bg(Color::Green).bold()),
        ]);
        frame.render_widget(Paragraph::new(line).style(Style::new().bg(Color::Black)), area);
        return;
    }

    if app.search_active {
        let prompt_style = if app.search_no_match {
            Style::new().fg(Color::Red).bold()
        } else {
            Style::new().fg(Color::Black).bg(Color::Yellow).bold()
        };
        let status = if app.search_no_match {
            "  (no match)".to_string()
        } else {
            match app.search_match_label() {
                Some(pos) => format!("  {}  [Enter/S-Enter] next/prev  [Esc] cancel", pos),
                None => "  [Enter] keep  [Esc] cancel".to_string(),
            }
        };
        let line = Line::from(vec![
            Span::styled(" /search: ", Style::new().fg(Color::Black).bg(Color::Yellow).bold()),
            Span::styled(format!("{}\u{2588}", app.search_query), prompt_style),
            Span::styled(status, Style::new().fg(Color::DarkGray)),
        ]);
        frame.render_widget(Paragraph::new(line).style(Style::new().bg(Color::Black)), area);
        return;
    }

    frame.render_widget(Paragraph::new(line).style(Style::new().bg(Color::Black)), area);
}

fn render_lane(frame: &mut Frame, area: Rect, app: &mut App) {
    let title = match app.mode_label() {
        Some(mode) => format!(" {} traces — {} ", app.traces.len(), mode),
        None if app.traces.len() > 1 => format!(" {} traces ", app.traces.len()),
        None => format!(
            " GPU Streams ({}) — active cuda:{} ",
            app.streams.len(),
            app.active_stream()
        ),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Blue))
        .title(title)
        .title_style(Style::new().fg(Color::Cyan).bold());

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let total_width = inner.width as usize;
    let label_width = stream_label_width(app);
    if total_width <= label_width + 4 {
        app.lane_layout = crate::app::LaneLayout::default();
        return;
    }
    let lane_width = total_width - label_width;

    let (ts_start, ts_end) = app.global_visible_window();
    let time_span = (ts_end - ts_start).max(1.0);

    let visible_rows = inner.height as usize;
    let start = app.lane_view_offset.min(app.lanes.len().saturating_sub(1));
    let end = (start + visible_rows).min(app.lanes.len());

    app.lane_layout = crate::app::LaneLayout {
        inner_x: inner.x,
        inner_y: inner.y,
        inner_h: inner.height,
        label_width: label_width as u16,
        lane_width: lane_width as u16,
        view_offset: start,
        ts_start,
        time_span,
    };

    let mut lines: Vec<Line> = Vec::new();

    let multi = app.traces.len() > 1;
    let short_labels = app.trace_display_labels();

    for lane_idx in start..end {
        let lane = &app.lanes[lane_idx];
        let is_active_lane = lane_idx == app.active_lane;

        let trace_prefix = if multi {
            format!("{} ", short_labels[lane.trace_id()])
        } else {
            String::new()
        };
        let label = if lane.is_annotations() {
            trace_prefix.trim_end().to_string()
        } else {
            format!("{}cuda:{}", trace_prefix, lane.stream_id())
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
    let multi = app.traces.len() > 1;
    let short_labels = app.trace_display_labels();
    let max = app
        .lanes
        .iter()
        .map(|lane| {
            let prefix = if multi {
                short_labels
                    .get(lane.trace_id())
                    .map(|l| l.chars().count() + 1)
                    .unwrap_or(0)
            } else {
                0
            };
            let body = if lane.is_annotations() {
                0
            } else {
                format!("cuda:{}", lane.stream_id()).chars().count()
            };
            prefix + body
        })
        .max()
        .unwrap_or(6);
    max + 2
}

pub(crate) fn build_lane(
    app: &App,
    lane_idx: usize,
    ts_start: f64,
    time_span: f64,
    width: usize,
) -> Vec<Span<'static>> {
    let lane = &app.lanes[lane_idx];
    if lane.is_annotations() {
        return build_annotation_lane(app, lane_idx, ts_start, time_span, width);
    }

    let ts_end = ts_start + time_span;
    let diff_active =
        app.traces.len() == 2 && app.align_mode == crate::app::AlignMode::Diff;
    let classes = app.lane_column_classes(lane_idx, ts_start, time_span, width, diff_active);

    // Per-column colour, authoritative: the priority-resolved class governs the
    // cell colour, so a wide matched (dimmed) block can never overwrite a
    // narrower added/deleted column.
    let mut fg = vec![Color::White; width];
    let mut bg = vec![Color::Black; width];
    for (col, cell) in classes.iter().enumerate() {
        let Some((class, owner_pos)) = *cell else {
            continue;
        };
        let base = lane
            .item_indices()
            .get(owner_pos)
            .map(|&i| kernel_color(app.kernels[i].cat.as_str(), owner_pos))
            .unwrap_or(Color::Black);
        let (f, b) = match class {
            crate::app::ColumnClass::Selected => (Color::Black, Color::White),
            crate::app::ColumnClass::Added => (Color::Black, DIFF_ADDED),
            crate::app::ColumnClass::Deleted => (Color::White, DIFF_DELETED),
            crate::app::ColumnClass::Matched if diff_active => {
                (Color::Rgb(140, 140, 140), dim_color(base))
            }
            crate::app::ColumnClass::Matched => (Color::Black, base),
        };
        fg[col] = f;
        bg[col] = b;
    }

    // Per-column label characters, drawn over each block's own column run. Later
    // blocks may overwrite earlier label chars in an overlap — cosmetic only; the
    // colour buffer above stays priority-correct.
    let mut chars = vec![' '; width];
    for &item_idx in lane.item_indices().iter() {
        let ts = app.kernel_render_ts(item_idx);
        let end_ts = app.kernel_render_end(item_idx);
        let Some((start_col, end_col)) =
            crate::app::kernel_columns(ts, end_ts, ts_start, ts_end, width)
        else {
            continue;
        };
        let block_width = end_col.saturating_sub(start_col).max(1);
        let label = pad_to(ellipsize(&app.kernels[item_idx].name, block_width), block_width);
        for (offset, ch) in label.chars().enumerate() {
            let col = start_col + offset;
            if col < end_col && col < width {
                chars[col] = ch;
            }
        }
    }

    coalesce_columns(&fg, &bg, &chars)
}

/// Merges per-column (fg, bg, char) cells into styled spans, one span per run of
/// identical style.
fn coalesce_columns(fg: &[Color], bg: &[Color], chars: &[char]) -> Vec<Span<'static>> {
    let width = fg.len();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut col = 0;
    while col < width {
        let (f, b) = (fg[col], bg[col]);
        let mut run = String::new();
        while col < width && fg[col] == f && bg[col] == b {
            run.push(chars[col]);
            col += 1;
        }
        spans.push(Span::styled(run, Style::new().fg(f).bg(b)));
    }
    spans
}

/// Annotation lanes render with a single background and no diff colouring; this
/// preserves the original block-based drawing for them.
fn build_annotation_lane(
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
        let name = app.annotations[item_idx].name.as_str();
        let ts = app.annotation_render_ts(item_idx);
        let end_ts = app.annotation_render_end(item_idx);
        let is_selected = is_active_lane && pos == app.selected_item;

        let Some((start_col, end_col)) =
            crate::app::kernel_columns(ts, end_ts, ts_start, ts_end, width)
        else {
            continue;
        };

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
        } else {
            Style::new().fg(Color::White).bg(ann_bg)
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

/// Darkens a block colour toward black so matched (unchanged) kernels recede and
/// the green/red added/deleted blocks stand out in diff mode.
fn dim_color(c: Color) -> Color {
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(r / 3, g / 3, b / 3),
        other => other,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use crate::trace::{AnnotationEvent, KernelEvent, Trace};

    fn k(stream: u64, ts: f64, name: &str, dur: f64) -> KernelEvent {
        KernelEvent {
            name: name.into(),
            cat: "kernel".into(),
            ts,
            dur,
            device: 0,
            stream,
            grid: None,
            block: None,
            shared_memory: None,
            registers_per_thread: None,
            correlation: None,
            trace_id: 0,
        }
    }

    fn ann(stream: u64, ts: f64, dur: f64, name: &str) -> AnnotationEvent {
        AnnotationEvent {
            name: name.into(),
            ts,
            dur,
            stream,
            trace_id: 0,
        }
    }

    fn bgs(spans: &[Span<'static>]) -> Vec<Color> {
        spans.iter().filter_map(|s| s.style.bg).collect()
    }

    // S8: an annotation lane renders with ann_bg and never the diff colours.
    #[test]
    fn test_annotation_lane_uses_ann_bg_never_diff_colors() {
        let t0 = Trace {
            kernels: vec![k(1, 100.0, "x", 10.0)],
            annotations: vec![ann(1, 100.0, 40.0, "ctx")],
        };
        let t1 = Trace {
            kernels: vec![k(1, 100.0, "x", 10.0)],
            annotations: vec![ann(1, 100.0, 40.0, "ctx")],
        };
        let app = App::new_multi(vec![("A".into(), t0), ("B".into(), t1)]);
        let ann_lane = app.lanes.iter().position(|l| l.is_annotations()).unwrap();
        let spans = build_lane(&app, ann_lane, 100.0, 40.0, 40);
        let colors = bgs(&spans);
        let ann_bg = Color::Rgb(90, 90, 110);
        assert!(colors.contains(&ann_bg), "annotation lane uses ann_bg");
        assert!(
            !colors.contains(&DIFF_ADDED) && !colors.contains(&DIFF_DELETED),
            "annotation lane never uses diff colours"
        );
    }

    // The fix at the UI layer: a wide MATCHED (dimmed) kernel that spans a
    // narrow ADDED kernel does NOT hide it — a green cell appears in the lane.
    #[test]
    fn test_wide_matched_does_not_hide_added() {
        let t0 = Trace {
            kernels: vec![k(1, 100.0, "reduce", 400.0), k(1, 200.0, "gone", 20.0)],
            annotations: vec![],
        };
        let t1 = Trace {
            kernels: vec![k(1, 100.0, "reduce", 400.0), k(1, 200.0, "extra", 20.0)],
            annotations: vec![],
        };
        let mut app = App::new_multi(vec![("A".into(), t0), ("B".into(), t1)]);
        // Park selection so nothing turns white.
        app.active_lane = usize::MAX;
        let lane1 = app
            .lanes
            .iter()
            .position(|l| !l.is_annotations() && l.trace_id() == 1)
            .unwrap();
        let spans = build_lane(&app, lane1, 100.0, 400.0, 400);
        let colors = bgs(&spans);
        assert!(
            colors.contains(&DIFF_ADDED),
            "green (added) cell visible over the wide matched reduce"
        );
    }
}
