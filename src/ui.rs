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
            "  [/] search  [Tab/S-Tab] lane  [A/D] item  [W/S] zoom  [F] fit  [G] align  [Q] quit",
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
    let title = match app.alignment_label() {
        Some(status) => format!(
            " {} traces — {} ",
            app.traces.len(),
            status
        ),
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
        return;
    }
    let lane_width = total_width - label_width;

    let (ts_start, ts_end) = app.global_visible_window();
    let time_span = (ts_end - ts_start).max(1.0);

    let mut lines: Vec<Line> = Vec::new();

    let visible_rows = inner.height as usize;
    let start = app.lane_view_offset.min(app.lanes.len().saturating_sub(1));
    let end = (start + visible_rows).min(app.lanes.len());

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
        if app.use_gap_columns_for_lane(lane_idx) {
            spans.extend(build_lane_columns(app, lane_idx, lane_width));
        } else {
            spans.extend(build_lane(app, lane_idx, ts_start, time_span, lane_width));
        }
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
                let base = kernel_color(k.cat.as_str(), pos);
                let bg = app.kernel_diff_color(item_idx).unwrap_or(base);
                (
                    k.name.as_str(),
                    app.kernel_render_ts(item_idx),
                    app.kernel_render_end(item_idx),
                    bg,
                )
            }
            crate::app::Lane::Annotations { .. } => {
                let a = &app.annotations[item_idx];
                (
                    a.name.as_str(),
                    app.annotation_render_ts(item_idx),
                    app.annotation_render_end(item_idx),
                    ann_bg,
                )
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

// Gap-aligned diff layout: render one lane by LCS slot index rather than time.
// Matched kernels in T0/T1 of a stream share the same slot -> same screen x;
// a kernel present only in one trace draws a colored block in its lane and a
// black blank gap at the same slot in the opposite lane.
fn build_lane_columns(app: &App, lane_idx: usize, width: usize) -> Vec<Span<'static>> {
    let lane = &app.lanes[lane_idx];
    let (stream_id, trace_id) = match lane {
        crate::app::Lane::Kernels { stream_id, trace_id, .. } => (*stream_id, *trace_id),
        crate::app::Lane::Annotations { .. } => {
            return build_annotation_columns(app, lane_idx, width);
        }
    };
    let Some(layout) = app.stream_layout(stream_id) else {
        return build_lane(app, lane_idx, 0.0, 1.0, width);
    };
    let slots = &app.diff_columns_by_stream[&stream_id].slots;
    if width == 0 || layout.total_cols == 0 {
        return vec![Span::styled(" ".repeat(width), Style::new().bg(Color::Black))];
    }

    let selected_visual_col = selected_visual_col_for_stream(app, &layout, stream_id);
    let vp = crate::app::resolve_viewport(app.zoom_mode, layout.total_cols, width, selected_visual_col);

    let is_active_lane = lane_idx == app.active_lane;
    let selected_kernel = if is_active_lane {
        lane.item_indices().get(app.selected_item).copied()
    } else {
        None
    };

    let mut spans: Vec<Span<'static>> = Vec::with_capacity(width);
    for x in 0..width {
        let a = vp.window_start + x as f64 / vp.scale;
        let b = vp.window_start + (x + 1) as f64 / vp.scale;
        let start = (a.floor().max(0.0) as usize).min(layout.total_cols);
        let end = (b.ceil() as usize).clamp(start, layout.total_cols);
        let cols = if start < end {
            &layout.columns[start..end]
        } else {
            &[][..]
        };

        // Single-slot cell that holds the selected kernel: highlight it.
        let selected_here = selected_kernel.is_some_and(|sk| {
            cols.iter().any(|c| matches!(c, crate::app::VisualColumn::Slot(i)
                if slot_kernel(slots, *i, trace_id) == Some(sk)))
        });

        let (label, style) = if selected_here {
            (label_for_cell(app, cols, slots, trace_id), Style::new().fg(Color::Black).bg(Color::White))
        } else {
            let color = aggregate_kernel_cell_color(app, cols, slots, trace_id);
            (label_for_cell(app, cols, slots, trace_id), Style::new().fg(Color::Black).bg(color))
        };
        spans.push(Span::styled(label, style));
    }
    spans
}

fn slot_kernel(slots: &[crate::app::DiffColumnSlot], slot_idx: usize, trace_id: usize) -> Option<usize> {
    let slot = slots.get(slot_idx)?;
    if trace_id == 0 { slot.t0_kernel } else { slot.t1_kernel }
}

// One label char for a cell: the first covered kernel's initial, else space.
fn label_for_cell(
    app: &App,
    cols: &[crate::app::VisualColumn],
    slots: &[crate::app::DiffColumnSlot],
    trace_id: usize,
) -> String {
    for c in cols {
        if let crate::app::VisualColumn::Slot(i) = c {
            if let Some(idx) = slot_kernel(slots, *i, trace_id) {
                let name = app.kernels[idx].name.as_str();
                return ellipsize(name, 1);
            }
        }
    }
    " ".to_string()
}

// Per-lane color for a compressed cell covering a range of visual columns.
// Diff priority (Removed>Added) preserves S1/S2 opposite-lane gaps under fit.
fn aggregate_kernel_cell_color(
    app: &App,
    cols: &[crate::app::VisualColumn],
    slots: &[crate::app::DiffColumnSlot],
    trace_id: usize,
) -> Color {
    let mut has_removed = false;
    let mut has_added = false;
    let mut matched_slot: Option<usize> = None;
    for c in cols {
        let crate::app::VisualColumn::Slot(i) = c else { continue };
        let slot = &slots[*i];
        match (trace_id, slot.t0_kernel, slot.t1_kernel) {
            (0, Some(_), None) => has_removed = true,
            (1, None, Some(_)) => has_added = true,
            (_, Some(_), Some(_)) if matched_slot.is_none() => {
                matched_slot = Some(*i);
            }
            _ => {}
        }
    }
    if has_removed {
        Color::Rgb(220, 38, 38)
    } else if has_added {
        Color::Rgb(34, 197, 94)
    } else if let Some(i) = matched_slot {
        if let Some(idx) = slot_kernel(slots, i, trace_id) {
            kernel_color(app.kernels[idx].cat.as_str(), i)
        } else {
            Color::Black
        }
    } else {
        Color::Black
    }
}

// Visual column of the selected kernel within `stream_id`'s layout, for panning.
// Same-stream lanes center on the exact column; other streams pan proportionally.
fn selected_visual_col_for_stream(
    app: &App,
    layout: &crate::app::StreamLayout,
    stream_id: u64,
) -> Option<usize> {
    let active = app
        .lanes
        .get(app.active_lane)
        .filter(|l| matches!(l, crate::app::Lane::Kernels { .. }))?;
    let sel_idx = *active.item_indices().get(app.selected_item)?;
    let kc = app.kernel_diff_column.get(sel_idx).copied().flatten()?;
    if kc.stream_id == stream_id {
        layout.slot_to_visual_col.get(kc.column).copied()
    } else {
        let active_slots = app
            .diff_columns_by_stream
            .get(&kc.stream_id)
            .map(|c| c.slots.len())
            .unwrap_or(1)
            .max(1);
        let frac = kc.column as f64 / active_slots as f64;
        Some(((frac * layout.total_cols as f64) as usize).min(layout.total_cols.saturating_sub(1)))
    }
}

// Render an annotation lane in column space: each annotation spans the visual
// columns of the kernels it covers (by ts), so its end aligns to a kernel column
// under the SAME viewport as the kernel lane of that stream (fit/zoom/idle-gaps).
fn build_annotation_columns(app: &App, lane_idx: usize, width: usize) -> Vec<Span<'static>> {
    let lane = &app.lanes[lane_idx];
    let stream_id = lane.stream_id();
    let Some(layout) = app.stream_layout(stream_id) else {
        return build_lane(app, lane_idx, 0.0, 1.0, width);
    };
    if width == 0 || layout.total_cols == 0 {
        return vec![Span::styled(" ".repeat(width), Style::new().bg(Color::Black))];
    }
    let selected_visual_col = selected_visual_col_for_stream(app, &layout, stream_id);
    let vp = crate::app::resolve_viewport(app.zoom_mode, layout.total_cols, width, selected_visual_col);
    let ann_bg = Color::Rgb(90, 90, 110);

    // Paint each screen cell that falls inside any annotation's visual span.
    let mut fills: Vec<Option<usize>> = vec![None; width];
    for (pos, &ann_idx) in lane.item_indices().iter().enumerate() {
        let Some((lo, hi)) = app.annotation_visual_span(ann_idx, &layout) else {
            continue;
        };
        let x0_raw = (((lo as f64) - vp.window_start) * vp.scale).floor();
        let x1_raw = ((((hi + 1) as f64) - vp.window_start) * vp.scale).ceil() - 1.0;
        // Skip spans entirely outside the viewport (else clamping bleeds into cell 0).
        if x1_raw < 0.0 || x0_raw >= width as f64 {
            continue;
        }
        let x0 = x0_raw.max(0.0) as usize;
        let x1 = (x1_raw.max(0.0) as usize).min(width.saturating_sub(1));
        if x0 > x1 {
            continue;
        }
        for cell in fills.iter_mut().take(x1 + 1).skip(x0) {
            *cell = Some(pos);
        }
    }

    let is_active_lane = lane_idx == app.active_lane;
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(width);
    for (x, fill) in fills.iter().enumerate() {
        match fill {
            Some(pos) => {
                let pos = *pos;
                let ann_idx = lane.item_indices()[pos];
                let name = app.annotations[ann_idx].name.as_str();
                let selected = is_active_lane && pos == app.selected_item;
                // First cell of this annotation's run shows its initial.
                let first = x == 0 || fills[x - 1] != Some(pos);
                let label = if first { ellipsize(name, 1) } else { " ".to_string() };
                let style = if selected {
                    Style::new().fg(Color::Black).bg(Color::White)
                } else {
                    Style::new().fg(Color::White).bg(ann_bg)
                };
                spans.push(Span::styled(label, style));
            }
            None => spans.push(Span::styled(" ".to_string(), Style::new().bg(Color::Black))),
        }
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
