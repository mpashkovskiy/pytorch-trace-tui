use crate::trace::{AnnotationEvent, KernelEvent, Trace};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffStatus {
    Matched,
    Added,
    Removed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffColumnSlot {
    pub t0_kernel: Option<usize>,
    pub t1_kernel: Option<usize>,
    pub lead_gap_cols: u16,
}

#[derive(Debug, Clone)]
pub struct DiffStreamColumns {
    pub slots: Vec<DiffColumnSlot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelColumn {
    pub stream_id: u64,
    pub column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualColumn {
    Gap,
    Slot(usize),
}

#[derive(Debug, Clone)]
pub struct StreamLayout {
    pub stream_id: u64,
    pub columns: Vec<VisualColumn>,
    pub slot_to_visual_col: Vec<usize>,
    pub total_cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZoomMode {
    Scale(f64),
    Fit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizontalViewport {
    pub scale: f64,
    pub window_start: f64,
    pub width: usize,
}

const ZOOM_MIN: f64 = 0.1;
const ZOOM_FACTOR: f64 = 1.5;

/// A single navigable row on the timeline. Every lane — whether it holds GPU
/// kernels or user annotations — behaves identically for navigation.
#[derive(Debug, Clone)]
pub enum Lane {
    Kernels {
        stream_id: u64,
        trace_id: usize,
        item_indices: Vec<usize>,
    },
    Annotations {
        stream_id: u64,
        trace_id: usize,
        item_indices: Vec<usize>,
    },
}

impl Lane {
    pub fn stream_id(&self) -> u64 {
        match self {
            Lane::Kernels { stream_id, .. } | Lane::Annotations { stream_id, .. } => *stream_id,
        }
    }

    pub fn trace_id(&self) -> usize {
        match self {
            Lane::Kernels { trace_id, .. } | Lane::Annotations { trace_id, .. } => *trace_id,
        }
    }

    pub fn item_indices(&self) -> &[usize] {
        match self {
            Lane::Kernels { item_indices, .. } | Lane::Annotations { item_indices, .. } => {
                item_indices
            }
        }
    }

    pub fn is_annotations(&self) -> bool {
        matches!(self, Lane::Annotations { .. })
    }
}

/// The currently selected item in the active lane, either a kernel or an annotation.
#[derive(Debug, Clone, Copy)]
pub enum SelectedTraceItem<'a> {
    Kernel(&'a KernelEvent),
    Annotation(&'a AnnotationEvent),
}

/// Kernels from the current one up to (exclusive) the next same-named kernel in
/// the lane. `median` holds per-position median duration across repeated blocks.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub rows: Vec<(usize, String, f64)>,
    pub median: Option<Vec<(String, f64)>>,
    pub reps_found: usize,
    pub source_lane: usize,
    pub source_start: usize,
    pub scroll: usize,
}

/// Per-trace metadata for multi-trace alignment. `offset_us` is added to every
/// raw timestamp of this trace so a shared anchor annotation lines up on the
/// common time axis.
#[derive(Debug, Clone)]
pub struct TraceMeta {
    pub label: String,
    pub offset_us: f64,
    pub anchor: Option<String>,
}

#[derive(Debug, Clone)]
pub struct App {
    pub kernels: Vec<KernelEvent>,
    pub annotations: Vec<AnnotationEvent>,
    pub streams: Vec<u64>,
    pub lanes: Vec<Lane>,
    pub traces: Vec<TraceMeta>,
    pub active_lane: usize,
    pub selected_item: usize,
    pub zoom_level: f64,
    pub zoom_mode: ZoomMode,
    pub lane_view_offset: usize,
    pub search_active: bool,
    pub search_query: String,
    pub search_no_match: bool,
    pub sequence: Option<Sequence>,
    pub sequence_status: Option<String>,
    pub diff_status: Vec<DiffStatus>,
    pub diff_columns_by_stream: BTreeMap<u64, DiffStreamColumns>,
    pub kernel_diff_column: Vec<Option<KernelColumn>>,
}

impl App {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(trace: Trace) -> Self {
        App::new_multi(vec![("T0".to_string(), trace)])
    }

    /// Build an App from one or more labelled traces. Each trace's events are
    /// stamped with its trace index, alignment offsets are computed from the
    /// first shared annotation name, and lanes are interleaved round-robin.
    pub fn new_multi(labelled: Vec<(String, Trace)>) -> Self {
        let mut kernels: Vec<KernelEvent> = Vec::new();
        let mut annotations: Vec<AnnotationEvent> = Vec::new();
        let mut labels: Vec<String> = Vec::new();

        for (trace_id, (label, trace)) in labelled.into_iter().enumerate() {
            labels.push(label);
            for mut k in trace.kernels {
                k.trace_id = trace_id;
                kernels.push(k);
            }
            for mut a in trace.annotations {
                a.trace_id = trace_id;
                annotations.push(a);
            }
        }

        if labels.is_empty() {
            labels.push("T0".to_string());
        }
        let trace_count = labels.len();

        let traces = compute_alignment(&annotations, &labels);

        let mut streams: Vec<u64> = kernels
            .iter()
            .map(|k| k.stream)
            .chain(annotations.iter().map(|a| a.stream))
            .collect();
        streams.sort_unstable();
        streams.dedup();
        if streams.is_empty() {
            streams.push(0);
        }

        let per_trace: Vec<Vec<Lane>> = (0..trace_count)
            .map(|tid| build_trace_lanes(&kernels, &annotations, tid))
            .collect();
        let lanes = interleave_lanes(per_trace);

        // Start on the first kernel lane so the current selection is a kernel
        // (N / sequence work immediately); fall back to lane 0 if none exist.
        let initial_lane = lanes
            .iter()
            .position(|l| !l.is_annotations())
            .unwrap_or(0);

        let n_kernels = kernels.len();
        let mut app = App {
            kernels,
            annotations,
            streams,
            lanes,
            traces,
            active_lane: initial_lane,
            selected_item: 0,
            zoom_level: 1.0,
            zoom_mode: ZoomMode::Scale(1.0),
            lane_view_offset: 0,
            search_active: false,
            search_query: String::new(),
            search_no_match: false,
            sequence: None,
            sequence_status: None,
            diff_status: vec![DiffStatus::Matched; n_kernels],
            diff_columns_by_stream: BTreeMap::new(),
            kernel_diff_column: vec![None; n_kernels],
        };
        app.clamp_selected_item();
        app.compute_diff();
        app
    }

    fn clamp_selected_item(&mut self) {
        let len = self.active_lane_len();
        if len == 0 {
            self.selected_item = 0;
        } else if self.selected_item >= len {
            self.selected_item = len - 1;
        }
    }

    pub fn active_lane_len(&self) -> usize {
        self.lanes
            .get(self.active_lane)
            .map(|l| l.item_indices().len())
            .unwrap_or(0)
    }

    /// Stream id of the active lane (used for the header/label).
    pub fn active_stream(&self) -> u64 {
        self.lanes
            .get(self.active_lane)
            .map(|l| l.stream_id())
            .unwrap_or_else(|| self.streams.first().copied().unwrap_or(0))
    }

    pub fn active_lane_is_annotations(&self) -> bool {
        self.lanes
            .get(self.active_lane)
            .map(|l| l.is_annotations())
            .unwrap_or(false)
    }

    fn trace_offset(&self, trace_id: usize) -> f64 {
        self.traces.get(trace_id).map(|t| t.offset_us).unwrap_or(0.0)
    }

    pub fn compute_diff(&mut self) {
        if self.traces.len() < 2 {
            return;
        }
        for s in &mut self.diff_status {
            *s = DiffStatus::Matched;
        }
        self.diff_columns_by_stream.clear();
        self.kernel_diff_column = vec![None; self.kernels.len()];

        let mut streams: Vec<u64> = self.kernels.iter().map(|k| k.stream).collect();
        streams.sort_unstable();
        streams.dedup();

        for stream in streams {
            let mut t0: Vec<usize> = self.kernels.iter().enumerate()
                .filter(|(_, k)| k.trace_id == 0 && k.stream == stream)
                .map(|(i, _)| i)
                .collect();
            let mut t1: Vec<usize> = self.kernels.iter().enumerate()
                .filter(|(_, k)| k.trace_id == 1 && k.stream == stream)
                .map(|(i, _)| i)
                .collect();
            if t0.is_empty() || t1.is_empty() {
                t0.sort_by(|&a, &b| self.kernels[a].ts.partial_cmp(&self.kernels[b].ts)
                    .unwrap_or(std::cmp::Ordering::Equal));
                t1.sort_by(|&a, &b| self.kernels[a].ts.partial_cmp(&self.kernels[b].ts)
                    .unwrap_or(std::cmp::Ordering::Equal));
                let mut slots: Vec<DiffColumnSlot> = Vec::new();
                for (col, &g) in t0.iter().enumerate() {
                    self.diff_status[g] = DiffStatus::Removed;
                    self.kernel_diff_column[g] = Some(KernelColumn { stream_id: stream, column: col });
                    slots.push(DiffColumnSlot { t0_kernel: Some(g), t1_kernel: None, lead_gap_cols: 0 });
                }
                for (idx, &g) in t1.iter().enumerate() {
                    let col = t0.len() + idx;
                    self.diff_status[g] = DiffStatus::Added;
                    self.kernel_diff_column[g] = Some(KernelColumn { stream_id: stream, column: col });
                    slots.push(DiffColumnSlot { t0_kernel: None, t1_kernel: Some(g), lead_gap_cols: 0 });
                }
                self.diff_columns_by_stream.insert(stream, DiffStreamColumns { slots });
                continue;
            }
            t0.sort_by(|&a, &b| self.kernels[a].ts.partial_cmp(&self.kernels[b].ts)
                .unwrap_or(std::cmp::Ordering::Equal));
            t1.sort_by(|&a, &b| self.kernels[a].ts.partial_cmp(&self.kernels[b].ts)
                .unwrap_or(std::cmp::Ordering::Equal));

            let pairs = {
                let names0: Vec<&str> = t0.iter().map(|&i| self.kernels[i].name.as_str()).collect();
                let names1: Vec<&str> = t1.iter().map(|&i| self.kernels[i].name.as_str()).collect();
                crate::diff::myers_lcs(&names0, &names1)
            };

            let mut slots: Vec<DiffColumnSlot> = Vec::with_capacity(pairs.len());
            for (slot_idx, (li, ri)) in pairs.into_iter().enumerate() {
                match (li, ri) {
                    (Some(a), Some(b)) => {
                        let g0 = t0[a];
                        let g1 = t1[b];
                        self.diff_status[g0] = DiffStatus::Matched;
                        self.diff_status[g1] = DiffStatus::Matched;
                        self.kernel_diff_column[g0] = Some(KernelColumn { stream_id: stream, column: slot_idx });
                        self.kernel_diff_column[g1] = Some(KernelColumn { stream_id: stream, column: slot_idx });
                        slots.push(DiffColumnSlot { t0_kernel: Some(g0), t1_kernel: Some(g1), lead_gap_cols: 0 });
                    }
                    (Some(a), None) => {
                        let g0 = t0[a];
                        self.diff_status[g0] = DiffStatus::Removed;
                        self.kernel_diff_column[g0] = Some(KernelColumn { stream_id: stream, column: slot_idx });
                        slots.push(DiffColumnSlot { t0_kernel: Some(g0), t1_kernel: None, lead_gap_cols: 0 });
                    }
                    (None, Some(b)) => {
                        let g1 = t1[b];
                        self.diff_status[g1] = DiffStatus::Added;
                        self.kernel_diff_column[g1] = Some(KernelColumn { stream_id: stream, column: slot_idx });
                        slots.push(DiffColumnSlot { t0_kernel: None, t1_kernel: Some(g1), lead_gap_cols: 0 });
                    }
                    (None, None) => {
                        slots.push(DiffColumnSlot { t0_kernel: None, t1_kernel: None, lead_gap_cols: 0 });
                    }
                }
            }
            self.diff_columns_by_stream.insert(stream, DiffStreamColumns { slots });
        }
        self.fill_lead_gaps();
    }

    // Idle time between consecutive T0 kernels becomes proportional blank columns
    // (lead_gap_cols) before the slot that starts the following kernel, so the
    // reference trace's real spacing survives contiguous column packing.
    fn fill_lead_gaps(&mut self) {
        let streams: Vec<u64> = self.diff_columns_by_stream.keys().copied().collect();
        for stream in streams {
            let n = self.diff_columns_by_stream[&stream].slots.len();
            // Ordered (prev_t0_end, this_slot_idx) for slots whose T0 kernel exists.
            let mut idles: Vec<f64> = Vec::new();
            let mut per_slot: Vec<(usize, f64)> = Vec::new();
            let mut prev_end: Option<f64> = None;
            for slot_idx in 0..n {
                let t0 = self.diff_columns_by_stream[&stream].slots[slot_idx].t0_kernel;
                let Some(g) = t0 else { continue };
                let start = self.kernels[g].ts;
                if let Some(pe) = prev_end {
                    let idle = (start - pe).max(0.0);
                    if idle > 0.0 {
                        idles.push(idle);
                    }
                    per_slot.push((slot_idx, idle));
                }
                prev_end = Some(self.kernels[g].end_ts());
            }
            let gap_unit = median_of(&idles);
            if gap_unit <= 0.0 {
                continue;
            }
            for (slot_idx, idle) in per_slot {
                if idle <= 0.0 {
                    continue;
                }
                let cells = ((idle / gap_unit).ceil()).clamp(1.0, 8.0) as u16;
                self.diff_columns_by_stream
                    .get_mut(&stream)
                    .unwrap()
                    .slots[slot_idx]
                    .lead_gap_cols = cells;
            }
        }
    }

    pub fn stream_layout(&self, stream_id: u64) -> Option<StreamLayout> {
        let slots = &self.diff_columns_by_stream.get(&stream_id)?.slots;
        let mut columns: Vec<VisualColumn> = Vec::new();
        let mut slot_to_visual_col = vec![0usize; slots.len()];
        for (slot_idx, slot) in slots.iter().enumerate() {
            for _ in 0..slot.lead_gap_cols {
                columns.push(VisualColumn::Gap);
            }
            slot_to_visual_col[slot_idx] = columns.len();
            columns.push(VisualColumn::Slot(slot_idx));
        }
        Some(StreamLayout {
            stream_id,
            total_cols: columns.len(),
            columns,
            slot_to_visual_col,
        })
    }

    // Visual-column [start,end] an annotation should span: the columns of the
    // kernels (same trace+stream) whose start ts falls within the annotation's
    // time span. Empty coverage falls back to the nearest kernel (1-col block).
    pub fn annotation_visual_span(
        &self,
        ann_idx: usize,
        layout: &StreamLayout,
    ) -> Option<(usize, usize)> {
        let ann = self.annotations.get(ann_idx)?;
        let ann_start = ann.ts;
        let ann_end = ann.end_ts();

        let mut covered_cols: Vec<usize> = Vec::new();
        for (kidx, k) in self.kernels.iter().enumerate() {
            if k.trace_id != ann.trace_id || k.stream != ann.stream {
                continue;
            }
            if k.ts >= ann_start && k.ts <= ann_end {
                if let Some(kc) = self.kernel_diff_column.get(kidx).copied().flatten() {
                    if kc.stream_id == layout.stream_id {
                        if let Some(&vc) = layout.slot_to_visual_col.get(kc.column) {
                            covered_cols.push(vc);
                        }
                    }
                }
            }
        }
        if let (Some(&lo), Some(&hi)) = (covered_cols.iter().min(), covered_cols.iter().max()) {
            return Some((lo, hi));
        }

        // Fallback: nearest kernel by ts in the same trace+stream, minimal 1 col.
        let nearest = self
            .kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| k.trace_id == ann.trace_id && k.stream == ann.stream)
            .min_by(|(_, a), (_, b)| {
                (a.ts - ann_start).abs().total_cmp(&(b.ts - ann_start).abs())
            })
            .map(|(i, _)| i)?;
        let kc = self.kernel_diff_column.get(nearest).copied().flatten()?;
        let vc = *layout.slot_to_visual_col.get(kc.column)?;
        Some((vc, vc))
    }

    pub fn kernel_diff_color(&self, idx: usize) -> Option<ratatui::style::Color> {
        match self.diff_status.get(idx) {
            Some(DiffStatus::Added)   => Some(ratatui::style::Color::Rgb(34, 197, 94)),
            Some(DiffStatus::Removed) => Some(ratatui::style::Color::Rgb(220, 38, 38)),
            _                         => None,
        }
    }

    pub fn use_gap_columns_for_lane(&self, lane_idx: usize) -> bool {
        let Some(lane) = self.lanes.get(lane_idx) else {
            return false;
        };
        match lane {
            Lane::Kernels { trace_id, stream_id, .. } => {
                (*trace_id == 0 || *trace_id == 1)
                    && self.traces.len() == 2
                    && self.diff_columns_by_stream.contains_key(stream_id)
            }
            Lane::Annotations { trace_id, stream_id, .. } => {
                (*trace_id == 0 || *trace_id == 1)
                    && self.traces.len() == 2
                    && self.diff_columns_by_stream.contains_key(stream_id)
            }
        }
    }

    /// Aligned start ts of kernel `idx`: raw ts + its trace's alignment offset.
    pub fn kernel_render_ts(&self, idx: usize) -> f64 {
        self.kernels
            .get(idx)
            .map(|k| k.ts + self.trace_offset(k.trace_id))
            .unwrap_or(0.0)
    }

    pub fn kernel_render_end(&self, idx: usize) -> f64 {
        self.kernels
            .get(idx)
            .map(|k| k.end_ts() + self.trace_offset(k.trace_id))
            .unwrap_or(0.0)
    }

    pub fn annotation_render_ts(&self, idx: usize) -> f64 {
        self.annotations
            .get(idx)
            .map(|a| a.ts + self.trace_offset(a.trace_id))
            .unwrap_or(0.0)
    }

    pub fn annotation_render_end(&self, idx: usize) -> f64 {
        self.annotations
            .get(idx)
            .map(|a| a.end_ts() + self.trace_offset(a.trace_id))
            .unwrap_or(0.0)
    }

    /// Human-readable alignment status for the header, or None for a single
    /// trace where alignment is not meaningful.
    pub fn alignment_label(&self) -> Option<String> {
        if self.traces.len() < 2 {
            return None;
        }
        match self.traces.first().and_then(|t| t.anchor.as_deref()) {
            Some(name) => Some(format!("aligned on {:?}", name)),
            None => Some("not aligned (no shared annotation)".to_string()),
        }
    }

    /// Per-trace display labels shortened to their differing part (common prefix
    /// and suffix stripped), for compact lane labels.
    pub fn trace_display_labels(&self) -> Vec<String> {
        let labels: Vec<String> = self.traces.iter().map(|t| t.label.clone()).collect();
        shorten_labels(&labels)
    }

    /// Realigns other traces to the currently-selected kernel: the selected
    /// trace stays fixed, and every other trace is shifted so its same-named
    /// kernel nearest (by aligned start) to the selection slides onto it. No-op
    /// unless a kernel is selected. Returns whether an alignment was performed.
    pub fn align_to_selected_kernel(&mut self) -> bool {
        let lane = match self.lanes.get(self.active_lane) {
            Some(l) if !l.is_annotations() => l,
            _ => return false,
        };
        let Some(&kidx) = lane.item_indices().get(self.selected_item) else {
            return false;
        };
        let Some(kernel) = self.kernels.get(kidx) else {
            return false;
        };
        let ref_trace = kernel.trace_id;
        let name = kernel.name.clone();
        let target = self.kernel_render_ts(kidx);

        for tid in 0..self.traces.len() {
            if tid == ref_trace {
                continue;
            }
            let nearest_raw = self
                .kernels
                .iter()
                .filter(|k| k.trace_id == tid && k.name == name)
                .map(|k| k.ts)
                .min_by(|a, b| {
                    (a - target)
                        .abs()
                        .partial_cmp(&(b - target).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(raw_start) = nearest_raw {
                if let Some(meta) = self.traces.get_mut(tid) {
                    meta.offset_us = target - raw_start;
                }
            }
        }

        for meta in &mut self.traces {
            meta.anchor = Some(name.clone());
        }
        true
    }

    pub fn selected_trace_item(&self) -> Option<SelectedTraceItem<'_>> {
        let lane = self.lanes.get(self.active_lane)?;
        let idx = *lane.item_indices().get(self.selected_item)?;
        match lane {
            Lane::Kernels { .. } => self.kernels.get(idx).map(SelectedTraceItem::Kernel),
            Lane::Annotations { .. } => {
                self.annotations.get(idx).map(SelectedTraceItem::Annotation)
            }
        }
    }

    // ── Item navigation (A/D) — no wrap, matches previous kernel navigation ──

    pub fn prev_item(&mut self) {
        if self.selected_item > 0 {
            self.selected_item -= 1;
        }
    }

    pub fn next_item(&mut self) {
        if self.selected_item + 1 < self.active_lane_len() {
            self.selected_item += 1;
        }
    }

    // ── Lane navigation (Tab / Shift+Tab) — wraps, picks nearest-ts item ─────

    pub fn next_lane(&mut self) {
        if self.lanes.len() <= 1 {
            return;
        }
        let target = (self.active_lane + 1) % self.lanes.len();
        self.move_to_lane(target);
    }

    pub fn prev_lane(&mut self) {
        if self.lanes.len() <= 1 {
            return;
        }
        let target = (self.active_lane + self.lanes.len() - 1) % self.lanes.len();
        self.move_to_lane(target);
    }

    fn move_to_lane(&mut self, target: usize) {
        // Gap-column lanes navigate in column space: carry the selected kernel's
        // LCS column so the target lands on the same slot (or nearest present).
        if self.use_gap_columns_for_lane(self.active_lane)
            && self.use_gap_columns_for_lane(target)
        {
            if let Some(kc) = self.selected_item_column() {
                self.active_lane = target;
                self.selected_item = self.nearest_present_item_by_column(kc);
                return;
            }
        }
        let prev_ts = self.selected_item_render_ts();
        self.active_lane = target;
        self.selected_item = match prev_ts {
            Some(ts) => self.nearest_item_in_active_lane(ts),
            None => 0,
        };
    }

    fn selected_item_column(&self) -> Option<KernelColumn> {
        let lane = self.lanes.get(self.active_lane)?;
        let idx = *lane.item_indices().get(self.selected_item)?;
        self.kernel_diff_column.get(idx).copied().flatten()
    }

    fn nearest_present_item_by_column(&self, target: KernelColumn) -> usize {
        let Some(lane) = self.lanes.get(self.active_lane) else {
            return 0;
        };
        let mut best = 0usize;
        let mut best_diff = usize::MAX;
        for (pos, &idx) in lane.item_indices().iter().enumerate() {
            if let Some(kc) = self.kernel_diff_column.get(idx).copied().flatten() {
                if kc.stream_id != target.stream_id {
                    continue;
                }
                let diff = kc.column.abs_diff(target.column);
                if diff < best_diff {
                    best_diff = diff;
                    best = pos;
                }
            }
        }
        best
    }

    #[cfg(test)]
    fn move_to_lane_for_test(&mut self, target: usize) {
        self.move_to_lane(target);
    }

    /// Aligned ts of the item at `pos` in `lane` (raw ts + trace offset), so
    /// cross-trace navigation compares positions on the shared visual axis.
    fn item_render_ts(&self, lane: &Lane, pos: usize) -> Option<f64> {
        let idx = *lane.item_indices().get(pos)?;
        match lane {
            Lane::Kernels { .. } => {
                self.kernels.get(idx).map(|_| self.kernel_render_ts(idx))
            }
            Lane::Annotations { .. } => {
                self.annotations.get(idx).map(|_| self.annotation_render_ts(idx))
            }
        }
    }

    fn selected_item_render_ts(&self) -> Option<f64> {
        let lane = self.lanes.get(self.active_lane)?;
        self.item_render_ts(lane, self.selected_item)
    }

    fn nearest_item_in_active_lane(&self, target_ts: f64) -> usize {
        let Some(lane) = self.lanes.get(self.active_lane) else {
            return 0;
        };
        let n = lane.item_indices().len();
        if n == 0 {
            return 0;
        }
        let mut best = 0usize;
        let mut best_diff = f64::MAX;
        for pos in 0..n {
            if let Some(ts) = self.item_render_ts(lane, pos) {
                let diff = (ts - target_ts).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best = pos;
                }
            }
        }
        best
    }

    pub fn zoom_in(&mut self) {
        self.zoom_level *= ZOOM_FACTOR;
        self.zoom_mode = ZoomMode::Scale(self.zoom_level);
    }

    pub fn zoom_out(&mut self) {
        self.zoom_level = (self.zoom_level / ZOOM_FACTOR).max(ZOOM_MIN);
        self.zoom_mode = ZoomMode::Scale(self.zoom_level);
    }

    pub fn zoom_fit(&mut self) {
        self.zoom_mode = ZoomMode::Fit;
    }

    // ── Search over BOTH lane kinds ──────────────────────────────────────────

    pub fn search_start(&mut self) {
        self.search_active = true;
        self.search_query.clear();
        self.search_no_match = false;
    }

    pub fn search_cancel(&mut self) {
        self.search_active = false;
        self.search_query.clear();
        self.search_no_match = false;
    }

    pub fn search_commit(&mut self) {
        self.search_active = false;
    }

    pub fn search_push(&mut self, c: char) {
        self.search_query.push(c);
        self.search_apply();
    }

    pub fn search_backspace(&mut self) {
        self.search_query.pop();
        self.search_apply();
    }

    fn search_apply(&mut self) {
        if self.search_query.is_empty() {
            self.search_no_match = false;
            return;
        }
        let needle = self.search_query.to_lowercase();

        for lane_idx in 0..self.lanes.len() {
            let lane = &self.lanes[lane_idx];
            for (pos, &item_idx) in lane.item_indices().iter().enumerate() {
                let name = match lane {
                    Lane::Kernels { .. } => self.kernels.get(item_idx).map(|k| &k.name),
                    Lane::Annotations { .. } => self.annotations.get(item_idx).map(|a| &a.name),
                };
                if let Some(name) = name {
                    if name.to_lowercase().contains(&needle) {
                        self.active_lane = lane_idx;
                        self.selected_item = pos;
                        self.search_no_match = false;
                        return;
                    }
                }
            }
        }
        self.search_no_match = true;
    }

    // ── Sequence between same-named kernels (N key) ──────────────────────────

    fn active_kernel_lane(&self) -> Option<(usize, &[usize])> {
        let lane = self.lanes.get(self.active_lane)?;
        match lane {
            Lane::Kernels { item_indices, .. } => Some((self.active_lane, item_indices)),
            Lane::Annotations { .. } => None,
        }
    }

    pub fn start_sequence(&mut self) -> bool {
        let Some((lane_idx, item_indices)) = self.active_kernel_lane() else {
            return false;
        };
        let start = self.selected_item;
        let Some(&start_kernel_idx) = item_indices.get(start) else {
            return false;
        };
        let Some(start_kernel) = self.kernels.get(start_kernel_idx) else {
            return false;
        };
        let name = start_kernel.name.clone();

        // End at the next kernel with the same name (exclusive), or the lane end.
        let mut end = item_indices.len();
        for pos in (start + 1)..item_indices.len() {
            if let Some(k) = item_indices.get(pos).and_then(|&i| self.kernels.get(i)) {
                if k.name == name {
                    end = pos;
                    break;
                }
            }
        }

        let rows: Vec<(usize, String, f64)> = item_indices[start..end]
            .iter()
            .enumerate()
            .filter_map(|(offset, &kidx)| {
                self.kernels
                    .get(kidx)
                    .map(|k| (offset + 1, k.name.clone(), k.dur))
            })
            .collect();

        self.sequence = Some(Sequence {
            rows,
            median: None,
            reps_found: 1,
            source_lane: lane_idx,
            source_start: start,
            scroll: 0,
        });
        self.sequence_status = None;
        self.extend_sequence_median();
        true
    }

    pub fn close_sequence(&mut self) {
        self.sequence = None;
        self.sequence_status = None;
    }

    pub fn sequence_scroll_up(&mut self, amount: usize) {
        if let Some(seq) = self.sequence.as_mut() {
            seq.scroll = seq.scroll.saturating_sub(amount);
        }
    }

    pub fn sequence_scroll_down(&mut self, amount: usize, viewport: usize) {
        if let Some(seq) = self.sequence.as_mut() {
            let max = seq.rows.len().saturating_sub(viewport);
            seq.scroll = (seq.scroll + amount).min(max);
        }
    }

    pub fn sequence_csv(&self) -> Option<String> {
        let seq = self.sequence.as_ref()?;
        let mut out = String::from("idx\tkernel name\tmedian\n");
        for (i, (idx, name, _dur)) in seq.rows.iter().enumerate() {
            let med = seq
                .median
                .as_ref()
                .and_then(|m| m.get(i))
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            out.push_str(&format!("{}\t{}\t{:.2}\n", idx, name, med));
        }
        Some(out)
    }

    /// Scan forward for every block whose ordered kernel names exactly match the
    /// captured sequence — occurrences may be non-contiguous (intervening
    /// non-matching kernels are skipped) — then compute the per-position median
    /// duration across every matching block, including the original.
    pub fn extend_sequence_median(&mut self) {
        let Some(seq) = self.sequence.as_ref() else {
            return;
        };
        let block_len = seq.rows.len();
        if block_len == 0 {
            return;
        }
        let pattern: Vec<String> = seq.rows.iter().map(|(_, n, _)| n.clone()).collect();
        let source_lane = seq.source_lane;
        let source_start = seq.source_start;

        let Some(Lane::Kernels { item_indices, .. }) = self.lanes.get(source_lane) else {
            return;
        };
        let item_indices = item_indices.clone();

        let mut per_pos: Vec<Vec<f64>> = vec![Vec::new(); block_len];
        let mut reps = 0usize;
        let mut start = source_start;
        while start + block_len <= item_indices.len() {
            let matches = (0..block_len).all(|off| {
                item_indices
                    .get(start + off)
                    .and_then(|&i| self.kernels.get(i))
                    .map(|k| k.name == pattern[off])
                    .unwrap_or(false)
            });
            if matches {
                for (off, slot) in per_pos.iter_mut().enumerate() {
                    if let Some(k) =
                        item_indices.get(start + off).and_then(|&i| self.kernels.get(i))
                    {
                        slot.push(k.dur);
                    }
                }
                reps += 1;
                start += block_len;
            } else {
                start += 1;
            }
        }

        let median: Vec<(String, f64)> = pattern
            .iter()
            .zip(per_pos.iter())
            .map(|(name, durs)| (name.clone(), median_of(durs)))
            .collect();

        if let Some(seq) = self.sequence.as_mut() {
            seq.median = Some(median);
            seq.reps_found = reps;
        }
    }

    // ── Vertical scroll — one lane == one rendered row ───────────────────────

    pub fn ensure_active_lane_visible(&mut self, visible_rows: usize) {
        if visible_rows == 0 {
            return;
        }
        if self.active_lane < self.lane_view_offset {
            self.lane_view_offset = self.active_lane;
        } else if self.active_lane >= self.lane_view_offset + visible_rows {
            self.lane_view_offset = self.active_lane + 1 - visible_rows;
        }
    }

    pub fn global_time_bounds(&self) -> (f64, f64) {
        let mut min_start = f64::MAX;
        let mut max_end = f64::MIN;
        for (i, _) in self.kernels.iter().enumerate() {
            min_start = min_start.min(self.kernel_render_ts(i));
            max_end = max_end.max(self.kernel_render_end(i));
        }
        for (i, _) in self.annotations.iter().enumerate() {
            min_start = min_start.min(self.annotation_render_ts(i));
            max_end = max_end.max(self.annotation_render_end(i));
        }
        if min_start > max_end {
            (0.0, 1.0)
        } else {
            (min_start, (max_end).max(min_start + 1.0))
        }
    }

    /// Aligned center ts of the selected item (raw ts + its trace's offset),
    /// used to center the visible window on the shared aligned axis.
    fn selected_render_center(&self) -> Option<f64> {
        let lane = self.lanes.get(self.active_lane)?;
        let idx = *lane.item_indices().get(self.selected_item)?;
        match lane {
            Lane::Kernels { .. } => {
                Some((self.kernel_render_ts(idx) + self.kernel_render_end(idx)) / 2.0)
            }
            Lane::Annotations { .. } => {
                Some((self.annotation_render_ts(idx) + self.annotation_render_end(idx)) / 2.0)
            }
        }
    }

    pub fn global_visible_window(&self) -> (f64, f64) {
        let (g_start, g_end) = self.global_time_bounds();
        let total_span = (g_end - g_start).max(1.0);
        let visible_span = (total_span / self.zoom_level).max(1e-3);

        let center = self
            .selected_render_center()
            .unwrap_or(g_start + total_span / 2.0);

        let ts_start = (center - visible_span / 2.0).max(g_start);
        let ts_end = (center + visible_span / 2.0).min(g_end);
        (ts_start, ts_end.max(ts_start + visible_span))
    }

    pub fn zoom_label(&self) -> String {
        if self.zoom_mode == ZoomMode::Fit {
            return "fit".to_string();
        }
        let z = self.zoom_level;
        if z >= 10.0 {
            format!("{:.0}x", z)
        } else {
            format!("{:.1}x", z)
        }
    }
}

/// Build the flat, ordered lane list. For each stream (sorted) we emit its
/// Earliest start ts of the given annotation name within one trace, if present.
fn anchor_ts(annotations: &[AnnotationEvent], trace_id: usize, name: &str) -> Option<f64> {
    annotations
        .iter()
        .filter(|a| a.trace_id == trace_id && a.name == name)
        .map(|a| a.ts)
        .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
}

/// Computes per-trace alignment offsets. The anchor is the first annotation name
/// (in trace 0's timestamp order) that appears in EVERY trace; each trace is then
/// shifted so that anchor's start coincides with trace 0's. If no name is shared,
/// all offsets are 0 and a warning is emitted to stderr.
fn compute_alignment(annotations: &[AnnotationEvent], labels: &[String]) -> Vec<TraceMeta> {
    let trace_count = labels.len();

    let mut t0_names: Vec<&str> = annotations
        .iter()
        .filter(|a| a.trace_id == 0)
        .map(|a| (a.ts, a.name.as_str()))
        .collect::<Vec<_>>()
        .into_iter()
        .fold(Vec::new(), |mut acc, (_, n)| {
            if !acc.contains(&n) {
                acc.push(n);
            }
            acc
        });
    // Order candidate names by earliest occurrence in trace 0.
    t0_names.sort_by(|a, b| {
        let ta = anchor_ts(annotations, 0, a).unwrap_or(f64::MAX);
        let tb = anchor_ts(annotations, 0, b).unwrap_or(f64::MAX);
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let anchor = t0_names.into_iter().find(|name| {
        (0..trace_count).all(|tid| anchor_ts(annotations, tid, name).is_some())
    });

    match anchor {
        Some(name) => {
            let ref_ts = anchor_ts(annotations, 0, name).unwrap_or(0.0);
            labels
                .iter()
                .enumerate()
                .map(|(tid, label)| {
                    let this_ts = anchor_ts(annotations, tid, name).unwrap_or(ref_ts);
                    TraceMeta {
                        label: label.clone(),
                        offset_us: ref_ts - this_ts,
                        anchor: Some(name.to_string()),
                    }
                })
                .collect()
        }
        None => {
            if trace_count > 1 {
                eprintln!(
                    "warning: no annotation name shared across all {} traces; \
                     rendering without alignment (raw timestamps).",
                    trace_count
                );
            }
            labels
                .iter()
                .map(|label| TraceMeta {
                    label: label.clone(),
                    offset_us: 0.0,
                    anchor: None,
                })
                .collect()
        }
    }
}

/// Shortens trace labels to just their differing middle by stripping the common
/// character prefix and suffix shared by all labels. Falls back to `T{index}`
/// when there is a single label that would become empty, or when any shortened
/// label is empty (labels identical, or one is a substring boundary of another).
fn shorten_labels(labels: &[String]) -> Vec<String> {
    if labels.len() < 2 {
        return labels.to_vec();
    }

    let cols: Vec<Vec<char>> = labels.iter().map(|s| s.chars().collect()).collect();
    let min_len = cols.iter().map(|c| c.len()).min().unwrap_or(0);

    let mut prefix = 0;
    while prefix < min_len && cols.iter().all(|c| c[prefix] == cols[0][prefix]) {
        prefix += 1;
    }

    let mut suffix = 0;
    while suffix < min_len - prefix
        && cols
            .iter()
            .all(|c| c[c.len() - 1 - suffix] == cols[0][cols[0].len() - 1 - suffix])
    {
        suffix += 1;
    }

    let shortened: Vec<String> = cols
        .iter()
        .map(|c| c[prefix..c.len() - suffix].iter().collect::<String>())
        .collect();

    if shortened.iter().any(|s| s.is_empty()) {
        return (0..labels.len()).map(|i| format!("T{}", i)).collect();
    }
    shortened
}

/// Builds the lane list for ONE trace: for each of that trace's streams, an
/// annotation lane (if any) then a kernel lane (if any). Item indices point into
/// the shared flat vecs and are filtered by `trace_id`, then sorted by timestamp.
fn build_trace_lanes(
    kernels: &[KernelEvent],
    annotations: &[AnnotationEvent],
    trace_id: usize,
) -> Vec<Lane> {
    let mut streams: Vec<u64> = kernels
        .iter()
        .filter(|k| k.trace_id == trace_id)
        .map(|k| k.stream)
        .chain(
            annotations
                .iter()
                .filter(|a| a.trace_id == trace_id)
                .map(|a| a.stream),
        )
        .collect();
    streams.sort_unstable();
    streams.dedup();

    let mut lanes = Vec::new();
    for &stream_id in &streams {
        let mut ann: Vec<usize> = annotations
            .iter()
            .enumerate()
            .filter(|(_, a)| a.trace_id == trace_id && a.stream == stream_id)
            .map(|(i, _)| i)
            .collect();
        ann.sort_by(|&a, &b| {
            annotations[a]
                .ts
                .partial_cmp(&annotations[b].ts)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if !ann.is_empty() {
            lanes.push(Lane::Annotations {
                stream_id,
                trace_id,
                item_indices: ann,
            });
        }

        let mut kern: Vec<usize> = kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| k.trace_id == trace_id && k.stream == stream_id)
            .map(|(i, _)| i)
            .collect();
        kern.sort_by(|&a, &b| {
            kernels[a]
                .ts
                .partial_cmp(&kernels[b].ts)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if !kern.is_empty() {
            lanes.push(Lane::Kernels {
                stream_id,
                trace_id,
                item_indices: kern,
            });
        }
    }
    lanes
}

/// Interleaves per-trace lane lists round-robin by lane index: row 0 of every
/// trace, then row 1 of every trace, etc. Traces that lack a given row are
/// skipped, so unequal lane counts collapse gracefully.
fn interleave_lanes(per_trace: Vec<Vec<Lane>>) -> Vec<Lane> {
    let max_len = per_trace.iter().map(|l| l.len()).max().unwrap_or(0);
    let mut lanes = Vec::new();
    for row in 0..max_len {
        for trace_lanes in &per_trace {
            if let Some(lane) = trace_lanes.get(row) {
                lanes.push(lane.clone());
            }
        }
    }
    if lanes.is_empty() {
        lanes.push(Lane::Kernels {
            stream_id: 0,
            trace_id: 0,
            item_indices: vec![],
        });
    }
    lanes
}

pub fn kernel_columns(
    k_ts: f64,
    k_end: f64,
    ts_start: f64,
    ts_end: f64,
    width: usize,
) -> Option<(usize, usize)> {
    if k_end <= ts_start || k_ts >= ts_end {
        return None;
    }
    let time_span = (ts_end - ts_start).max(1.0);
    let col_of = |ts: f64| -> f64 { (ts - ts_start) / time_span * width as f64 };
    let start_col = col_of(k_ts.max(ts_start)).floor().max(0.0) as usize;
    let end_col = (col_of(k_end.min(ts_end)).ceil() as usize)
        .max(start_col + 1)
        .min(width);
    if start_col >= width {
        return None;
    }
    Some((start_col, end_col))
}

/// Median of the durations. Even counts average the two middle values; only
/// finite values are considered, and comparison uses `total_cmp` for ordering.
fn median_of(durs: &[f64]) -> f64 {
    let mut finite: Vec<f64> = durs.iter().copied().filter(|d| d.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let n = finite.len();
    if !n.is_multiple_of(2) {
        finite[n / 2]
    } else {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    }
}

// Horizontal viewport for column rendering. Fit compresses every visual column
// into `width` cells (scale may be < 1, many columns per cell); Scale keeps
// `scale` cells per column and centers the window on the selected column.
pub fn resolve_viewport(
    mode: ZoomMode,
    total_cols: usize,
    width: usize,
    selected_visual_col: Option<usize>,
) -> HorizontalViewport {
    if total_cols == 0 || width == 0 {
        return HorizontalViewport { scale: 1.0, window_start: 0.0, width };
    }
    match mode {
        ZoomMode::Fit => HorizontalViewport {
            scale: (width as f64 / total_cols as f64).min(1.0),
            window_start: 0.0,
            width,
        },
        ZoomMode::Scale(s) => {
            let scale = s.max(1e-6);
            let visible_cols = width as f64 / scale;
            let center = selected_visual_col.unwrap_or(0) as f64;
            let max_start = (total_cols as f64 - visible_cols).max(0.0);
            let window_start = (center - visible_cols / 2.0).clamp(0.0, max_start);
            HorizontalViewport { scale, window_start, width }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_from(kernels: Vec<KernelEvent>) -> App {
        App::new(Trace {
            kernels,
            annotations: vec![],
        })
    }

    fn make_kernel(stream: u64, ts: f64, dur: f64) -> KernelEvent {
        KernelEvent {
            name: format!("kernel_s{}_t{}", stream, ts as u64),
            cat: "kernel".to_string(),
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

    fn named_kernel(stream: u64, ts: f64, name: &str) -> KernelEvent {
        KernelEvent {
            name: name.to_string(),
            cat: "kernel".to_string(),
            ts,
            dur: 5.0,
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

    fn sample_app() -> App {
        let kernels = vec![
            make_kernel(7, 100.0, 10.0),
            make_kernel(7, 200.0, 20.0),
            make_kernel(7, 300.0, 5.0),
            make_kernel(3, 150.0, 8.0),
            make_kernel(3, 250.0, 12.0),
        ];
        app_from(kernels)
    }

    #[test]
    fn test_streams_sorted() {
        let app = sample_app();
        assert_eq!(app.streams, vec![3, 7]);
    }

    #[test]
    fn test_lanes_kernel_only() {
        // Two streams, no annotations → two kernel lanes, stream 3 first.
        let app = sample_app();
        assert_eq!(app.lanes.len(), 2);
        assert_eq!(app.lanes[0].stream_id(), 3);
        assert!(!app.lanes[0].is_annotations());
        assert_eq!(app.lanes[0].item_indices().len(), 2);
        assert_eq!(app.lanes[1].stream_id(), 7);
        assert_eq!(app.lanes[1].item_indices().len(), 3);
    }

    #[test]
    fn test_active_lane_is_first() {
        let app = sample_app();
        assert_eq!(app.active_lane, 0);
        assert_eq!(app.active_stream(), 3);
        assert_eq!(app.active_lane_len(), 2);
    }

    // ── S3 regression: pure-kernel navigation identical to before ────────────

    #[test]
    fn test_navigation_ad() {
        let mut app = sample_app();
        app.next_lane(); // go to stream 7 kernel lane (3 kernels)
        assert_eq!(app.active_stream(), 7);
        assert_eq!(app.selected_item, 0);

        app.next_item();
        assert_eq!(app.selected_item, 1);
        app.next_item();
        assert_eq!(app.selected_item, 2);
        app.next_item(); // clamps
        assert_eq!(app.selected_item, 2);

        app.prev_item();
        assert_eq!(app.selected_item, 1);
        app.prev_item();
        assert_eq!(app.selected_item, 0);
        app.prev_item(); // clamps
        assert_eq!(app.selected_item, 0);
    }

    #[test]
    fn test_lane_wraps() {
        let mut app = sample_app();
        app.next_lane();
        app.next_lane(); // wraps back to first lane
        assert_eq!(app.active_lane, 0);
    }

    #[test]
    fn test_prev_lane_wraps() {
        let mut app = sample_app();
        assert_eq!(app.active_lane, 0);
        app.prev_lane(); // wraps to last lane
        assert_eq!(app.active_lane, app.lanes.len() - 1);
        app.prev_lane();
        assert_eq!(app.active_lane, app.lanes.len() - 2);
        app.next_lane();
        assert_eq!(app.active_lane, app.lanes.len() - 1);
    }

    #[test]
    fn test_zoom() {
        let mut app = sample_app();
        let initial = app.zoom_level;
        app.zoom_in();
        assert!(app.zoom_level > initial);
        app.zoom_out();
        app.zoom_out();
        assert!(app.zoom_level < initial);
        for _ in 0..100 {
            app.zoom_out();
        }
        assert!(app.zoom_level >= 0.1);
    }

    #[test]
    fn test_zoom_in_unbounded() {
        let mut app = sample_app();
        for _ in 0..200 {
            app.zoom_in();
        }
        assert!(app.zoom_level > 1e6);
        assert!(app.zoom_level.is_finite());
    }

    // Z: zoom_fit sets Fit mode; any zoom in/out returns to Scale tracking zoom_level.
    #[test]
    fn zoom_mode_toggle_and_return() {
        let mut app = sample_app();
        assert!(matches!(app.zoom_mode, ZoomMode::Scale(_)), "default Scale");
        app.zoom_fit();
        assert_eq!(app.zoom_mode, ZoomMode::Fit, "zoom_fit -> Fit");
        app.zoom_in();
        assert!(matches!(app.zoom_mode, ZoomMode::Scale(_)), "zoom_in returns to Scale");
        app.zoom_fit();
        assert_eq!(app.zoom_mode, ZoomMode::Fit);
        app.zoom_out();
        assert!(matches!(app.zoom_mode, ZoomMode::Scale(_)), "zoom_out returns to Scale");
        // Scale tracks zoom_level.
        if let ZoomMode::Scale(s) = app.zoom_mode {
            assert!((s - app.zoom_level).abs() < 1e-9, "Scale mirrors zoom_level");
        }
    }

    #[test]
    fn zoom_label_shows_fit() {
        let mut app = sample_app();
        assert_ne!(app.zoom_label(), "fit");
        app.zoom_fit();
        assert_eq!(app.zoom_label(), "fit", "Fit mode labels as 'fit'");
    }

    #[test]
    fn test_global_window_covers_all_streams() {
        let app = sample_app();
        let (start, end) = app.global_time_bounds();
        assert!(start <= 100.0);
        assert!(end >= 305.0);
    }

    #[test]
    fn test_high_zoom_narrows_window_with_large_timestamps() {
        let base = 2_451_606_365_000_000.0;
        let kernels = vec![
            make_kernel(4, base, 4.0),
            make_kernel(4, base + 100_000_000.0, 4.0),
            make_kernel(4, base + 200_000_000.0, 4.0),
        ];
        let mut app = app_from(kernels);

        let (s0, e0) = app.global_visible_window();
        let full_span = e0 - s0;
        assert!(full_span > 100_000_000.0);

        for _ in 0..30 {
            app.zoom_in();
        }
        let (s1, e1) = app.global_visible_window();
        let zoomed_span = e1 - s1;

        assert!(
            zoomed_span < full_span / 1000.0,
            "zoom should shrink window: full={} zoomed={}",
            full_span,
            zoomed_span
        );
        assert!(s1 >= s0 && e1 <= e0);
    }

    // ── S1 happy path: annotation lane is a first-class navigable lane ───────

    fn app_with_annotations() -> App {
        let kernels = vec![
            named_kernel(4, 100.0, "kernel_a"),
            named_kernel(4, 200.0, "kernel_b"),
            named_kernel(4, 300.0, "kernel_c"),
        ];
        let annotations = vec![
            AnnotationEvent { name: "ctx_0".to_string(), ts: 90.0, dur: 60.0, stream: 4, trace_id: 0 },
            AnnotationEvent { name: "ctx_1".to_string(), ts: 180.0, dur: 40.0, stream: 4, trace_id: 0 },
        ];
        App::new(Trace { kernels, annotations })
    }

    #[test]
    fn test_annotation_lane_before_kernel_lane() {
        let app = app_with_annotations();
        assert_eq!(app.lanes.len(), 2);
        assert!(app.lanes[0].is_annotations());
        assert_eq!(app.lanes[0].item_indices().len(), 2);
        assert!(!app.lanes[1].is_annotations());
        assert_eq!(app.lanes[1].item_indices().len(), 3);
    }

    #[test]
    fn test_tab_from_kernel_lane_reaches_annotation_lane() {
        // Lane order: [cuda:1 kernels] [cuda:4 annotations] [cuda:4 kernels].
        let kernels = vec![
            named_kernel(1, 10.0, "k1"),
            named_kernel(4, 100.0, "k4"),
        ];
        let annotations = vec![AnnotationEvent {
            name: "ctx".to_string(),
            ts: 95.0,
            dur: 20.0,
            stream: 4,
            trace_id: 0,
        }];
        let mut app = App::new(Trace { kernels, annotations });
        assert_eq!(app.lanes.len(), 3);

        assert_eq!(app.active_lane, 0);
        assert!(!app.active_lane_is_annotations());
        assert_eq!(app.active_stream(), 1);

        // Tab from a kernel lane must land on the annotation lane.
        app.next_lane();
        assert_eq!(app.active_lane, 1);
        assert!(app.active_lane_is_annotations());
        assert_eq!(app.active_stream(), 4);
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Annotation(a) => assert_eq!(a.name, "ctx"),
            _ => panic!("Tab must select the annotation lane"),
        }
    }

    #[test]
    fn test_tab_selects_annotation_lane_and_ad_navigates() {
        let mut app = app_with_annotations();
        // Startup is the kernel lane; Tab wraps to the annotation lane.
        assert!(!app.active_lane_is_annotations());
        app.next_lane();
        assert!(app.active_lane_is_annotations());
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Annotation(a) => assert_eq!(a.name, "ctx_0"),
            _ => panic!("expected annotation selected"),
        }

        // A/D moves within the annotation lane.
        app.next_item();
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Annotation(a) => assert_eq!(a.name, "ctx_1"),
            _ => panic!("expected annotation selected"),
        }
        app.next_item(); // clamp
        assert_eq!(app.selected_item, 1);

        // Tab moves to the kernel lane of the same stream.
        app.next_lane();
        assert!(!app.active_lane_is_annotations());
        assert!(matches!(
            app.selected_trace_item().unwrap(),
            SelectedTraceItem::Kernel(_)
        ));
    }

    #[test]
    fn test_lane_change_selects_nearest_ts() {
        // Annotation ctx_1 at ts=180; nearest kernel is kernel_b at ts=200.
        let mut app = app_with_annotations();
        app.next_lane(); // to annotation lane
        assert!(app.active_lane_is_annotations());
        app.next_item(); // annotation ctx_1 (ts 180)
        app.next_lane(); // kernel lane; should land near ts 180 → kernel_b (200)
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "kernel_b"),
            _ => panic!("expected kernel selected"),
        }
    }

    #[test]
    fn test_zoom_center_follows_annotation() {
        let mut app = app_with_annotations();
        // Zoom in so the window is narrow enough to center on the selection
        // (otherwise the window clamps to global bounds and skews the center).
        for _ in 0..5 {
            app.zoom_in();
        }
        let (s, e) = app.global_visible_window();
        let center = (s + e) / 2.0;
        assert!(
            (center - 120.0).abs() < 20.0,
            "expected window centered near annotation ctx_0 midpoint 120, got center={}",
            center
        );
    }

    // ── S2 edge: annotation-only stream survives; empty stream handling ──────

    #[test]
    fn test_annotation_only_stream_survives() {
        let kernels = vec![named_kernel(1, 100.0, "k1")];
        let annotations = vec![AnnotationEvent {
            name: "solo".to_string(),
            ts: 50.0,
            dur: 10.0,
            stream: 9,
            trace_id: 0,
        }];
        let app = App::new(Trace { kernels, annotations });
        assert_eq!(app.streams, vec![1, 9]);
        // Lanes: stream 1 kernel lane, stream 9 annotation lane.
        assert_eq!(app.lanes.len(), 2);
        assert!(app.lanes.iter().any(|l| l.stream_id() == 9 && l.is_annotations()));
    }

    #[test]
    fn test_stream_without_annotations_is_single_lane() {
        let kernels = vec![
            named_kernel(1, 0.0, "a"),
            named_kernel(2, 0.0, "b"),
        ];
        let annotations = vec![AnnotationEvent {
            name: "ctx".to_string(),
            ts: 0.0,
            dur: 1.0,
            stream: 2,
            trace_id: 0,
        }];
        let app = App::new(Trace { kernels, annotations });
        // stream 1: 1 lane (kernels). stream 2: 2 lanes (annotations + kernels).
        assert_eq!(app.lanes.len(), 3);
        assert_eq!(app.lanes[0].stream_id(), 1);
        assert!(!app.lanes[0].is_annotations());
        assert_eq!(app.lanes[1].stream_id(), 2);
        assert!(app.lanes[1].is_annotations());
        assert_eq!(app.lanes[2].stream_id(), 2);
        assert!(!app.lanes[2].is_annotations());
    }

    // ── Scroll: one lane == one row ──────────────────────────────────────────

    #[test]
    fn test_scroll_keeps_active_lane_visible() {
        let kernels = vec![
            make_kernel(1, 0.0, 1.0),
            make_kernel(2, 0.0, 1.0),
            make_kernel(3, 0.0, 1.0),
            make_kernel(4, 0.0, 1.0),
        ];
        let mut app = app_from(kernels);
        assert_eq!(app.lanes.len(), 4);

        let visible_rows = 2;
        app.ensure_active_lane_visible(visible_rows);
        assert_eq!(app.lane_view_offset, 0);

        app.next_lane();
        app.next_lane();
        app.ensure_active_lane_visible(visible_rows);
        assert_eq!(app.active_lane, 2);
        assert_eq!(app.lane_view_offset, 1);

        app.next_lane();
        app.ensure_active_lane_visible(visible_rows);
        assert_eq!(app.lane_view_offset, 2);
    }

    // ── Search over both kinds ───────────────────────────────────────────────

    #[test]
    fn test_search_jumps_to_first_match_across_lanes() {
        let kernels = vec![
            named_kernel(3, 100.0, "elementwise_add"),
            named_kernel(3, 200.0, "reduce_sum"),
            named_kernel(7, 150.0, "volta_sgemm_128"),
            named_kernel(7, 250.0, "ampere_gemm"),
        ];
        let mut app = app_from(kernels);

        app.search_start();
        assert!(app.search_active);
        app.search_push('v');
        app.search_push('o');
        app.search_push('l');

        assert!(!app.search_no_match);
        assert_eq!(app.active_stream(), 7);
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "volta_sgemm_128"),
            _ => panic!("expected kernel"),
        }

        app.search_commit();
        assert!(!app.search_active);
    }

    #[test]
    fn test_search_finds_annotation() {
        let mut app = app_with_annotations();
        app.search_start();
        app.search_push('c');
        app.search_push('t');
        app.search_push('x');
        app.search_push('_');
        app.search_push('1');
        assert!(!app.search_no_match);
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Annotation(a) => assert_eq!(a.name, "ctx_1"),
            _ => panic!("expected annotation"),
        }
    }

    #[test]
    fn test_search_case_insensitive_and_no_match() {
        let kernels = vec![
            named_kernel(3, 100.0, "Reduce_Sum"),
            named_kernel(7, 150.0, "GEMM"),
        ];
        let mut app = app_from(kernels);

        app.search_start();
        app.search_push('g');
        app.search_push('e');
        assert!(!app.search_no_match);
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "GEMM"),
            _ => panic!("expected kernel"),
        }

        app.search_push('z');
        assert!(app.search_no_match);
    }

    #[test]
    fn test_search_cancel_resets() {
        let kernels = vec![named_kernel(3, 100.0, "foo"), named_kernel(7, 150.0, "bar")];
        let mut app = app_from(kernels);
        app.search_start();
        app.search_push('b');
        assert_eq!(app.active_stream(), 7);
        app.search_cancel();
        assert!(!app.search_active);
        assert!(app.search_query.is_empty());
    }

    #[test]
    fn test_kernel_columns_skips_out_of_window() {
        let ts_start = 1000.0;
        let ts_end = 1100.0;
        let width = 100;

        assert_eq!(kernel_columns(500.0, 900.0, ts_start, ts_end, width), None);
        assert_eq!(kernel_columns(1200.0, 1300.0, ts_start, ts_end, width), None);

        let (s, e) = kernel_columns(1050.0, 1060.0, ts_start, ts_end, width).unwrap();
        assert_eq!(s, 50);
        assert_eq!(e, 60);
    }

    #[test]
    fn test_kernel_columns_proportional_width_at_high_zoom() {
        let ts_start = 1000.0;
        let ts_end = 1013.0;
        let width = 200;

        let big = kernel_columns(1000.0, 1012.4, ts_start, ts_end, width).unwrap();
        let small = kernel_columns(1006.0, 1006.5, ts_start, ts_end, width).unwrap();

        let big_w = big.1 - big.0;
        let small_w = small.1 - small.0;
        assert!(
            big_w > small_w,
            "12.4us kernel should be wider than 0.5us kernel: big={} small={}",
            big_w,
            small_w
        );
        assert!(big_w > 100, "12.4us over 13us window in 200 cols should be wide: {}", big_w);
    }

    #[test]
    fn test_kernel_columns_clips_partial_overlap() {
        let (s, e) = kernel_columns(950.0, 1050.0, 1000.0, 1100.0, 100).unwrap();
        assert_eq!(s, 0);
        assert_eq!(e, 50);
    }

    // ── Sequence feature (N key) ─────────────────────────────────────────────

    fn kd(stream: u64, ts: f64, name: &str, dur: f64) -> KernelEvent {
        KernelEvent {
            name: name.to_string(),
            cat: "kernel".to_string(),
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

    // S1: sequence runs from current up to (not including) next same-named kernel.
    #[test]
    fn test_sequence_between_same_named() {
        let kernels = vec![
            kd(1, 0.0, "foo", 10.0),
            kd(1, 20.0, "bar", 20.0),
            kd(1, 50.0, "baz", 5.0),
            kd(1, 60.0, "foo", 8.0),
            kd(1, 80.0, "qux", 3.0),
        ];
        let mut app = app_from(kernels);
        assert_eq!(app.selected_item, 0);
        assert!(app.start_sequence());
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(
            seq.rows,
            vec![
                (1, "foo".to_string(), 10.0),
                (2, "bar".to_string(), 20.0),
                (3, "baz".to_string(), 5.0),
            ]
        );
        assert_eq!(
            app.sequence_csv().unwrap(),
            "idx\tkernel name\tmedian\n1\tfoo\t10.00\n2\tbar\t20.00\n3\tbaz\t5.00\n"
        );
    }

    // S2a: no next same-named → sequence to end of lane.
    #[test]
    fn test_sequence_to_end_of_lane() {
        let kernels = vec![
            kd(1, 0.0, "a", 1.0),
            kd(1, 10.0, "b", 2.0),
            kd(1, 20.0, "c", 3.0),
        ];
        let mut app = app_from(kernels);
        assert!(app.start_sequence());
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(
            seq.rows,
            vec![
                (1, "a".to_string(), 1.0),
                (2, "b".to_string(), 2.0),
                (3, "c".to_string(), 3.0),
            ]
        );
    }

    // S2b: N on an annotation lane is a no-op (no sequence).
    #[test]
    fn test_sequence_none_on_annotation_lane() {
        let mut app = app_with_annotations();
        app.next_lane();
        assert!(app.active_lane_is_annotations());
        assert!(!app.start_sequence());
        assert!(app.sequence.is_none());
    }

    // Bug #2: N works from a kernel lane WITHOUT any prior search. The current
    // selected kernel drives the sequence even right after startup.
    #[test]
    fn test_sequence_works_without_prior_search() {
        let mut app = app_with_annotations();
        assert!(!app.active_lane_is_annotations());
        assert!(app.start_sequence());
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(seq.rows.first().map(|(_, n, _)| n.as_str()), Some("kernel_a"));
    }

    // Bug #2 root cause: on startup the initial active lane must be a kernel lane
    // (not annotations), so N works immediately without navigating or searching.
    #[test]
    fn test_initial_active_lane_is_kernel_when_available() {
        let app = app_with_annotations();
        assert!(!app.active_lane_is_annotations());
        assert!(app.selected_trace_item().is_some());
    }

    #[test]
    fn test_sequence_works_on_startup_without_navigation() {
        let mut app = app_with_annotations();
        assert!(app.start_sequence());
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(seq.rows.first().map(|(_, n, _)| n.as_str()), Some("kernel_a"));
    }

    // S3: per-position median over exact repeated blocks.
    #[test]
    fn test_sequence_median_per_position() {
        let kernels = vec![
            kd(1, 0.0, "foo", 10.0),
            kd(1, 10.0, "bar", 20.0),
            kd(1, 20.0, "foo", 12.0),
            kd(1, 30.0, "bar", 24.0),
            kd(1, 40.0, "foo", 11.0),
            kd(1, 50.0, "bar", 22.0),
            kd(1, 60.0, "foo", 9.0),
        ];
        let mut app = app_from(kernels);
        assert!(app.start_sequence());
        // Block = [foo, bar] (stops before next foo).
        assert_eq!(app.sequence.as_ref().unwrap().rows.len(), 2);

        app.extend_sequence_median();
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(seq.reps_found, 3, "3 exact [foo,bar] blocks");
        let median = seq.median.as_ref().unwrap();
        assert_eq!(median.len(), 2);
        assert_eq!(median[0].0, "foo");
        assert!((median[0].1 - 11.0).abs() < 1e-9, "median(10,12,11)=11 got {}", median[0].1);
        assert_eq!(median[1].0, "bar");
        assert!((median[1].1 - 22.0).abs() < 1e-9, "median(20,24,22)=22 got {}", median[1].1);
    }

    // Median of an even count = average of the two middle values.
    #[test]
    fn test_sequence_median_even_count() {
        let kernels = vec![
            kd(1, 0.0, "foo", 10.0),
            kd(1, 10.0, "bar", 100.0),
            kd(1, 20.0, "foo", 20.0),
            kd(1, 30.0, "bar", 200.0),
            kd(1, 40.0, "foo", 30.0),
            kd(1, 50.0, "bar", 300.0),
            kd(1, 60.0, "foo", 40.0),
            kd(1, 70.0, "bar", 400.0),
            kd(1, 80.0, "foo", 99.0),
        ];
        let mut app = app_from(kernels);
        assert!(app.start_sequence());
        app.extend_sequence_median();
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(seq.reps_found, 4);
        let median = seq.median.as_ref().unwrap();
        // foo durs (10,20,30,40) sorted → median = (20+30)/2 = 25
        assert!((median[0].1 - 25.0).abs() < 1e-9, "even median foo got {}", median[0].1);
        // bar durs (100,200,300,400) → (200+300)/2 = 250
        assert!((median[1].1 - 250.0).abs() < 1e-9, "even median bar got {}", median[1].1);
    }

    // Median scan finds repeats ANYWHERE later, skipping intervening kernels.
    #[test]
    fn test_sequence_median_finds_non_contiguous_repeats() {
        let kernels = vec![
            kd(1, 0.0, "foo", 10.0),
            kd(1, 10.0, "bar", 20.0),
            kd(1, 20.0, "foo", 12.0),
            kd(1, 30.0, "bar", 24.0),
            kd(1, 40.0, "foo", 11.0),
            kd(1, 50.0, "qux", 99.0), // gap between blocks — must be skipped, not a stop
            kd(1, 60.0, "foo", 8.0),
            kd(1, 70.0, "bar", 8.0),
        ];
        let mut app = app_from(kernels);
        assert!(app.start_sequence());
        app.extend_sequence_median();
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(seq.reps_found, 3, "3 [foo,bar] blocks: at 0, 20, 60 (qux skipped)");
        let median = seq.median.as_ref().unwrap();
        // foo durs (10,12,8) sorted (8,10,12) → 10
        assert!((median[0].1 - 10.0).abs() < 1e-9, "foo median got {}", median[0].1);
        // bar durs (20,24,8) sorted (8,20,24) → 20
        assert!((median[1].1 - 20.0).abs() < 1e-9, "bar median got {}", median[1].1);
    }

    // S4 regression: no sequence by default; close clears it.
    #[test]
    fn test_sequence_default_none_and_close() {
        let mut app = sample_app();
        assert!(app.sequence.is_none());
        assert!(app.start_sequence());
        assert!(app.sequence.is_some());
        app.close_sequence();
        assert!(app.sequence.is_none());
    }

    #[test]
    fn test_sequence_scroll_clamps_to_bounds() {
        let kernels: Vec<KernelEvent> = (0..10)
            .map(|i| kd(1, i as f64 * 10.0, &format!("k{}", i), i as f64))
            .collect();
        let mut app = app_from(kernels);
        assert!(app.start_sequence());
        assert_eq!(app.sequence.as_ref().unwrap().rows.len(), 10);
        assert_eq!(app.sequence.as_ref().unwrap().scroll, 0);

        // Scroll up at the top is a no-op.
        app.sequence_scroll_up(3);
        assert_eq!(app.sequence.as_ref().unwrap().scroll, 0);

        // With a viewport of 4, max scroll is 10 - 4 = 6.
        app.sequence_scroll_down(3, 4);
        assert_eq!(app.sequence.as_ref().unwrap().scroll, 3);
        app.sequence_scroll_down(100, 4);
        assert_eq!(app.sequence.as_ref().unwrap().scroll, 6);

        app.sequence_scroll_up(2);
        assert_eq!(app.sequence.as_ref().unwrap().scroll, 4);
    }

    #[test]
    fn test_sequence_end_to_end_copy_writes_file_artifact() {
        let kernels = vec![
            kd(1, 0.0, "gemm", 10.0),
            kd(1, 20.0, "relu", 20.0),
            kd(1, 40.0, "gemm", 30.0),
            kd(1, 60.0, "relu", 40.0),
            kd(1, 80.0, "gemm", 50.0),
        ];
        let mut app = app_from(kernels);
        assert!(app.start_sequence());
        assert_eq!(
            app.sequence.as_ref().unwrap().rows,
            vec![(1, "gemm".to_string(), 10.0), (2, "relu".to_string(), 20.0)]
        );

        app.extend_sequence_median();
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(seq.reps_found, 2);
        assert_eq!(
            seq.median.as_ref().unwrap(),
            &vec![("gemm".to_string(), 20.0), ("relu".to_string(), 30.0)]
        );

        let csv = app.sequence_csv().unwrap();
        let mut sink: Vec<u8> = Vec::new();
        let mut mgr = crate::clipboard::ClipboardManager::new();
        let outcome = mgr.copy(&csv, &mut sink).unwrap();

        assert!(outcome.via_osc52);
        assert!(outcome.file_path.exists());
        assert_eq!(std::fs::read_to_string(&outcome.file_path).unwrap(), csv);
        let _ = std::fs::remove_file(&outcome.file_path);
    }

    // ── Multi-trace alignment + interleaving ─────────────────────────────────

    fn ann(stream: u64, ts: f64, name: &str) -> AnnotationEvent {
        AnnotationEvent {
            name: name.to_string(),
            ts,
            dur: 1.0,
            stream,
            trace_id: 0,
        }
    }

    fn trace_of(kernels: Vec<KernelEvent>, annotations: Vec<AnnotationEvent>) -> Trace {
        Trace {
            kernels,
            annotations,
        }
    }

    // S1: two traces share annotation "step" at different abs ts -> aligned
    // render_ts of that annotation is equal across both traces.
    #[test]
    fn test_alignment_offsets_anchor_on_first_shared_annotation() {
        let t0 = trace_of(vec![kd(1, 100.0, "foo", 5.0)], vec![ann(1, 100.0, "step")]);
        let t1 = trace_of(vec![kd(1, 500.0, "foo", 5.0)], vec![ann(1, 500.0, "step")]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        assert_eq!(app.traces.len(), 2);
        assert_eq!(app.traces[0].offset_us, 0.0, "reference trace offset is 0");
        assert_eq!(app.traces[1].offset_us, -400.0, "shift t1 back by 400");
        assert_eq!(app.traces[0].anchor.as_deref(), Some("step"));

        // Aligned anchor timestamps coincide.
        let a0 = app
            .annotations
            .iter()
            .find(|a| a.trace_id == 0 && a.name == "step")
            .unwrap();
        let a1 = app
            .annotations
            .iter()
            .find(|a| a.trace_id == 1 && a.name == "step")
            .unwrap();
        let r0 = a0.ts + app.traces[0].offset_us;
        let r1 = a1.ts + app.traces[1].offset_us;
        assert!((r0 - r1).abs() < 1e-9, "anchor aligned: {} vs {}", r0, r1);
    }

    // S1b: the render-ts accessors add the per-trace offset to raw ts.
    #[test]
    fn test_render_ts_accessors_apply_offset() {
        let t0 = trace_of(vec![kd(1, 100.0, "foo", 5.0)], vec![ann(1, 100.0, "step")]);
        let t1 = trace_of(vec![kd(1, 500.0, "bar", 7.0)], vec![ann(1, 500.0, "step")]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        let k1 = app.kernels.iter().position(|k| k.trace_id == 1).unwrap();
        // raw 500 + offset -400 = 100
        assert!((app.kernel_render_ts(k1) - 100.0).abs() < 1e-9);
        assert!((app.kernel_render_end(k1) - 107.0).abs() < 1e-9);
    }

    // S2: interleave order = row0 of every trace, then row1, etc. with correct
    // trace_id per lane.
    #[test]
    fn test_lanes_interleaved_by_row_across_traces() {
        // Each trace: stream 1 has kernels only -> 1 lane per trace.
        let t0 = trace_of(vec![kd(1, 0.0, "a", 1.0), kd(2, 0.0, "b", 1.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 0.0, "c", 1.0), kd(2, 0.0, "d", 1.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        // t0 has 2 lanes (stream1, stream2), t1 has 2 lanes -> interleave:
        // [t0.row0, t1.row0, t0.row1, t1.row1]
        assert_eq!(app.lanes.len(), 4);
        assert_eq!(app.lanes[0].trace_id(), 0);
        assert_eq!(app.lanes[1].trace_id(), 1);
        assert_eq!(app.lanes[2].trace_id(), 0);
        assert_eq!(app.lanes[3].trace_id(), 1);
        assert_eq!(app.lanes[0].stream_id(), 1);
        assert_eq!(app.lanes[1].stream_id(), 1);
        assert_eq!(app.lanes[2].stream_id(), 2);
        assert_eq!(app.lanes[3].stream_id(), 2);
    }

    // S3: no shared annotation name -> all offsets 0, lanes still interleaved.
    #[test]
    fn test_no_common_annotation_falls_back_to_zero_offset() {
        let t0 = trace_of(vec![kd(1, 100.0, "a", 1.0)], vec![ann(1, 100.0, "alpha")]);
        let t1 = trace_of(vec![kd(1, 500.0, "b", 1.0)], vec![ann(1, 500.0, "beta")]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        assert_eq!(app.traces[0].offset_us, 0.0);
        assert_eq!(app.traces[1].offset_us, 0.0);
        assert!(app.traces[0].anchor.is_none(), "no anchor chosen");
        // Each trace: annotation lane + kernel lane = 2 lanes; interleaved = 4,
        // in [t0.ann, t1.ann, t0.kern, t1.kern] order.
        assert_eq!(app.lanes.len(), 4, "still interleaved by row across traces");
        assert_eq!(app.lanes[0].trace_id(), 0);
        assert_eq!(app.lanes[1].trace_id(), 1);
        assert!(app.lanes[0].is_annotations());
        assert!(app.lanes[1].is_annotations());
        assert!(!app.lanes[2].is_annotations());
        assert!(!app.lanes[3].is_annotations());
    }

    // S4: unequal lane counts -> round-robin to max, skip missing rows.
    #[test]
    fn test_unequal_lane_counts_round_robin_skips_missing() {
        // t0: 3 streams -> 3 lanes. t1: 1 stream -> 1 lane.
        let t0 = trace_of(
            vec![
                kd(1, 0.0, "a", 1.0),
                kd(2, 0.0, "b", 1.0),
                kd(3, 0.0, "c", 1.0),
            ],
            vec![],
        );
        let t1 = trace_of(vec![kd(1, 0.0, "d", 1.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        // Expected: [t0.row0, t1.row0, t0.row1, t0.row2]
        assert_eq!(app.lanes.len(), 4);
        assert_eq!(app.lanes[0].trace_id(), 0);
        assert_eq!(app.lanes[1].trace_id(), 1);
        assert_eq!(app.lanes[2].trace_id(), 0);
        assert_eq!(app.lanes[3].trace_id(), 0);
    }

    // S5 regression: single-trace App::new behaves as before (offset 0, 1 trace).
    #[test]
    fn test_single_trace_regression_offsets_and_lanes() {
        let app = app_with_annotations();
        assert_eq!(app.traces.len(), 1);
        assert_eq!(app.traces[0].offset_us, 0.0);
        // Same lane layout as before: annotation lane + kernel lane for stream 4.
        assert_eq!(app.lanes.len(), 2);
        assert!(app.lanes.iter().all(|l| l.trace_id() == 0));
    }

    // S1c: global_time_bounds spans the ALIGNED window across traces, so the far
    // trace's shifted events fall inside the reference trace's coordinate range.
    #[test]
    fn test_global_bounds_use_aligned_timestamps() {
        // t0 anchor "step" at 100; t1 anchor at 500 -> offset -400.
        // t1 kernel raw 500 -> aligned 100, so aligned max end ~ 105, not 505.
        let t0 = trace_of(vec![kd(1, 100.0, "foo", 5.0)], vec![ann(1, 100.0, "step")]);
        let t1 = trace_of(vec![kd(1, 500.0, "bar", 5.0)], vec![ann(1, 500.0, "step")]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let (g_start, g_end) = app.global_time_bounds();
        // Both traces' aligned events sit near ts=100..105.
        assert!((g_start - 100.0).abs() < 1e-9, "start {}", g_start);
        assert!(g_end <= 106.0, "aligned end should be ~105 not 505, got {}", g_end);
    }

    // Alignment label surfaces the chosen anchor for multi-trace, empty otherwise.
    #[test]
    fn test_alignment_label_reports_anchor() {
        let t0 = trace_of(vec![kd(1, 100.0, "foo", 5.0)], vec![ann(1, 100.0, "step")]);
        let t1 = trace_of(vec![kd(1, 500.0, "bar", 5.0)], vec![ann(1, 500.0, "step")]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert_eq!(app.alignment_label().as_deref(), Some("aligned on \"step\""));

        let single = app_with_annotations();
        assert!(single.alignment_label().is_none(), "single trace: no label");

        let d0 = trace_of(vec![kd(1, 1.0, "a", 1.0)], vec![ann(1, 1.0, "x")]);
        let d1 = trace_of(vec![kd(1, 9.0, "b", 1.0)], vec![ann(1, 9.0, "y")]);
        let disjoint = App::new_multi(vec![("T0".into(), d0), ("T1".into(), d1)]);
        assert_eq!(
            disjoint.alignment_label().as_deref(),
            Some("not aligned (no shared annotation)")
        );
    }

    // Tabbing across offset traces lands on the ALIGNED-nearest item, so the
    // visually-adjacent kernel is selected (this feature is for visual compare).
    #[test]
    fn test_tab_across_traces_uses_aligned_position() {
        // t0: foo@100, bar@200 (annotation@100). t1: foo@9100, bar@9200 (annotation@9000).
        // No shared annotation name → both offsets 0.0 (raw timestamps).
        // Selected T0.bar at render_ts 200; nearest in T1 is foo@9100 (distance 8900 vs bar@9200 distance 9000).
        let t0 = trace_of(
            vec![kd(1, 100.0, "foo", 5.0), kd(1, 200.0, "bar", 5.0)],
            vec![ann(1, 100.0, "step")],
        );
        let t1 = trace_of(
            vec![kd(1, 9100.0, "foo", 5.0), kd(1, 9200.0, "bar", 5.0)],
            vec![ann(1, 9000.0, "step")],
        );
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        let bar_lane = app
            .lanes
            .iter()
            .position(|l| {
                l.trace_id() == 0
                    && !l.is_annotations()
                    && l.item_indices()
                        .iter()
                        .any(|&i| app.kernels[i].name == "bar")
            })
            .unwrap();
        app.active_lane = bar_lane;
        let bar_pos = app.lanes[bar_lane]
            .item_indices()
            .iter()
            .position(|&i| app.kernels[i].name == "bar")
            .unwrap();
        app.selected_item = bar_pos;

        let t1_kern_lane = app
            .lanes
            .iter()
            .position(|l| l.trace_id() == 1 && !l.is_annotations())
            .unwrap();
        app.move_to_lane_for_test(t1_kern_lane);
        let sel = app.selected_trace_item().unwrap();
        match sel {
            // Under gap-column layout, tab carries the LCS column: T0.bar is at
            // slot 1 (foo=slot0, bar=slot1, both matched), so it lands on T1.bar.
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "bar"),
            _ => panic!("expected kernel"),
        }
    }

    // ── Dynamic align to the selected kernel (g key) ─────────────────────────

    /// Aligned start ts of trace `tid`'s nearest same-named kernel to `target`.
    fn nearest_same_name_aligned(app: &App, tid: usize, name: &str, target: f64) -> Option<f64> {
        app.kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| k.trace_id == tid && k.name == name)
            .map(|(i, _)| app.kernel_render_ts(i))
            .min_by(|a, b| {
                (a - target)
                    .abs()
                    .partial_cmp(&(b - target).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn select_kernel(app: &mut App, tid: usize, name: &str) {
        let lane = app
            .lanes
            .iter()
            .position(|l| {
                l.trace_id() == tid
                    && !l.is_annotations()
                    && l.item_indices().iter().any(|&i| app.kernels[i].name == name)
            })
            .unwrap();
        let pos = app.lanes[lane]
            .item_indices()
            .iter()
            .position(|&i| app.kernels[i].name == name)
            .unwrap();
        app.active_lane = lane;
        app.selected_item = pos;
    }

    // S1: aligning to a selected kernel shifts other traces so their nearest
    // same-named kernel's start coincides with the selected kernel's start.
    #[test]
     fn test_align_to_selected_kernel_shifts_others() {
         // compute_diff no longer auto-aligns: no shared annotations → offset 0.0.
         // t0: gemm@200. t1: gemm@700.
         let t0 = trace_of(vec![kd(1, 200.0, "gemm", 8.0)], vec![]);
         let t1 = trace_of(vec![kd(1, 700.0, "gemm", 8.0)], vec![]);
         let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
         assert_eq!(app.traces[1].offset_us, 0.0, "no auto-offset under gap layout");

        select_kernel(&mut app, 0, "gemm");
        let ok = app.align_to_selected_kernel();
        assert!(ok, "align succeeds on a selected kernel");

        // Selected trace unchanged; t1 shifted so its gemm sits at 200.
        assert_eq!(app.traces[0].offset_us, 0.0, "selected/reference trace fixed");
        assert_eq!(app.traces[1].offset_us, -500.0, "t1 shifts by 200-700");
        let t1_gemm = app.kernels.iter().position(|k| k.trace_id == 1).unwrap();
        assert!((app.kernel_render_ts(t1_gemm) - 200.0).abs() < 1e-9);
        // Header anchor reflects the kernel used.
        assert_eq!(app.alignment_label().as_deref(), Some("aligned on \"gemm\""));
        let _ = nearest_same_name_aligned(&app, 1, "gemm", 200.0);
    }

    // S2: when a trace has multiple same-named kernels, align picks the one
    // whose aligned start is NEAREST to the selected kernel, not the first.
    #[test]
    fn test_align_picks_nearest_same_named_kernel() {
        // t0: bar@500 selected. t1 has gemm@100, bar@480, bar@900.
        // Nearest bar to 500 is 480 -> shift by 500-480 = +20.
        let t0 = trace_of(vec![kd(1, 500.0, "bar", 5.0)], vec![]);
        let t1 = trace_of(
            vec![
                kd(1, 100.0, "gemm", 5.0),
                kd(1, 480.0, "bar", 5.0),
                kd(1, 900.0, "bar", 5.0),
            ],
            vec![],
        );
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        select_kernel(&mut app, 0, "bar");
        assert!(app.align_to_selected_kernel());
        assert_eq!(app.traces[1].offset_us, 20.0, "nearest bar (480) -> +20");
    }

    // S3: aligning when another trace has NO same-named kernel leaves that trace
    // untouched (no panic, offset unchanged).
    #[test]
    fn test_align_no_match_leaves_trace_untouched() {
        let t0 = trace_of(vec![kd(1, 200.0, "gemm", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 700.0, "relu", 5.0)], vec![]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        select_kernel(&mut app, 0, "gemm");
        assert!(app.align_to_selected_kernel());
        assert_eq!(app.traces[1].offset_us, 0.0, "no 'gemm' in t1 -> unchanged");
    }

    // S4: aligning with an annotation (non-kernel) selected is a no-op.
    #[test]
    fn test_align_noop_when_selection_not_kernel() {
        let t0 = trace_of(vec![kd(1, 100.0, "k", 5.0)], vec![ann(1, 90.0, "ctx")]);
        let t1 = trace_of(vec![kd(1, 900.0, "k", 5.0)], vec![ann(1, 800.0, "ctx")]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let load_offset = app.traces[1].offset_us;
        // Select the annotation lane.
        let ann_lane = app.lanes.iter().position(|l| l.is_annotations()).unwrap();
        app.active_lane = ann_lane;
        app.selected_item = 0;
        assert!(!app.align_to_selected_kernel(), "no-op on annotation");
        assert_eq!(app.traces[1].offset_us, load_offset, "offsets unchanged");
    }

    // ── Short trace labels (strip common prefix + suffix) ────────────────────

    #[test]
    fn test_short_labels_strip_common_prefix_and_suffix() {
        let labels = vec![
            "resnet_baseline.pt.trace.json".to_string(),
            "resnet_tuned.pt.trace.json".to_string(),
        ];
        assert_eq!(shorten_labels(&labels), vec!["baseline", "tuned"]);
    }

    #[test]
    fn test_short_labels_prefix_only() {
        let labels = vec!["run_a".to_string(), "run_bb".to_string()];
        assert_eq!(shorten_labels(&labels), vec!["a", "bb"]);
    }

    #[test]
    fn test_short_labels_identical_fall_back_to_index() {
        let labels = vec!["same.json".to_string(), "same.json".to_string()];
        assert_eq!(shorten_labels(&labels), vec!["T0", "T1"]);
    }

    #[test]
    fn test_short_labels_single_unchanged() {
        let labels = vec!["only_one.json".to_string()];
        assert_eq!(shorten_labels(&labels), vec!["only_one.json"]);
    }

    #[test]
    fn test_short_labels_empty_diff_falls_back_to_index() {
        // Common prefix "a" + suffix "c" would leave one empty -> index fallback.
        let labels = vec!["ac".to_string(), "abc".to_string()];
        assert_eq!(shorten_labels(&labels), vec!["T0", "T1"]);
    }

    #[test]
    fn diff_status_defaults_matched() {
        let app = app_from(vec![
            make_kernel(1, 0.0, 5.0),
            make_kernel(1, 10.0, 5.0),
        ]);
        assert_eq!(app.diff_status.len(), app.kernels.len());
        assert!(app.diff_status.iter().all(|s| *s == crate::DiffStatus::Matched));
    }

    // ── compute_diff scenarios ───────────────────────────────────────────────

    fn make_two_trace_app(t0_names: &[&str], t1_names: &[&str]) -> App {
        let t0 = trace_of(
            t0_names.iter().enumerate().map(|(i, n)| kd(1, i as f64 * 10.0, n, 5.0)).collect(),
            vec![],
        );
        let t1 = trace_of(
            t1_names.iter().enumerate().map(|(i, n)| kd(1, i as f64 * 10.0, n, 5.0)).collect(),
            vec![],
        );
        App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)])
    }

    // S2: T1 has extra kernel absent in T0 -> Added
    #[test]
    fn compute_diff_added() {
        let mut app = make_two_trace_app(&["gemm", "relu"], &["gemm", "extra", "relu"]);
        app.compute_diff();
        let extra_idx = app.kernels.iter().position(|k| k.trace_id == 1 && k.name == "extra").unwrap();
        assert_eq!(app.diff_status[extra_idx], DiffStatus::Added);
        let gemm_t1 = app.kernels.iter().position(|k| k.trace_id == 1 && k.name == "gemm").unwrap();
        assert_eq!(app.diff_status[gemm_t1], DiffStatus::Matched);
    }

    // S3: T0 has kernel absent in T1 -> Removed
    #[test]
    fn compute_diff_removed() {
        let mut app = make_two_trace_app(&["gemm", "bn", "relu"], &["gemm", "relu"]);
        app.compute_diff();
        let bn_idx = app.kernels.iter().position(|k| k.trace_id == 0 && k.name == "bn").unwrap();
        assert_eq!(app.diff_status[bn_idx], DiffStatus::Removed);
        let relu_t0 = app.kernels.iter().position(|k| k.trace_id == 0 && k.name == "relu").unwrap();
        assert_eq!(app.diff_status[relu_t0], DiffStatus::Matched);
    }

    // S5: T0=[gemm,bn,relu], T1=[gemm,relu] - bn removed, gemm+relu matched
    #[test]
    fn compute_diff_mixed() {
        let mut app = make_two_trace_app(&["gemm", "bn", "relu"], &["gemm", "relu"]);
        app.compute_diff();
        let gemm_t0 = app.kernels.iter().position(|k| k.trace_id == 0 && k.name == "gemm").unwrap();
        let bn_t0   = app.kernels.iter().position(|k| k.trace_id == 0 && k.name == "bn").unwrap();
        let relu_t0 = app.kernels.iter().position(|k| k.trace_id == 0 && k.name == "relu").unwrap();
        assert_eq!(app.diff_status[gemm_t0], DiffStatus::Matched);
        assert_eq!(app.diff_status[bn_t0],   DiffStatus::Removed);
        assert_eq!(app.diff_status[relu_t0], DiffStatus::Matched);
    }

    // S4: single trace -> all remain Matched after compute_diff
    #[test]
    fn compute_diff_single_all_matched() {
        let mut app = app_from(vec![make_kernel(1, 0.0, 5.0), make_kernel(1, 10.0, 5.0)]);
        app.compute_diff();
        assert!(app.diff_status.iter().all(|s| *s == DiffStatus::Matched));
    }

    // S4: single-trace lanes never use gap columns (fall back to time layout).
    #[test]
    fn surface_s4_single_trace_no_gap_columns() {
        let app = app_from(vec![make_kernel(1, 0.0, 5.0), make_kernel(1, 10.0, 5.0)]);
        for idx in 0..app.lanes.len() {
            assert!(!app.use_gap_columns_for_lane(idx), "single-trace lane {idx} must not use gap columns");
        }
    }

    // S5: with N>2 traces, no lane uses gap columns (v1 supports exactly 2).
    #[test]
    fn surface_s5_three_traces_fallback() {
        let t0 = trace_of(vec![kd(1, 0.0, "A", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 0.0, "A", 5.0)], vec![]);
        let t2 = trace_of(vec![kd(1, 0.0, "A", 5.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1), ("T2".into(), t2)]);
        for idx in 0..app.lanes.len() {
            assert!(!app.use_gap_columns_for_lane(idx), "N>2 lane {idx} must fall back to time layout");
        }
    }

    // S8: navigating between diff lanes carries the LCS column; a present slot is
    // selected, never a gap. Removed B (t0-only) maps to nearest present in t1.
    #[test]
    fn nav_diff_lane_carries_column() {
        let mut app = make_two_trace_app(&["A", "B", "C"], &["A", "C"]);

        select_kernel(&mut app, 0, "C");
        let src = app.active_lane;
        let t1_lane = app.lanes.iter().position(|l| l.trace_id() == 1 && !l.is_annotations()).unwrap();
        app.move_to_lane_for_test(t1_lane);
        let landed = app.lanes[app.active_lane].item_indices()[app.selected_item];
        assert_eq!(app.kernels[landed].name, "C", "matched C must carry across to t1 C");

        app.move_to_lane_for_test(src);
        select_kernel(&mut app, 0, "B");
        app.move_to_lane_for_test(t1_lane);
        let landed = app.lanes[app.active_lane].item_indices()[app.selected_item];
        assert!(
            app.kernels[landed].name == "A" || app.kernels[landed].name == "C",
            "removed B must land on nearest present kernel in t1, got {}",
            app.kernels[landed].name,
        );
    }

    #[test]
    fn kernel_diff_color_variants() {
        use ratatui::style::Color;
        let mut app = make_two_trace_app(&["gemm"], &["relu"]);
        app.compute_diff();
        let removed = app.kernels.iter().position(|k| k.trace_id == 0).unwrap();
        let added   = app.kernels.iter().position(|k| k.trace_id == 1).unwrap();
        assert_eq!(app.kernel_diff_color(removed), Some(Color::Rgb(220, 38, 38)));
        assert_eq!(app.kernel_diff_color(added),   Some(Color::Rgb(34, 197, 94)));
        assert_eq!(app.kernel_diff_color(999),     None);
    }

    // S1: after compute_diff, matched kernels share a column (no auto-offset under gap layout)
    #[test]
    fn compute_diff_no_offset_matched_share_column() {
        let t0 = trace_of(vec![kd(1, 100.0, "gemm", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 900.0, "gemm", 5.0)], vec![]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        app.compute_diff();
        // No shared annotations, so offset stays 0.0 (no auto-offset under gap layout)
        assert!((app.traces[1].offset_us - 0.0).abs() < 1e-6, "no auto-offset: offset={}", app.traces[1].offset_us);
        let t0_gemm = app.kernels.iter().position(|k| k.trace_id == 0 && k.name == "gemm").unwrap();
        let t1_gemm = app.kernels.iter().position(|k| k.trace_id == 1 && k.name == "gemm").unwrap();
        // Matched kernels share the same column in kernel_diff_column
        let col0 = app.kernel_diff_column[t0_gemm].unwrap().column;
        let col1 = app.kernel_diff_column[t1_gemm].unwrap().column;
        assert_eq!(col0, col1, "matched kernels share column");
    }

    // G1: idle between consecutive T0 kernels -> proportional lead_gap_cols on the
    // slot starting the following kernel. Larger idle => more lead columns.
    #[test]
    fn compute_idle_gap_cols() {
        // T0: A@0-10, B@100-110 (idle 90), C@120-130 (idle 10). T1 matches names.
        let t0 = trace_of(
            vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0), kd(1, 120.0, "C", 10.0)],
            vec![],
        );
        let t1 = trace_of(
            vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0), kd(1, 120.0, "C", 10.0)],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let slots = &app.diff_columns_by_stream[&1].slots;
        // 3 matched slots for A, B, C in order.
        let gap_b = slots[1].lead_gap_cols;
        let gap_c = slots[2].lead_gap_cols;
        assert!(gap_b > 0, "large idle before B must produce a gap, got {gap_b}");
        assert!(gap_b > gap_c, "idle before B (90) > before C (10): {gap_b} vs {gap_c}");
        assert!(gap_b <= 8, "lead_gap_cols must be clamped to <=8, got {gap_b}");
    }

    // Shared visual layer: stream_layout expands each slot's lead_gap_cols into
    // Gap columns before the Slot, giving one coordinate system for kernels+annotations.
    #[test]
    fn stream_layout_expands_gaps() {
        let t0 = trace_of(
            vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0)],
            vec![],
        );
        let t1 = trace_of(
            vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0)],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let layout = app.stream_layout(1).expect("layout for stream 1");
        // Slot 0 (A) has no lead gap; slot 1 (B) has lead_gap_cols > 0 before it.
        let gap_b = app.diff_columns_by_stream[&1].slots[1].lead_gap_cols as usize;
        assert!(gap_b > 0, "B must have a lead gap");
        // Visual columns: Slot(0), then gap_b Gaps, then Slot(1).
        assert_eq!(layout.columns[0], VisualColumn::Slot(0));
        for g in 1..=gap_b {
            assert_eq!(layout.columns[g], VisualColumn::Gap, "col {g} must be Gap");
        }
        assert_eq!(layout.columns[gap_b + 1], VisualColumn::Slot(1));
        assert_eq!(layout.slot_to_visual_col[0], 0);
        assert_eq!(layout.slot_to_visual_col[1], gap_b + 1);
        assert_eq!(layout.total_cols, layout.columns.len());
    }

    // Z: resolve_viewport maps zoom to a horizontal scale. Fit compresses all
    // visual columns into width (scale<=1, window_start 0); Scale centers on selection.
    #[test]
    fn resolve_viewport_fit_and_scale() {
        // Fit with more columns than width -> scale = width/total, from column 0.
        let vp = resolve_viewport(ZoomMode::Fit, 1000, 100, Some(500));
        assert!((vp.scale - 0.1).abs() < 1e-9, "fit scale = 100/1000, got {}", vp.scale);
        assert_eq!(vp.window_start, 0.0, "fit anchors at 0");
        assert_eq!(vp.width, 100);

        // Scale(4.0): 4 cells per column, window centered on selected col, clamped.
        let vp = resolve_viewport(ZoomMode::Scale(4.0), 1000, 100, Some(500));
        assert!(vp.scale >= 1.0, "scale >= 1 stays 1+");
        let visible = 100.0 / vp.scale;
        assert!(vp.window_start >= 0.0 && vp.window_start <= 1000.0 - visible, "clamped: {}", vp.window_start);
        // Centered near 500.
        assert!((vp.window_start + visible / 2.0 - 500.0).abs() <= visible, "centered on selection");
    }

    // A1: annotation_visual_span maps an annotation's covered kernels (by ts span)
    // to the visual-column range of those kernels, so its end aligns to a kernel.
    #[test]
    fn annotation_visual_span_maps_column_range() {
        // T0 kernels A@0, B@100, C@200 (all matched); annotation covers [50,250] -> B,C.
        let mut a0 = ann(1, 50.0, "region");
        a0.dur = 200.0; // covers ts 50..250 -> B(100) and C(200)
        let t0 = trace_of(
            vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0), kd(1, 200.0, "C", 10.0)],
            vec![a0],
        );
        let t1 = trace_of(
            vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0), kd(1, 200.0, "C", 10.0)],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let layout = app.stream_layout(1).unwrap();
        let ann_idx = app.annotations.iter().position(|a| a.name == "region").unwrap();
        let (lo, hi) = app.annotation_visual_span(ann_idx, &layout).expect("span");
        // B is slot 1, C is slot 2; their visual cols.
        let col_b = layout.slot_to_visual_col[1];
        let col_c = layout.slot_to_visual_col[2];
        assert_eq!(lo, col_b, "span start at B's visual col");
        assert_eq!(hi, col_c, "span end at C's visual col (annotation end aligns to C)");
    }

    // T6: new_multi automatically runs compute_diff (no manual call needed)
    #[test]
    fn new_multi_runs_diff() {
        let t0 = trace_of(vec![kd(1, 100.0, "gemm", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 900.0, "extra", 5.0), kd(1, 910.0, "gemm", 5.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let extra_idx = app.kernels.iter().position(|k| k.trace_id == 1 && k.name == "extra").unwrap();
        assert_eq!(app.diff_status[extra_idx], DiffStatus::Added,
            "new_multi must auto-run compute_diff");
    }

    // ── TestBackend render proofs (S1/S2/S3) ─────────────────────────────────

    fn render_buffer(app: &App) -> ratatui::buffer::Buffer {
        use ratatui::{backend::TestBackend, Terminal};
        let mut t = Terminal::new(TestBackend::new(120, 20)).unwrap();
        t.draw(|f| crate::ui::render(f, app)).unwrap();
        t.backend().buffer().clone()
    }

    fn bg_at(buf: &ratatui::buffer::Buffer, x: u16, y: u16) -> Option<ratatui::style::Color> {
        buf.cell(ratatui::prelude::Position { x, y })
            .map(|c| c.style().bg)
            .and_then(|b| b)
    }

    fn first_kernel_col_in_row(buf: &ratatui::buffer::Buffer, y: u16, w: u16)
        -> Option<(u16, ratatui::style::Color)>
    {
        let sep = (0..w).find(|&x| {
            buf.cell(ratatui::prelude::Position { x, y })
               .map(|c| c.symbol() == "\u{2502}")  // │
               .unwrap_or(false)
        })?;
        // First non-space cell after the separator
        for x in (sep + 1)..w {
            if let Some(cell) = buf.cell(ratatui::prelude::Position { x, y }) {
                if cell.symbol() != " " {
                    if let Some(bg) = cell.style().bg {
                        return Some((x, bg));
                    }
                }
            }
        }
        None
    }

    // S2: Added kernel renders with GREEN_DIFF background
    #[test]
    fn render_added_cell_green() {
        use ratatui::style::Color;
        let t0 = trace_of(vec![kd(1, 0.0, "gemm", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 0.0, "gemm", 5.0), kd(1, 10.0, "extra", 5.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let buf = render_buffer(&app);
        let found_green = (0..20u16).flat_map(|y| (0..120u16).map(move |x| (x, y)))
            .any(|(x, y)| bg_at(&buf, x, y) == Some(Color::Rgb(34, 197, 94)));
        assert!(found_green, "Added kernel must render with GREEN_DIFF background");
    }

    // S3: Removed kernel renders with RED_DIFF background
    #[test]
    fn render_removed_cell_red() {
        use ratatui::style::Color;
        let t0 = trace_of(vec![kd(1, 0.0, "gemm", 5.0), kd(1, 10.0, "bn", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 0.0, "gemm", 5.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let buf = render_buffer(&app);
        let found_red = (0..20u16).flat_map(|y| (0..120u16).map(move |x| (x, y)))
            .any(|(x, y)| bg_at(&buf, x, y) == Some(Color::Rgb(220, 38, 38)));
        assert!(found_red, "Removed kernel must render with RED_DIFF background");
    }

    // S1: matched kernels render at same column, not with diff colors
    #[test]
    fn render_matched_same_column() {
        use ratatui::style::Color;
        // Both traces have gemm at different raw ts; compute_diff matches them
        let t0 = trace_of(vec![kd(1, 100.0, "gemm", 8.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 500.0, "gemm", 8.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let buf = render_buffer(&app);

        // Find T0 and T1 kernel lane rows
        let rows: Vec<(String, u16)> = (0..20u16)
            .filter_map(|y| {
                let row: String = (0..120u16).map(|x|
                    buf.cell(ratatui::prelude::Position { x, y })
                       .map(|c| c.symbol())
                       .unwrap_or(" ")
                       .chars().next().unwrap_or(' ')
                ).collect();
                if row.contains("cuda:1") { Some((row, y)) } else { None }
            })
            .collect();
        assert!(rows.len() >= 2, "need 2 kernel lanes: {:?}", rows.iter().map(|r| r.1).collect::<Vec<_>>());

        let col0 = first_kernel_col_in_row(&buf, rows[0].1, 120);
        let col1 = first_kernel_col_in_row(&buf, rows[1].1, 120);
        if let (Some((c0, bg0)), Some((c1, bg1))) = (col0, col1) {
            assert_eq!(c0, c1, "matched gemm must start at same column; T0 col={c0}, T1 col={c1}");
            assert_ne!(bg0, Color::Rgb(34, 197, 94), "matched T0 must not be green");
            assert_ne!(bg0, Color::Rgb(220, 38, 38), "matched T0 must not be red");
            assert_ne!(bg1, Color::Rgb(34, 197, 94), "matched T1 must not be green");
            assert_ne!(bg1, Color::Rgb(220, 38, 38), "matched T1 must not be red");
        }
    }

    // S4: single-trace renders no diff colors in the timeline
    #[test]
    fn render_single_trace_no_diff_color() {
        use ratatui::style::Color;
        let app = app_from(vec![
            make_kernel(1, 0.0, 5.0),
            make_kernel(1, 10.0, 5.0),
        ]);
        let buf = render_buffer(&app);
        let green = (0..20u16).flat_map(|y| (0..120u16).map(move |x| (x, y)))
            .any(|(x, y)| bg_at(&buf, x, y) == Some(Color::Rgb(34, 197, 94)));
        let red = (0..20u16).flat_map(|y| (0..120u16).map(move |x| (x, y)))
            .any(|(x, y)| bg_at(&buf, x, y) == Some(Color::Rgb(220, 38, 38)));
        assert!(!green, "single trace must have no green diff cells");
        assert!(!red,   "single trace must have no red diff cells");
    }

    // ── gap-aligned diff-layout render proofs (S1/S2/S3) ─────────────────────

    // Locate the two stream-1 kernel-lane rows (T0 then T1, round-robin order).
    // Lane rows sit inside the bordered block (│ at x=0) and carry a colored
    // kernel block; the header line also mentions cuda:1 but has no border.
    fn kernel_lane_rows(buf: &ratatui::buffer::Buffer) -> (u16, u16) {
        use ratatui::style::Color;
        let ys: Vec<u16> = (0..20u16)
            .filter(|&y| {
                let border = buf
                    .cell(ratatui::prelude::Position { x: 0, y })
                    .map(|c| c.symbol() == "\u{2502}")
                    .unwrap_or(false);
                let has_block = (0..120u16)
                    .any(|x| matches!(bg_at(buf, x, y), Some(c) if c != Color::Black));
                border && has_block
            })
            .collect();
        assert!(ys.len() >= 2, "need 2 kernel lane rows, got {ys:?}");
        (ys[0], ys[1])
    }

    // Start x of every contiguous colored (non-black, bg-set) block in the lane
    // area. Lane cells carry an explicit bg; the label/border/separator do not,
    // so scanning bg-set cells across the whole row isolates kernel blocks.
    fn block_starts_in_row(buf: &ratatui::buffer::Buffer, y: u16) -> Vec<u16> {
        use ratatui::style::Color;
        let mut starts = Vec::new();
        let mut prev_colored = false;
        for x in 0..120u16 {
            let colored = matches!(bg_at(buf, x, y), Some(c) if c != Color::Black);
            if colored && !prev_colored {
                starts.push(x);
            }
            prev_colored = colored;
        }
        starts
    }

    // Build a 2-trace app with `n` matched kernels per trace (more than lane width).
    fn make_many_kernel_app(n: usize) -> App {
        let names: Vec<String> = (0..n).map(|i| format!("k{i}")).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        make_two_trace_app(&refs, &refs)
    }

    // Z1: in Fit mode every slot is represented within the lane width (window at 0,
    // no clipping off the right edge).
    #[test]
    fn surface_fit_shows_all() {
        use ratatui::style::Color;
        let mut app = make_many_kernel_app(300);
        app.zoom_fit();
        let layout = app.stream_layout(1).unwrap();
        assert!(layout.total_cols > 118, "test needs more cols than width");
        let buf = render_buffer(&app);
        let (yr, _yg) = kernel_lane_rows(&buf);
        // Fit compresses the whole sequence: real (non-black) kernel color must reach
        // near the right edge of the lane, proving every slot is represented.
        let rightmost = (0..118u16)
            .rev()
            .find(|&x| matches!(bg_at(&buf, x, yr), Some(c) if c != Color::Black));
        assert!(rightmost.unwrap_or(0) > 100, "fit must fill lane width with kernels, got {rightmost:?}");
    }

    // Z1: a removed/added kernel merged into a compressed cell still shows its diff color.
    #[test]
    fn surface_fit_preserves_diff_color() {
        use ratatui::style::Color;
        // 200 matched + one removed (T0 only) + one added (T1 only).
        let mut t0n: Vec<String> = (0..200).map(|i| format!("k{i}")).collect();
        let mut t1n = t0n.clone();
        t0n.push("REMOVED_ONLY".to_string());
        t1n.push("ADDED_ONLY".to_string());
        let r0: Vec<&str> = t0n.iter().map(|s| s.as_str()).collect();
        let r1: Vec<&str> = t1n.iter().map(|s| s.as_str()).collect();
        let mut app = make_two_trace_app(&r0, &r1);
        app.zoom_fit();
        let buf = render_buffer(&app);
        let (yr, yg) = kernel_lane_rows(&buf);
        let has_red = (0..120u16).any(|x| bg_at(&buf, x, yr) == Some(Color::Rgb(220, 38, 38)));
        let has_green = (0..120u16).any(|x| bg_at(&buf, x, yg) == Some(Color::Rgb(34, 197, 94)));
        assert!(has_red, "removed kernel must stay red even when merged in fit mode");
        assert!(has_green, "added kernel must stay green even when merged in fit mode");
    }

    // A1: an annotation whose span is entirely left of the viewport must NOT paint
    // cell 0 (no offscreen-left bleed).
    #[test]
    fn surface_annotation_no_offscreen_left_bleed() {
        use ratatui::style::Color;
        let n = 300usize;
        let mk = || -> Vec<KernelEvent> {
            (0..n).map(|i| kd(1, i as f64 * 10.0, &format!("k{i}"), 5.0)).collect()
        };
        let mut a0 = ann(1, 0.0, "early");
        a0.dur = 5.0; // covers only k0 (ts 0..5), far left of a right-panned viewport
        let t0 = trace_of(mk(), vec![a0]);
        let t1 = trace_of(mk(), vec![]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        for _ in 0..6 {
            app.zoom_in();
        }
        select_kernel(&mut app, 0, &format!("k{}", n - 1));
        let buf = render_buffer(&app);
        let ann_lane = app.lanes.iter().position(|l| l.is_annotations() && l.trace_id() == 0).unwrap();
        let ann_y = 2u16 + ann_lane as u16;
        let ann_bg = Color::Rgb(90, 90, 110);
        let bled = (0..120u16).any(|x| bg_at(&buf, x, ann_y) == Some(ann_bg));
        assert!(!bled, "offscreen-left annotation must not bleed into the viewport");
    }

    // A1: an annotation covering a late kernel renders (in column space) with its
    // right edge at that kernel's column, not at the annotation's raw timestamp.
    #[test]
    fn surface_annotation_aligns_to_kernel_column() {
        use ratatui::style::Color;
        let mut a0 = ann(1, 50.0, "region");
        a0.dur = 200.0;
        let ks = || vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0), kd(1, 200.0, "C", 10.0)];
        let t0 = trace_of(ks(), vec![a0]);
        let t1 = trace_of(ks(), vec![]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        for _ in 0..4 {
            app.zoom_in();
        }
        let buf = render_buffer(&app);

        let ann_lane = app.lanes.iter().position(|l| l.is_annotations() && l.trace_id() == 0).unwrap();
        let kern_lane = app.lanes.iter().position(|l| !l.is_annotations() && l.trace_id() == 0).unwrap();
        let ann_y = 2u16 + ann_lane as u16;
        let kern_y = 2u16 + kern_lane as u16;

        let ann_right = (0..120u16).rev().find(|&x| {
            x < 118 && matches!(bg_at(&buf, x, ann_y), Some(c) if c != Color::Black)
        });
        let kern_starts: Vec<u16> = block_starts_in_row(&buf, kern_y).into_iter().filter(|&x| x < 118).collect();
        let c_col = *kern_starts.last().unwrap();
        let ar = ann_right.expect("annotation must render a colored block");
        assert!(ar >= c_col.saturating_sub(2), "annotation end must align to C's column: ann_right={ar} c_col={c_col}");
    }

    // G1: idle before B (large) renders more leading blank columns than before C
    // (small); both lanes mirror identical columns so matched kernels stay aligned.
    #[test]
    fn surface_idle_gap_visible() {
        use ratatui::style::Color;
        let ks = || vec![kd(1, 0.0, "A", 10.0), kd(1, 100.0, "B", 10.0), kd(1, 120.0, "C", 10.0)];
        let t0 = trace_of(ks(), vec![]);
        let t1 = trace_of(ks(), vec![]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        for _ in 0..4 {
            app.zoom_in();
        }
        let buf = render_buffer(&app);
        let (yr, yg) = kernel_lane_rows(&buf);

        // Ignore the right border column (x>=118) which can bleed a colored cell.
        let content = |v: Vec<u16>| -> Vec<u16> { v.into_iter().filter(|&x| x < 118).collect() };
        let starts = content(block_starts_in_row(&buf, yr));
        assert_eq!(starts.len(), 3, "3 kernel blocks in T0: {starts:?}");
        let gap_before = |bi: usize| -> u16 {
            let mut g = 0u16;
            let mut x = starts[bi];
            while x > 0 {
                x -= 1;
                if bg_at(&buf, x, yr) == Some(Color::Black) {
                    g += 1;
                } else {
                    break;
                }
            }
            g
        };
        let gap_b = gap_before(1);
        let gap_c = gap_before(2);
        assert!(gap_b > gap_c, "idle before B (90) wider than before C (10): {gap_b} vs {gap_c}");

        let starts_t1 = content(block_starts_in_row(&buf, yg));
        assert_eq!(starts, starts_t1, "T0/T1 matched kernels must align: {starts:?} vs {starts_t1:?}");
    }

    // Z2: zooming in gives each slot >=1 cell so individual kernels are distinct.
    #[test]
    fn surface_zoom_in_separates() {
        let mut app = make_two_trace_app(&["A", "B", "C"], &["A", "B", "C"]);
        for _ in 0..6 {
            app.zoom_in();
        }
        let buf = render_buffer(&app);
        let (yr, _yg) = kernel_lane_rows(&buf);
        // At high zoom the 3 matched kernels render as 3 separate colored blocks
        // (ignoring the right-border bleed at x>=118).
        let starts: Vec<u16> = block_starts_in_row(&buf, yr).into_iter().filter(|&x| x < 118).collect();
        assert_eq!(starts.len(), 3, "zoomed-in kernels must render 3 distinct blocks, got {starts:?}");
    }

    // S1 (gap): removed B is red in T0 lane; SAME column is a black gap in T1 lane.
    #[test]
    fn surface_s1_removed_gap() {
        use ratatui::style::Color;
        let app = make_two_trace_app(&["A", "B", "C"], &["A", "C"]);
        let buf = render_buffer(&app);
        let (yr, yg) = kernel_lane_rows(&buf);

        let red_x = (0..120u16)
            .find(|&x| bg_at(&buf, x, yr) == Some(Color::Rgb(220, 38, 38)))
            .expect("removed B must render red in T0 lane");
        assert_eq!(
            bg_at(&buf, red_x, yg),
            Some(Color::Black),
            "T1 lane must be a black gap at removed-B column x={red_x}",
        );

        let s0 = block_starts_in_row(&buf, yr);
        let s1 = block_starts_in_row(&buf, yg);
        assert_eq!(s0.first(), s1.first(), "matched A must align: t0={s0:?} t1={s1:?}");
        assert_eq!(s0.last(), s1.last(), "matched C must align: t0={s0:?} t1={s1:?}");
    }

    // S2 (gap): added B is green in T1 lane; SAME column is a black gap in T0 lane.
    #[test]
    fn surface_s2_added_gap() {
        use ratatui::style::Color;
        let app = make_two_trace_app(&["A", "C"], &["A", "B", "C"]);
        let buf = render_buffer(&app);
        let (yr, yg) = kernel_lane_rows(&buf);

        let green_x = (0..120u16)
            .find(|&x| bg_at(&buf, x, yg) == Some(Color::Rgb(34, 197, 94)))
            .expect("added B must render green in T1 lane");
        assert_eq!(
            bg_at(&buf, green_x, yr),
            Some(Color::Black),
            "T0 lane must be a black gap at added-B column x={green_x}",
        );

        let s0 = block_starts_in_row(&buf, yr);
        let s1 = block_starts_in_row(&buf, yg);
        assert_eq!(s0.first(), s1.first(), "matched A must align: t0={s0:?} t1={s1:?}");
        assert_eq!(s0.last(), s1.last(), "matched C must align: t0={s0:?} t1={s1:?}");
    }

    // S3 (gap): all matched -> no red/green; T0 and T1 lanes render an identical
    // contiguous colored run (matched slots pack to the same columns in both).
    #[test]
    fn surface_s3_matched_no_gap() {
        use ratatui::style::Color;
        let app = make_two_trace_app(&["A", "B", "C"], &["A", "B", "C"]);
        let buf = render_buffer(&app);
        let (yr, yg) = kernel_lane_rows(&buf);

        for y in [yr, yg] {
            for x in 0..120u16 {
                assert_ne!(bg_at(&buf, x, y), Some(Color::Rgb(220, 38, 38)), "no red at {x},{y}");
                assert_ne!(bg_at(&buf, x, y), Some(Color::Rgb(34, 197, 94)), "no green at {x},{y}");
            }
        }

        let s0 = block_starts_in_row(&buf, yr);
        let s1 = block_starts_in_row(&buf, yg);
        assert_eq!(s0, s1, "matched lanes must render identical columns: t0={s0:?} t1={s1:?}");
        assert!(!s0.is_empty(), "matched kernels must render a colored run");

        // Contiguous packing: the colored run has no interior black cell.
        let first = *s0.first().unwrap();
        let last_colored = (first..120u16)
            .take_while(|&x| matches!(bg_at(&buf, x, yr), Some(c) if c != Color::Black))
            .last()
            .unwrap_or(first);
        let interior_black = (first..last_colored)
            .any(|x| bg_at(&buf, x, yr) == Some(Color::Black));
        assert!(!interior_black, "matched slots must pack contiguously (no interior black gap)");
    }

    // ── diff-column data model (T2 RED phase) ────────────────────────────────

    // D1: three matched kernels -> 3 slots, all with both sides present, and
    // corresponding kernels share the same column index.
    #[test]
    fn diff_columns_matched_share_column() {
        // Given: both traces carry A, B, C on stream 1
        let app = make_two_trace_app(&["A", "B", "C"], &["A", "B", "C"]);
        // Then: 3 diff column slots exist for stream 1
        assert_eq!(app.diff_columns_by_stream[&1].slots.len(), 3);
        // Then: every slot has both sides present (matched)
        assert!(
            app.diff_columns_by_stream[&1]
                .slots
                .iter()
                .all(|s| s.t0_kernel.is_some() && s.t1_kernel.is_some()),
            "all slots must be matched (both sides present)",
        );
        // Then: the "A" kernel from each trace maps to the same column
        let a0 = app
            .kernels
            .iter()
            .position(|k| k.name == "A" && k.trace_id == 0)
            .expect("A in t0");
        let a1 = app
            .kernels
            .iter()
            .position(|k| k.name == "A" && k.trace_id == 1)
            .expect("A in t1");
        assert_eq!(
            app.kernel_diff_column[a0].expect("a0 must have a column").column,
            app.kernel_diff_column[a1].expect("a1 must have a column").column,
            "matched A kernels must share a column index",
        );
    }

    // D2: T0=[A,B,C], T1=[A,C] -> slot for B has t1_kernel=None; C kernels
    // still share the same column in both traces.
    #[test]
    fn diff_columns_removed_makes_gap() {
        // Given: T0 has B, T1 does not
        let app = make_two_trace_app(&["A", "B", "C"], &["A", "C"]);
        let slots = &app.diff_columns_by_stream[&1].slots;
        // Then: at least one slot is a t0-only gap (B removed)
        assert!(
            slots.iter().any(|s| s.t1_kernel.is_none() && s.t0_kernel.is_some()),
            "B (removed) must produce a slot with t0_kernel=Some, t1_kernel=None",
        );
        // Then: the C kernels in both traces share the same column
        let c0 = app
            .kernels
            .iter()
            .position(|k| k.name == "C" && k.trace_id == 0)
            .expect("C in t0");
        let c1 = app
            .kernels
            .iter()
            .position(|k| k.name == "C" && k.trace_id == 1)
            .expect("C in t1");
        assert_eq!(
            app.kernel_diff_column[c0].expect("c0 must have a column").column,
            app.kernel_diff_column[c1].expect("c1 must have a column").column,
            "matched C kernels must share a column index despite B gap",
        );
    }

    // D3: T0=[A,C], T1=[A,B,C] -> slot for B has t0_kernel=None (B added).
    #[test]
    fn diff_columns_added_makes_gap() {
        // Given: T1 has B, T0 does not
        let app = make_two_trace_app(&["A", "C"], &["A", "B", "C"]);
        let slots = &app.diff_columns_by_stream[&1].slots;
        // Then: at least one slot is a t1-only gap (B added)
        assert!(
            slots.iter().any(|s| s.t0_kernel.is_none() && s.t1_kernel.is_some()),
            "B (added) must produce a slot with t0_kernel=None, t1_kernel=Some",
        );
    }

    // D4: compute_diff must NOT touch traces[1].offset_us (stays 0.0 from load).
    #[test]
    fn compute_diff_does_not_set_offset() {
        // Given: two traces with "gemm" at very different timestamps, no shared annotation
        let t0 = trace_of(vec![kd(1, 100.0, "gemm", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 900.0, "gemm", 5.0)], vec![]);
        // When: App::new_multi constructs the app (compute_diff runs inside)
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        // Then: offset_us is 0.0 — compute_diff must not write it
        assert_eq!(
            app.traces[1].offset_us,
            0.0,
            "compute_diff must not modify offset_us; got {}",
            app.traces[1].offset_us,
        );
    }
}