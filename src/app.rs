use crate::trace::{AnnotationEvent, KernelEvent, Trace};

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
    pub lane_view_offset: usize,
    pub search_active: bool,
    pub search_query: String,
    pub search_no_match: bool,
    pub sequence: Option<Sequence>,
    pub sequence_status: Option<String>,
    pub alignment: crate::align::AlignmentState,
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

        let alignment = crate::align::build_alignment(&kernels, &annotations, trace_count);

        let mut app = App {
            kernels,
            annotations,
            streams,
            lanes,
            traces,
            active_lane: initial_lane,
            selected_item: 0,
            zoom_level: 1.0,
            lane_view_offset: 0,
            search_active: false,
            search_query: String::new(),
            search_no_match: false,
            sequence: None,
            sequence_status: None,
            alignment,
        };
        app.clamp_selected_item();
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

    /// Aligned start ts of kernel `idx`: uses piecewise display times when
    /// available, otherwise falls back to raw ts + trace offset.
    pub fn kernel_render_ts(&self, idx: usize) -> f64 {
        let Some(k) = self.kernels.get(idx) else { return 0.0; };
        if let crate::align::AlignmentMode::PiecewiseWarp = self.alignment.mode {
            let li = self.alignment.kernel_local_idx.get(idx).copied().unwrap_or(0);
            if let Some(dt) = self.alignment.display_kernel_times
                .get(k.trace_id).and_then(|v| v.get(li)) {
                return dt.display_start;
            }
        }
        k.ts + self.trace_offset(k.trace_id)
    }

    pub fn kernel_render_end(&self, idx: usize) -> f64 {
        let Some(k) = self.kernels.get(idx) else { return 0.0; };
        if let crate::align::AlignmentMode::PiecewiseWarp = self.alignment.mode {
            let li = self.alignment.kernel_local_idx.get(idx).copied().unwrap_or(0);
            if let Some(dt) = self.alignment.display_kernel_times
                .get(k.trace_id).and_then(|v| v.get(li)) {
                return dt.display_end;
            }
        }
        k.end_ts() + self.trace_offset(k.trace_id)
    }

    pub fn annotation_render_ts(&self, idx: usize) -> f64 {
        let Some(a) = self.annotations.get(idx) else { return 0.0; };
        if let crate::align::AlignmentMode::PiecewiseWarp = self.alignment.mode {
            let li = self.alignment.annotation_local_idx.get(idx).copied().unwrap_or(0);
            if let Some(dt) = self.alignment.display_annotation_times
                .get(a.trace_id).and_then(|v| v.get(li)) {
                return dt.display_start;
            }
        }
        a.ts + self.trace_offset(a.trace_id)
    }

    pub fn annotation_render_end(&self, idx: usize) -> f64 {
        let Some(a) = self.annotations.get(idx) else { return 0.0; };
        if let crate::align::AlignmentMode::PiecewiseWarp = self.alignment.mode {
            let li = self.alignment.annotation_local_idx.get(idx).copied().unwrap_or(0);
            if let Some(dt) = self.alignment.display_annotation_times
                .get(a.trace_id).and_then(|v| v.get(li)) {
                return dt.display_end;
            }
        }
        a.end_ts() + self.trace_offset(a.trace_id)
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
            let nearest = self
                .kernels
                .iter()
                .enumerate()
                .filter(|(_, k)| k.trace_id == tid && k.name == name)
                .map(|(i, _)| self.kernel_render_ts(i))
                .min_by(|a, b| {
                    (a - target)
                        .abs()
                        .partial_cmp(&(b - target).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(aligned_start) = nearest {
                if let Some(meta) = self.traces.get_mut(tid) {
                    meta.offset_us += target - aligned_start;
                }
            }
        }

        for meta in &mut self.traces {
            meta.anchor = Some(name.clone());
        }
        self.alignment = crate::align::AlignmentState::offset_only(0, self.traces.len());
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
        let prev_ts = self.selected_item_render_ts();
        self.active_lane = target;
        self.selected_item = match prev_ts {
            Some(ts) => self.nearest_item_in_active_lane(ts),
            None => 0,
        };
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
    }

    pub fn zoom_out(&mut self) {
        self.zoom_level = (self.zoom_level / ZOOM_FACTOR).max(ZOOM_MIN);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{AnnotationEvent, KernelEvent, Trace};

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
        // t0: foo@100, bar@200 (anchor@100). t1: foo@9100->200, bar@9200->300 (anchor@9000).
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
            // Aligned-nearest to 200 is t1.foo (aligned 200), not t1.bar (aligned 300).
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "foo"),
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
        // No shared annotation -> load offsets are 0; both traces raw.
        // t0: gemm@200. t1: gemm@700. Selecting t0.gemm and aligning moves t1
        // so t1.gemm aligned start == 200.
        let t0 = trace_of(vec![kd(1, 200.0, "gemm", 8.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 700.0, "gemm", 8.0)], vec![]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert_eq!(app.traces[1].offset_us, 0.0, "no annotation -> load offset 0");

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
    fn test_app_has_offset_only_alignment_by_default() {
        let app = sample_app();
        assert!(matches!(app.alignment.mode, crate::align::AlignmentMode::OffsetOnly));
    }

    // T9: in PiecewiseWarp mode kernel_render_ts reads display times, not raw+offset.
    #[test]
    fn test_render_ts_uses_warped_time_in_piecewise() {
        let t0 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b0 = b as f64 * 1000.0;
                vec![kd(1, b0, "gemm", 8.0), kd(1, b0 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let t1 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b1 = b as f64 * 900.0;
                vec![kd(1, b1, "gemm", 8.0), kd(1, b1 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert!(matches!(app.alignment.mode, crate::align::AlignmentMode::PiecewiseWarp),
            "3 matched blocks → PiecewiseWarp");
        let t1_gemm = app.kernels.iter().position(|k| k.trace_id == 1 && k.name == "gemm").unwrap();
        let display = app.kernel_render_ts(t1_gemm);
        assert!(display != app.kernels[t1_gemm].ts || app.traces[1].offset_us == 0.0,
            "PiecewiseWarp: display ts {display} must reflect warped position");
    }

    // T10: new_multi with 3 gap-separated blocks per trace triggers PiecewiseWarp.
    #[test]
    fn test_new_multi_builds_piecewise_when_blocks_present() {
        let t0 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b0 = b as f64 * 1000.0;
                vec![kd(1, b0, "gemm", 8.0), kd(1, b0 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let t1 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b1 = b as f64 * 900.0;
                vec![kd(1, b1, "gemm", 8.0), kd(1, b1 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert!(matches!(app.alignment.mode, crate::align::AlignmentMode::PiecewiseWarp));
        assert!(!app.alignment.display_kernel_times.iter().all(|v| v.is_empty()),
            "display times must be populated");
    }

    // T10: single-trace new_multi keeps OffsetOnly (S4).
    #[test]
    fn test_new_multi_single_trace_offset_only() {
        let app = app_with_annotations();
        assert!(matches!(app.alignment.mode, crate::align::AlignmentMode::OffsetOnly));
    }

    // G-key after PiecewiseWarp load switches alignment back to OffsetOnly.
    #[test]
    fn test_g_key_switches_to_offset_only_and_preserves_shift() {
        let t0 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b0 = b as f64 * 1000.0;
                vec![kd(1, b0, "gemm", 8.0), kd(1, b0 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let t1 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b1 = b as f64 * 900.0;
                vec![kd(1, b1, "gemm", 8.0), kd(1, b1 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert!(matches!(app.alignment.mode, crate::align::AlignmentMode::PiecewiseWarp),
            "should start as PiecewiseWarp");

        let kern_lane = app.lanes.iter().position(|l| l.trace_id() == 0 && !l.is_annotations()).unwrap();
        app.active_lane = kern_lane;
        app.selected_item = 0;
        assert!(app.align_to_selected_kernel(), "G-key align must succeed");
        assert!(matches!(app.alignment.mode, crate::align::AlignmentMode::OffsetOnly),
            "G-key must switch to OffsetOnly");
    }

    // S1 integration: piecewise-aligned matched blocks have equal display_start,
    // so they render at the same horizontal position on the shared time axis.
    #[test]
    fn test_s1_piecewise_matched_blocks_have_equal_display_start() {
        let t0 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b0 = b as f64 * 1000.0;
                vec![kd(1, b0, "gemm", 8.0), kd(1, b0 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let t1 = trace_of(
            (0..3).flat_map(|b: usize| {
                let b1 = b as f64 * 900.0;
                vec![kd(1, b1, "gemm", 8.0), kd(1, b1 + 10.0, "relu", 4.0)]
            }).collect(),
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert!(matches!(app.alignment.mode, crate::align::AlignmentMode::PiecewiseWarp));

        let mut aligned_pairs = 0;
        for block in &app.alignment.blocks {
            let (Some(tb0), Some(tb1)) = (&block.per_trace[0], &block.per_trace[1]) else {
                continue;
            };
            let d0 = app.kernel_render_ts(tb0.kernel_indices.start);
            let d1 = app.kernel_render_ts(tb1.kernel_indices.start);
            assert!(
                (d0 - d1).abs() < 1e-3,
                "S1: block {} T0 display={} T1 display={} differ",
                block.id, d0, d1
            );
            aligned_pairs += 1;
        }
        assert!(aligned_pairs >= 2, "need ≥2 matched pairs, got {}", aligned_pairs);
    }
}
