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

/// Per-trace metadata. Holds only the display label; render positions live in
/// the per-event override vectors on `App`.
#[derive(Debug, Clone)]
pub struct TraceMeta {
    pub label: String,
}

/// How exactly two traces are laid out on the shared time axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMode {
    /// Kernel-sequence git diff: matched kernels snap onto the anchor trace and
    /// unmatched kernels open inserted gaps.
    Diff,
    /// Every trace zero-based to a common start; internal timing preserved.
    Normal,
}

/// Per-kernel git-diff classification in two-trace diff mode. Matched kernels
/// carry no status (`None` in `kernel_diff_status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDiff {
    /// Present only in the second trace (a diff insertion).
    Added,
    /// Present only in the anchor trace (a diff deletion).
    Deleted,
}

/// Winning render class for a single terminal column in a kernel lane. When
/// several kernels overlap one column, the highest-ranked class wins so a wide
/// matched (dimmed) block never hides a narrower added/deleted one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnClass {
    Selected,
    Deleted,
    Added,
    /// Unchanged kernel: dimmed in diff mode, normal base colour otherwise.
    Matched,
}

impl ColumnClass {
    fn rank(self) -> u8 {
        match self {
            ColumnClass::Selected => 4,
            ColumnClass::Deleted => 3,
            ColumnClass::Added => 2,
            ColumnClass::Matched => 1,
        }
    }
}

/// Screen geometry of the last-rendered lane area, recorded by the UI so mouse
/// clicks can be reverse-mapped to a lane and an item. All fields are in
/// terminal cell coordinates except the time window.
#[derive(Debug, Clone, Copy, Default)]
pub struct LaneLayout {
    /// Top-left of the lane block interior (inside the border).
    pub inner_x: u16,
    pub inner_y: u16,
    /// Interior height in rows (one lane per row).
    pub inner_h: u16,
    /// Width of the label gutter (label text plus the `│` separator).
    pub label_width: u16,
    /// Width of the timeline area to the right of the gutter.
    pub lane_width: u16,
    /// First lane index shown at `inner_y` (mirrors `lane_view_offset`).
    pub view_offset: usize,
    /// Aligned time window mapped across `lane_width`.
    pub ts_start: f64,
    pub time_span: f64,
}

#[derive(Debug, Clone)]
pub struct App {
    pub kernels: Vec<KernelEvent>,
    pub annotations: Vec<AnnotationEvent>,
    pub streams: Vec<u64>,
    pub lanes: Vec<Lane>,
    pub traces: Vec<TraceMeta>,
    pub align_mode: AlignMode,
    pub active_lane: usize,
    pub selected_item: usize,
    pub zoom_level: f64,
    pub lane_view_offset: usize,
    pub search_active: bool,
    pub search_query: String,
    pub search_no_match: bool,
    /// All items matching the current query as `(lane_idx, item_pos)`, ordered by
    /// lane then position; cycled by `search_next` / `search_prev`.
    pub search_matches: Vec<(usize, usize)>,
    /// Index into `search_matches` of the currently selected match.
    pub search_match_idx: usize,
    pub sequence: Option<Sequence>,
    pub sequence_status: Option<String>,
    /// Transient one-line status shown in the header (e.g. after a CSV export),
    /// cleared on the next key press.
    pub status: Option<String>,
    /// Absolute render-start per kernel (by flat index), always populated by
    /// `recompute_alignment`.
    pub kernel_render_overrides: Vec<Option<f64>>,
    /// Absolute (start, end) render span per annotation (by flat index), always
    /// populated by `recompute_alignment`.
    pub annotation_render_overrides: Vec<Option<(f64, f64)>>,
    /// Per-kernel git-diff status (by flat index) in two-trace diff mode; `None`
    /// for matched kernels and in every non-diff layout.
    pub kernel_diff_status: Vec<Option<KernelDiff>>,
    /// Screen geometry of the last-rendered lane area, for mouse hit-testing.
    pub lane_layout: LaneLayout,
}

impl App {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(trace: Trace) -> Self {
        App::new_multi(vec![("T0".to_string(), trace)])
    }

    /// Build an App from one or more labelled traces. Each trace's events are
    /// stamped with its trace index, lanes are interleaved round-robin, and
    /// render positions are computed by `recompute_alignment` (default: diff
    /// mode for exactly two traces, normal zero-base otherwise).
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

        let traces: Vec<TraceMeta> = labels.into_iter().map(|label| TraceMeta { label }).collect();

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

        let kernel_render_overrides = vec![None; kernels.len()];
        let annotation_render_overrides = vec![None; annotations.len()];
        let kernel_diff_status = vec![None; kernels.len()];

        let mut app = App {
            kernels,
            annotations,
            streams,
            lanes,
            traces,
            align_mode: AlignMode::Diff,
            active_lane: initial_lane,
            selected_item: 0,
            zoom_level: 1.0,
            lane_view_offset: 0,
            search_active: false,
            search_query: String::new(),
            search_no_match: false,
            search_matches: Vec::new(),
            search_match_idx: 0,
            sequence: None,
            sequence_status: None,
            status: None,
            kernel_render_overrides,
            annotation_render_overrides,
            kernel_diff_status,
            lane_layout: LaneLayout::default(),
        };
        app.recompute_alignment();
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

    /// Absolute render-start of kernel `idx` on the shared axis.
    pub fn kernel_render_ts(&self, idx: usize) -> f64 {
        match self.kernel_render_overrides.get(idx) {
            Some(Some(ts)) => *ts,
            _ => self.kernels.get(idx).map(|k| k.ts).unwrap_or(0.0),
        }
    }

    pub fn kernel_render_end(&self, idx: usize) -> f64 {
        let dur = self.kernels.get(idx).map(|k| k.dur).unwrap_or(0.0);
        self.kernel_render_ts(idx) + dur
    }

    pub fn annotation_render_ts(&self, idx: usize) -> f64 {
        match self.annotation_render_overrides.get(idx) {
            Some(Some((start, _))) => *start,
            _ => self.annotations.get(idx).map(|a| a.ts).unwrap_or(0.0),
        }
    }

    pub fn annotation_render_end(&self, idx: usize) -> f64 {
        match self.annotation_render_overrides.get(idx) {
            Some(Some((_, end))) => *end,
            _ => self.annotations.get(idx).map(|a| a.end_ts()).unwrap_or(0.0),
        }
    }

    /// Git-diff status of kernel `idx`: `Some(Added|Deleted)` only for unmatched
    /// kernels in two-trace diff mode, `None` otherwise.
    pub fn kernel_diff(&self, idx: usize) -> Option<KernelDiff> {
        self.kernel_diff_status.get(idx).copied().flatten()
    }

    /// Per-column winning `(class, owner_pos)` for a kernel lane at the given
    /// window and `width`, applying the priority order
    /// Selected > Deleted > Added > Matched. `owner_pos` indexes the lane's
    /// `item_indices()`. Columns with no kernel are `None`. When `diff_active` is
    /// false, every non-selected kernel column is `Matched` (the UI maps that to
    /// the normal base colour rather than a dimmed one).
    pub fn lane_column_classes(
        &self,
        lane_idx: usize,
        ts_start: f64,
        time_span: f64,
        width: usize,
        diff_active: bool,
    ) -> Vec<Option<(ColumnClass, usize)>> {
        let mut cols: Vec<Option<(ColumnClass, usize)>> = vec![None; width];
        if width == 0 {
            return cols;
        }
        let Some(lane) = self.lanes.get(lane_idx) else {
            return cols;
        };
        let is_active_lane = lane_idx == self.active_lane;
        let ts_end = ts_start + time_span;

        for (pos, &item_idx) in lane.item_indices().iter().enumerate() {
            let class = if is_active_lane && pos == self.selected_item {
                ColumnClass::Selected
            } else if !diff_active {
                ColumnClass::Matched
            } else {
                match self.kernel_diff(item_idx) {
                    Some(KernelDiff::Added) => ColumnClass::Added,
                    Some(KernelDiff::Deleted) => ColumnClass::Deleted,
                    None => ColumnClass::Matched,
                }
            };
            let (start, end) = match lane {
                Lane::Kernels { .. } => (
                    self.kernel_render_ts(item_idx),
                    self.kernel_render_end(item_idx),
                ),
                Lane::Annotations { .. } => (
                    self.annotation_render_ts(item_idx),
                    self.annotation_render_end(item_idx),
                ),
            };
            let Some((start_col, end_col)) =
                kernel_columns(start, end, ts_start, ts_end, width)
            else {
                continue;
            };
            for cell in cols.iter_mut().take(end_col).skip(start_col) {
                let win = match cell {
                    Some((c, _)) => class.rank() > c.rank(),
                    None => true,
                };
                if win {
                    *cell = Some((class, pos));
                }
            }
        }
        cols
    }

    /// Header label for the current alignment mode when exactly two traces are
    /// open (`"diff"` / `"normal"`), or `None` otherwise.
    pub fn mode_label(&self) -> Option<String> {
        if self.traces.len() != 2 {
            return None;
        }
        Some(
            match self.align_mode {
                AlignMode::Diff => "diff",
                AlignMode::Normal => "normal",
            }
            .to_string(),
        )
    }

    /// Per-trace display labels shortened to their differing part (common prefix
    /// and suffix stripped), for compact lane labels.
    pub fn trace_display_labels(&self) -> Vec<String> {
        let labels: Vec<String> = self.traces.iter().map(|t| t.label.clone()).collect();
        shorten_labels(&labels)
    }

    /// Toggles between diff and normal alignment for exactly two traces and
    /// recomputes render positions. No-op (returns false) for any other count.
    pub fn toggle_align_mode(&mut self) -> bool {
        if self.traces.len() != 2 {
            return false;
        }
        self.align_mode = match self.align_mode {
            AlignMode::Diff => AlignMode::Normal,
            AlignMode::Normal => AlignMode::Diff,
        };
        self.recompute_alignment();
        true
    }

    /// Rebuilds both render-override vectors from scratch for the current mode.
    /// Every event is first zero-based to the common global start (normal); for
    /// exactly two traces in diff mode, each common stream's kernels are then
    /// remapped by the git-diff and its annotations carried along a per-stream
    /// `TimeMap` built from the kernel control points.
    pub fn recompute_alignment(&mut self) {
        let trace_count = self.traces.len();
        let normal_off = self.normal_offsets();

        for (i, k) in self.kernels.iter().enumerate() {
            self.kernel_render_overrides[i] = Some(k.ts + normal_off[k.trace_id]);
        }
        for (i, a) in self.annotations.iter().enumerate() {
            let off = normal_off[a.trace_id];
            self.annotation_render_overrides[i] = Some((a.ts + off, a.end_ts() + off));
        }
        for s in &mut self.kernel_diff_status {
            *s = None;
        }

        if trace_count != 2 || self.align_mode != AlignMode::Diff {
            return;
        }

        let mut streams: Vec<u64> = self.kernels.iter().map(|k| k.stream).collect();
        streams.sort_unstable();
        streams.dedup();

        for stream in streams {
            let idx0 = stream_kernel_indices(&self.kernels, 0, stream);
            let idx1 = stream_kernel_indices(&self.kernels, 1, stream);
            if idx0.is_empty() || idx1.is_empty() {
                continue;
            }
            let (ctrl0, ctrl1) = remap_stream(
                &self.kernels,
                &idx0,
                &idx1,
                normal_off[0],
                normal_off[1],
                &mut self.kernel_render_overrides,
                &mut self.kernel_diff_status,
            );
            let map0 = TimeMap::new(ctrl0, normal_off[0]);
            let map1 = TimeMap::new(ctrl1, normal_off[1]);
            self.remap_stream_annotations(stream, 0, &map0);
            self.remap_stream_annotations(stream, 1, &map1);
        }
    }

    /// Per-trace normal offset that zero-bases every trace to the earliest raw
    /// timestamp across all traces: `offset[t] = global_min - trace_min[t]`.
    fn normal_offsets(&self) -> Vec<f64> {
        let n = self.traces.len();
        let mut trace_min = vec![f64::MAX; n];
        for k in &self.kernels {
            if k.ts < trace_min[k.trace_id] {
                trace_min[k.trace_id] = k.ts;
            }
        }
        for a in &self.annotations {
            if a.ts < trace_min[a.trace_id] {
                trace_min[a.trace_id] = a.ts;
            }
        }
        let global_min = trace_min
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::MAX, f64::min);
        let global_min = if global_min.is_finite() {
            global_min
        } else {
            0.0
        };
        trace_min
            .iter()
            .map(|&m| if m.is_finite() { global_min - m } else { 0.0 })
            .collect()
    }

    /// Overwrites annotation overrides on one (trace, stream) by mapping each
    /// annotation's start and end through the stream's diff `TimeMap`.
    fn remap_stream_annotations(&mut self, stream: u64, trace_id: usize, map: &TimeMap) {
        for i in 0..self.annotations.len() {
            let a = &self.annotations[i];
            if a.trace_id != trace_id || a.stream != stream {
                continue;
            }
            let start = map.map(a.ts);
            let end = start.max(map.map(a.end_ts()));
            self.annotation_render_overrides[i] = Some((start, end));
        }
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

    // ── Mouse ────────────────────────────────────────────────────────────────

    /// Lane index shown at terminal row `row`, if the row falls within the
    /// rendered lane area and maps to an existing lane.
    fn lane_at_row(&self, row: u16) -> Option<usize> {
        let lo = &self.lane_layout;
        if row < lo.inner_y || row >= lo.inner_y + lo.inner_h {
            return None;
        }
        let lane_idx = lo.view_offset + (row - lo.inner_y) as usize;
        (lane_idx < self.lanes.len()).then_some(lane_idx)
    }

    /// Item position within `lane_idx` whose rendered columns cover terminal
    /// column `col`, or the nearest item's position when the click lands on
    /// empty timeline space. `None` only if the lane has no items or the click
    /// is left of the timeline gutter.
    fn item_at_col(&self, lane_idx: usize, col: u16) -> Option<usize> {
        let lo = &self.lane_layout;
        let lane = self.lanes.get(lane_idx)?;
        let n = lane.item_indices().len();
        if n == 0 {
            return None;
        }
        let lane_x0 = lo.inner_x + lo.label_width;
        if col < lane_x0 || lo.lane_width == 0 {
            return None;
        }
        let click = (col - lane_x0) as usize;
        let width = lo.lane_width as usize;
        let ts_end = lo.ts_start + lo.time_span;

        let mut nearest = 0usize;
        let mut nearest_dist = usize::MAX;
        for pos in 0..n {
            let (ts, end) = self.item_render_span(lane, pos);
            let Some((s, e)) = kernel_columns(ts, end, lo.ts_start, ts_end, width) else {
                continue;
            };
            if click >= s && click < e {
                return Some(pos);
            }
            let center = (s + e) / 2;
            let dist = center.abs_diff(click);
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest = pos;
            }
        }
        Some(nearest)
    }

    /// Rendered (start, end) timestamps of the item at `pos` in `lane`.
    fn item_render_span(&self, lane: &Lane, pos: usize) -> (f64, f64) {
        let idx = lane.item_indices().get(pos).copied().unwrap_or(0);
        match lane {
            Lane::Kernels { .. } => (self.kernel_render_ts(idx), self.kernel_render_end(idx)),
            Lane::Annotations { .. } => (
                self.annotation_render_ts(idx),
                self.annotation_render_end(idx),
            ),
        }
    }

    /// Handles a left click at terminal `(col, row)`. A click in a lane's
    /// timeline selects the item under (or nearest to) the cursor; a click in
    /// the label gutter just activates that lane. Returns whether anything was
    /// selected.
    pub fn click_select(&mut self, col: u16, row: u16) -> bool {
        let Some(lane_idx) = self.lane_at_row(row) else {
            return false;
        };
        let lo = self.lane_layout;
        let in_gutter = col < lo.inner_x + lo.label_width;

        self.active_lane = lane_idx;
        if in_gutter {
            self.clamp_selected_item();
            return true;
        }
        match self.item_at_col(lane_idx, col) {
            Some(pos) => {
                self.selected_item = pos;
                true
            }
            None => {
                self.clamp_selected_item();
                true
            }
        }
    }

    // ── Search over BOTH lane kinds ──────────────────────────────────────────

    pub fn search_start(&mut self) {
        self.search_active = true;
        self.search_query.clear();
        self.search_no_match = false;
        self.search_matches.clear();
        self.search_match_idx = 0;
    }

    pub fn search_cancel(&mut self) {
        self.search_active = false;
        self.search_query.clear();
        self.search_no_match = false;
        self.search_matches.clear();
        self.search_match_idx = 0;
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

    /// Rebuilds the ordered match list for the current query and jumps to the
    /// first match. Empty query clears matches and the no-match flag.
    fn search_apply(&mut self) {
        self.search_matches.clear();
        self.search_match_idx = 0;
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
                        self.search_matches.push((lane_idx, pos));
                    }
                }
            }
        }

        if let Some(&(lane_idx, pos)) = self.search_matches.first() {
            self.active_lane = lane_idx;
            self.selected_item = pos;
            self.search_no_match = false;
        } else {
            self.search_no_match = true;
        }
    }

    /// Moves the selection to the next match, wrapping past the last to the
    /// first. No-op when there are no matches.
    pub fn search_next(&mut self) {
        if self.search_matches.is_empty() {
            return;
        }
        self.search_match_idx = (self.search_match_idx + 1) % self.search_matches.len();
        self.goto_current_match();
    }

    /// Moves the selection to the previous match, wrapping past the first to the
    /// last. No-op when there are no matches.
    pub fn search_prev(&mut self) {
        if self.search_matches.is_empty() {
            return;
        }
        let n = self.search_matches.len();
        self.search_match_idx = (self.search_match_idx + n - 1) % n;
        self.goto_current_match();
    }

    fn goto_current_match(&mut self) {
        if let Some(&(lane_idx, pos)) = self.search_matches.get(self.search_match_idx) {
            self.active_lane = lane_idx;
            self.selected_item = pos;
        }
    }

    /// `"i/N"` position of the current match among all matches, or `None` when
    /// the query has no matches.
    pub fn search_match_label(&self) -> Option<String> {
        if self.search_matches.is_empty() {
            return None;
        }
        Some(format!(
            "{}/{}",
            self.search_match_idx + 1,
            self.search_matches.len()
        ))
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

    /// CSV of every kernel in the active lane (raw timestamps, full fields), or
    /// `None` when the active lane is not a kernel lane. Row order matches the
    /// lane's on-screen order.
    pub fn lane_kernels_csv(&self) -> Option<String> {
        let lane = self.lanes.get(self.active_lane)?;
        let Lane::Kernels { item_indices, .. } = lane else {
            return None;
        };
        let mut out = String::from(
            "idx,annotation,stage,name,ts,dur,end_ts,stream,device,grid,block,\
             shared_memory,registers_per_thread,correlation\n",
        );
        let opt_u64 = |v: Option<u64>| v.map(|n| n.to_string()).unwrap_or_default();
        for (row, &ki) in item_indices.iter().enumerate() {
            let Some(k) = self.kernels.get(ki) else {
                continue;
            };
            let ann =
                annotation_for_kernel(&self.annotations, k.stream, k.trace_id, k.ts);
            let ann_name = ann.map(|a| a.name.as_str()).unwrap_or("");
            let stage = ann.map(|a| stage_for_annotation_name(&a.name)).unwrap_or("");
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                row + 1,
                csv_escape(ann_name),
                stage,
                csv_escape(&k.name),
                k.ts,
                k.dur,
                k.end_ts(),
                k.stream,
                k.device,
                csv_escape(k.grid.as_deref().unwrap_or("")),
                csv_escape(k.block.as_deref().unwrap_or("")),
                opt_u64(k.shared_memory),
                opt_u64(k.registers_per_thread),
                opt_u64(k.correlation),
            ));
        }
        Some(out)
    }

    /// Suggested filename for the active kernel lane's CSV dump, e.g.
    /// `lane-cuda4-trace0.csv` (trace suffix only when multiple traces are open).
    pub fn lane_csv_filename(&self) -> String {
        let Some(lane) = self.lanes.get(self.active_lane) else {
            return "lane.csv".to_string();
        };
        if self.traces.len() > 1 {
            format!(
                "lane-cuda{}-trace{}.csv",
                lane.stream_id(),
                lane.trace_id()
            )
        } else {
            format!("lane-cuda{}.csv", lane.stream_id())
        }
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

/// A `(raw_ts, render_ts)` control point pair for the annotation `TimeMap`.
type ControlPoints = Vec<(f64, f64)>;

/// Flat kernel indices for one trace's one stream, sorted by raw timestamp.
fn stream_kernel_indices(kernels: &[KernelEvent], trace_id: usize, stream: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = kernels
        .iter()
        .enumerate()
        .filter(|(_, k)| k.trace_id == trace_id && k.stream == stream)
        .map(|(i, _)| i)
        .collect();
    idx.sort_by(|&a, &b| {
        kernels[a]
            .ts
            .partial_cmp(&kernels[b].ts)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// A monotonic piecewise-linear map from raw timestamps to render timestamps,
/// built from `(raw_kernel_start, assigned_render_start)` control points. Used to
/// carry annotations along the same remap their stream's kernels underwent in
/// diff mode. Below the first / above the last control it extrapolates by that
/// control's constant shift.
struct TimeMap {
    controls: Vec<(f64, f64)>,
    normal_off: f64,
}

impl TimeMap {
    fn new(mut controls: Vec<(f64, f64)>, normal_off: f64) -> Self {
        controls.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        TimeMap {
            controls,
            normal_off,
        }
    }

    fn map(&self, t: f64) -> f64 {
        let n = self.controls.len();
        if n == 0 {
            return t + self.normal_off;
        }
        let (first_raw, first_render) = self.controls[0];
        if t <= first_raw {
            return t + (first_render - first_raw);
        }
        let (last_raw, last_render) = self.controls[n - 1];
        if t >= last_raw {
            return t + (last_render - last_raw);
        }
        for w in self.controls.windows(2) {
            let (raw_i, render_i) = w[0];
            let (raw_j, render_j) = w[1];
            if raw_i <= t && t <= raw_j {
                if raw_j == raw_i {
                    return t + (render_i - raw_i);
                }
                let alpha = (t - raw_i) / (raw_j - raw_i);
                return render_i + alpha * (render_j - render_i);
            }
        }
        t + (last_render - last_raw)
    }
}

/// Diffs the two streams' kernel-name sequences, writing absolute render-start
/// overrides for their kernels and returning the trace-0 and trace-1 control
/// points `(raw_start, render_start)` so annotations can follow the same remap.
/// `idx0`/`idx1` are ts-sorted flat kernel indices for trace 0 / 1; `base0`/
/// `base1` are the per-trace normal offsets used as the un-shifted origin.
fn remap_stream(
    kernels: &[KernelEvent],
    idx0: &[usize],
    idx1: &[usize],
    off0: f64,
    off1: f64,
    overrides: &mut [Option<f64>],
    diff_status: &mut [Option<KernelDiff>],
) -> (ControlPoints, ControlPoints) {
    use similar::{capture_diff_slices, Algorithm, DiffOp};

    let names0: Vec<&str> = idx0.iter().map(|&i| kernels[i].name.as_str()).collect();
    let names1: Vec<&str> = idx1.iter().map(|&i| kernels[i].name.as_str()).collect();

    let base0 = |local: usize| kernels[idx0[local]].ts + off0;
    let base1 = |local: usize| kernels[idx1[local]].ts + off1;
    let dur0 = |local: usize| kernels[idx0[local]].dur;
    let dur1 = |local: usize| kernels[idx1[local]].dur;
    let raw0 = |local: usize| kernels[idx0[local]].ts;
    let raw1 = |local: usize| kernels[idx1[local]].ts;
    let len0 = idx0.len();

    let mut ctrl0: Vec<(f64, f64)> = Vec::new();
    let mut ctrl1: Vec<(f64, f64)> = Vec::new();
    let mut inserted_shift = 0.0f64;

    let assign0 =
        |local: usize, render: f64, ov: &mut [Option<f64>], c0: &mut Vec<(f64, f64)>| {
            ov[idx0[local]] = Some(render);
            c0.push((raw0(local), render));
        };

    // Places a run of trace-1-only kernels into a newly inserted anchor gap whose
    // width is the run's span, advancing `shift` and recording trace-1 controls.
    let apply_insert = |old_index: usize,
                        new_index: usize,
                        new_len: usize,
                        shift: &mut f64,
                        ov: &mut [Option<f64>],
                        ds: &mut [Option<KernelDiff>],
                        c1: &mut Vec<(f64, f64)>| {
        if new_len == 0 {
            return;
        }
        let run = new_index..new_index + new_len;
        let insert_run_start = base1(new_index);
        let insert_run_end = run
            .clone()
            .map(|j| base1(j) + dur1(j))
            .fold(f64::MIN, f64::max);
        let insert_width = (insert_run_end - insert_run_start).max(0.0);
        let gap_start = if old_index < len0 {
            base0(old_index) + *shift
        } else {
            base0(len0 - 1) + *shift + dur0(len0 - 1)
        };
        for j in run {
            let render = gap_start + (base1(j) - insert_run_start);
            ov[idx1[j]] = Some(render);
            ds[idx1[j]] = Some(KernelDiff::Added);
            c1.push((raw1(j), render));
        }
        *shift += insert_width;
    };

    for op in capture_diff_slices(Algorithm::Myers, &names0, &names1) {
        match op {
            DiffOp::Equal {
                old_index,
                new_index,
                len,
            } => {
                for k in 0..len {
                    let pos = base0(old_index + k) + inserted_shift;
                    assign0(old_index + k, pos, overrides, &mut ctrl0);
                    overrides[idx1[new_index + k]] = Some(pos);
                    ctrl1.push((raw1(new_index + k), pos));
                }
            }
            DiffOp::Delete {
                old_index, old_len, ..
            } => {
                for k in 0..old_len {
                    let pos = base0(old_index + k) + inserted_shift;
                    assign0(old_index + k, pos, overrides, &mut ctrl0);
                    diff_status[idx0[old_index + k]] = Some(KernelDiff::Deleted);
                }
            }
            DiffOp::Insert {
                old_index,
                new_index,
                new_len,
            } => {
                apply_insert(
                    old_index,
                    new_index,
                    new_len,
                    &mut inserted_shift,
                    overrides,
                    diff_status,
                    &mut ctrl1,
                );
            }
            DiffOp::Replace {
                old_index,
                old_len,
                new_index,
                new_len,
            } => {
                for k in 0..old_len {
                    let pos = base0(old_index + k) + inserted_shift;
                    assign0(old_index + k, pos, overrides, &mut ctrl0);
                    diff_status[idx0[old_index + k]] = Some(KernelDiff::Deleted);
                }
                apply_insert(
                    old_index + old_len,
                    new_index,
                    new_len,
                    &mut inserted_shift,
                    overrides,
                    diff_status,
                    &mut ctrl1,
                );
            }
        }
    }
    (ctrl0, ctrl1)
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

/// Derives the vLLM iteration stage (`prefill`/`decode`/`mixed`) from a
/// `gpu_user_annotation` name emitted by vLLM's `gpu_worker.py`.
///
/// Two formats are recognised, both carrying a *request* count before each
/// parenthesised *token* count:
///   simple:   `execute_context_<ctx>(<n>)_generation_<gen>(<n>)`
///   detailed: `execute_<total>_context_<ctx>(...)_generation_<gen>(...)`
///
/// Classification uses the request counts (`ctx`, `gen`):
/// `ctx>0,gen==0`→prefill, `ctx==0,gen>0`→decode, both>0→mixed. Names that do
/// not match (e.g. `nccl:_all_gather_base`) yield `""`.
pub(crate) fn stage_for_annotation_name(name: &str) -> &'static str {
    let parse_counts = |s: &str| -> Option<(u64, u64)> {
        let after_prefix = s.strip_prefix("execute_")?;
        // Detailed form has a numeric total before `context_`; skip it if present.
        let rest = match after_prefix.strip_prefix("context_") {
            Some(r) => r,
            None => {
                let (total, r) = after_prefix.split_once("_context_")?;
                if !total.chars().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                r
            }
        };
        // `rest` = `<ctx>(...)_generation_<gen>(...)`. Read leading ctx request count.
        let ctx: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if ctx.is_empty() {
            return None;
        }
        let gen_marker = "_generation_";
        let gpos = rest.find(gen_marker)?;
        let after_gen = &rest[gpos + gen_marker.len()..];
        let gen: String = after_gen
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if gen.is_empty() {
            return None;
        }
        Some((ctx.parse().ok()?, gen.parse().ok()?))
    };

    match parse_counts(name) {
        Some((ctx, gen)) if ctx > 0 && gen == 0 => "prefill",
        Some((ctx, gen)) if ctx == 0 && gen > 0 => "decode",
        Some((ctx, gen)) if ctx > 0 && gen > 0 => "mixed",
        _ => "",
    }
}

/// Finds the annotation whose time window `[ts, end_ts)` contains `kernel_ts`,
/// restricted to the same `stream` and `trace_id`. When several overlap, the
/// tightest (smallest `dur`) wins; ties break toward the later-starting (inner)
/// annotation, then the lowest flat index for determinism.
pub(crate) fn annotation_for_kernel(
    annotations: &[AnnotationEvent],
    stream: u64,
    trace_id: usize,
    kernel_ts: f64,
) -> Option<&AnnotationEvent> {
    let mut best: Option<&AnnotationEvent> = None;
    for a in annotations {
        if a.stream != stream || a.trace_id != trace_id {
            continue;
        }
        if !(a.ts <= kernel_ts && kernel_ts < a.end_ts()) {
            continue;
        }
        best = match best {
            None => Some(a),
            Some(b) => {
                let tighter = a.dur < b.dur || (a.dur == b.dur && a.ts > b.ts);
                if tighter {
                    Some(a)
                } else {
                    Some(b)
                }
            }
        };
    }
    best
}

/// Quotes a CSV field per RFC 4180 when it contains a comma, quote, or newline
/// (grid/block values like `[32,1,1]` contain commas). Inner quotes are doubled.
pub(crate) fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
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

    // ── Search prev/next navigation ──────────────────────────────────────────

    fn search_app() -> App {
        // Four kernels matching "gemm" across two streams, plus one non-match.
        app_from(vec![
            named_kernel(3, 100.0, "gemm_a"),
            named_kernel(3, 200.0, "relu"),
            named_kernel(3, 300.0, "gemm_b"),
            named_kernel(7, 150.0, "gemm_c"),
            named_kernel(7, 250.0, "gemm_d"),
        ])
    }

    fn selected_name(app: &App) -> String {
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Kernel(k) => k.name.clone(),
            SelectedTraceItem::Annotation(a) => a.name.clone(),
        }
    }

    #[test]
    fn test_search_collects_all_matches_and_jumps_to_first() {
        let mut app = search_app();
        app.search_start();
        for c in "gemm".chars() {
            app.search_push(c);
        }
        assert_eq!(app.search_matches.len(), 4, "four gemm_* matches");
        assert_eq!(app.search_match_idx, 0);
        assert_eq!(selected_name(&app), "gemm_a");
        assert_eq!(app.search_match_label().as_deref(), Some("1/4"));
    }

    #[test]
    fn test_search_next_advances_through_matches() {
        let mut app = search_app();
        app.search_start();
        for c in "gemm".chars() {
            app.search_push(c);
        }
        // Lane order: stream 3 lane (gemm_a, gemm_b), then stream 7 (gemm_c, gemm_d).
        app.search_next();
        assert_eq!(selected_name(&app), "gemm_b");
        assert_eq!(app.search_match_label().as_deref(), Some("2/4"));
        app.search_next();
        assert_eq!(selected_name(&app), "gemm_c");
        app.search_next();
        assert_eq!(selected_name(&app), "gemm_d");
        assert_eq!(app.search_match_label().as_deref(), Some("4/4"));
    }

    #[test]
    fn test_search_next_wraps_to_first() {
        let mut app = search_app();
        app.search_start();
        for c in "gemm".chars() {
            app.search_push(c);
        }
        for _ in 0..3 {
            app.search_next();
        }
        assert_eq!(selected_name(&app), "gemm_d"); // 4/4
        app.search_next(); // wraps
        assert_eq!(selected_name(&app), "gemm_a");
        assert_eq!(app.search_match_label().as_deref(), Some("1/4"));
    }

    #[test]
    fn test_search_prev_wraps_to_last() {
        let mut app = search_app();
        app.search_start();
        for c in "gemm".chars() {
            app.search_push(c);
        }
        // At first match; prev wraps to the last.
        app.search_prev();
        assert_eq!(selected_name(&app), "gemm_d");
        assert_eq!(app.search_match_label().as_deref(), Some("4/4"));
        app.search_prev();
        assert_eq!(selected_name(&app), "gemm_c");
    }

    #[test]
    fn test_search_next_prev_noop_without_matches() {
        let mut app = search_app();
        app.search_start();
        for c in "zzz".chars() {
            app.search_push(c);
        }
        assert!(app.search_no_match);
        assert!(app.search_matches.is_empty());
        assert!(app.search_match_label().is_none());
        // No panic, no movement.
        app.search_next();
        app.search_prev();
    }

    #[test]
    fn test_search_refine_resets_match_index() {
        let mut app = search_app();
        app.search_start();
        for c in "gemm".chars() {
            app.search_push(c);
        }
        app.search_next();
        app.search_next();
        assert_eq!(app.search_match_idx, 2);
        // Refining the query rebuilds matches and jumps back to the first.
        app.search_push('_');
        app.search_push('c');
        assert_eq!(app.search_matches.len(), 1);
        assert_eq!(app.search_match_idx, 0);
        assert_eq!(selected_name(&app), "gemm_c");
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

    // ── Lane CSV export (E key) ──────────────────────────────────────────────

    #[test]
    fn test_csv_escape_quotes_commas_and_quotes() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("[32,1,1]"), "\"[32,1,1]\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
        assert_eq!(csv_escape("line\nbreak"), "\"line\nbreak\"");
    }

    // S1 happy: active kernel lane -> header + one row per kernel, full fields,
    // grid/block comma-quoted.
    #[test]
    fn test_lane_kernels_csv_happy() {
        let mut k0 = kd(4, 100.0, "gemm", 30.0);
        k0.device = 7;
        k0.grid = Some("[32,1,1]".into());
        k0.block = Some("[128,1,1]".into());
        k0.shared_memory = Some(2048);
        k0.registers_per_thread = Some(64);
        k0.correlation = Some(999);
        let app = app_from(vec![k0, kd(4, 200.0, "relu", 10.0)]);

        let csv = app.lane_kernels_csv().expect("kernel lane produces CSV");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "idx,annotation,stage,name,ts,dur,end_ts,stream,device,grid,block,shared_memory,registers_per_thread,correlation"
        );
        // Row 1: gemm with all fields; grid/block quoted (contain commas). No
        // annotations in this fixture, so annotation+stage are empty.
        assert_eq!(
            lines[1],
            "1,,,gemm,100,30,130,4,7,\"[32,1,1]\",\"[128,1,1]\",2048,64,999"
        );
        // Row 2: relu with empty optional fields.
        assert_eq!(lines[2], "2,,,relu,200,10,210,4,0,,,,,");
        assert_eq!(lines.len(), 3, "header + 2 kernels");
    }

    // S2 edge: annotation lane -> None; empty is handled (header only).
    #[test]
    fn test_lane_kernels_csv_none_on_annotation_lane() {
        let kernels = vec![named_kernel(4, 100.0, "k")];
        let annotations = vec![AnnotationEvent {
            name: "ctx".to_string(),
            ts: 90.0,
            dur: 20.0,
            stream: 4,
            trace_id: 0,
        }];
        let mut app = App::new(Trace {
            kernels,
            annotations,
        });
        let ann_lane = app.lanes.iter().position(|l| l.is_annotations()).unwrap();
        app.active_lane = ann_lane;
        assert!(
            app.lane_kernels_csv().is_none(),
            "annotation lane is not exportable"
        );
    }

    // S1/S2/S3/S4: `annotation` + `stage` are columns 2 and 3. Each kernel is
    // tagged with the annotation whose [ts,end_ts) window (same stream+trace)
    // contains its ts, and the derived prefill/decode/mixed stage. Kernels with
    // no covering annotation get empty annotation+stage. Overlapping annotations
    // resolve to the tightest (shortest-span) one.
    #[test]
    fn test_lane_kernels_csv_annotation_stage_columns() {
        // Stream 4: three kernels.
        //  k@100 -> covered by decode annotation [90,140)
        //  k@200 -> covered by BOTH a wide [150,300) and a tight prefill [195,215);
        //           the tight one wins.
        //  k@400 -> no covering annotation -> empty annotation+stage.
        let kernels = vec![
            named_kernel(4, 100.0, "gemm"),
            named_kernel(4, 200.0, "relu"),
            named_kernel(4, 400.0, "add"),
        ];
        let annotations = vec![
            AnnotationEvent {
                name: "execute_context_0(0)_generation_1(1)".to_string(),
                ts: 90.0,
                dur: 50.0, // [90,140)
                stream: 4,
                trace_id: 0,
            },
            AnnotationEvent {
                name: "wide_noise".to_string(),
                ts: 150.0,
                dur: 150.0, // [150,300) — wider, must lose to the tight one
                stream: 4,
                trace_id: 0,
            },
            AnnotationEvent {
                name: "execute_context_1(5)_generation_0(0)".to_string(),
                ts: 195.0,
                dur: 20.0, // [195,215) — tight, wins for k@200
                stream: 4,
                trace_id: 0,
            },
        ];
        let app = App::new(Trace {
            kernels,
            annotations,
        });

        let csv = app.lane_kernels_csv().expect("kernel lane produces CSV");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "idx,annotation,stage,name,ts,dur,end_ts,stream,device,grid,block,shared_memory,registers_per_thread,correlation"
        );
        // k@100: decode annotation.
        assert_eq!(
            lines[1],
            "1,execute_context_0(0)_generation_1(1),decode,gemm,100,5,105,4,0,,,,,"
        );
        // k@200: tightest overlap = prefill annotation (wide_noise loses).
        assert_eq!(
            lines[2],
            "2,execute_context_1(5)_generation_0(0),prefill,relu,200,5,205,4,0,,,,,"
        );
        // k@400: no covering annotation -> empty annotation + stage.
        assert_eq!(lines[3], "3,,,add,400,5,405,4,0,,,,,");
        assert_eq!(lines.len(), 4, "header + 3 kernels");
    }

    #[test]
    fn test_stage_prefill() {
        assert_eq!(
            stage_for_annotation_name("execute_context_1(5)_generation_0(0)"),
            "prefill"
        );
    }

    #[test]
    fn test_stage_decode() {
        assert_eq!(
            stage_for_annotation_name("execute_context_0(0)_generation_1(1)"),
            "decode"
        );
    }

    #[test]
    fn test_stage_mixed() {
        assert_eq!(
            stage_for_annotation_name("execute_context_2(8)_generation_3(3)"),
            "mixed"
        );
    }

    #[test]
    fn test_stage_none_on_nccl() {
        assert_eq!(stage_for_annotation_name("nccl:_all_gather_base"), "");
    }

    #[test]
    fn test_stage_none_on_garbage() {
        assert_eq!(stage_for_annotation_name("execute_context_x(0)"), "");
        assert_eq!(stage_for_annotation_name(""), "");
        assert_eq!(stage_for_annotation_name("execute_context_1(5)"), "");
    }

    #[test]
    fn test_stage_detailed_form() {
        assert_eq!(
            stage_for_annotation_name(
                "execute_9_context_1(sq4sk10sqsq0sqsk0)_generation_0(sq0sk0sqsq0sqsk0)"
            ),
            "prefill"
        );
        assert_eq!(
            stage_for_annotation_name(
                "execute_5_context_0(sq0sk0sqsq0sqsk0)_generation_5(sq5sk20sqsq0sqsk0)"
            ),
            "decode"
        );
    }

    #[test]
    fn test_annotation_tightest_overlap_tie_break() {
        // Two equal-duration annotations both cover ts=100; the later-starting
        // (inner) one wins the tie.
        let anns = vec![
            AnnotationEvent {
                name: "outer".into(),
                ts: 90.0,
                dur: 20.0,
                stream: 4,
                trace_id: 0,
            },
            AnnotationEvent {
                name: "inner".into(),
                ts: 95.0,
                dur: 20.0,
                stream: 4,
                trace_id: 0,
            },
        ];
        let picked = annotation_for_kernel(&anns, 4, 0, 100.0).unwrap();
        assert_eq!(picked.name, "inner");
        // Wrong stream / trace_id are excluded.
        assert!(annotation_for_kernel(&anns, 7, 0, 100.0).is_none());
        assert!(annotation_for_kernel(&anns, 4, 1, 100.0).is_none());
    }

    #[test]
    fn test_lane_csv_filename_single_and_multi_trace() {
        let app = app_from(vec![kd(4, 0.0, "a", 1.0)]);
        assert_eq!(app.lane_csv_filename(), "lane-cuda4.csv");

        let t0 = trace_of(vec![kd(4, 0.0, "a", 1.0)], vec![]);
        let t1 = trace_of(vec![kd(4, 0.0, "a", 1.0)], vec![]);
        let app2 = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert!(
            app2.lane_csv_filename().starts_with("lane-cuda4-trace"),
            "multi-trace name carries the trace id: {}",
            app2.lane_csv_filename()
        );
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

    // ── Multi-trace: interleaving, diff/normal modes, TimeMap ────────────────

    fn ann_dur(stream: u64, ts: f64, dur: f64, name: &str) -> AnnotationEvent {
        AnnotationEvent {
            name: name.to_string(),
            ts,
            dur,
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

    // Interleave order = row0 of every trace, then row1, etc. with correct
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

    // Unequal lane counts -> round-robin to max, skip missing rows.
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

    // Accessors read the (always-populated) override vectors, not a scalar
    // offset. Two traces default to diff mode: t1.foo snaps onto t0.foo start.
    #[test]
    fn test_render_ts_accessors_read_overrides() {
        let t0 = trace_of(vec![kd(1, 100.0, "foo", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 500.0, "foo", 7.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        let k1 = app.kernels.iter().position(|k| k.trace_id == 1).unwrap();
        // Diff snap: start == anchor start (zero-based to global min 100).
        assert!((app.kernel_render_ts(k1) - 100.0).abs() < 1e-9);
        // t1 keeps its own duration 7 -> end 107.
        assert!((app.kernel_render_end(k1) - 107.0).abs() < 1e-9);
    }

    // Single trace zero-bases to identity (its own min), G is a no-op, no mode.
    #[test]
    fn test_single_trace_regression() {
        let app = app_with_annotations();
        assert_eq!(app.traces.len(), 1);
        // Zero-base of one trace is identity: earliest event stays put.
        let k0 = app.kernels.iter().position(|k| k.name == "kernel_a").unwrap();
        assert!((app.kernel_render_ts(k0) - 100.0).abs() < 1e-9);
        assert_eq!(app.lanes.len(), 2);
        assert!(app.lanes.iter().all(|l| l.trace_id() == 0));
    }

    // global_time_bounds flows through the overridden accessors: in diff mode two
    // traces with matched kernels collapse to a compact aligned window.
    #[test]
    fn test_global_bounds_use_override_positions() {
        let t0 = trace_of(vec![kd(1, 100.0, "foo", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 500.0, "foo", 5.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let (g_start, g_end) = app.global_time_bounds();
        // Both traces' events sit near ts=100..105 (zero-based + diff-snapped).
        assert!((g_start - 100.0).abs() < 1e-9, "start {}", g_start);
        assert!(g_end <= 106.0, "aligned end ~105 not 505, got {}", g_end);
    }

    // Tabbing across diff-aligned traces lands on the aligned-nearest item; with
    // matched kernels snapped, selecting t0.bar and tabbing to t1 lands on t1.bar.
    #[test]
    fn test_tab_across_traces_uses_aligned_position() {
        let t0 = trace_of(
            vec![kd(1, 100.0, "foo", 5.0), kd(1, 200.0, "bar", 5.0)],
            vec![],
        );
        let t1 = trace_of(
            vec![kd(1, 9100.0, "foo", 5.0), kd(1, 9200.0, "bar", 5.0)],
            vec![],
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
            // t1.bar is snapped onto t0.bar (aligned 200) -> it is the nearest.
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "bar"),
            _ => panic!("expected kernel"),
        }
    }

    // ── Two-trace kernel-diff alignment (default mode) ───────────────────────

    /// Flat kernel index of trace `tid`'s kernel named `name` at raw ts `ts`.
    fn kidx(app: &App, tid: usize, name: &str, ts: f64) -> usize {
        app.kernels
            .iter()
            .position(|k| k.trace_id == tid && k.name == name && (k.ts - ts).abs() < 1e-9)
            .unwrap_or_else(|| panic!("kernel {name}@{ts} (trace {tid}) not found"))
    }

    fn aidx(app: &App, tid: usize, name: &str) -> usize {
        app.annotations
            .iter()
            .position(|a| a.trace_id == tid && a.name == name)
            .unwrap_or_else(|| panic!("annotation {name} (trace {tid}) not found"))
    }

    // Diff S1: identical name sequence, differing durations. Trace-1 starts snap
    // onto trace-0 starts; each trace keeps its own duration (ends differ).
    #[test]
    fn test_diff_align_s1_equal_snaps_start_keeps_own_duration() {
        let t0 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 200.0, "b", 10.0)],
            vec![],
        );
        let t1 = trace_of(
            vec![kd(1, 5100.0, "a", 40.0), kd(1, 5200.0, "b", 40.0)],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        let a0 = kidx(&app, 0, "a", 100.0);
        let a1 = kidx(&app, 1, "a", 5100.0);
        let b0 = kidx(&app, 0, "b", 200.0);
        let b1 = kidx(&app, 1, "b", 5200.0);

        assert!((app.kernel_render_ts(a1) - app.kernel_render_ts(a0)).abs() < 1e-9);
        assert!((app.kernel_render_ts(b1) - app.kernel_render_ts(b0)).abs() < 1e-9);
        // Anchor keeps its zero-based position (global min 100) and own duration.
        assert!((app.kernel_render_ts(a0) - 100.0).abs() < 1e-9);
        assert!((app.kernel_render_end(a0) - 110.0).abs() < 1e-9);
        // Trace 1 keeps its own (longer) duration -> ends differ.
        assert!((app.kernel_render_end(a1) - 140.0).abs() < 1e-9);
        assert!(app.kernel_render_end(a1) != app.kernel_render_end(a0));
    }

    // Diff S2: middle Insert. Trace-1-only "x" fills a gap before the next anchor
    // kernel; later anchor kernels shift right by the inserted run's span.
    #[test]
    fn test_diff_align_s2_middle_insert_shifts_later_anchor() {
        let t0 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 200.0, "b", 10.0)],
            vec![],
        );
        let t1 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 130.0, "x", 25.0),
                kd(1, 200.0, "b", 10.0),
            ],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        let a0 = kidx(&app, 0, "a", 100.0);
        let b0 = kidx(&app, 0, "b", 200.0);
        let x1 = kidx(&app, 1, "x", 130.0);

        // "a" unchanged at anchor origin.
        assert!((app.kernel_render_ts(a0) - 100.0).abs() < 1e-9);
        // Insert span = single kernel dur = 25 -> "b" shifts 200 -> 225.
        assert!(
            (app.kernel_render_ts(b0) - 225.0).abs() < 1e-9,
            "b shifted by insert span, got {}",
            app.kernel_render_ts(b0)
        );
        // "x" fills the inserted gap [200,225); "b" is pushed past it to 225.
        assert!(
            (app.kernel_render_ts(x1) - 200.0).abs() < 1e-9,
            "x at gap start, got {}",
            app.kernel_render_ts(x1)
        );
        assert!((app.kernel_render_end(x1) - app.kernel_render_ts(b0)).abs() < 1e-9);
    }

    // Diff S5: Delete. Trace-0-only kernel keeps anchor position; trace 1 has no
    // kernel mapped there; surrounding matches stay aligned.
    #[test]
    fn test_diff_align_s5_delete_keeps_anchor_only() {
        let t0 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 200.0, "gone", 10.0),
                kd(1, 300.0, "b", 10.0),
            ],
            vec![],
        );
        let t1 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 300.0, "b", 10.0)],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        let gone0 = kidx(&app, 0, "gone", 200.0);
        let b0 = kidx(&app, 0, "b", 300.0);
        let b1 = kidx(&app, 1, "b", 300.0);

        assert!((app.kernel_render_ts(gone0) - 200.0).abs() < 1e-9);
        assert!((app.kernel_render_ts(b0) - 300.0).abs() < 1e-9);
        assert!((app.kernel_render_ts(b1) - 300.0).abs() < 1e-9);
    }

    // Diff Replace = Delete(old) + Insert(new): anchor-only old kernel stays put,
    // new trace-1 kernel fills the inserted gap, later anchors shift by its span.
    #[test]
    fn test_diff_align_replace_is_delete_plus_insert() {
        let t0 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 200.0, "old", 10.0),
                kd(1, 300.0, "b", 10.0),
            ],
            vec![],
        );
        let t1 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 200.0, "new", 40.0),
                kd(1, 300.0, "b", 10.0),
            ],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);

        let old0 = kidx(&app, 0, "old", 200.0);
        let new1 = kidx(&app, 1, "new", 200.0);
        let b0 = kidx(&app, 0, "b", 300.0);

        assert!((app.kernel_render_ts(old0) - 200.0).abs() < 1e-9);
        // "new" span 40 -> "b" 300 -> 340; "new" fills gap [300,340).
        assert!(
            (app.kernel_render_ts(b0) - 340.0).abs() < 1e-9,
            "b shifted by new span, got {}",
            app.kernel_render_ts(b0)
        );
        assert!(
            (app.kernel_render_ts(new1) - 300.0).abs() < 1e-9,
            "new at gap start, got {}",
            app.kernel_render_ts(new1)
        );
        assert!((app.kernel_render_end(new1) - app.kernel_render_ts(b0)).abs() < 1e-9);
    }

    // ── New diff/normal-mode contract (S1–S8) ────────────────────────────────

    // S1: diff mode remaps an annotation via the per-stream TimeMap: its start
    // snaps to the anchor and its end stays >= start (monotonic).
    #[test]
    fn test_mode_s1_diff_remaps_annotation_via_timemap() {
        // Shared stream 1: t0 [a@100, b@200]; t1 [a@100, x@130 (insert), b@200].
        // Annotation "ctx" on t0 spans [a..b]; on t1 spans over the inserted x.
        let t0 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 200.0, "b", 10.0)],
            vec![ann_dur(1, 100.0, 100.0, "ctx")],
        );
        let t1 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 130.0, "x", 25.0),
                kd(1, 200.0, "b", 10.0),
            ],
            vec![ann_dur(1, 100.0, 100.0, "ctx")],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert_eq!(app.align_mode, AlignMode::Diff);

        // t1 kernels: a@100, x fills [200,225), b@225 (shifted by span 25).
        let b0 = kidx(&app, 0, "b", 200.0);
        assert!((app.kernel_render_ts(b0) - 225.0).abs() < 1e-9);

        // t1 "ctx" raw [100,200] maps through controls a:(100->100), b:(200->225):
        // start=100, end interpolates to 225 (b's render). end >= start.
        let c1 = aidx(&app, 1, "ctx");
        assert!((app.annotation_render_ts(c1) - 100.0).abs() < 1e-9);
        assert!(
            (app.annotation_render_end(c1) - 225.0).abs() < 1e-9,
            "ctx end remapped through TimeMap, got {}",
            app.annotation_render_end(c1)
        );
        assert!(app.annotation_render_end(c1) >= app.annotation_render_ts(c1));
    }

    // S2: normal mode zero-bases every trace to the common start; both traces'
    // earliest events coincide. Applies for >2 traces and the 2-trace toggle.
    #[test]
    fn test_mode_s2_normal_zero_base() {
        // Three traces at wildly different absolute ts -> all earliest coincide.
        let t0 = trace_of(vec![kd(1, 1000.0, "a", 5.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 5000.0, "b", 5.0)], vec![]);
        let t2 = trace_of(vec![kd(1, 9000.0, "c", 5.0)], vec![]);
        let app =
            App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1), ("T2".into(), t2)]);
        let a = kidx(&app, 0, "a", 1000.0);
        let b = kidx(&app, 1, "b", 5000.0);
        let c = kidx(&app, 2, "c", 9000.0);
        // global_min = 1000 -> a stays 1000, b/c shift to 1000 too.
        assert!((app.kernel_render_ts(a) - 1000.0).abs() < 1e-9);
        assert!((app.kernel_render_ts(b) - 1000.0).abs() < 1e-9);
        assert!((app.kernel_render_ts(c) - 1000.0).abs() < 1e-9);

        // 2-trace normal toggle: same zero-base rule.
        let u0 = trace_of(vec![kd(1, 1000.0, "a", 5.0)], vec![]);
        let u1 = trace_of(vec![kd(1, 5000.0, "z", 5.0)], vec![]);
        let mut app2 = App::new_multi(vec![("T0".into(), u0), ("T1".into(), u1)]);
        assert!(app2.toggle_align_mode());
        assert_eq!(app2.align_mode, AlignMode::Normal);
        let za = kidx(&app2, 0, "a", 1000.0);
        let zz = kidx(&app2, 1, "z", 5000.0);
        assert!((app2.kernel_render_ts(za) - 1000.0).abs() < 1e-9);
        assert!((app2.kernel_render_ts(zz) - 1000.0).abs() < 1e-9);
    }

    // S3: Diff -> Normal -> Diff returns identical kernel + annotation overrides.
    #[test]
    fn test_mode_s3_toggle_idempotency() {
        let t0 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 200.0, "b", 10.0)],
            vec![ann_dur(1, 100.0, 100.0, "ctx")],
        );
        let t1 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 130.0, "x", 25.0),
                kd(1, 200.0, "b", 10.0),
            ],
            vec![ann_dur(1, 100.0, 120.0, "ctx")],
        );
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let k_before = app.kernel_render_overrides.clone();
        let a_before = app.annotation_render_overrides.clone();

        assert!(app.toggle_align_mode());
        assert!(app.toggle_align_mode());
        assert_eq!(app.align_mode, AlignMode::Diff);
        assert_eq!(app.kernel_render_overrides, k_before, "kernels restored");
        assert_eq!(
            app.annotation_render_overrides, a_before,
            "annotations restored"
        );
    }

    // S4: an annotation-only stream (no kernels) uses normal mapping in both modes.
    #[test]
    fn test_mode_s4_annotation_only_stream_normal_in_both_modes() {
        // Stream 1 has kernels in both traces (drives diff); stream 9 is
        // annotation-only in t1. global_min = 100.
        let t0 = trace_of(vec![kd(1, 100.0, "a", 10.0)], vec![]);
        let t1 = trace_of(
            vec![kd(1, 100.0, "a", 10.0)],
            vec![ann_dur(9, 400.0, 10.0, "solo")],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let s = aidx(&app, 1, "solo");
        // Normal offset for t1 = global_min(100) - trace1_min(100) = 0 -> raw.
        assert!((app.annotation_render_ts(s) - 400.0).abs() < 1e-9);
        assert!((app.annotation_render_end(s) - 410.0).abs() < 1e-9);
    }

    // S5: a stream present in only one trace uses normal mapping (no diff shift).
    #[test]
    fn test_mode_s5_one_sided_stream_normal() {
        // Stream 2 only exists in t0. global_min = 100 -> t0 offset 0.
        let t0 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(2, 300.0, "solo", 10.0)],
            vec![],
        );
        let t1 = trace_of(vec![kd(1, 100.0, "a", 10.0)], vec![]);
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let s = kidx(&app, 0, "solo", 300.0);
        assert!((app.kernel_render_ts(s) - 300.0).abs() < 1e-9, "one-sided raw");
    }

    // S6: single trace zero-bases to identity, G is a no-op, no mode label.
    #[test]
    fn test_mode_s6_single_trace() {
        let mut app = app_from(vec![kd(1, 100.0, "a", 10.0), kd(1, 200.0, "b", 10.0)]);
        let a = kidx(&app, 0, "a", 100.0);
        let b = kidx(&app, 0, "b", 200.0);
        assert!((app.kernel_render_ts(a) - 100.0).abs() < 1e-9);
        assert!((app.kernel_render_ts(b) - 200.0).abs() < 1e-9);
        assert!(!app.toggle_align_mode(), "G no-op for single trace");
        assert!(app.mode_label().is_none());
    }

    // S7: >2 traces -> G is a no-op and layout stays normal (no diff snapping).
    #[test]
    fn test_mode_s7_g_noop_for_more_than_two() {
        let t0 = trace_of(vec![kd(1, 100.0, "a", 10.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 500.0, "a", 10.0)], vec![]);
        let t2 = trace_of(vec![kd(1, 900.0, "a", 10.0)], vec![]);
        let mut app =
            App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1), ("T2".into(), t2)]);
        let before = app.kernel_render_overrides.clone();
        assert!(!app.toggle_align_mode(), "G no-op for 3 traces");
        assert_eq!(app.kernel_render_overrides, before, "layout unchanged");
        // Normal zero-base: all "a" coincide at global_min 100 (no diff snap).
        let a1 = kidx(&app, 1, "a", 500.0);
        assert!((app.kernel_render_ts(a1) - 100.0).abs() < 1e-9);
    }

    // S8: mode_label is Some("diff")/Some("normal") for 2 traces (flips on
    // toggle) and None for 1 and 3 traces.
    #[test]
    fn test_mode_s8_header_label() {
        let t0 = trace_of(vec![kd(1, 100.0, "a", 10.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 200.0, "a", 10.0)], vec![]);
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert_eq!(app.mode_label().as_deref(), Some("diff"));
        assert!(app.toggle_align_mode());
        assert_eq!(app.mode_label().as_deref(), Some("normal"));
        assert!(app.toggle_align_mode());
        assert_eq!(app.mode_label().as_deref(), Some("diff"));

        let single = app_from(vec![kd(1, 0.0, "a", 1.0)]);
        assert!(single.mode_label().is_none());

        let u0 = trace_of(vec![kd(1, 0.0, "a", 1.0)], vec![]);
        let u1 = trace_of(vec![kd(1, 0.0, "b", 1.0)], vec![]);
        let u2 = trace_of(vec![kd(1, 0.0, "c", 1.0)], vec![]);
        let three =
            App::new_multi(vec![("T0".into(), u0), ("T1".into(), u1), ("T2".into(), u2)]);
        assert!(three.mode_label().is_none());
    }

    // Every event has a populated override after construction (invariant).
    #[test]
    fn test_overrides_fully_populated() {
        let t0 = trace_of(
            vec![kd(1, 100.0, "a", 10.0)],
            vec![ann_dur(1, 100.0, 20.0, "ctx")],
        );
        let t1 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 130.0, "x", 10.0)],
            vec![ann_dur(1, 100.0, 40.0, "ctx")],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        assert!(app.kernel_render_overrides.iter().all(|o| o.is_some()));
        assert!(app.annotation_render_overrides.iter().all(|o| o.is_some()));
    }

    // ── Diff status: added / deleted / matched ───────────────────────────────

    // Insert -> trace-1-only kernel is Added; matched kernels carry no status.
    #[test]
    fn test_diff_status_insert_is_added() {
        let t0 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 200.0, "b", 10.0)],
            vec![],
        );
        let t1 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 130.0, "x", 25.0),
                kd(1, 200.0, "b", 10.0),
            ],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let x1 = kidx(&app, 1, "x", 130.0);
        let a1 = kidx(&app, 1, "a", 100.0);
        let a0 = kidx(&app, 0, "a", 100.0);
        assert_eq!(app.kernel_diff(x1), Some(KernelDiff::Added));
        assert_eq!(app.kernel_diff(a1), None, "matched kernel has no status");
        assert_eq!(app.kernel_diff(a0), None);
    }

    // Delete -> trace-0-only kernel is Deleted.
    #[test]
    fn test_diff_status_delete_is_deleted() {
        let t0 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 200.0, "gone", 10.0),
                kd(1, 300.0, "b", 10.0),
            ],
            vec![],
        );
        let t1 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 300.0, "b", 10.0)],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let gone0 = kidx(&app, 0, "gone", 200.0);
        assert_eq!(app.kernel_diff(gone0), Some(KernelDiff::Deleted));
    }

    // Replace -> old kernel Deleted, new kernel Added.
    #[test]
    fn test_diff_status_replace_marks_both() {
        let t0 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 200.0, "old", 10.0),
                kd(1, 300.0, "b", 10.0),
            ],
            vec![],
        );
        let t1 = trace_of(
            vec![
                kd(1, 100.0, "a", 10.0),
                kd(1, 200.0, "new", 40.0),
                kd(1, 300.0, "b", 10.0),
            ],
            vec![],
        );
        let app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let old0 = kidx(&app, 0, "old", 200.0);
        let new1 = kidx(&app, 1, "new", 200.0);
        assert_eq!(app.kernel_diff(old0), Some(KernelDiff::Deleted));
        assert_eq!(app.kernel_diff(new1), Some(KernelDiff::Added));
    }

    // Normal mode clears all diff status; toggling back restores it.
    #[test]
    fn test_diff_status_cleared_in_normal_mode() {
        let t0 = trace_of(vec![kd(1, 100.0, "a", 10.0)], vec![]);
        let t1 = trace_of(
            vec![kd(1, 100.0, "a", 10.0), kd(1, 130.0, "x", 10.0)],
            vec![],
        );
        let mut app = App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)]);
        let x1 = kidx(&app, 1, "x", 130.0);
        assert_eq!(app.kernel_diff(x1), Some(KernelDiff::Added));

        assert!(app.toggle_align_mode());
        assert!(app.kernel_diff_status.iter().all(|s| s.is_none()), "normal clears status");

        assert!(app.toggle_align_mode());
        assert_eq!(app.kernel_diff(x1), Some(KernelDiff::Added), "diff restores status");
    }

    // No diff status for >2 traces (no diff runs).
    #[test]
    fn test_diff_status_none_for_more_than_two_traces() {
        let t0 = trace_of(vec![kd(1, 100.0, "a", 10.0)], vec![]);
        let t1 = trace_of(vec![kd(1, 100.0, "a", 10.0), kd(1, 130.0, "x", 10.0)], vec![]);
        let t2 = trace_of(vec![kd(1, 100.0, "a", 10.0)], vec![]);
        let app =
            App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1), ("T2".into(), t2)]);
        assert!(app.kernel_diff_status.iter().all(|s| s.is_none()));
    }

    // ── Per-column priority render (lane_column_classes) ─────────────────────

    // Diff app where a wide MATCHED kernel spans a narrow ADDED / DELETED one so
    // the priority resolution is exercised. Stream 1:
    //   t0: reduce@100 dur 400 (matched), gone@200 dur 20 (deleted-only in t0)
    //   t1: reduce@100 dur 400 (matched), extra@200 dur 20 (added-only in t1)
    // Under diff, reduce matches; gone is Deleted (t0), extra is Added (t1).
    fn overlap_diff_app() -> App {
        let t0 = trace_of(
            vec![kd(1, 100.0, "reduce", 400.0), kd(1, 200.0, "gone", 20.0)],
            vec![],
        );
        let t1 = trace_of(
            vec![kd(1, 100.0, "reduce", 400.0), kd(1, 200.0, "extra", 20.0)],
            vec![],
        );
        App::new_multi(vec![("T0".into(), t0), ("T1".into(), t1)])
    }

    fn kernel_lane(app: &App, tid: usize) -> usize {
        app.lanes
            .iter()
            .position(|l| !l.is_annotations() && l.trace_id() == tid)
            .unwrap()
    }

    // Parks the active lane out of range so nothing is the selected item; tests
    // that exercise class/geometry (not selection) start from a clean slate.
    fn no_selection(app: &mut App) {
        app.active_lane = usize::MAX;
    }

    // S1: non-overlapping kernels each own their own columns; owner_pos correct;
    // gaps are None.
    #[test]
    fn test_columns_s1_non_overlapping_own_columns() {
        let mut app = app_from(vec![kd(1, 0.0, "a", 10.0), kd(1, 50.0, "b", 10.0)]);
        no_selection(&mut app);
        let lane = kernel_lane(&app, 0);
        // Window [0,100) over width 100 -> a: cols 0..10, b: cols 50..60.
        let cols = app.lane_column_classes(lane, 0.0, 100.0, 100, false);
        assert_eq!(cols.len(), 100);
        assert_eq!(cols[5], Some((ColumnClass::Matched, 0)), "a owns col 5");
        assert_eq!(cols[55], Some((ColumnClass::Matched, 1)), "b owns col 55");
        assert_eq!(cols[30], None, "gap is empty");
    }

    // S2 (the bug): a wide MATCHED kernel overlapping a narrow ADDED kernel's
    // columns -> those columns are Added, not Matched. This is the column the old
    // cursor-clamp dropped.
    #[test]
    fn test_columns_s2_added_wins_over_wide_matched() {
        let mut app = overlap_diff_app();
        no_selection(&mut app);
        let lane1 = kernel_lane(&app, 1);
        let extra_pos = app.lanes[lane1]
            .item_indices()
            .iter()
            .position(|&i| app.kernels[i].name == "extra")
            .unwrap();
        // reduce spans render cols 0..400 (rs 100..500); the diff shifts extra to
        // render 220..240, i.e. cols 120..140 in this [100,500) width-400 window.
        let cols = app.lane_column_classes(lane1, 100.0, 400.0, 400, true);
        assert_eq!(
            cols[130],
            Some((ColumnClass::Added, extra_pos)),
            "added kernel wins its column over the wide matched reduce"
        );
        // A column the reduce covers but extra does not -> Matched.
        assert_eq!(
            cols[10].map(|(c, _)| c),
            Some(ColumnClass::Matched),
            "reduce-only column stays matched"
        );
    }

    // S3: wide Matched over a narrow Deleted -> those columns Deleted.
    #[test]
    fn test_columns_s3_deleted_wins_over_wide_matched() {
        let mut app = overlap_diff_app();
        no_selection(&mut app);
        let lane0 = kernel_lane(&app, 0);
        let gone_pos = app.lanes[lane0]
            .item_indices()
            .iter()
            .position(|&i| app.kernels[i].name == "gone")
            .unwrap();
        let cols = app.lane_column_classes(lane0, 100.0, 400.0, 400, true);
        assert_eq!(cols[110], Some((ColumnClass::Deleted, gone_pos)));
    }

    // S4: selected kernel columns win over an overlapping matched kernel.
    #[test]
    fn test_columns_s4_selected_wins() {
        let mut app = overlap_diff_app();
        let lane1 = kernel_lane(&app, 1);
        // Select the "extra" (added) kernel; its columns must become Selected.
        app.active_lane = lane1;
        let extra_pos = app.lanes[lane1]
            .item_indices()
            .iter()
            .position(|&i| app.kernels[i].name == "extra")
            .unwrap();
        app.selected_item = extra_pos;
        let cols = app.lane_column_classes(lane1, 100.0, 400.0, 400, true);
        assert_eq!(cols[130], Some((ColumnClass::Selected, extra_pos)));
        // The reduce-only column stays matched (not the active selection).
        assert_eq!(cols[10].map(|(c, _)| c), Some(ColumnClass::Matched));
    }

    // S5: a column overlapped by both an Added and a Deleted kernel -> Deleted
    // (rank 3 > 2). Build a synthetic app by overriding diff status directly.
    #[test]
    fn test_columns_s5_deleted_beats_added() {
        // Two kernels at the SAME columns in one lane, one Added one Deleted.
        let mut app = app_from(vec![kd(1, 100.0, "x", 50.0), kd(1, 100.0, "y", 50.0)]);
        no_selection(&mut app);
        let lane = kernel_lane(&app, 0);
        let items = app.lanes[lane].item_indices().to_vec();
        // Force diff statuses: items[0]=Added, items[1]=Deleted.
        app.kernel_diff_status[items[0]] = Some(KernelDiff::Added);
        app.kernel_diff_status[items[1]] = Some(KernelDiff::Deleted);
        let cols = app.lane_column_classes(lane, 100.0, 50.0, 50, true);
        assert_eq!(
            cols[10].map(|(c, _)| c),
            Some(ColumnClass::Deleted),
            "deleted outranks added on a contested column"
        );
    }

    // S6: gaps between kernels are None.
    #[test]
    fn test_columns_s6_gaps_are_empty() {
        let mut app = app_from(vec![kd(1, 0.0, "a", 5.0), kd(1, 80.0, "b", 5.0)]);
        no_selection(&mut app);
        let lane = kernel_lane(&app, 0);
        let cols = app.lane_column_classes(lane, 0.0, 100.0, 100, false);
        assert!(cols[40].is_none(), "mid gap empty");
        assert!(cols[0].is_some(), "a present at start");
    }

    // S7: non-diff mode collapses every non-selected kernel to Matched even when
    // kernel_diff would report Added/Deleted.
    #[test]
    fn test_columns_s7_non_diff_all_matched() {
        let mut app = overlap_diff_app();
        let lane1 = kernel_lane(&app, 1);
        // Sanity: in diff, extra is Added.
        let cols_diff = app.lane_column_classes(lane1, 100.0, 400.0, 400, true);
        assert_eq!(cols_diff[130].map(|(c, _)| c), Some(ColumnClass::Added));
        // Non-diff: same column is Matched (base colour in the UI), not Added.
        app.active_lane = usize::MAX; // ensure nothing is the active/selected lane
        let cols_norm = app.lane_column_classes(lane1, 100.0, 400.0, 400, false);
        assert_eq!(cols_norm[130].map(|(c, _)| c), Some(ColumnClass::Matched));
        assert!(cols_norm
            .iter()
            .flatten()
            .all(|(c, _)| *c == ColumnClass::Matched));
    }

    // ── Mouse hit-testing ────────────────────────────────────────────────────

    // Single trace, one stream, three kernels laid out across a known window.
    // A click inside a kernel's columns selects that kernel.
    fn mouse_app() -> App {
        // Kernels a@0..10, b@20..30, c@40..50 on stream 1.
        let mut app = app_from(vec![
            kd(1, 0.0, "a", 10.0),
            kd(1, 20.0, "b", 10.0),
            kd(1, 40.0, "c", 10.0),
        ]);
        // Simulate a render: gutter 10 wide, timeline 50 wide, window [0,50).
        app.lane_layout = LaneLayout {
            inner_x: 0,
            inner_y: 1,
            inner_h: 4,
            label_width: 10,
            lane_width: 50,
            view_offset: 0,
            ts_start: 0.0,
            time_span: 50.0,
        };
        app
    }

    #[test]
    fn test_click_selects_kernel_under_cursor() {
        let mut app = mouse_app();
        // Timeline x0 = inner_x + label_width = 10. Kernel "b" spans ts 20..30,
        // which in a 50-wide window maps to cols 20..30 -> screen 30..40.
        assert!(app.click_select(32, 1), "click on kernel lane row");
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "b"),
            _ => panic!("expected kernel b"),
        }
    }

    #[test]
    fn test_click_first_and_last_kernel() {
        let mut app = mouse_app();
        // "a" spans ts 0..10 -> screen cols 10..20.
        assert!(app.click_select(11, 1));
        assert_eq!(app.selected_item, 0);
        // "c" spans ts 40..50 -> screen cols 50..60.
        assert!(app.click_select(52, 1));
        assert_eq!(app.selected_item, 2);
    }

    #[test]
    fn test_click_empty_space_selects_nearest() {
        let mut app = mouse_app();
        // Gap between "a" (screen 10..20) and "b" (screen 30..40): click at 26
        // is nearest to "b" (center 35, dist 9) vs "a" (center 15, dist 11).
        assert!(app.click_select(26, 1));
        assert_eq!(app.selected_item, 1, "nearest item is b");
    }

    #[test]
    fn test_click_in_label_gutter_activates_lane_only() {
        let mut app = mouse_app();
        app.selected_item = 2;
        // Column 3 is inside the 10-wide gutter -> activates lane, keeps item.
        assert!(app.click_select(3, 1));
        assert_eq!(app.active_lane, 0);
        assert_eq!(app.selected_item, 2, "gutter click preserves selection");
    }

    #[test]
    fn test_click_outside_lane_area_is_noop() {
        let mut app = mouse_app();
        app.selected_item = 1;
        // Row 0 is above inner_y (1); row 5 is below inner_y+inner_h (5).
        assert!(!app.click_select(30, 0));
        assert!(!app.click_select(30, 5));
        assert_eq!(app.selected_item, 1, "out-of-area clicks change nothing");
    }

    #[test]
    fn test_click_selects_correct_lane_across_rows() {
        // Two streams -> two kernel lanes stacked at rows 1 and 2.
        let mut app = app_from(vec![kd(1, 0.0, "s1k", 10.0), kd(2, 0.0, "s2k", 10.0)]);
        assert_eq!(app.lanes.len(), 2);
        app.lane_layout = LaneLayout {
            inner_x: 0,
            inner_y: 1,
            inner_h: 4,
            label_width: 10,
            lane_width: 50,
            view_offset: 0,
            ts_start: 0.0,
            time_span: 50.0,
        };
        // Click row 2 (second lane), inside its kernel columns.
        assert!(app.click_select(12, 2));
        assert_eq!(app.active_lane, 1);
        match app.selected_trace_item().unwrap() {
            SelectedTraceItem::Kernel(k) => assert_eq!(k.name, "s2k"),
            _ => panic!("expected kernel s2k"),
        }
    }

    #[test]
    fn test_click_honors_view_offset() {
        let mut app = app_from(vec![
            kd(1, 0.0, "s1", 10.0),
            kd(2, 0.0, "s2", 10.0),
            kd(3, 0.0, "s3", 10.0),
        ]);
        assert_eq!(app.lanes.len(), 3);
        // Scrolled so lane 1 is the first visible row.
        app.lane_layout = LaneLayout {
            inner_x: 0,
            inner_y: 1,
            inner_h: 2,
            label_width: 10,
            lane_width: 50,
            view_offset: 1,
            ts_start: 0.0,
            time_span: 50.0,
        };
        // Row 1 now maps to lane index 1 (view_offset 1 + 0).
        assert!(app.click_select(12, 1));
        assert_eq!(app.active_lane, 1);
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
}
