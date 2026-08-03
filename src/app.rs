use crate::trace::{AnnotationEvent, KernelEvent, Trace};

const ZOOM_MIN: f64 = 0.1;
const ZOOM_FACTOR: f64 = 1.5;

/// A single navigable row on the timeline. Every lane — whether it holds GPU
/// kernels or user annotations — behaves identically for navigation.
#[derive(Debug, Clone)]
pub enum Lane {
    Kernels {
        stream_id: u64,
        item_indices: Vec<usize>,
    },
    Annotations {
        stream_id: u64,
        item_indices: Vec<usize>,
    },
}

impl Lane {
    pub fn stream_id(&self) -> u64 {
        match self {
            Lane::Kernels { stream_id, .. } | Lane::Annotations { stream_id, .. } => *stream_id,
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

impl SelectedTraceItem<'_> {
    pub fn ts(&self) -> f64 {
        match self {
            SelectedTraceItem::Kernel(k) => k.ts,
            SelectedTraceItem::Annotation(a) => a.ts,
        }
    }

    pub fn dur(&self) -> f64 {
        match self {
            SelectedTraceItem::Kernel(k) => k.dur,
            SelectedTraceItem::Annotation(a) => a.dur,
        }
    }
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
}

#[derive(Debug, Clone)]
pub struct App {
    pub kernels: Vec<KernelEvent>,
    pub annotations: Vec<AnnotationEvent>,
    pub streams: Vec<u64>,
    pub lanes: Vec<Lane>,
    pub active_lane: usize,
    pub selected_item: usize,
    pub zoom_level: f64,
    pub lane_view_offset: usize,
    pub search_active: bool,
    pub search_query: String,
    pub search_no_match: bool,
    pub sequence: Option<Sequence>,
    pub sequence_status: Option<String>,
}

impl App {
    pub fn new(trace: Trace) -> Self {
        let kernels = trace.kernels;
        let annotations = trace.annotations;

        // Streams come from BOTH kernels and annotations so annotation-only
        // streams are never dropped.
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

        let lanes = build_lanes(&kernels, &annotations, &streams);

        let mut app = App {
            kernels,
            annotations,
            streams,
            lanes,
            active_lane: 0,
            selected_item: 0,
            zoom_level: 1.0,
            lane_view_offset: 0,
            search_active: false,
            search_query: String::new(),
            search_no_match: false,
            sequence: None,
            sequence_status: None,
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

    /// ts of the item at position `pos` within the given lane, if any.
    fn item_ts(&self, lane: &Lane, pos: usize) -> Option<f64> {
        let idx = *lane.item_indices().get(pos)?;
        match lane {
            Lane::Kernels { .. } => self.kernels.get(idx).map(|k| k.ts),
            Lane::Annotations { .. } => self.annotations.get(idx).map(|a| a.ts),
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
        let prev_ts = self.selected_trace_item().map(|i| i.ts());
        self.active_lane = target;
        self.selected_item = match prev_ts {
            Some(ts) => self.nearest_item_in_active_lane(ts),
            None => 0,
        };
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
            if let Some(ts) = self.item_ts(lane, pos) {
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
        });
        self.sequence_status = None;
        true
    }

    pub fn close_sequence(&mut self) {
        self.sequence = None;
        self.sequence_status = None;
    }

    pub fn sequence_csv(&self) -> Option<String> {
        let seq = self.sequence.as_ref()?;
        let mut out = String::from("idx,kernel name,duration\n");
        for (idx, name, dur) in &seq.rows {
            out.push_str(&format!("{},{},{}\n", idx, name, format_dur(*dur)));
        }
        Some(out)
    }

    /// Scan forward for contiguous blocks whose ordered kernel names exactly
    /// match the captured sequence, then compute the per-position median
    /// duration across every matching block (including the original).
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
            if !matches {
                break;
            }
            for (off, slot) in per_pos.iter_mut().enumerate() {
                if let Some(k) = item_indices.get(start + off).and_then(|&i| self.kernels.get(i)) {
                    slot.push(k.dur);
                }
            }
            reps += 1;
            start += block_len;
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
        for k in &self.kernels {
            min_start = min_start.min(k.ts);
            max_end = max_end.max(k.end_ts());
        }
        for a in &self.annotations {
            min_start = min_start.min(a.ts);
            max_end = max_end.max(a.end_ts());
        }
        if min_start > max_end {
            (0.0, 1.0)
        } else {
            (min_start, (max_end).max(min_start + 1.0))
        }
    }

    pub fn global_visible_window(&self) -> (f64, f64) {
        let (g_start, g_end) = self.global_time_bounds();
        let total_span = (g_end - g_start).max(1.0);
        let visible_span = (total_span / self.zoom_level).max(1e-3);

        let center = self
            .selected_trace_item()
            .map(|i| i.ts() + i.dur() / 2.0)
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
/// annotation lane first (if it has any annotations) then its kernel lane (if
/// it has any kernels). Each lane's item indices are sorted by timestamp so
/// left/right navigation maps to the timeline.
fn build_lanes(
    kernels: &[KernelEvent],
    annotations: &[AnnotationEvent],
    streams: &[u64],
) -> Vec<Lane> {
    let mut lanes = Vec::new();
    for &stream_id in streams {
        let mut ann: Vec<usize> = annotations
            .iter()
            .enumerate()
            .filter(|(_, a)| a.stream == stream_id)
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
                item_indices: ann,
            });
        }

        let mut kern: Vec<usize> = kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| k.stream == stream_id)
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
                item_indices: kern,
            });
        }
    }
    if lanes.is_empty() {
        lanes.push(Lane::Kernels {
            stream_id: streams.first().copied().unwrap_or(0),
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

fn format_dur(dur: f64) -> String {
    format!("{}", dur)
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
            AnnotationEvent { name: "ctx_0".to_string(), ts: 90.0, dur: 60.0, stream: 4 },
            AnnotationEvent { name: "ctx_1".to_string(), ts: 180.0, dur: 40.0, stream: 4 },
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
        // Active lane starts on the annotation lane (rendered first).
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
            "idx,kernel name,duration\n1,foo,10\n2,bar,20\n3,baz,5\n"
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
        assert!(app.active_lane_is_annotations());
        assert!(!app.start_sequence());
        assert!(app.sequence.is_none());
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

    // Median scan stops at the first block that doesn't match the pattern exactly.
    #[test]
    fn test_sequence_median_stops_at_mismatch() {
        let kernels = vec![
            kd(1, 0.0, "foo", 10.0),
            kd(1, 10.0, "bar", 20.0),
            kd(1, 20.0, "foo", 12.0),
            kd(1, 30.0, "bar", 24.0),
            kd(1, 40.0, "foo", 11.0),
            kd(1, 50.0, "qux", 99.0), // breaks the [foo,bar] pattern
            kd(1, 60.0, "foo", 8.0),
            kd(1, 70.0, "bar", 8.0),
        ];
        let mut app = app_from(kernels);
        assert!(app.start_sequence());
        app.extend_sequence_median();
        let seq = app.sequence.as_ref().unwrap();
        assert_eq!(seq.reps_found, 2, "only first two [foo,bar] blocks are contiguous matches");
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
}
