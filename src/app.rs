use crate::trace::{AnnotationEvent, KernelEvent, Trace};

const ZOOM_MIN: f64 = 0.1;
const ZOOM_FACTOR: f64 = 1.5;

#[derive(Debug, Clone)]
pub struct App {
    pub kernels: Vec<KernelEvent>,
    pub annotations: Vec<AnnotationEvent>,
    pub streams: Vec<u64>,
    pub active_stream_idx: usize,
    pub filtered_indices: Vec<usize>,
    pub selected_kernel: usize,
    pub zoom_level: f64,
    pub view_offset: usize,
    pub stream_view_offset: usize,
    pub search_active: bool,
    pub search_query: String,
    pub search_no_match: bool,
}

impl App {
    pub fn new(trace: Trace) -> Self {
        let kernels = trace.kernels;
        let annotations = trace.annotations;
        // Collect unique streams
        let mut streams: Vec<u64> = {
            let mut s: Vec<u64> = kernels.iter().map(|k| k.stream).collect();
            s.sort_unstable();
            s.dedup();
            s
        };
        if streams.is_empty() {
            streams.push(0);
        }

        let mut app = App {
            kernels,
            annotations,
            streams,
            active_stream_idx: 0,
            filtered_indices: vec![],
            selected_kernel: 0,
            zoom_level: 1.0,
            view_offset: 0,
            stream_view_offset: 0,
            search_active: false,
            search_query: String::new(),
            search_no_match: false,
        };
        app.rebuild_filter();
        app
    }

    pub fn rebuild_filter(&mut self) {
        let stream_id = self.active_stream();
        self.filtered_indices = self
            .kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| k.stream == stream_id)
            .map(|(i, _)| i)
            .collect();
        if self.filtered_indices.is_empty() {
            self.selected_kernel = 0;
        } else {
            self.selected_kernel = self.selected_kernel.min(self.filtered_indices.len() - 1);
        }
        self.view_offset = 0;
    }

    pub fn active_stream(&self) -> u64 {
        self.streams[self.active_stream_idx]
    }

    pub fn kernel_indices_for_stream(&self, stream_id: u64) -> Vec<usize> {
        self.kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| k.stream == stream_id)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn selected_event(&self) -> Option<&KernelEvent> {
        let idx = *self.filtered_indices.get(self.selected_kernel)?;
        self.kernels.get(idx)
    }

    pub fn annotation_indices_for_stream(&self, stream_id: u64) -> Vec<usize> {
        self.annotations
            .iter()
            .enumerate()
            .filter(|(_, a)| a.stream == stream_id)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn prev_kernel(&mut self) {
        if self.filtered_indices.is_empty() {
            return;
        }
        if self.selected_kernel > 0 {
            self.selected_kernel -= 1;
        }
        self.clamp_view_offset();
    }

    pub fn next_kernel(&mut self) {
        if self.filtered_indices.is_empty() {
            return;
        }
        if self.selected_kernel + 1 < self.filtered_indices.len() {
            self.selected_kernel += 1;
        }
        self.clamp_view_offset();
    }

    pub fn zoom_in(&mut self) {
        self.zoom_level *= ZOOM_FACTOR;
    }

    pub fn zoom_out(&mut self) {
        self.zoom_level = (self.zoom_level / ZOOM_FACTOR).max(ZOOM_MIN);
    }

    pub fn next_stream(&mut self) {
        if self.streams.len() <= 1 {
            return;
        }
        self.active_stream_idx = (self.active_stream_idx + 1) % self.streams.len();
        self.rebuild_filter();
    }

    pub fn prev_stream(&mut self) {
        if self.streams.len() <= 1 {
            return;
        }
        self.active_stream_idx =
            (self.active_stream_idx + self.streams.len() - 1) % self.streams.len();
        self.rebuild_filter();
    }

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

        for (s_idx, &stream_id) in self.streams.iter().enumerate() {
            let indices = self.kernel_indices_for_stream(stream_id);
            for (list_idx, &kernel_idx) in indices.iter().enumerate() {
                if self.kernels[kernel_idx]
                    .name
                    .to_lowercase()
                    .contains(&needle)
                {
                    self.active_stream_idx = s_idx;
                    self.rebuild_filter();
                    self.selected_kernel = list_idx.min(self.filtered_indices.len().saturating_sub(1));
                    self.search_no_match = false;
                    return;
                }
            }
        }
        self.search_no_match = true;
    }

    pub fn rows_for_stream(&self, stream_idx: usize) -> usize {
        let stream_id = self.streams[stream_idx];
        if self.annotations.iter().any(|a| a.stream == stream_id) {
            2
        } else {
            1
        }
    }

    pub fn ensure_active_stream_visible(&mut self, visible_rows: usize) {
        if visible_rows == 0 {
            return;
        }
        if self.active_stream_idx < self.stream_view_offset {
            self.stream_view_offset = self.active_stream_idx;
            return;
        }
        loop {
            let mut used = 0usize;
            let mut last_fit = self.stream_view_offset;
            for idx in self.stream_view_offset..self.streams.len() {
                let need = self.rows_for_stream(idx);
                if used + need > visible_rows {
                    break;
                }
                used += need;
                last_fit = idx;
            }
            if self.active_stream_idx <= last_fit {
                break;
            }
            self.stream_view_offset += 1;
        }
    }

    fn clamp_view_offset(&mut self) {
        if self.selected_kernel < self.view_offset {
            self.view_offset = self.selected_kernel;
        }
    }

    pub fn global_time_bounds(&self) -> (f64, f64) {
        let mut min_start = f64::MAX;
        let mut max_end = f64::MIN;
        for k in &self.kernels {
            min_start = min_start.min(k.ts);
            max_end = max_end.max(k.end_ts());
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
            .selected_event()
            .map(|k| k.ts + k.dur / 2.0)
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
    fn test_filter_by_stream() {
        let mut app = sample_app();
        // First stream is 3
        assert_eq!(app.active_stream(), 3);
        assert_eq!(app.filtered_indices.len(), 2);

        app.next_stream();
        assert_eq!(app.active_stream(), 7);
        assert_eq!(app.filtered_indices.len(), 3);
    }

    #[test]
    fn test_navigation_ad() {
        let mut app = sample_app();
        app.next_stream(); // go to stream 7 (3 kernels)
        assert_eq!(app.selected_kernel, 0);

        app.next_kernel();
        assert_eq!(app.selected_kernel, 1);
        app.next_kernel();
        assert_eq!(app.selected_kernel, 2);
        // Can't go past last
        app.next_kernel();
        assert_eq!(app.selected_kernel, 2);

        app.prev_kernel();
        assert_eq!(app.selected_kernel, 1);
        app.prev_kernel();
        assert_eq!(app.selected_kernel, 0);
        // Can't go before first
        app.prev_kernel();
        assert_eq!(app.selected_kernel, 0);
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
        // Clamp at min
        for _ in 0..100 {
            app.zoom_out();
        }
        assert!(app.zoom_level >= 0.1);
    }

    #[test]
    fn test_tab_wraps() {
        let mut app = sample_app();
        app.next_stream();
        app.next_stream(); // wraps back to first
        assert_eq!(app.active_stream_idx, 0);
    }

    #[test]
    fn test_prev_stream_wraps() {
        let mut app = sample_app();
        assert_eq!(app.active_stream_idx, 0);
        app.prev_stream(); // wraps to last
        assert_eq!(app.active_stream_idx, app.streams.len() - 1);
        app.prev_stream();
        assert_eq!(app.active_stream_idx, app.streams.len() - 2);
        app.next_stream();
        assert_eq!(app.active_stream_idx, app.streams.len() - 1);
    }

    #[test]
    fn test_kernel_indices_for_stream() {
        let app = sample_app();
        let s7 = app.kernel_indices_for_stream(7);
        let s3 = app.kernel_indices_for_stream(3);
        assert_eq!(s7.len(), 3);
        assert_eq!(s3.len(), 2);
    }

    #[test]
    fn test_stream_scroll_keeps_active_visible() {
        let kernels = vec![
            make_kernel(1, 0.0, 1.0),
            make_kernel(2, 0.0, 1.0),
            make_kernel(3, 0.0, 1.0),
            make_kernel(4, 0.0, 1.0),
        ];
        let mut app = app_from(kernels);
        assert_eq!(app.streams, vec![1, 2, 3, 4]);

        let visible_rows = 2;
        app.ensure_active_stream_visible(visible_rows);
        assert_eq!(app.stream_view_offset, 0);

        app.next_stream();
        app.next_stream();
        app.ensure_active_stream_visible(visible_rows);
        assert_eq!(app.active_stream_idx, 2);
        assert_eq!(app.stream_view_offset, 1);

        app.next_stream();
        app.ensure_active_stream_visible(visible_rows);
        assert_eq!(app.stream_view_offset, 2);
    }

    #[test]
    fn test_scroll_accounts_for_annotation_rows() {
        let kernels = vec![
            make_kernel(1, 0.0, 1.0),
            make_kernel(2, 0.0, 1.0),
            make_kernel(3, 0.0, 1.0),
        ];
        let annotations = vec![AnnotationEvent {
            name: "ctx".to_string(),
            ts: 0.0,
            dur: 1.0,
            stream: 2,
        }];
        let mut app = App::new(Trace { kernels, annotations });
        assert_eq!(app.streams, vec![1, 2, 3]);
        assert_eq!(app.rows_for_stream(0), 1);
        assert_eq!(app.rows_for_stream(1), 2);
        assert_eq!(app.rows_for_stream(2), 1);

        let visible_rows = 3;
        while app.active_stream_idx != 2 {
            app.next_stream();
        }
        app.ensure_active_stream_visible(visible_rows);
        assert_eq!(app.stream_view_offset, 1);
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

    #[test]
    fn test_search_jumps_to_first_match_across_streams() {
        let kernels = vec![
            named_kernel(3, 100.0, "elementwise_add"),
            named_kernel(3, 200.0, "reduce_sum"),
            named_kernel(7, 150.0, "volta_sgemm_128"),
            named_kernel(7, 250.0, "ampere_gemm"),
        ];
        let mut app = app_from(kernels);
        assert_eq!(app.active_stream(), 3);

        app.search_start();
        assert!(app.search_active);

        app.search_push('v');
        app.search_push('o');
        app.search_push('l');

        assert!(!app.search_no_match);
        assert_eq!(app.active_stream(), 7);
        assert_eq!(app.selected_event().unwrap().name, "volta_sgemm_128");

        app.search_commit();
        assert!(!app.search_active);
        assert_eq!(app.active_stream(), 7);
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
        assert_eq!(app.selected_event().unwrap().name, "GEMM");

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

    fn app_with_annotations() -> App {
        let kernels = vec![
            named_kernel(4, 100.0, "kernel_a"),
            named_kernel(4, 200.0, "kernel_b"),
        ];
        let annotations = vec![
            AnnotationEvent { name: "ctx_0".to_string(), ts: 90.0, dur: 60.0, stream: 4 },
            AnnotationEvent { name: "ctx_1".to_string(), ts: 180.0, dur: 40.0, stream: 4 },
        ];
        App::new(Trace { kernels, annotations })
    }

    #[test]
    fn test_annotations_available_for_rendering() {
        let app = app_with_annotations();
        assert_eq!(app.active_stream(), 4);
        assert_eq!(app.annotation_indices_for_stream(4).len(), 2);
        assert_eq!(app.annotation_indices_for_stream(999).len(), 0);
    }

    #[test]
    fn test_navigation_ignores_annotations() {
        let mut app = app_with_annotations();
        assert_eq!(app.selected_kernel, 0);
        app.next_kernel();
        assert_eq!(app.selected_kernel, 1);
        assert_eq!(app.selected_event().unwrap().name, "kernel_b");
        app.next_kernel();
        assert_eq!(app.selected_kernel, 1);
        app.prev_kernel();
        assert_eq!(app.selected_kernel, 0);
    }
}
