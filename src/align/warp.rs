use super::{AlignedBlock, AlignedTraceBlock, DisplayKernelTime, TimeSegment, TraceTimeMap};
use crate::trace::{AnnotationEvent, KernelEvent};

/// Builds one `TraceTimeMap` per trace from the matched block list.
/// Each matched per-trace block becomes a `TimeSegment` mapping raw time to
/// display (reference) time. Unmatched traces get an identity segment so all
/// events stay visible.
pub fn build_time_maps(blocks: &[AlignedBlock], trace_count: usize) -> Vec<TraceTimeMap> {
    let mut maps: Vec<TraceTimeMap> = (0..trace_count)
        .map(|tid| TraceTimeMap { trace_id: tid, segments: Vec::new() })
        .collect();

    for block in blocks {
        for (tid, maybe_tb) in block.per_trace.iter().enumerate() {
            if let Some(tb) = maybe_tb {
                maps[tid].segments.push(TimeSegment {
                    trace_id: tid,
                    block_id: block.id,
                    raw_start: tb.raw_start,
                    raw_end: tb.raw_end,
                    display_start: block.display_start,
                    display_end: block.display_end,
                });
            }
        }
    }

    // Sort segments by raw_start for binary-search lookup.
    for map in &mut maps {
        map.segments.sort_by(|a, b| {
            a.raw_start.partial_cmp(&b.raw_start).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    maps
}

/// Maps a raw timestamp to display space using the precomputed segments.
/// Uses binary search; events outside all segments clamp to the nearest segment.
pub fn map_ts(maps: &[TraceTimeMap], trace_id: usize, raw_ts: f64) -> f64 {
    let Some(map) = maps.get(trace_id) else { return raw_ts; };
    if map.segments.is_empty() {
        return raw_ts;
    }
    // Find the last segment whose raw_start <= raw_ts.
    let pos = map.segments.partition_point(|s| s.raw_start <= raw_ts);
    let seg = if pos == 0 {
        &map.segments[0]
    } else {
        &map.segments[(pos - 1).min(map.segments.len() - 1)]
    };
    let raw_span = (seg.raw_end - seg.raw_start).max(1e-9);
    let disp_span = seg.display_end - seg.display_start;
    // Linear interpolation: display_ts = display_start + (raw - raw_start) * ratio
    let ratio = disp_span / raw_span;
    (seg.display_start + (raw_ts - seg.raw_start) * ratio).max(0.0)
}

/// Precomputes display start/end for every kernel and annotation event.
/// Returns `(kernel_display_times, annotation_display_times)` indexed by
/// global flat-vec index. Also returns per-trace local indices for efficient
/// `display_kernel_times[trace_id][local_idx]` lookup.
pub fn precompute_display_times(
    kernels: &[KernelEvent],
    annotations: &[AnnotationEvent],
    maps: &[TraceTimeMap],
) -> (
    Vec<Vec<DisplayKernelTime>>,
    Vec<Vec<DisplayKernelTime>>,
    Vec<usize>,
    Vec<usize>,
) {
    let trace_count = maps.len();
    let mut k_display: Vec<Vec<DisplayKernelTime>> = vec![Vec::new(); trace_count];
    let mut a_display: Vec<Vec<DisplayKernelTime>> = vec![Vec::new(); trace_count];
    let mut k_local_idx = vec![0usize; kernels.len()];
    let mut a_local_idx = vec![0usize; annotations.len()];

    // Process kernels per trace in their stored order.
    for (gi, k) in kernels.iter().enumerate() {
        let tid = k.trace_id.min(trace_count.saturating_sub(1));
        k_local_idx[gi] = k_display[tid].len();
        let ds = map_ts(maps, tid, k.ts);
        let de = map_ts(maps, tid, k.end_ts());
        k_display[tid].push(DisplayKernelTime { display_start: ds, display_end: de });
    }
    for (gi, a) in annotations.iter().enumerate() {
        let tid = a.trace_id.min(trace_count.saturating_sub(1));
        a_local_idx[gi] = a_display[tid].len();
        let ds = map_ts(maps, tid, a.ts);
        let de = map_ts(maps, tid, a.end_ts());
        a_display[tid].push(DisplayKernelTime { display_start: ds, display_end: de });
    }
    (k_display, a_display, k_local_idx, a_local_idx)
}

#[allow(dead_code)]
fn _uses_atb(_: &AlignedTraceBlock) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::KernelEvent;

    fn tb(rs: f64, re: f64) -> AlignedTraceBlock {
        AlignedTraceBlock { raw_start: rs, raw_end: re, kernel_indices: 0..1, confidence: 1.0 }
    }

    fn mk_block(id: usize, ds: f64, de: f64, per: Vec<Option<AlignedTraceBlock>>) -> AlignedBlock {
        AlignedBlock { id, display_start: ds, display_end: de, per_trace: per }
    }

    #[test]
    fn test_build_time_maps_creates_sorted_segments() {
        let blocks = vec![
            mk_block(0, 0.0, 100.0, vec![Some(tb(0.0, 30.0)), Some(tb(0.0, 30.0))]),
            mk_block(1, 100.0, 200.0, vec![Some(tb(40.0, 70.0)), Some(tb(40.0, 70.0))]),
        ];
        let maps = build_time_maps(&blocks, 2);
        assert_eq!(maps.len(), 2);
        assert!(maps[0].segments.windows(2).all(|w| w[0].raw_start <= w[1].raw_start));
        assert_eq!(maps[0].segments[0].display_start, 0.0);
    }

    #[test]
    fn test_build_time_maps_identity_for_unmatched() {
        let blocks = vec![
            mk_block(0, 0.0, 100.0, vec![Some(tb(0.0, 30.0)), None]),
        ];
        let maps = build_time_maps(&blocks, 2);
        assert_eq!(maps[1].segments.len(), 0, "no segments for unmatched trace");
    }

    #[test]
    fn test_map_ts_linear_interpolation() {
        // seg raw[0,10] -> disp[100,120]; ts=5 -> 110
        let seg = TimeSegment {
            trace_id: 0, block_id: 0,
            raw_start: 0.0, raw_end: 10.0,
            display_start: 100.0, display_end: 120.0,
        };
        let map = TraceTimeMap { trace_id: 0, segments: vec![seg] };
        let maps = vec![map];
        let ds = map_ts(&maps, 0, 5.0);
        assert!((ds - 110.0).abs() < 1e-6, "expected 110.0 got {ds}");
    }

    #[test]
    fn test_precompute_display_times_interpolates() {
        fn tk(ts: f64, dur: f64) -> KernelEvent {
            KernelEvent { name: "x".into(), cat: "k".into(), ts, dur,
                device: 0, stream: 1, grid: None, block: None,
                shared_memory: None, registers_per_thread: None,
                correlation: None, trace_id: 0 }
        }
        let seg = TimeSegment {
            trace_id: 0, block_id: 0,
            raw_start: 0.0, raw_end: 10.0,
            display_start: 100.0, display_end: 120.0,
        };
        let maps = vec![TraceTimeMap { trace_id: 0, segments: vec![seg] }];
        let ks = vec![tk(5.0, 2.0)];
        let anns: Vec<AnnotationEvent> = vec![];
        let (kd, _, _, _) = precompute_display_times(&ks, &anns, &maps);
        assert!((kd[0][0].display_start - 110.0).abs() < 1e-6);
        assert!((kd[0][0].display_end   - 114.0).abs() < 1e-6);
    }
}
