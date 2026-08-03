use crate::trace::KernelEvent;

mod match_blocks;
mod nw;
mod pipeline;
mod warp;
pub use match_blocks::match_blocks;
pub use nw::align_block_pair;
pub use pipeline::build_alignment;
pub use warp::{build_time_maps, precompute_display_times};

#[derive(Debug, Clone)]
pub struct TimeSegment {
    pub trace_id: usize,
    pub block_id: usize,
    pub raw_start: f64,
    pub raw_end: f64,
    pub display_start: f64,
    pub display_end: f64,
}

#[derive(Debug, Clone)]
pub struct TraceTimeMap {
    pub trace_id: usize,
    pub segments: Vec<TimeSegment>,
}

#[derive(Debug, Clone)]
pub struct DisplayKernelTime {
    pub display_start: f64,
    pub display_end: f64,
}

#[derive(Debug, Clone)]
pub struct AlignedTraceBlock {
    pub raw_start: f64,
    pub raw_end: f64,
    pub kernel_indices: std::ops::Range<usize>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct AlignedBlock {
    pub id: usize,
    pub display_start: f64,
    pub display_end: f64,
    pub per_trace: Vec<Option<AlignedTraceBlock>>,
}

#[derive(Debug, Clone)]
pub enum AlignmentMode {
    OffsetOnly,
    PiecewiseWarp,
}

#[derive(Debug, Clone)]
pub struct AlignmentState {
    pub mode: AlignmentMode,
    pub reference_trace_id: usize,
    pub blocks: Vec<AlignedBlock>,
    pub time_maps: Vec<TraceTimeMap>,
    pub display_kernel_times: Vec<Vec<DisplayKernelTime>>,
    pub display_annotation_times: Vec<Vec<DisplayKernelTime>>,
    pub kernel_local_idx: Vec<usize>,
    pub annotation_local_idx: Vec<usize>,
}

impl AlignmentState {
    pub fn offset_only(reference_trace_id: usize, trace_count: usize) -> Self {
        Self {
            mode: AlignmentMode::OffsetOnly,
            reference_trace_id,
            blocks: Vec::new(),
            time_maps: Vec::new(),
            display_kernel_times: vec![Vec::new(); trace_count],
            display_annotation_times: vec![Vec::new(); trace_count],
            kernel_local_idx: Vec::new(),
            annotation_local_idx: Vec::new(),
        }
    }
}


/// Returns `(raw_start, raw_end, global_index_range)` blocks for one trace.
/// Primary: split at idle gaps where gap > median_gap * 10.
/// Fallback: detect the shortest repeating name period and tile into equal blocks.
pub fn detect_blocks(kernels: &[KernelEvent], trace_id: usize) -> Vec<(f64, f64, std::ops::Range<usize>)> {
    let mut indexed: Vec<(usize, &KernelEvent)> = kernels
        .iter()
        .enumerate()
        .filter(|(_, k)| k.trace_id == trace_id)
        .collect();
    if indexed.is_empty() {
        return Vec::new();
    }
    indexed.sort_by(|a, b| a.1.ts.partial_cmp(&b.1.ts).unwrap_or(std::cmp::Ordering::Equal));

    if indexed.len() == 1 {
        let (gi, k) = indexed[0];
        return vec![(k.ts, k.end_ts(), gi..gi + 1)];
    }

    // Compute inter-kernel gaps and their median.
    let mut gaps: Vec<f64> = indexed
        .windows(2)
        .map(|w| (w[1].1.ts - w[0].1.end_ts()).max(0.0))
        .collect();
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted_gaps.len();
    let median_gap = if n.is_multiple_of(2) {
        (sorted_gaps[n / 2 - 1] + sorted_gaps[n / 2]) / 2.0
    } else {
        sorted_gaps[n / 2]
    };
    let threshold = (median_gap * 10.0).max(1.0);

    // Primary: split at gaps exceeding threshold.
    let split_points: Vec<usize> = gaps
        .iter()
        .enumerate()
        .filter(|(_, &g)| g > threshold)
        .map(|(i, _)| i + 1)
        .collect();

    if !split_points.is_empty() {
        let mut blocks = Vec::new();
        let mut start = 0;
        let mut boundaries: Vec<usize> = split_points;
        boundaries.push(indexed.len());
        for end in boundaries {
            let slice = &indexed[start..end];
            if !slice.is_empty() {
                let raw_start = slice[0].1.ts;
                let raw_end = slice.last().map(|(_, k)| k.end_ts()).unwrap_or(raw_start);
                let gi_first = slice[0].0;
                let gi_last = slice.last().unwrap().0;
                blocks.push((raw_start, raw_end, gi_first..gi_last + 1));
            }
            start = end;
        }
        return blocks;
    }

    // Fallback: find the shortest repeating period in the name sequence.
    let names: Vec<&str> = indexed.iter().map(|(_, k)| k.name.as_str()).collect();
    let len = names.len();
    let period = (1..=len / 2).find(|&p| {
        len.is_multiple_of(p)
            && (p..len).all(|i| names[i] == names[i % p])
    });

    if let Some(p) = period {
        return (0..len / p)
            .map(|b| {
                let s = b * p;
                let e = s + p;
                let raw_start = indexed[s].1.ts;
                let raw_end = indexed[e - 1].1.end_ts();
                let gi_first = indexed[s].0;
                let gi_last = indexed[e - 1].0;
                (raw_start, raw_end, gi_first..gi_last + 1)
            })
            .collect();
    }

    // No structure found: one block for the whole trace.
    let raw_start = indexed[0].1.ts;
    let raw_end = indexed.last().map(|(_, k)| k.end_ts()).unwrap_or(raw_start);
    let gi_first = indexed[0].0;
    let gi_last = indexed.last().unwrap().0;
    vec![(raw_start, raw_end, gi_first..gi_last + 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::KernelEvent;

    fn test_kernel(trace_id: usize, ts: f64, dur: f64, name: &str) -> KernelEvent {
        KernelEvent {
            name: name.to_string(),
            cat: "kernel".to_string(),
            ts,
            dur,
            device: 0,
            stream: 1,
            grid: None,
            block: None,
            shared_memory: None,
            registers_per_thread: None,
            correlation: None,
            trace_id,
        }
    }

    #[test]
    fn test_alignment_state_constructs_offset_only() {
        let s = AlignmentState::offset_only(1, 3);
        assert!(matches!(s.mode, AlignmentMode::OffsetOnly));
        assert_eq!(s.reference_trace_id, 1);
        assert!(s.blocks.is_empty());
        assert_eq!(s.display_kernel_times.len(), 3);
    }

    // S6: [A,B,C] repeated 3 times without gaps → 3 blocks each of size 3.
    #[test]
    fn test_detect_blocks_repeated_subsequence() {
        let ks: Vec<KernelEvent> = ["A","B","C","A","B","C","A","B","C"]
            .iter()
            .enumerate()
            .map(|(i, n)| test_kernel(0, i as f64 * 10.0, 5.0, n))
            .collect();
        let blocks = detect_blocks(&ks, 0);
        assert_eq!(blocks.len(), 3, "period 3 → 3 blocks");
        assert_eq!(blocks[0].2.len(), 3);
        assert_eq!(blocks[1].2.len(), 3);
        assert_eq!(blocks[2].2.len(), 3);
    }

    // Three clusters separated by large idle gaps → 3 blocks.
    #[test]
    fn test_detect_blocks_idle_gap() {
        let mut ks = Vec::new();
        for i in 0..3usize {
            let base = i as f64 * 1000.0;
            ks.push(test_kernel(0, base,        5.0, "gemm"));
            ks.push(test_kernel(0, base + 10.0, 5.0, "relu"));
        }
        let blocks = detect_blocks(&ks, 0);
        assert_eq!(blocks.len(), 3);
    }
}
