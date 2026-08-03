use crate::trace::{AnnotationEvent, KernelEvent};

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

pub fn build_alignment(
    _kernels: &[KernelEvent],
    _annotations: &[AnnotationEvent],
    trace_count: usize,
) -> AlignmentState {
    AlignmentState::offset_only(0, trace_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_state_constructs_offset_only() {
        let s = AlignmentState::offset_only(1, 3);
        assert!(matches!(s.mode, AlignmentMode::OffsetOnly));
        assert_eq!(s.reference_trace_id, 1);
        assert!(s.blocks.is_empty());
        assert_eq!(s.display_kernel_times.len(), 3);
    }
}
