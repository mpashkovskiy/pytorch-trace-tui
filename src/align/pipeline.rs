use super::{
    AlignedBlock, AlignedTraceBlock, AlignmentMode, AlignmentState,
    align_block_pair, build_time_maps, detect_blocks, match_blocks, precompute_display_times,
};
use crate::trace::{AnnotationEvent, KernelEvent};

/// Full pipeline: detect per-trace blocks → match across traces → NW-align pairs
/// → build piecewise time maps → precompute display times.
/// Falls back to OffsetOnly when trace_count < 2 or no multi-block structure found.
pub fn build_alignment(
    kernels: &[KernelEvent],
    annotations: &[AnnotationEvent],
    trace_count: usize,
) -> AlignmentState {
    if trace_count < 2 {
        return AlignmentState::offset_only(0, trace_count);
    }

    let per_trace_blocks: Vec<Vec<(f64, f64, std::ops::Range<usize>)>> = (0..trace_count)
        .map(|tid| detect_blocks(kernels, tid))
        .collect();

    let ref_blocks = &per_trace_blocks[0];
    if ref_blocks.len() < 2 {
        return AlignmentState::offset_only(0, trace_count);
    }
    if !per_trace_blocks[1..].iter().any(|b| b.len() >= 2) {
        return AlignmentState::offset_only(0, trace_count);
    }

    let mut aligned_blocks: Vec<AlignedBlock> = ref_blocks
        .iter()
        .enumerate()
        .map(|(id, (rs, re, _))| AlignedBlock {
            id,
            display_start: *rs,
            display_end: *re,
            per_trace: vec![None; trace_count],
        })
        .collect();

    for (id, (rs, re, rng)) in ref_blocks.iter().enumerate() {
        aligned_blocks[id].per_trace[0] = Some(AlignedTraceBlock {
            raw_start: *rs,
            raw_end: *re,
            kernel_indices: rng.clone(),
            confidence: 1.0,
        });
    }

    for (tid, other_blocks) in per_trace_blocks.iter().enumerate().skip(1) {
        if other_blocks.is_empty() {
            continue;
        }
        let ref_ks_owned: Vec<KernelEvent> =
            kernels.iter().filter(|k| k.trace_id == 0).cloned().collect();
        let other_ks_owned: Vec<KernelEvent> =
            kernels.iter().filter(|k| k.trace_id == tid).cloned().collect();

        let pairs = match_blocks(ref_blocks, other_blocks, &ref_ks_owned, &other_ks_owned);

        for (ri, oi) in pairs {
            if ri >= aligned_blocks.len() {
                continue;
            }
            let (rs, re, rng) = &other_blocks[oi];
            let ref_slice: Vec<KernelEvent> = kernels
                [ref_blocks[ri].2.start..ref_blocks[ri].2.end.min(kernels.len())]
                .iter()
                .filter(|k| k.trace_id == 0)
                .cloned()
                .collect();
            let oth_slice: Vec<KernelEvent> = kernels
                [other_blocks[oi].2.start..other_blocks[oi].2.end.min(kernels.len())]
                .iter()
                .filter(|k| k.trace_id == tid)
                .cloned()
                .collect();
            let confidence = align_block_pair(&ref_slice, &oth_slice);
            aligned_blocks[ri].per_trace[tid] = Some(AlignedTraceBlock {
                raw_start: *rs,
                raw_end: *re,
                kernel_indices: rng.clone(),
                confidence,
            });
        }
    }

    let time_maps = build_time_maps(&aligned_blocks, trace_count);
    let (k_disp, a_disp, k_local, a_local) =
        precompute_display_times(kernels, annotations, &time_maps);

    AlignmentState {
        mode: AlignmentMode::PiecewiseWarp,
        reference_trace_id: 0,
        blocks: aligned_blocks,
        time_maps,
        display_kernel_times: k_disp,
        display_annotation_times: a_disp,
        kernel_local_idx: k_local,
        annotation_local_idx: a_local,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::KernelEvent;

    fn tk(trace_id: usize, ts: f64, dur: f64, name: &str) -> KernelEvent {
        KernelEvent { name: name.into(), cat: "k".into(), ts, dur, device: 0, stream: 1,
            grid: None, block: None, shared_memory: None, registers_per_thread: None,
            correlation: None, trace_id }
    }

    // S5: no multi-block structure (single block per trace) → OffsetOnly.
    #[test]
    fn test_build_alignment_no_blocks_falls_back_offset_only() {
        let all: Vec<KernelEvent> =
            ["A","B","C","D"].iter().enumerate().map(|(i,n)| tk(0,i as f64*10.0,5.0,n))
            .chain(["E","F","G","H"].iter().enumerate().map(|(i,n)| tk(1,i as f64*10.0,5.0,n)))
            .collect();
        let st = build_alignment(&all, &[], 2);
        assert!(matches!(st.mode, AlignmentMode::OffsetOnly));
    }

    // S4: single trace → OffsetOnly.
    #[test]
    fn test_build_alignment_single_trace_offset_only() {
        let ks: Vec<KernelEvent> = ["A","B"].iter().enumerate()
            .map(|(i,n)| tk(0,i as f64*10.0,5.0,n)).collect();
        assert!(matches!(build_alignment(&ks, &[], 1).mode, AlignmentMode::OffsetOnly));
    }

    // S1: 2 traces × 3 gap-separated [gemm,relu] blocks → PiecewiseWarp.
    #[test]
    fn test_build_alignment_happy_sets_piecewise() {
        let mut ks = Vec::new();
        for b in 0..3usize {
            let b0 = b as f64 * 1000.0; let b1 = b as f64 * 900.0;
            ks.push(tk(0, b0,       8.0, "gemm")); ks.push(tk(0, b0+10.0, 4.0, "relu"));
            ks.push(tk(1, b1,       8.0, "gemm")); ks.push(tk(1, b1+10.0, 4.0, "relu"));
        }
        let st = build_alignment(&ks, &[], 2);
        assert!(matches!(st.mode, AlignmentMode::PiecewiseWarp));
        assert!(!st.blocks.is_empty());
    }
}
