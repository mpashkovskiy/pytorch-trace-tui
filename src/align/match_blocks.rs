use super::AlignedTraceBlock;
use crate::trace::KernelEvent;

/// Greedily pairs reference blocks with other-trace blocks.
/// Equal counts → pair by index. Unequal → score by name-bag overlap, pick
/// highest non-conflicting pairs (greedy, ties broken by index order).
pub fn match_blocks(
    ref_blocks: &[(f64, f64, std::ops::Range<usize>)],
    other_blocks: &[(f64, f64, std::ops::Range<usize>)],
    ref_kernels: &[KernelEvent],
    other_kernels: &[KernelEvent],
) -> Vec<(usize, usize)> {
    if ref_blocks.is_empty() || other_blocks.is_empty() {
        return Vec::new();
    }
    if ref_blocks.len() == other_blocks.len() {
        return (0..ref_blocks.len()).map(|i| (i, i)).collect();
    }
    // Score every (ref_idx, other_idx) pair by name multiset overlap ratio.
    let mut scores: Vec<(f32, usize, usize)> = ref_blocks
        .iter()
        .enumerate()
        .flat_map(|(ri, (_, _, rr))| {
            other_blocks
                .iter()
                .enumerate()
                .map(move |(oi, (_, _, or_))| {
                    let s = block_similarity(rr, or_, ref_kernels, other_kernels);
                    (s, ri, oi)
                })
        })
        .collect();
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut used_ref = vec![false; ref_blocks.len()];
    let mut used_other = vec![false; other_blocks.len()];
    let mut pairs = Vec::new();
    for (_, ri, oi) in scores {
        if !used_ref[ri] && !used_other[oi] {
            pairs.push((ri, oi));
            used_ref[ri] = true;
            used_other[oi] = true;
        }
    }
    pairs.sort_by_key(|&(ri, _)| ri);
    pairs
}

fn block_similarity(
    rr: &std::ops::Range<usize>,
    or_: &std::ops::Range<usize>,
    ref_kernels: &[KernelEvent],
    other_kernels: &[KernelEvent],
) -> f32 {
    let rnames: Vec<&str> = ref_kernels
        .get(rr.clone())
        .into_iter()
        .flatten()
        .map(|k| k.name.as_str())
        .collect();
    let onames: Vec<&str> = other_kernels
        .get(or_.clone())
        .into_iter()
        .flatten()
        .map(|k| k.name.as_str())
        .collect();
    if rnames.is_empty() && onames.is_empty() {
        return 1.0;
    }
    let matches = rnames.iter().filter(|n| onames.contains(n)).count();
    let total = rnames.len().max(onames.len());
    if total == 0 { 1.0 } else { matches as f32 / total as f32 }
}

// Suppress unused-import warning — AlignedTraceBlock used by callers.
#[allow(dead_code)]
fn _uses_atb(_: AlignedTraceBlock) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(s: f64, e: f64, r: std::ops::Range<usize>) -> (f64, f64, std::ops::Range<usize>) {
        (s, e, r)
    }

    fn tk(name: &str) -> KernelEvent {
        KernelEvent {
            name: name.to_string(), cat: "kernel".to_string(),
            ts: 0.0, dur: 1.0, device: 0, stream: 1,
            grid: None, block: None, shared_memory: None,
            registers_per_thread: None, correlation: None, trace_id: 0,
        }
    }

    #[test]
    fn test_match_blocks_equal_count_pairs_by_index() {
        let r = vec![blk(0.0,30.0,0..3), blk(40.0,70.0,3..6)];
        let o = vec![blk(0.0,30.0,0..3), blk(40.0,70.0,3..6)];
        let pairs = match_blocks(&r, &o, &[], &[]);
        assert_eq!(pairs, vec![(0,0),(1,1)]);
    }

    #[test]
    fn test_match_blocks_unequal_picks_best() {
        let ref_ks: Vec<KernelEvent> = ["gemm","relu","gemm","relu"].iter().map(|n| tk(n)).collect();
        let oth_ks: Vec<KernelEvent> = ["gemm","relu"].iter().map(|n| tk(n)).collect();
        let r = vec![blk(0.0,20.0,0..2), blk(20.0,40.0,2..4)];
        let o = vec![blk(0.0,20.0,0..2)];
        let pairs = match_blocks(&r, &o, &ref_ks, &oth_ks);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].1, 0);
    }
}
