use crate::trace::KernelEvent;

const K: usize = 3;
const GAP_PENALTY: f32 = -0.4;
const MISMATCH: f32 = -0.2;

fn normalize_tokens(name: &str) -> Vec<String> {
    name.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn token_overlap_score(a_names: &[&str], b_names: &[&str]) -> f32 {
    if a_names.is_empty() && b_names.is_empty() {
        return 1.0;
    }
    let a_toks: Vec<String> = a_names.iter().flat_map(|n| normalize_tokens(n)).collect();
    let b_toks: Vec<String> = b_names.iter().flat_map(|n| normalize_tokens(n)).collect();
    if a_toks.is_empty() && b_toks.is_empty() {
        return 1.0;
    }
    let matches = a_toks.iter().filter(|t| b_toks.contains(t)).count();
    let total = a_toks.len().max(b_toks.len());
    if total == 0 { 1.0 } else { matches as f32 / total as f32 }
}

/// Bounded NW alignment with K-wide fusion groups.
/// Consumes 1..=K consecutive kernels from either side per step, allowing
/// `[gemm, relu]` to match `[fused_gemm_relu]` when token overlap is high.
/// Returns confidence in [0, 1].
pub fn align_block_pair(
    ref_kernels: &[KernelEvent],
    other_kernels: &[KernelEvent],
) -> f32 {
    let rn = ref_kernels.len();
    let on = other_kernels.len();
    if rn == 0 && on == 0 {
        return 1.0;
    }
    if rn == 0 || on == 0 {
        return 0.0;
    }

    let rnames: Vec<&str> = ref_kernels.iter().map(|k| k.name.as_str()).collect();
    let onames: Vec<&str> = other_kernels.iter().map(|k| k.name.as_str()).collect();

    // dp[i][j] = best score aligning rnames[0..i] vs onames[0..j]
    let mut dp = vec![vec![f32::NEG_INFINITY; on + 1]; rn + 1];
    dp[0][0] = 0.0;
    for i in 1..=rn {
        dp[i][0] = dp[i-1][0] + GAP_PENALTY;
    }
    for j in 1..=on {
        dp[0][j] = dp[0][j-1] + GAP_PENALTY;
    }

    for i in 1..=rn {
        for j in 1..=on {
            let mut best = f32::NEG_INFINITY;
            // gap in ref (consume k from other)
            for k in 1..=K.min(j) {
                let v = dp[i][j - k] + GAP_PENALTY * k as f32;
                if v > best { best = v; }
            }
            // gap in other (consume k from ref)
            for k in 1..=K.min(i) {
                let v = dp[i - k][j] + GAP_PENALTY * k as f32;
                if v > best { best = v; }
            }
            // match groups: consume ri from ref and oi from other (1..=K each)
            for ri in 1..=K.min(i) {
                for oi in 1..=K.min(j) {
                    let rs = &rnames[i - ri..i];
                    let os = &onames[j - oi..j];
                    let score = token_overlap_score(rs, os);
                    let base = if score > 0.5 { score } else { MISMATCH };
                    let v = dp[i - ri][j - oi] + base;
                    if v > best { best = v; }
                }
            }
            dp[i][j] = best;
        }
    }

    let raw = dp[rn][on];
    let max_possible = rn.max(on) as f32;
    if max_possible <= 0.0 { return 0.0; }
    ((raw / max_possible + 1.0) / 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::KernelEvent;

    fn tk(name: &str) -> KernelEvent {
        KernelEvent {
            name: name.to_string(), cat: "kernel".to_string(),
            ts: 0.0, dur: 1.0, device: 0, stream: 1,
            grid: None, block: None, shared_memory: None,
            registers_per_thread: None, correlation: None, trace_id: 0,
        }
    }

    // S2: [gemm, relu] should match [fused_gemm_relu] with high confidence.
    #[test]
    fn test_align_block_pair_fusion_s2() {
        let r = vec![tk("gemm"), tk("relu")];
        let o = vec![tk("fused_gemm_relu")];
        let conf = align_block_pair(&r, &o);
        assert!(conf > 0.5, "fusion match confidence={conf}");
    }

    // S3: [gemm, bn, relu] vs [gemm, relu] — bn aligns to gap, rest match.
    #[test]
    fn test_align_block_pair_missing_kernel_s3() {
        let r = vec![tk("gemm"), tk("bn"), tk("relu")];
        let o = vec![tk("gemm"), tk("relu")];
        let conf = align_block_pair(&r, &o);
        assert!(conf > 0.5, "missing-kernel match confidence={conf}");
    }

    #[test]
    fn test_align_block_pair_identical() {
        let r = vec![tk("gemm"), tk("relu")];
        let o = vec![tk("gemm"), tk("relu")];
        let conf = align_block_pair(&r, &o);
        assert!(conf > 0.8, "identical blocks confidence={conf}");
    }
}
