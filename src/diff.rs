use similar::{capture_diff_slices, Algorithm, ChangeTag};

/// Myers O(ND) diff over string slices.
/// Returns `(Option<left_idx>, Option<right_idx>)` pairs in order:
/// `(Some,Some)` = matched, `(Some,None)` = removed from left,
/// `(None,Some)` = added on right.
pub fn myers_lcs(left: &[&str], right: &[&str]) -> Vec<(Option<usize>, Option<usize>)> {
    capture_diff_slices(Algorithm::Myers, left, right)
        .iter()
        .flat_map(|op| op.iter_changes(left, right))
        .map(|change| match change.tag() {
            ChangeTag::Equal => (change.old_index(), change.new_index()),
            ChangeTag::Delete => (change.old_index(), None),
            ChangeTag::Insert => (None, change.new_index()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn myers_empty_both() {
        assert!(myers_lcs(&[], &[]).is_empty());
    }

    #[test]
    fn myers_all_match() {
        let r = myers_lcs(&["a", "b"], &["a", "b"]);
        assert_eq!(r, vec![(Some(0), Some(0)), (Some(1), Some(1))]);
    }

    #[test]
    fn myers_no_common() {
        let r = myers_lcs(&["a"], &["b"]);
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r.iter().any(|p| p == &(Some(0), None)), "a removed: {r:?}");
        assert!(r.iter().any(|p| p == &(None, Some(0))), "b added: {r:?}");
    }

    // S5: gemm+bn+relu vs gemm+relu -> bn removed, rest matched
    #[test]
    fn myers_mixed_remove() {
        let r = myers_lcs(&["gemm", "bn", "relu"], &["gemm", "relu"]);
        assert_eq!(r.len(), 3, "{r:?}");
        assert!(r.iter().any(|p| p == &(Some(0), Some(0))), "gemm matched: {r:?}");
        assert!(r.iter().any(|p| p == &(Some(1), None)), "bn removed: {r:?}");
        assert!(r.iter().any(|p| p == &(Some(2), Some(1))), "relu matched: {r:?}");
    }

    #[test]
    fn myers_diff_len_right_longer() {
        let r = myers_lcs(&["a"], &["a", "b"]);
        assert!(r.iter().any(|p| p == &(Some(0), Some(0))), "a matched: {r:?}");
        assert!(r.iter().any(|p| p == &(None, Some(1))), "b added: {r:?}");
    }

    #[test]
    fn myers_empty_left() {
        assert_eq!(myers_lcs(&[], &["x", "y"]), vec![(None, Some(0)), (None, Some(1))]);
    }

    #[test]
    fn myers_empty_right() {
        assert_eq!(myers_lcs(&["x", "y"], &[]), vec![(Some(0), None), (Some(1), None)]);
    }
}
