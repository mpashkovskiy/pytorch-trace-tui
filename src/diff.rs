/// Myers O(ND) diff over string slices.
/// Returns `(Option<left_idx>, Option<right_idx>)` pairs in order:
/// `(Some,Some)` = matched, `(Some,None)` = removed from left,
/// `(None,Some)` = added on right.
pub fn myers_lcs(left: &[&str], right: &[&str]) -> Vec<(Option<usize>, Option<usize>)> {
    let n = left.len();
    let m = right.len();
    if n == 0 && m == 0 { return vec![]; }
    if n == 0 { return (0..m).map(|j| (None, Some(j))).collect(); }
    if m == 0 { return (0..n).map(|i| (Some(i), None)).collect(); }

    let max_d = n + m;
    // v[k + max_d] = furthest-reaching x-coordinate on diagonal k
    let mut v = vec![0i64; 2 * max_d + 2];
    // Capture v snapshot BEFORE each round (indexed by round d)
    let mut trace: Vec<Vec<i64>> = Vec::with_capacity(max_d + 1);

    for d in 0i64..=(max_d as i64) {
        trace.push(v.clone());
        let mut k = -d;
        while k <= d {
            let ki = (k + max_d as i64) as usize;
            // Choose move: down (insert) if forced, or down if yields further x
            let go_down = k == -d
                || (k != d && v[ki - 1] < v[ki + 1]);

            let x_start = if go_down { v[ki + 1] } else { v[ki - 1] + 1 };
            let mut x = x_start;
            let mut y = x - k;
            // Extend snake (matches)
            while x < n as i64 && y < m as i64 && left[x as usize] == right[y as usize] {
                x += 1;
                y += 1;
            }
            v[ki] = x;
            if x >= n as i64 && y >= m as i64 {
                return backtrack(&trace, n, m, max_d, d);
            }
            k += 2;
        }
    }
    backtrack(&trace, n, m, max_d, max_d as i64)
}

fn backtrack(
    trace: &[Vec<i64>],
    n: usize,
    m: usize,
    max_d: usize,
    found_d: i64,
) -> Vec<(Option<usize>, Option<usize>)> {
    let mut x = n as i64;
    let mut y = m as i64;
    let mut ops: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let off = max_d as i64;

    // Walk backwards from d=found_d down to d=0
    let max_step = (found_d as usize).min(trace.len().saturating_sub(1));
    for d in (0..=max_step).rev() {
        let v = &trace[d]; // snapshot taken BEFORE this round
        let d_i = d as i64;
        let k = x - y;

        // Determine which move was made at round d to arrive at (x,y)
        let go_down = k == -d_i
            || (k != d_i && {
                let vm1 = if k - 1 + off >= 0 { v[(k - 1 + off) as usize] } else { -1 };
                let vp1 = if k + 1 + off >= 0 { v[(k + 1 + off) as usize] } else { -1 };
                vm1 < vp1
            });

        let prev_k = if go_down { k + 1 } else { k - 1 };
        let prev_x = if prev_k + off >= 0 { v[(prev_k + off) as usize] } else { 0 };
        let prev_y = prev_x - prev_k;

        // Walk back the snake (matches between prev and the non-diagonal step)
        while x > prev_x && y > prev_y && x > 0 && y > 0 {
            x -= 1; y -= 1;
            ops.push((Some(x as usize), Some(y as usize)));
        }

        if d > 0 {
            if go_down {
                // We moved down: y was decremented (insert from right)
                if y > 0 {
                    ops.push((None, Some((y - 1) as usize)));
                }
            } else {
                // We moved right: x was decremented (delete from left)
                if x > 0 {
                    ops.push((Some((x - 1) as usize), None));
                }
            }
        }

        x = prev_x;
        y = prev_y;
    }

    ops.reverse();
    ops
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
