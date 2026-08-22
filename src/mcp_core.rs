//! Pure, bounded trace-analysis functions shared by the TUI and the MCP server.
//!
//! Every list-returning function enforces `limit` (rows returned) and `offset`
//! (rows skipped) so callers — especially an LLM over MCP — cannot pull an
//! unbounded dump from an 800MB / multi-million-kernel trace into their context.

use crate::app::{annotation_for_kernel, csv_escape, stage_for_annotation_name};
use crate::trace::{parse_trace, KernelEvent, Trace};
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The vLLM iteration stage a kernel executes under, derived from the covering
/// `gpu_user_annotation`. `None` means the kernel has no stage-bearing
/// annotation (e.g. covered only by an nccl annotation, or none at all).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Prefill,
    Decode,
    Mixed,
    None,
}

impl Stage {
    fn from_ann_str(s: &str) -> Stage {
        match s {
            "prefill" => Stage::Prefill,
            "decode" => Stage::Decode,
            "mixed" => Stage::Mixed,
            _ => Stage::None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
            Stage::Mixed => "mixed",
            Stage::None => "none",
        }
    }

    /// Parses an MCP filter argument. Only the three real stages are accepted;
    /// `"none"` is rejected because "no filter" is expressed as `Option::None`,
    /// not a filter that selects unannotated kernels.
    pub fn parse_arg(s: &str) -> Result<Stage, String> {
        match s {
            "prefill" => Ok(Stage::Prefill),
            "decode" => Ok(Stage::Decode),
            "mixed" => Ok(Stage::Mixed),
            other => Err(format!(
                "invalid stage {other:?}: expected one of prefill, decode, mixed"
            )),
        }
    }
}

/// Classifies a kernel's stage. This is the single classification path reused by
/// every stage-aware feature (lane filter, summary breakdown, sequence filter,
/// stage_summary) so the vLLM name parsing lives in exactly one place.
pub(crate) fn stage_of_kernel(trace: &Trace, k: &KernelEvent) -> Stage {
    annotation_for_kernel(&trace.annotations, k.stream, k.trace_id, k.ts)
        .map(|a| Stage::from_ann_str(stage_for_annotation_name(&a.name)))
        .unwrap_or(Stage::None)
}

/// Fixed stage ordering for deterministic per-stage output rows.
const STAGE_ORDER: [Stage; 4] = [Stage::Prefill, Stage::Decode, Stage::Mixed, Stage::None];

fn stage_slot(s: Stage) -> usize {
    match s {
        Stage::Prefill => 0,
        Stage::Decode => 1,
        Stage::Mixed => 2,
        Stage::None => 3,
    }
}

/// Hard ceiling on rows any single tool call may return, regardless of the
/// requested `limit`. Keeps MCP responses bounded even on a caller mistake.
pub const MAX_ROWS: usize = 1000;

/// Default page size when a caller omits `limit`.
pub const DEFAULT_LIMIT: usize = 100;

fn effective_limit(limit: Option<usize>) -> usize {
    limit.unwrap_or(DEFAULT_LIMIT).min(MAX_ROWS)
}

/// Recursively scans `dir` for PyTorch profiler traces (`*.pt.trace.json.gz`)
/// and returns their paths relative to `dir`, sorted for determinism. The user
/// workflow nests traces at `logs/<run>/traces/*.pt.trace.json.gz`, so a flat
/// scan misses them; this walks the tree. Returns an empty vec (not an error)
/// when none are found. The returned relative paths are directly reusable as the
/// `path` argument of the other MCP tools.
pub fn list_traces_in_dir(dir: &str) -> Result<Vec<String>> {
    let root = Path::new(dir);
    let mut hits: Vec<PathBuf> = Vec::new();
    walk_traces(root, 0, &mut hits);
    let mut rel: Vec<String> = hits
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .unwrap_or(p)
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    rel.sort();
    Ok(rel)
}

/// Bounded depth-first walk collecting `*.pt.trace.json.gz` files. Skips
/// symlinks (so an ancestor-pointing symlink can't cycle), hidden directories
/// (names starting with `.`), and unreadable entries; caps recursion at
/// `MAX_DEPTH`.
fn walk_traces(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    const MAX_DEPTH: usize = 8;
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        let path = entry.path();
        if ft.is_dir() {
            let hidden = entry
                .file_name()
                .to_str()
                .map(|n| n.starts_with('.'))
                .unwrap_or(true);
            if hidden {
                continue;
            }
            walk_traces(&path, depth + 1, out);
        } else if entry
            .file_name()
            .to_str()
            .map(|n| n.ends_with(".pt.trace.json.gz"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// Compact display label for a trace, disambiguating traces that share a
/// filename across runs. The user's layout is `logs/<run>/traces/<file>`, where
/// `<file>` (e.g. `dp0_..._rank7....pt.trace.json.gz`) repeats every run — so
/// the label combines the run folder (the segment before `traces/`, else the
/// immediate parent) with the rank (`rankN` parsed from the filename, else the
/// filename stem). A flat trace with no meaningful parent falls back to its stem.
pub fn trace_display_label(path: &str) -> String {
    let p = Path::new(path);
    let file = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let stem = file
        .strip_suffix(".pt.trace.json.gz")
        .or_else(|| file.strip_suffix(".json.gz"))
        .or_else(|| file.strip_suffix(".json"))
        .unwrap_or(file);

    let run = run_folder(p);
    let leaf = rank_of(stem).unwrap_or_else(|| stem.to_string());

    match run {
        Some(r) if !leaf.is_empty() => format!("{r}/{leaf}"),
        _ if !stem.is_empty() => stem.to_string(),
        _ => "trace".to_string(),
    }
}

/// The run folder for a trace path: the component immediately before `traces/`,
/// else the immediate parent directory name (unless it is `.`).
fn run_folder(p: &Path) -> Option<String> {
    let comps: Vec<&str> = p
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    if let Some(idx) = comps.iter().rposition(|&c| c == "traces") {
        if idx > 0 {
            return Some(comps[idx - 1].to_string());
        }
    }
    p.parent()
        .and_then(|par| par.file_name())
        .and_then(|n| n.to_str())
        .filter(|n| *n != ".")
        .map(|n| n.to_string())
}

/// Extracts a `rankN` token from a filename stem: the literal `rank` followed by
/// one or more digits (stopping at the first non-digit).
fn rank_of(stem: &str) -> Option<String> {
    let idx = stem.find("rank")?;
    let digits: String = stem[idx + 4..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        Some(format!("rank{digits}"))
    }
}

/// Per-stage kernel count and summed duration, one entry per stage.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StageStat {
    pub stage: &'static str,
    pub kernel_count: usize,
    pub total_dur: f64,
}

/// Aggregate, always-bounded overview of a trace: never returns per-kernel rows.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TraceSummary {
    pub kernel_count: usize,
    pub annotation_count: usize,
    pub streams: Vec<u64>,
    pub start_ts: f64,
    pub end_ts: f64,
    pub duration: f64,
    pub per_stage: Vec<StageStat>,
}

pub fn summary(trace: &Trace) -> TraceSummary {
    let streams: BTreeSet<u64> = trace
        .kernels
        .iter()
        .map(|k| k.stream)
        .chain(trace.annotations.iter().map(|a| a.stream))
        .collect();
    let start = trace
        .kernels
        .iter()
        .map(|k| k.ts)
        .fold(f64::INFINITY, f64::min);
    let end = trace
        .kernels
        .iter()
        .map(|k| k.end_ts())
        .fold(f64::NEG_INFINITY, f64::max);
    let (start_ts, end_ts) = if trace.kernels.is_empty() {
        (0.0, 0.0)
    } else {
        (start, end)
    };

    let mut counts = [(0usize, 0f64); 4];
    for k in &trace.kernels {
        let slot = stage_slot(stage_of_kernel(trace, k));
        counts[slot].0 += 1;
        counts[slot].1 += k.dur;
    }
    let per_stage = STAGE_ORDER
        .iter()
        .map(|&s| StageStat {
            stage: s.as_str(),
            kernel_count: counts[stage_slot(s)].0,
            total_dur: counts[stage_slot(s)].1,
        })
        .collect();

    TraceSummary {
        kernel_count: trace.kernels.len(),
        annotation_count: trace.annotations.len(),
        streams: streams.into_iter().collect(),
        start_ts,
        end_ts,
        duration: end_ts - start_ts,
        per_stage,
    }
}

/// CSV (same columns as the TUI `E` export, including `annotation` and `stage`)
/// for the kernels of one stream, paginated by `offset`/`limit`. Rows preserve
/// timeline order (kernels are pre-sorted by `ts` at parse time).
pub fn lane_kernels_csv_for(
    trace: &Trace,
    stream: u64,
    offset: usize,
    limit: Option<usize>,
    stage: Option<Stage>,
) -> String {
    let cap = effective_limit(limit);
    let mut out = String::from(
        "idx,annotation,stage,name,ts,dur,end_ts,stream,device,grid,block,\
         shared_memory,registers_per_thread,correlation\n",
    );
    let opt_u64 = |v: Option<u64>| v.map(|n| n.to_string()).unwrap_or_default();
    for (row, k) in trace
        .kernels
        .iter()
        .filter(|k| k.stream == stream)
        .filter(|k| stage.is_none_or(|s| stage_of_kernel(trace, k) == s))
        .enumerate()
        .skip(offset)
        .take(cap)
    {
        let ann = annotation_for_kernel(&trace.annotations, k.stream, k.trace_id, k.ts);
        let ann_name = ann.map(|a| a.name.as_str()).unwrap_or("");
        let stage = ann.map(|a| stage_for_annotation_name(&a.name)).unwrap_or("");
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            row + 1,
            csv_escape(ann_name),
            stage,
            csv_escape(&k.name),
            k.ts,
            k.dur,
            k.end_ts(),
            k.stream,
            k.device,
            csv_escape(k.grid.as_deref().unwrap_or("")),
            csv_escape(k.block.as_deref().unwrap_or("")),
            opt_u64(k.shared_memory),
            opt_u64(k.registers_per_thread),
            opt_u64(k.correlation),
        ));
    }
    out
}

/// Tab-separated kernel sequence for `stream` starting at the first kernel named
/// `kernel_name`, up to (excluding) the next kernel with that same name — the
/// same slice the TUI `N` popup shows, paginated by `offset`/`limit`.
pub fn kernel_sequence_for(
    trace: &Trace,
    stream: u64,
    kernel_name: &str,
    offset: usize,
    limit: Option<usize>,
    stage: Option<Stage>,
) -> String {
    let cap = effective_limit(limit);
    let lane: Vec<&crate::trace::KernelEvent> =
        trace.kernels.iter().filter(|k| k.stream == stream).collect();
    let start = lane.iter().position(|k| k.name == kernel_name);
    let mut out = String::from("idx\tname\tdur\n");
    let Some(start) = start else {
        return out;
    };
    let mut end = lane.len();
    for (pos, k) in lane.iter().enumerate().skip(start + 1) {
        if k.name == kernel_name {
            end = pos;
            break;
        }
    }
    for (row, k) in lane[start..end]
        .iter()
        .filter(|k| stage.is_none_or(|s| stage_of_kernel(trace, k) == s))
        .enumerate()
        .skip(offset)
        .take(cap)
    {
        out.push_str(&format!("{}\t{}\t{:.2}\n", row + 1, k.name, k.dur));
    }
    out
}

/// Per-stage aggregate stats for one stream: one row per stage present, with
/// kernel count, summed duration, and median duration. Bounded by construction
/// (at most four stage rows). Median averages the two middle values for an even
/// count; the row order is fixed (prefill, decode, mixed, none).
pub fn stage_summary_for(trace: &Trace, stream: u64) -> String {
    let mut durs: [Vec<f64>; 4] = Default::default();
    for k in trace.kernels.iter().filter(|k| k.stream == stream) {
        durs[stage_slot(stage_of_kernel(trace, k))].push(k.dur);
    }
    let mut out = String::from("stage\tkernel_count\ttotal_dur\tmedian_dur\n");
    for &s in &STAGE_ORDER {
        let v = &mut durs[stage_slot(s)];
        if v.is_empty() {
            continue;
        }
        let total: f64 = v.iter().sum();
        v.sort_by(|a, b| a.total_cmp(b));
        let mid = v.len() / 2;
        let median = if v.len().is_multiple_of(2) {
            (v[mid - 1] + v[mid]) / 2.0
        } else {
            v[mid]
        };
        out.push_str(&format!(
            "{}\t{}\t{:.2}\t{:.2}\n",
            s.as_str(),
            v.len(),
            total,
            median
        ));
    }
    out
}

/// Loads a trace from disk (`.json` or `.json.gz`) for the MCP tools.
pub fn load(path: &str) -> Result<Trace> {
    parse_trace(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{AnnotationEvent, KernelEvent};

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
            trace_id: 0,
        }
    }

    fn trace_with(stream: u64, n: usize) -> Trace {
        let kernels = (0..n)
            .map(|i| kd(stream, i as f64 * 10.0, &format!("k{}", i), 5.0))
            .collect();
        Trace {
            kernels,
            annotations: vec![],
        }
    }

    const DECODE_ANN: &str = "execute_context_0(0)_generation_1(1)";
    const PREFILL_ANN: &str = "execute_context_1(5)_generation_0(0)";

    fn ann(stream: u64, ts: f64, dur: f64, name: &str) -> AnnotationEvent {
        AnnotationEvent {
            name: name.into(),
            ts,
            dur,
            stream,
            trace_id: 0,
        }
    }

    fn trace_with_ann(stream: u64, n: usize, ann_name: &str) -> Trace {
        let mut t = trace_with(stream, n);
        t.annotations.push(ann(stream, -1.0, 1e9, ann_name));
        t
    }

    #[test]
    fn test_stage_parse_arg_valid_and_invalid() {
        assert_eq!(Stage::parse_arg("prefill"), Ok(Stage::Prefill));
        assert_eq!(Stage::parse_arg("decode"), Ok(Stage::Decode));
        assert_eq!(Stage::parse_arg("mixed"), Ok(Stage::Mixed));
        assert!(Stage::parse_arg("none").is_err());
        assert!(Stage::parse_arg("bogus").is_err());
    }

    #[test]
    fn test_stage_of_kernel_classifies() {
        let decode = trace_with_ann(4, 1, DECODE_ANN);
        assert_eq!(stage_of_kernel(&decode, &decode.kernels[0]), Stage::Decode);
        let prefill = trace_with_ann(4, 1, PREFILL_ANN);
        assert_eq!(stage_of_kernel(&prefill, &prefill.kernels[0]), Stage::Prefill);
        let bare = trace_with(4, 1);
        assert_eq!(stage_of_kernel(&bare, &bare.kernels[0]), Stage::None);
    }

    #[test]
    fn test_summary_core_counts() {
        let mut t = trace_with(4, 3);
        t.annotations.push(AnnotationEvent {
            name: "a".into(),
            ts: 0.0,
            dur: 1.0,
            stream: 4,
            trace_id: 0,
        });
        let s = summary(&t);
        assert_eq!(s.kernel_count, 3);
        assert_eq!(s.annotation_count, 1);
        assert_eq!(s.streams, vec![4]);
        assert_eq!(s.start_ts, 0.0);
        assert_eq!(s.end_ts, 25.0);
        assert_eq!(s.duration, 25.0);
    }

    #[test]
    fn test_lane_csv_core_respects_limit_offset() {
        let t = trace_with(4, 10);
        let csv = lane_kernels_csv_for(&t, 4, 2, Some(3), None);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "idx,annotation,stage,name,ts,dur,end_ts,stream,device,grid,block,shared_memory,registers_per_thread,correlation"
        );
        assert_eq!(lines.len(), 4, "header + exactly 3 rows");
        assert!(lines[1].contains(",k2,"), "offset=2 -> first row is k2");
    }

    #[test]
    fn test_lane_csv_core_includes_annotation_stage() {
        let mut t = trace_with(4, 1);
        t.annotations.push(AnnotationEvent {
            name: "execute_context_0(0)_generation_1(1)".into(),
            ts: -1.0,
            dur: 10.0,
            stream: 4,
            trace_id: 0,
        });
        let csv = lane_kernels_csv_for(&t, 4, 0, None, None);
        let row = csv.lines().nth(1).unwrap();
        assert!(row.contains("execute_context_0(0)_generation_1(1),decode,"));
    }

    #[test]
    fn test_lane_csv_core_caps_at_max_rows() {
        let t = trace_with(4, MAX_ROWS + 500);
        let csv = lane_kernels_csv_for(&t, 4, 0, Some(usize::MAX), None);
        assert_eq!(csv.lines().count(), MAX_ROWS + 1, "header + MAX_ROWS");
    }

    #[test]
    fn test_kernel_sequence_core_bounded() {
        let kernels = vec![
            kd(1, 0.0, "foo", 10.0),
            kd(1, 20.0, "bar", 20.0),
            kd(1, 50.0, "baz", 5.0),
            kd(1, 60.0, "foo", 8.0),
        ];
        let t = Trace {
            kernels,
            annotations: vec![],
        };
        let seq = kernel_sequence_for(&t, 1, "foo", 0, None, None);
        let lines: Vec<&str> = seq.lines().collect();
        assert_eq!(lines[0], "idx\tname\tdur");
        assert_eq!(lines.len(), 4, "header + foo,bar,baz (up to next foo)");
        assert!(lines[1].starts_with("1\tfoo\t"));
        assert!(lines[3].starts_with("3\tbaz\t"));
    }

    /// Builds a stream-4 trace with `decode_n` decode-annotated kernels followed
    /// by `prefill_n` prefill-annotated kernels (non-overlapping windows) plus
    /// one bare (unannotated) kernel. Used by the stage-aware feature tests.
    fn mixed_stage_trace(decode_n: usize, prefill_n: usize) -> Trace {
        let mut kernels = Vec::new();
        for i in 0..decode_n {
            kernels.push(kd(4, 100.0 + i as f64, &format!("d{i}"), 2.0));
        }
        for i in 0..prefill_n {
            kernels.push(kd(4, 500.0 + i as f64, &format!("p{i}"), 4.0));
        }
        kernels.push(kd(4, 900.0, "bare", 8.0));
        let annotations = vec![
            ann(4, 99.0, (decode_n as f64) + 2.0, DECODE_ANN),
            ann(4, 499.0, (prefill_n as f64) + 2.0, PREFILL_ANN),
        ];
        Trace {
            kernels,
            annotations,
        }
    }

    #[test]
    fn test_lane_csv_core_stage_filter() {
        let t = mixed_stage_trace(3, 2);
        // No filter -> all 6 kernels (3 decode + 2 prefill + 1 bare).
        assert_eq!(
            lane_kernels_csv_for(&t, 4, 0, None, None).lines().count(),
            7,
            "header + 6 kernels"
        );
        // stage=decode -> only the 3 decode rows, each stage column == decode.
        let csv = lane_kernels_csv_for(&t, 4, 0, None, Some(Stage::Decode));
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4, "header + 3 decode rows");
        for row in &lines[1..] {
            let stage = row.split(',').nth(2).unwrap();
            assert_eq!(stage, "decode");
        }
        // Filter applies BEFORE pagination: limit=2 over the decode set -> 2 rows.
        let paged = lane_kernels_csv_for(&t, 4, 0, Some(2), Some(Stage::Decode));
        assert_eq!(paged.lines().count(), 3, "header + 2 of 3 decode rows");
    }

    #[test]
    fn test_summary_core_per_stage() {
        let t = mixed_stage_trace(3, 2);
        let s = summary(&t);
        // Existing fields unchanged.
        assert_eq!(s.kernel_count, 6);
        // per_stage counts sum to kernel_count; durations are per-stage sums.
        let get = |name: &str| s.per_stage.iter().find(|x| x.stage == name).unwrap();
        assert_eq!(get("decode").kernel_count, 3);
        assert_eq!(get("prefill").kernel_count, 2);
        assert_eq!(get("mixed").kernel_count, 0);
        assert_eq!(get("none").kernel_count, 1);
        assert_eq!(get("decode").total_dur, 6.0, "3 decode * 2.0");
        assert_eq!(get("prefill").total_dur, 8.0, "2 prefill * 4.0");
        assert_eq!(get("none").total_dur, 8.0, "1 bare * 8.0");
        let sum: usize = s.per_stage.iter().map(|x| x.kernel_count).sum();
        assert_eq!(sum, s.kernel_count);
    }

    #[test]
    fn test_kernel_sequence_core_stage_filter() {
        // foo(decode), bar(decode), baz(prefill) between two foos.
        let kernels = vec![
            kd(4, 100.0, "foo", 10.0),
            kd(4, 101.0, "bar", 20.0),
            kd(4, 500.0, "baz", 5.0),
            kd(4, 900.0, "foo", 8.0),
        ];
        let annotations = vec![
            ann(4, 99.0, 5.0, DECODE_ANN),
            ann(4, 499.0, 5.0, PREFILL_ANN),
        ];
        let t = Trace {
            kernels,
            annotations,
        };
        // No filter -> foo,bar,baz.
        assert_eq!(
            kernel_sequence_for(&t, 4, "foo", 0, None, None).lines().count(),
            4
        );
        // stage=decode -> only foo,bar (baz is prefill, dropped).
        let seq = kernel_sequence_for(&t, 4, "foo", 0, None, Some(Stage::Decode));
        let lines: Vec<&str> = seq.lines().collect();
        assert_eq!(lines.len(), 3, "header + foo,bar");
        assert!(lines[1].contains("foo"));
        assert!(lines[2].contains("bar"));
    }

    #[test]
    fn test_stage_summary_core_median() {
        // decode durations [1,3,5] -> median 3 (odd); prefill [2,4] -> median 3 (even avg).
        let kernels = vec![
            kd(4, 100.0, "d0", 1.0),
            kd(4, 101.0, "d1", 3.0),
            kd(4, 102.0, "d2", 5.0),
            kd(4, 500.0, "p0", 2.0),
            kd(4, 501.0, "p1", 4.0),
        ];
        let annotations = vec![
            ann(4, 99.0, 5.0, DECODE_ANN),
            ann(4, 499.0, 5.0, PREFILL_ANN),
        ];
        let t = Trace {
            kernels,
            annotations,
        };
        let out = stage_summary_for(&t, 4);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "stage\tkernel_count\ttotal_dur\tmedian_dur");
        // Only present stages (decode, prefill); no mixed/none rows.
        assert_eq!(lines.len(), 3, "header + decode + prefill only");
        let decode = lines.iter().find(|l| l.starts_with("decode\t")).unwrap();
        assert_eq!(*decode, "decode\t3\t9.00\t3.00");
        let prefill = lines.iter().find(|l| l.starts_with("prefill\t")).unwrap();
        assert_eq!(*prefill, "prefill\t2\t6.00\t3.00");
    }

    fn unique_tmp(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ptt-{tag}-{}-{n}", std::process::id()))
    }

    fn touch(path: &std::path::Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn test_list_traces_in_dir_finds_nested() {
        let root = unique_tmp("nested");
        touch(&root.join("logs/run_a/traces/x.pt.trace.json.gz"));
        touch(&root.join("logs/run_b/traces/x.pt.trace.json.gz"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(
            got,
            vec![
                "logs/run_a/traces/x.pt.trace.json.gz".to_string(),
                "logs/run_b/traces/x.pt.trace.json.gz".to_string(),
            ]
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_finds_flat_and_nested() {
        let root = unique_tmp("flatnested");
        touch(&root.join("flat.pt.trace.json.gz"));
        touch(&root.join("logs/run/traces/deep.pt.trace.json.gz"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(
            got,
            vec![
                "flat.pt.trace.json.gz".to_string(),
                "logs/run/traces/deep.pt.trace.json.gz".to_string(),
            ]
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_empty_is_ok_not_err() {
        let root = unique_tmp("empty");
        std::fs::create_dir_all(&root).unwrap();
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert!(got.is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_ignores_non_traces() {
        let root = unique_tmp("ignore");
        touch(&root.join("a.json"));
        touch(&root.join("b.txt"));
        touch(&root.join("c.pt.trace.json"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert!(got.is_empty(), "only *.pt.trace.json.gz counts: {got:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_list_traces_in_dir_skips_hidden_dirs() {
        let root = unique_tmp("hidden");
        touch(&root.join(".cache/traces/x.pt.trace.json.gz"));
        touch(&root.join("visible/traces/y.pt.trace.json.gz"));
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(got, vec!["visible/traces/y.pt.trace.json.gz".to_string()]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn test_list_traces_in_dir_does_not_follow_symlink_cycle() {
        let root = unique_tmp("cycle");
        touch(&root.join("logs/traces/x.pt.trace.json.gz"));
        // A symlink pointing back to an ancestor would loop a naive walker.
        std::os::unix::fs::symlink(&root, root.join("logs/loop")).unwrap();
        let got = list_traces_in_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(got, vec!["logs/traces/x.pt.trace.json.gz".to_string()]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_label_run_and_rank() {
        assert_eq!(
            trace_display_label(
                "logs/20250115_1200-tp8/traces/dp0_pp0_tp7_dcp0_ep7_rank7.178.pt.trace.json.gz"
            ),
            "20250115_1200-tp8/rank7"
        );
    }

    #[test]
    fn test_label_run_no_rank() {
        assert_eq!(
            trace_display_label("logs/run/traces/foo.pt.trace.json.gz"),
            "run/foo"
        );
    }

    #[test]
    fn test_label_flat_stem() {
        assert_eq!(trace_display_label("flat.pt.trace.json.gz"), "flat");
    }

    #[test]
    fn test_label_no_traces_parent_uses_parent() {
        assert_eq!(
            trace_display_label("logs/run/foo_rank3.pt.trace.json.gz"),
            "run/rank3"
        );
    }

    #[test]
    fn test_label_rank_parsing_boundaries() {
        assert_eq!(
            trace_display_label("logs/r/traces/a_rank12_b.pt.trace.json.gz"),
            "r/rank12"
        );
        assert_eq!(
            trace_display_label("logs/r/traces/norankhere.pt.trace.json.gz"),
            "r/norankhere"
        );
    }

    #[test]
    fn test_label_empty_stem_fallback() {
        assert_eq!(trace_display_label(".pt.trace.json.gz"), "trace");
    }
}
