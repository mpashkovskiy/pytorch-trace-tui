//! Pure, bounded trace-analysis functions shared by the TUI and the MCP server.
//!
//! Every list-returning function enforces `limit` (rows returned) and `offset`
//! (rows skipped) so callers — especially an LLM over MCP — cannot pull an
//! unbounded dump from an 800MB / multi-million-kernel trace into their context.

use crate::app::{annotation_for_kernel, csv_escape, stage_for_annotation_name};
use crate::trace::{parse_trace, Trace};
use anyhow::Result;
use std::collections::BTreeSet;

/// Hard ceiling on rows any single tool call may return, regardless of the
/// requested `limit`. Keeps MCP responses bounded even on a caller mistake.
pub const MAX_ROWS: usize = 1000;

/// Default page size when a caller omits `limit`.
pub const DEFAULT_LIMIT: usize = 100;

fn effective_limit(limit: Option<usize>) -> usize {
    limit.unwrap_or(DEFAULT_LIMIT).min(MAX_ROWS)
}

/// Scans a directory for PyTorch profiler traces (`*.pt.trace.json.gz`),
/// returning their file names sorted for determinism.
pub fn list_traces_in_dir(dir: &str) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(".pt.trace.json.gz"))
        .collect();
    names.sort();
    Ok(names)
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
    TraceSummary {
        kernel_count: trace.kernels.len(),
        annotation_count: trace.annotations.len(),
        streams: streams.into_iter().collect(),
        start_ts,
        end_ts,
        duration: end_ts - start_ts,
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
    for (row, k) in lane[start..end].iter().enumerate().skip(offset).take(cap) {
        out.push_str(&format!("{}\t{}\t{:.2}\n", row + 1, k.name, k.dur));
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
        let csv = lane_kernels_csv_for(&t, 4, 2, Some(3));
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
        let csv = lane_kernels_csv_for(&t, 4, 0, None);
        let row = csv.lines().nth(1).unwrap();
        assert!(row.contains("execute_context_0(0)_generation_1(1),decode,"));
    }

    #[test]
    fn test_lane_csv_core_caps_at_max_rows() {
        let t = trace_with(4, MAX_ROWS + 500);
        let csv = lane_kernels_csv_for(&t, 4, 0, Some(usize::MAX));
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
        let seq = kernel_sequence_for(&t, 1, "foo", 0, None);
        let lines: Vec<&str> = seq.lines().collect();
        assert_eq!(lines[0], "idx\tname\tdur");
        assert_eq!(lines.len(), 4, "header + foo,bar,baz (up to next foo)");
        assert!(lines[1].starts_with("1\tfoo\t"));
        assert!(lines[3].starts_with("3\tbaz\t"));
    }
}
