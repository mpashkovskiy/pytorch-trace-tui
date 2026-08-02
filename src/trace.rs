use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// GPU kernel categories in PyTorch Chrome trace format
const GPU_CATS: &[&str] = &["kernel", "gpu_memcpy", "gpu_memset"];

/// Raw trace event as deserialized from JSON
#[derive(Debug, Clone, Deserialize)]
pub struct RawEvent {
    pub name: Option<String>,
    pub cat: Option<String>,
    pub ph: Option<String>,
    pub ts: Option<f64>,
    pub dur: Option<f64>,
    pub pid: Option<serde_json::Value>,
    pub tid: Option<serde_json::Value>,
    pub args: Option<HashMap<String, serde_json::Value>>,
}

/// A parsed GPU kernel event with all relevant fields
#[derive(Debug, Clone, Serialize)]
pub struct KernelEvent {
    pub name: String,
    pub cat: String,
    /// Timestamp in microseconds
    pub ts: f64,
    /// Duration in microseconds
    pub dur: f64,
    /// GPU device id
    pub device: u64,
    /// CUDA stream id
    pub stream: u64,
    // Args fields (all optional in trace)
    pub grid: Option<String>,
    pub block: Option<String>,
    pub shared_memory: Option<u64>,
    pub registers_per_thread: Option<u64>,
    pub correlation: Option<u64>,
}

impl KernelEvent {
    pub fn end_ts(&self) -> f64 {
        self.ts + self.dur
    }
}

/// Parse a trace file and return only GPU kernel events, sorted by timestamp.
/// Transparently handles both plain `.json` and gzipped `.json.gz` files.
/// Uses streaming JSON deserialization — never loads the full document into memory.
pub fn parse_trace(path: &str) -> Result<Vec<KernelEvent>> {
    let file = File::open(path).with_context(|| format!("Cannot open trace file: {}", path))?;

    let is_gz = path.ends_with(".gz");

    let kernels = if is_gz {
        let gz = GzDecoder::new(file);
        let reader = BufReader::new(gz);
        parse_from_reader(reader)
    } else {
        let reader = BufReader::new(file);
        parse_from_reader(reader)
    }
    .context("Failed to parse trace JSON")?;

    Ok(kernels)
}

/// Core streaming parser — works on any `BufRead`.
///
/// PyTorch traces come in two shapes:
///   1. Bare array:  `[ {...}, {...}, ... ]`
///   2. Object wrapper: `{ "traceEvents": [ {...}, ... ], ... }`
///
/// We detect the first non-whitespace character to decide, then stream
/// individual event objects without buffering the whole document.
fn parse_from_reader<R: BufRead>(mut reader: R) -> Result<Vec<KernelEvent>> {
    // Peek at the first non-whitespace byte to decide top-level shape.
    let first = peek_first_nonws(&mut reader)?;

    let mut kernels: Vec<KernelEvent> = match first {
        b'[' => {
            // Bare array — stream directly.
            stream_array(reader)?
        }
        b'{' => {
            // Object wrapper — find the "traceEvents" key then stream its array.
            stream_object_trace_events(reader)?
        }
        other => {
            anyhow::bail!(
                "Unexpected top-level JSON character '{}' (0x{:02x})",
                other as char,
                other
            );
        }
    };

    // Sort by timestamp ascending
    kernels.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));

    Ok(kernels)
}

/// Read bytes until we find a non-whitespace character; put it back via a
/// chain so the reader still starts at that character.
fn peek_first_nonws<R: BufRead>(reader: &mut R) -> Result<u8> {
    loop {
        let buf = reader.fill_buf().context("Failed to read trace")?;
        if buf.is_empty() {
            anyhow::bail!("Trace file is empty");
        }
        // Find first non-whitespace in current buffer
        for (i, &b) in buf.iter().enumerate() {
            if !b.is_ascii_whitespace() {
                // Consume up to but NOT including this byte — we want to
                // leave the reader positioned at it.
                reader.consume(i);
                return Ok(b);
            }
        }
        // All whitespace — consume entire buffer and keep looking
        let len = buf.len();
        reader.consume(len);
    }
}

use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

struct KernelSeqSeed;

impl<'de> DeserializeSeed<'de> for KernelSeqSeed {
    type Value = Vec<KernelEvent>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for KernelSeqSeed {
    type Value = Vec<KernelEvent>;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "an array of trace events")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut kernels = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1 << 20));
        while let Some(raw) = seq.next_element::<RawEvent>()? {
            if let Some(k) = try_into_kernel(raw) {
                kernels.push(k);
            }
        }
        Ok(kernels)
    }
}

fn stream_array<R: BufRead>(reader: R) -> Result<Vec<KernelEvent>> {
    let mut de = serde_json::Deserializer::from_reader(reader);
    let kernels = KernelSeqSeed
        .deserialize(&mut de)
        .context("Failed to stream top-level array")?;
    Ok(kernels)
}

fn stream_object_trace_events<R: BufRead>(reader: R) -> Result<Vec<KernelEvent>> {
    struct TraceEventsExtractor;

    impl<'de> Visitor<'de> for TraceEventsExtractor {
        type Value = Vec<KernelEvent>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "a PyTorch trace object with traceEvents key")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut kernels: Option<Vec<KernelEvent>> = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "traceEvents" {
                    kernels = Some(map.next_value_seed(KernelSeqSeed)?);
                } else {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(kernels.unwrap_or_default())
        }
    }

    let mut de = serde_json::Deserializer::from_reader(reader);
    let kernels = de
        .deserialize_map(TraceEventsExtractor)
        .context("Failed to stream traceEvents from object")?;
    Ok(kernels)
}

/// Try to convert a raw event into a KernelEvent; returns None if it's not a GPU kernel
fn try_into_kernel(e: RawEvent) -> Option<KernelEvent> {
    // Must not be a metadata / instant event
    let ph = e.ph.as_deref().unwrap_or("");
    if ph == "M" || ph == "i" || ph == "I" {
        return None;
    }

    // Must be a GPU category
    let cat = e.cat.as_deref().unwrap_or("");
    if !GPU_CATS.contains(&cat) {
        return None;
    }

    // Must have a timestamp
    let ts = e.ts?;
    let dur = e.dur.unwrap_or(0.0);

    let name = e.name.unwrap_or_else(|| "(unnamed)".to_string());

    // Stream = tid (thread id encodes the CUDA stream on GPU events)
    let stream = tid_to_u64(&e.tid).unwrap_or(0);

    // Device from args first, pid as fallback
    let device = args_u64(&e.args, "device")
        .or_else(|| tid_to_u64(&e.pid))
        .unwrap_or(0);

    let grid = args_string(&e.args, "grid");
    let block = args_string(&e.args, "block");
    let shared_memory = args_u64(&e.args, "shared memory");
    let registers_per_thread = args_u64(&e.args, "registers per thread");
    let correlation = args_u64(&e.args, "correlation");

    Some(KernelEvent {
        name,
        cat: cat.to_string(),
        ts,
        dur,
        device,
        stream,
        grid,
        block,
        shared_memory,
        registers_per_thread,
        correlation,
    })
}

fn tid_to_u64(v: &Option<serde_json::Value>) -> Option<u64> {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().or_else(|| n.as_f64().map(|f| f as u64)),
        Some(serde_json::Value::String(s)) => s.parse().ok(),
        _ => None,
    }
}

fn args_u64(args: &Option<HashMap<String, serde_json::Value>>, key: &str) -> Option<u64> {
    let map = args.as_ref()?;
    match map.get(key)? {
        serde_json::Value::Number(n) => n.as_u64().or_else(|| n.as_f64().map(|f| f as u64)),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn args_string(args: &Option<HashMap<String, serde_json::Value>>, key: &str) -> Option<String> {
    let map = args.as_ref()?;
    match map.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw(cat: &str, ph: &str, ts: f64, dur: f64, tid: u64) -> RawEvent {
        RawEvent {
            name: Some("test_kernel".to_string()),
            cat: Some(cat.to_string()),
            ph: Some(ph.to_string()),
            ts: Some(ts),
            dur: Some(dur),
            pid: Some(serde_json::Value::Number(0.into())),
            tid: Some(serde_json::Value::Number(tid.into())),
            args: None,
        }
    }

    #[test]
    fn test_gpu_kernel_accepted() {
        let e = make_raw("kernel", "X", 1000.0, 50.0, 7);
        let k = try_into_kernel(e).expect("Should parse GPU kernel");
        assert_eq!(k.name, "test_kernel");
        assert_eq!(k.stream, 7);
        assert!((k.ts - 1000.0).abs() < f64::EPSILON);
        assert!((k.dur - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cpu_op_rejected() {
        let e = make_raw("cpu_op", "X", 1000.0, 50.0, 0);
        assert!(try_into_kernel(e).is_none());
    }

    #[test]
    fn test_metadata_rejected() {
        let e = make_raw("kernel", "M", 0.0, 0.0, 0);
        assert!(try_into_kernel(e).is_none());
    }

    #[test]
    fn test_gpu_memcpy_accepted() {
        let e = make_raw("gpu_memcpy", "X", 2000.0, 10.0, 3);
        let k = try_into_kernel(e).expect("Should parse gpu_memcpy");
        assert_eq!(k.cat, "gpu_memcpy");
        assert_eq!(k.stream, 3);
    }

    /// Parse a plain-JSON trace that wraps events in {"traceEvents": [...]}
    #[test]
    fn test_parse_object_wrapper() {
        let json = r#"{
            "traceEvents": [
                {"name":"cpu_fn","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
                {"name":"my_kernel","cat":"kernel","ph":"X","ts":10.0,"dur":3.0,"pid":0,"tid":7,
                 "args":{"device":0,"stream":7,"grid":"[32,1,1]","block":"[128,1,1]",
                         "shared memory":0,"registers per thread":32,"correlation":99}},
                {"name":"process_name","ph":"M","pid":0,"tid":0,"args":{"name":"GPU 0"}}
            ],
            "displayTimeUnit": "ms"
        }"#;
        let reader = BufReader::new(json.as_bytes());
        let kernels = parse_from_reader(reader).expect("Should parse object wrapper");
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "my_kernel");
        assert_eq!(kernels[0].stream, 7);
        assert_eq!(kernels[0].grid.as_deref(), Some("[32,1,1]"));
    }

    /// Parse a bare-array trace
    #[test]
    fn test_parse_bare_array() {
        let json = r#"[
            {"name":"cpu_op","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
            {"name":"kern_a","cat":"kernel","ph":"X","ts":5.0,"dur":2.0,"pid":0,"tid":3},
            {"name":"kern_b","cat":"gpu_memcpy","ph":"X","ts":8.0,"dur":1.0,"pid":0,"tid":3}
        ]"#;
        let reader = BufReader::new(json.as_bytes());
        let kernels = parse_from_reader(reader).expect("Should parse bare array");
        assert_eq!(kernels.len(), 2);
        // Sorted by ts
        assert_eq!(kernels[0].name, "kern_a");
        assert_eq!(kernels[1].name, "kern_b");
    }

    /// Parse a gzipped trace end-to-end via parse_trace()
    #[test]
    fn test_parse_gz_trace() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let json = r#"{"traceEvents":[
            {"name":"cpu_fn","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
            {"name":"gz_kernel","cat":"kernel","ph":"X","ts":10.0,"dur":3.0,"pid":0,"tid":7}
        ]}"#;

        // Write gzipped JSON to a temp file
        let path = "/tmp/test_trace_rust.json.gz";
        {
            let f = File::create(path).unwrap();
            let mut gz = GzEncoder::new(f, Compression::default());
            gz.write_all(json.as_bytes()).unwrap();
            gz.finish().unwrap();
        }

        let kernels = parse_trace(path).expect("Should parse .json.gz");
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "gz_kernel");
    }
}
