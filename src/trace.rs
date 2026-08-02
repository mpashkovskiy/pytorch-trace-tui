use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

const GPU_CATS: &[&str] = &["kernel", "gpu_memcpy", "gpu_memset"];
const GPU_CATS_BYTES: &[&[u8]] = &[b"kernel", b"gpu_memcpy", b"gpu_memset"];

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

#[derive(Debug, Clone, Serialize)]
pub struct KernelEvent {
    pub name: String,
    pub cat: String,
    pub ts: f64,
    pub dur: f64,
    pub device: u64,
    pub stream: u64,
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

pub fn parse_trace(path: &str) -> Result<Vec<KernelEvent>> {
    let file = File::open(path).with_context(|| format!("Cannot open trace file: {}", path))?;

    let mut kernels = if path.ends_with(".gz") {
        let gz = GzDecoder::new(file);
        let reader = BufReader::with_capacity(1 << 16, gz);
        parse_lexer(reader)
    } else {
        let reader = BufReader::with_capacity(1 << 16, file);
        parse_lexer(reader)
    }
    .context("Failed to parse trace")?;

    kernels.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));
    Ok(kernels)
}

fn parse_lexer<R: BufRead>(mut reader: R) -> Result<Vec<KernelEvent>> {
    const CHUNK: usize = 1 << 16;

    let first = peek_first_nonws(&mut reader)?;

    match first {
        b'[' => lex_array(reader, CHUNK),
        b'{' => lex_object(reader, CHUNK),
        other => anyhow::bail!(
            "Unexpected top-level JSON character '{}' (0x{:02x})",
            other as char,
            other
        ),
    }
}

fn lex_object<R: BufRead>(reader: R, chunk: usize) -> Result<Vec<KernelEvent>> {
    let te_marker = b"traceEvents";
    let mut window: Vec<u8> = Vec::with_capacity(chunk * 2);
    let mut rdr = reader;
    let mut found = false;

    'scan: loop {
        let buf = rdr.fill_buf().context("read error")?;
        if buf.is_empty() {
            break;
        }
        let n = buf.len().min(chunk);
        window.extend_from_slice(&buf[..n]);
        rdr.consume(n);

        if let Some(pos) = window.windows(te_marker.len()).position(|w| w == te_marker) {
            let after = pos + te_marker.len();
            if let Some(off) = window[after..].iter().position(|&b| b == b'[') {
                let remainder = window[after + off + 1..].to_vec();
                window = remainder;
                found = true;
                break 'scan;
            }
        }
        if window.len() > te_marker.len() + 64 {
            let keep = window.len() - te_marker.len() - 16;
            window.drain(..keep);
        }
    }

    if !found {
        return Ok(vec![]);
    }

    lex_events_array(rdr, window, chunk)
}

fn lex_array<R: BufRead>(reader: R, chunk: usize) -> Result<Vec<KernelEvent>> {
    let window = Vec::new();
    lex_events_array(reader, window, chunk)
}

fn lex_events_array<R: BufRead>(
    mut rdr: R,
    window: Vec<u8>,
    _chunk: usize,
) -> Result<Vec<KernelEvent>> {
    let mut kernels: Vec<KernelEvent> = Vec::with_capacity(65536);

    let mut depth: u32 = 0;
    let mut in_string = false;
    let mut escape_next = false;

    const MAX_HEADER: usize = 512;
    let mut header: Vec<u8> = Vec::with_capacity(MAX_HEADER);

    #[derive(PartialEq, Clone, Copy)]
    enum State {
        Unknown,
        Keep,
        Skip,
    }
    let mut state = State::Unknown;
    let mut event_buf: Vec<u8> = Vec::with_capacity(1024);

    macro_rules! feed_byte {
        ($b:expr) => {{
            let b: u8 = $b;

            if escape_next {
                escape_next = false;
                match state {
                    State::Keep => event_buf.push(b),
                    State::Unknown if header.len() < MAX_HEADER => header.push(b),
                    _ => {}
                }
                continue;
            }

            if in_string {
                if b == b'\\' {
                    escape_next = true;
                } else if b == b'"' {
                    in_string = false;
                }
                match state {
                    State::Keep => event_buf.push(b),
                    State::Unknown if header.len() < MAX_HEADER => header.push(b),
                    _ => {}
                }
                continue;
            }

            match b {
                b'"' => {
                    in_string = true;
                    match state {
                        State::Keep => event_buf.push(b),
                        State::Unknown if header.len() < MAX_HEADER => header.push(b),
                        _ => {}
                    }
                }
                b'{' => {
                    depth += 1;
                    if depth == 1 {
                        state = State::Unknown;
                        header.clear();
                        event_buf.clear();
                        header.push(b);
                    } else {
                        match state {
                            State::Keep => event_buf.push(b),
                            State::Unknown if header.len() < MAX_HEADER => header.push(b),
                            _ => {}
                        }
                    }
                }
                b'}' => {
                    match state {
                        State::Keep => event_buf.push(b),
                        State::Unknown if header.len() < MAX_HEADER => header.push(b),
                        _ => {}
                    }
                    if depth == 1 {
                        if state == State::Keep && !event_buf.is_empty() {
                            if let Ok(raw) = serde_json::from_slice::<RawEvent>(&event_buf) {
                                if let Some(k) = try_into_kernel(raw) {
                                    kernels.push(k);
                                }
                            }
                        }
                        state = State::Unknown;
                        header.clear();
                        event_buf.clear();
                        depth = 0;
                    } else if depth > 0 {
                        depth -= 1;
                    }
                }
                _ => {
                    match state {
                        State::Keep => event_buf.push(b),
                        State::Unknown if header.len() < MAX_HEADER => header.push(b),
                        _ => {}
                    }
                }
            }

            if state == State::Unknown && depth == 1 {
                if let Some(is_gpu) = cat_in_header(&header) {
                    if is_gpu {
                        state = State::Keep;
                        event_buf.extend_from_slice(&header);
                        header.clear();
                    } else {
                        state = State::Skip;
                        header.clear();
                    }
                }
            }
        }};
    }

    for &b in &window {
        feed_byte!(b);
    }

    loop {
        let buf = rdr.fill_buf().context("read error")?;
        if buf.is_empty() {
            break;
        }
        let n = buf.len();
        let bytes: Vec<u8> = buf.to_vec();
        rdr.consume(n);
        for &b in &bytes {
            feed_byte!(b);
        }
    }

    Ok(kernels)
}

fn cat_in_header(buf: &[u8]) -> Option<bool> {
    let marker = b"\"cat\"";
    let pos = buf.windows(marker.len()).position(|w| w == marker)?;
    let after = &buf[pos + marker.len()..];
    let mut i = 0;
    while i < after.len() && matches!(after[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    if i >= after.len() || after[i] != b':' {
        return None;
    }
    i += 1;
    while i < after.len() && matches!(after[i], b' ' | b'\t') {
        i += 1;
    }
    if i >= after.len() || after[i] != b'"' {
        return None;
    }
    i += 1;
    let val_start = i;
    while i < after.len() && after[i] != b'"' {
        i += 1;
    }
    if i >= after.len() {
        return None;
    }
    let val = &after[val_start..i];
    Some(GPU_CATS_BYTES.contains(&val))
}

fn peek_first_nonws<R: BufRead>(reader: &mut R) -> Result<u8> {
    loop {
        let buf = reader.fill_buf().context("Failed to read trace")?;
        if buf.is_empty() {
            anyhow::bail!("Trace file is empty");
        }
        for (i, &b) in buf.iter().enumerate() {
            if !b.is_ascii_whitespace() {
                reader.consume(i);
                return Ok(b);
            }
        }
        let len = buf.len();
        reader.consume(len);
    }
}

fn try_into_kernel(e: RawEvent) -> Option<KernelEvent> {
    let ph = e.ph.as_deref().unwrap_or("");
    if ph == "M" || ph == "i" || ph == "I" {
        return None;
    }
    let cat = e.cat.as_deref().unwrap_or("");
    if !GPU_CATS.contains(&cat) {
        return None;
    }
    let ts = e.ts?;
    let dur = e.dur.unwrap_or(0.0);
    let name = e.name.unwrap_or_else(|| "(unnamed)".to_string());
    let stream = tid_to_u64(&e.tid).unwrap_or(0);
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
        let kernels = parse_lexer(reader).expect("Should parse object wrapper");
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "my_kernel");
        assert_eq!(kernels[0].stream, 7);
        assert_eq!(kernels[0].grid.as_deref(), Some("[32,1,1]"));
    }

    #[test]
    fn test_parse_bare_array() {
        let json = r#"[
            {"name":"cpu_op","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
            {"name":"kern_a","cat":"kernel","ph":"X","ts":5.0,"dur":2.0,"pid":0,"tid":3},
            {"name":"kern_b","cat":"gpu_memcpy","ph":"X","ts":8.0,"dur":1.0,"pid":0,"tid":3}
        ]"#;
        let reader = BufReader::new(json.as_bytes());
        let kernels = parse_lexer(reader).expect("Should parse bare array");
        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[0].name, "kern_a");
        assert_eq!(kernels[1].name, "kern_b");
    }

    #[test]
    fn test_parse_gz_trace() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let json = r#"{"traceEvents":[
            {"name":"cpu_fn","cat":"cpu_op","ph":"X","ts":1.0,"dur":5.0,"pid":1,"tid":1},
            {"name":"gz_kernel","cat":"kernel","ph":"X","ts":10.0,"dur":3.0,"pid":0,"tid":7}
        ]}"#;

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
