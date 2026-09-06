use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn write_trace(path: &std::path::Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let json = r#"{"traceEvents":[
        {"name":"gemm","cat":"kernel","ph":"X","ts":100.0,"dur":5.0,"pid":0,"tid":4,
         "args":{"device":0,"stream":4}}
    ]}"#;
    let f = std::fs::File::create(path).unwrap();
    let mut gz = flate2::write::GzEncoder::new(f, flate2::Compression::default());
    gz.write_all(json.as_bytes()).unwrap();
    gz.finish().unwrap();
}

fn newest_debug_log_after(start_nanos: u128) -> Option<std::path::PathBuf> {
    let tmp = std::env::temp_dir();
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for entry in std::fs::read_dir(&tmp).ok()?.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("pytorch-trace-tui-") {
            continue;
        }
        let log = entry.path().join("debug.log");
        if !log.exists() {
            continue;
        }
        let created = entry
            .metadata()
            .ok()
            .and_then(|m| m.created().or_else(|_| m.modified()).ok());
        let Some(created) = created else { continue };
        let created_nanos = created
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        if created_nanos + 5_000_000_000 < start_nanos {
            continue;
        }
        if best.as_ref().map(|(t, _)| created > *t).unwrap_or(true) {
            best = Some((created, log));
        }
    }
    best.map(|(_, p)| p)
}

#[test]
fn mcp_debug_keeps_stdout_pure_and_logs_to_file() {
    let root = std::env::temp_dir().join(format!(
        "ptt-mcpdbg-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    write_trace(&root.join("logs/run/traces/x_rank0.pt.trace.json.gz"));

    let start_nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();

    let mut child = Command::new(env!("CARGO_BIN_EXE_pytorch-trace-tui"))
        .args(["--mcp", "--debug"])
        .current_dir(&root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary runs");

    let handshake = concat!(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
        "\n",
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_traces","arguments":{}}}"#,
        "\n",
    );
    child
        .stdin
        .take()
        .unwrap()
        .write_all(handshake.as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("waits");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");

    for banned in [
        "MCP request:",
        "Loading trace:",
        "MCP response:",
        "MCP error:",
    ] {
        assert!(
            !stdout.contains(banned),
            "stdout leaked debug text {banned:?}:\n{stdout}"
        );
    }

    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|e| panic!("stdout line is not JSON-RPC ({e}): {line}"));
    }

    let log = newest_debug_log_after(start_nanos).expect("debug.log created under temp dir");
    let contents = std::fs::read_to_string(&log).expect("log readable");
    assert!(
        contents.contains("MCP request:"),
        "debug.log has human-readable request lines:\n{contents}"
    );
    assert!(
        contents.contains("list_traces"),
        "debug.log names the invoked tool:\n{contents}"
    );

    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(log.parent().unwrap()).ok();
}
