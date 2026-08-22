use std::io::Write;
use std::process::{Command, Stdio};

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

#[test]
fn picker_lists_nested_runs_with_distinct_labels() {
    let root = std::env::temp_dir().join(format!("ptt-picker-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    write_trace(&root.join("logs/run_a/traces/x_rank0.pt.trace.json.gz"));
    write_trace(&root.join("logs/run_b/traces/x_rank0.pt.trace.json.gz"));

    let mut child = Command::new(env!("CARGO_BIN_EXE_pytorch-trace-tui"))
        .current_dir(&root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary runs");

    // The picker prints its rows before reading stdin; `q` quits without opening.
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"q\n")
        .unwrap();
    let out = child.wait_with_output().expect("process completes");
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains("run_a/rank0"),
        "picker should show run_a/rank0; got:\n{stdout}"
    );
    assert!(
        stdout.contains("run_b/rank0"),
        "picker should show run_b/rank0; got:\n{stdout}"
    );

    std::fs::remove_dir_all(&root).ok();
}
