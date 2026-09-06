use std::io::Write;
use std::process::Command;

/// Writes a fixture with `reps` identical decode blocks — each a two-kernel
/// (`mm`/`add`) block under the same decode annotation name — so the collapse
/// grouping key `(trace, stream, name, kernel-name signature)` folds them.
fn write_repeated_fixture(path: &std::path::Path, reps: usize) {
    let mut events = String::new();
    let mut t = 0.0;
    for r in 0..reps {
        if r > 0 {
            events.push(',');
        }
        let ann_start = t;
        let mm_ts = t;
        let add_ts = t + 20.0;
        events.push_str(&format!(
            r#"{{"name":"mm","cat":"kernel","ph":"X","ts":{mm_ts},"dur":10.0,"pid":0,"tid":4,"args":{{"device":0,"stream":4}}}},
               {{"name":"add","cat":"kernel","ph":"X","ts":{add_ts},"dur":4.0,"pid":0,"tid":4,"args":{{"device":0,"stream":4}}}},
               {{"name":"execute_context_0(0)_generation_1(1)","cat":"gpu_user_annotation","ph":"X","ts":{ann_start},"dur":40.0,"pid":0,"tid":4}}"#
        ));
        t += 60.0;
    }
    let json = format!(r#"{{"traceEvents":[{events}]}}"#);
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(json.as_bytes()).unwrap();
}

/// End-to-end via the headless CSV dump: the raw (un-collapsed) lane must list
/// every repetition, confirming collapse is a non-destructive projection over
/// the same parsed data the CSV exports.
#[test]
fn repeated_blocks_are_all_present_in_raw_csv() {
    let dir = std::env::temp_dir().join(format!("ptt-collapse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let trace = dir.join("repeated.json");
    write_repeated_fixture(&trace, 15);

    let out = Command::new(env!("CARGO_BIN_EXE_pytorch-trace-tui"))
        .arg("--dump-lane-csv")
        .arg("4")
        .arg(&trace)
        .output()
        .expect("binary runs");

    assert!(out.status.success(), "exit ok: {:?}", out.status);
    let stdout = String::from_utf8(out.stdout).unwrap();
    let mm = stdout.lines().filter(|l| l.contains(",mm,")).count();
    let add = stdout.lines().filter(|l| l.contains(",add,")).count();
    assert_eq!(mm, 15, "raw CSV lists all 15 mm kernels");
    assert_eq!(add, 15, "raw CSV lists all 15 add kernels");

    std::fs::remove_dir_all(&dir).ok();
}
