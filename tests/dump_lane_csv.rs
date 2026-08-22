use std::io::Write;
use std::process::Command;

fn write_fixture(path: &std::path::Path) {
    let json = r#"{"traceEvents":[
        {"name":"gemm","cat":"kernel","ph":"X","ts":100.0,"dur":5.0,"pid":0,"tid":4,
         "args":{"device":0,"stream":4}},
        {"name":"add","cat":"kernel","ph":"X","ts":400.0,"dur":5.0,"pid":0,"tid":4,
         "args":{"device":0,"stream":4}},
        {"name":"execute_context_0(0)_generation_1(1)","cat":"gpu_user_annotation",
         "ph":"X","ts":90.0,"dur":50.0,"pid":0,"tid":4}
    ]}"#;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(json.as_bytes()).unwrap();
}

#[test]
fn dump_lane_csv_emits_annotation_and_stage_columns() {
    let dir = std::env::temp_dir().join(format!("ptt-dump-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let trace = dir.join("fixture.json");
    write_fixture(&trace);

    let out = Command::new(env!("CARGO_BIN_EXE_pytorch-trace-tui"))
        .arg("--dump-lane-csv")
        .arg("4")
        .arg(&trace)
        .output()
        .expect("binary runs");

    assert!(out.status.success(), "exit ok: {:?}", out.status);
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines[0],
        "idx,annotation,stage,name,ts,dur,end_ts,stream,device,grid,block,shared_memory,registers_per_thread,correlation"
    );
    // gemm@100 is inside the decode annotation window [90,140).
    assert_eq!(
        lines[1],
        "1,execute_context_0(0)_generation_1(1),decode,gemm,100,5,105,4,0,,,,,"
    );
    // add@400 is outside any annotation window -> empty annotation + stage.
    assert_eq!(lines[2], "2,,,add,400,5,405,4,0,,,,,");

    std::fs::remove_dir_all(&dir).ok();
}
