//! MCP (Model Context Protocol) server exposing read-only, bounded
//! trace-analysis tools over stdio, so an AI assistant can inspect PyTorch
//! profiler traces without a human driving the TUI.
//!
//! Every tool delegates to the pure cores in [`crate::mcp_core`], which enforce
//! `limit`/`offset` bounds — traces are 800MB+ / millions of kernels, so an
//! unbounded dump would overflow the caller's context.

use crate::debug_log;
use crate::mcp_core;
use crate::mcp_core::Stage;
use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{tool, tool_router, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct TraceMcpServer;

/// Logs one summarized argument under a request. Bodies are never passed here —
/// only short scalars (path, stream, name, stage, offset/limit) — so huge/sensitive
/// payloads stay out of the log.
fn log_arg(label: &str, value: &str) {
    debug_log::log_indented(&format!("{label}: {value}"));
}

fn log_loaded(path: &str, trace: &crate::trace::Trace) {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    debug_log::log_line(&format!("Loading trace: {}", debug_log::human_size(size)));
    debug_log::log_line(&format!(
        "Parsed {} kernels, {} annotations",
        trace.kernels.len(),
        trace.annotations.len()
    ));
}

fn log_response(tool: &str, summary: &str) {
    debug_log::log_line(&format!("MCP response: {tool} \u{2192} {summary}"));
}

fn log_error(tool: &str, msg: &str) {
    debug_log::log_line(&format!("MCP error: {tool} \u{2192} {msg}"));
}

fn load_logged(tool: &str, path: &str) -> Result<crate::trace::Trace, String> {
    match mcp_core::load(path) {
        Ok(trace) => {
            log_loaded(path, &trace);
            Ok(trace)
        }
        Err(e) => {
            let msg = e.to_string();
            log_error(tool, &msg);
            Err(msg)
        }
    }
}

/// Maps an optional stage filter string to `Option<Stage>`, rejecting values
/// other than prefill/decode/mixed. Absent means "no filter" (all stages).
fn parse_stage_filter(stage: &Option<String>) -> Result<Option<Stage>, String> {
    stage.as_deref().map(Stage::parse_arg).transpose()
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PathParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StageSummaryParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
    /// CUDA stream id to aggregate per-stage stats for.
    pub stream: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LaneParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
    /// CUDA stream id whose kernel lane to export.
    pub stream: u64,
    /// Rows to skip from the start of the lane. Defaults to 0.
    #[serde(default)]
    pub offset: usize,
    /// Max rows to return (capped by the server). Defaults to 100.
    pub limit: Option<usize>,
    /// Optional stage filter: prefill, decode, or mixed. Omit for all kernels.
    pub stage: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SequenceParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
    /// CUDA stream id to scan.
    pub stream: u64,
    /// Kernel name that starts the sequence.
    pub kernel_name: String,
    /// Rows to skip from the start of the sequence. Defaults to 0.
    #[serde(default)]
    pub offset: usize,
    /// Max rows to return (capped by the server). Defaults to 100.
    pub limit: Option<usize>,
    /// Optional stage filter: prefill, decode, or mixed. Omit for all kernels.
    pub stage: Option<String>,
}

#[tool_router(server_handler)]
impl TraceMcpServer {
    #[tool(
        name = "list_traces",
        description = "List PyTorch profiler traces (*.pt.trace.json.gz) in the current directory."
    )]
    fn list_traces(&self) -> Result<String, String> {
        debug_log::log_line("MCP request: tools/call \u{2192} list_traces");
        match mcp_core::list_traces_in_dir(".") {
            Ok(names) => {
                log_response("list_traces", &format!("{} traces", names.len()));
                Ok(names.join("\n"))
            }
            Err(e) => {
                let msg = e.to_string();
                log_error("list_traces", &msg);
                Err(msg)
            }
        }
    }

    #[tool(
        name = "summary",
        description = "Aggregate overview of a trace: kernel/annotation counts, streams, and total duration. Never returns per-kernel rows."
    )]
    fn summary(&self, Parameters(p): Parameters<PathParam>) -> Result<String, String> {
        debug_log::log_line("MCP request: tools/call \u{2192} summary");
        log_arg("trace", &p.path);
        let trace = load_logged("summary", &p.path)?;
        let s = mcp_core::summary(&trace);
        match serde_json::to_string_pretty(&s) {
            Ok(out) => {
                log_response(
                    "summary",
                    &format!("{} kernels, {} streams", s.kernel_count, s.streams.len()),
                );
                Ok(out)
            }
            Err(e) => {
                let msg = e.to_string();
                log_error("summary", &msg);
                Err(msg)
            }
        }
    }

    #[tool(
        name = "lane_kernels_csv",
        description = "CSV of kernels on one CUDA stream, including the covering annotation and derived prefill/decode/mixed stage. Optional `stage` filter (prefill|decode|mixed) returns only matching kernels. Paginated by offset/limit."
    )]
    fn lane_kernels_csv(
        &self,
        Parameters(p): Parameters<LaneParam>,
    ) -> Result<String, String> {
        debug_log::log_line("MCP request: tools/call \u{2192} lane_kernels_csv");
        log_arg("trace", &p.path);
        log_arg("stream", &p.stream.to_string());
        if let Some(stage) = &p.stage {
            log_arg("stage", stage);
        }
        let stage = match parse_stage_filter(&p.stage) {
            Ok(s) => s,
            Err(msg) => {
                log_error("lane_kernels_csv", &msg);
                return Err(msg);
            }
        };
        let trace = load_logged("lane_kernels_csv", &p.path)?;
        let csv = mcp_core::lane_kernels_csv_for(&trace, p.stream, p.offset, p.limit, stage);
        log_response("lane_kernels_csv", &debug_log::summarize_rows(&csv));
        Ok(csv)
    }

    #[tool(
        name = "kernel_sequence",
        description = "Tab-separated kernel sequence on a stream from the first kernel of the given name up to the next occurrence of that name. Optional `stage` filter (prefill|decode|mixed) restricts rows to that stage. Paginated by offset/limit."
    )]
    fn kernel_sequence(
        &self,
        Parameters(p): Parameters<SequenceParam>,
    ) -> Result<String, String> {
        debug_log::log_line("MCP request: tools/call \u{2192} kernel_sequence");
        log_arg("trace", &p.path);
        log_arg("stream", &p.stream.to_string());
        log_arg("kernel", &p.kernel_name);
        if let Some(stage) = &p.stage {
            log_arg("stage", stage);
        }
        let stage = match parse_stage_filter(&p.stage) {
            Ok(s) => s,
            Err(msg) => {
                log_error("kernel_sequence", &msg);
                return Err(msg);
            }
        };
        let trace = load_logged("kernel_sequence", &p.path)?;
        let seq = mcp_core::kernel_sequence_for(
            &trace,
            p.stream,
            &p.kernel_name,
            p.offset,
            p.limit,
            stage,
        );
        log_response(
            "kernel_sequence",
            &format!("{} kernels", debug_log::count_rows(&seq)),
        );
        Ok(seq)
    }

    #[tool(
        name = "stage_summary",
        description = "Per-stage aggregate stats for one CUDA stream: kernel count, total duration, and median duration for each of prefill/decode/mixed/none present on the stream."
    )]
    fn stage_summary(
        &self,
        Parameters(p): Parameters<StageSummaryParam>,
    ) -> Result<String, String> {
        debug_log::log_line("MCP request: tools/call \u{2192} stage_summary");
        log_arg("trace", &p.path);
        log_arg("stream", &p.stream.to_string());
        let trace = load_logged("stage_summary", &p.path)?;
        let out = mcp_core::stage_summary_for(&trace, p.stream);
        log_response(
            "stage_summary",
            &format!("{} stages", debug_log::count_rows(&out)),
        );
        Ok(out)
    }
}

/// Runs the MCP server over stdio until the client disconnects.
pub async fn serve() -> Result<()> {
    let service = TraceMcpServer.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
