//! MCP (Model Context Protocol) server exposing read-only, bounded
//! trace-analysis tools over stdio, so an AI assistant can inspect PyTorch
//! profiler traces without a human driving the TUI.
//!
//! Every tool delegates to the pure cores in [`crate::mcp_core`], which enforce
//! `limit`/`offset` bounds — traces are 800MB+ / millions of kernels, so an
//! unbounded dump would overflow the caller's context.

use crate::mcp_core;
use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{tool, tool_router, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct TraceMcpServer;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PathParam {
    /// Path to a `.pt.trace.json` or `.pt.trace.json.gz` file.
    pub path: String,
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
}

#[tool_router(server_handler)]
impl TraceMcpServer {
    #[tool(
        name = "list_traces",
        description = "List PyTorch profiler traces (*.pt.trace.json.gz) in the current directory."
    )]
    fn list_traces(&self) -> Result<String, String> {
        let names = mcp_core::list_traces_in_dir(".").map_err(|e| e.to_string())?;
        Ok(names.join("\n"))
    }

    #[tool(
        name = "summary",
        description = "Aggregate overview of a trace: kernel/annotation counts, streams, and total duration. Never returns per-kernel rows."
    )]
    fn summary(&self, Parameters(p): Parameters<PathParam>) -> Result<String, String> {
        let trace = mcp_core::load(&p.path).map_err(|e| e.to_string())?;
        let s = mcp_core::summary(&trace);
        serde_json::to_string_pretty(&s).map_err(|e| e.to_string())
    }

    #[tool(
        name = "lane_kernels_csv",
        description = "CSV of kernels on one CUDA stream, including the covering annotation and derived prefill/decode/mixed stage. Paginated by offset/limit."
    )]
    fn lane_kernels_csv(
        &self,
        Parameters(p): Parameters<LaneParam>,
    ) -> Result<String, String> {
        let trace = mcp_core::load(&p.path).map_err(|e| e.to_string())?;
        Ok(mcp_core::lane_kernels_csv_for(
            &trace, p.stream, p.offset, p.limit,
        ))
    }

    #[tool(
        name = "kernel_sequence",
        description = "Tab-separated kernel sequence on a stream from the first kernel of the given name up to the next occurrence of that name. Paginated by offset/limit."
    )]
    fn kernel_sequence(
        &self,
        Parameters(p): Parameters<SequenceParam>,
    ) -> Result<String, String> {
        let trace = mcp_core::load(&p.path).map_err(|e| e.to_string())?;
        Ok(mcp_core::kernel_sequence_for(
            &trace,
            p.stream,
            &p.kernel_name,
            p.offset,
            p.limit,
        ))
    }
}

/// Runs the MCP server over stdio until the client disconnects.
pub async fn serve() -> Result<()> {
    let service = TraceMcpServer.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
