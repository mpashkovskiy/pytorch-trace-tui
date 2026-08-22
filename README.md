# pytorch-trace-tui

![](demo.gif)

A terminal UI for exploring **GPU kernels** in PyTorch profiler traces, rendered
Perfetto-style as horizontal timeline lanes — one lane per CUDA stream, all
sharing a single time axis.

Reads Chrome-trace JSON produced by `torch.profiler`, filters out everything
except GPU kernels (`kernel`, `gpu_memcpy`, `gpu_memset`), and lets you scrub and
zoom the timeline directly in your terminal. Handles both plain `.json` and
gzipped `.json.gz` traces via streaming deserialization, so 800 MB+ traces load
in seconds.

## Install

Download the latest release and run:

```bash
wget https://github.com/mpashkovskiy/pytorch-trace-tui/releases/download/latest/pytorch-trace-tui-x86_64-unknown-linux-musl
chmod +x pytorch-trace-tui-x86_64-unknown-linux-musl
# make sure you run from folder with traces
./pytorch-trace-tui-x86_64-unknown-linux-musl
```

The released binary is statically linked against musl, so it bundles libc and
runs on any Linux regardless of the host glibc version — no more
`version `GLIBC_2.39' not found` on older distros.

Or build and install from source

```bash
git clone https://github.com/mpashkovskiy/pytorch-trace-tui.git
rustup target add x86_64-unknown-linux-musl   # once
cargo build --release
cargo install --path .
```

Local builds default to the statically-linked `x86_64-unknown-linux-musl` target
(via `.cargo/config.toml`), so the binary you build bundles libc and is just as
portable as the released one.

## Usage

```bash
# Open a specific trace
pytorch-trace-tui my_trace.pt.trace.json.gz

# Overlay several traces to compare them side by side
pytorch-trace-tui baseline.pt.trace.json.gz tuned.pt.trace.json.gz

# Or run with no argument to scan the current directory for
# *.pt.trace.json.gz and pick one — or several — interactively.
# At the picker, enter one number, or several ("1 3" or "1,3") to overlay.
pytorch-trace-tui
```

## Controls

| Key             | Action                          |
| --------------- | ------------------------------- |
| `A` / `←`       | Select previous item in the current lane |
| `D` / `→`       | Select next item in the current lane |
| `W` / `↑`       | Zoom in                         |
| `S` / `↓`       | Zoom out                        |
| `Tab`           | Next lane                       |
| `Shift+Tab`     | Previous lane                   |
| `/`             | Incremental name search (kernels and annotations); `Enter` / `Shift+Enter` cycle to the next / previous match, `Tab` keeps the current match, `Esc` cancels |
| `N`             | Show the kernel sequence from the selected kernel (see below) |
| `E`             | Export all kernels of the current lane to a CSV file in the working directory |
| `G`             | Toggle diff / normal alignment (two traces only) |
| `Q`             | Quit                            |

You can also use the **mouse**: left-click a kernel or annotation to select it
(clicking empty space in a lane selects the nearest item, and clicking a lane's
label just switches to that lane), and scroll the wheel to zoom in and out.

In the **sequence popup** (opened with `N`):

| Key                 | Action                              |
| ------------------- | ----------------------------------- |
| `↑` / `↓`           | Scroll one row                      |
| `PageUp` / `PageDn` | Scroll one page                     |
| `Y`                 | Copy the sequence (tab-separated) to the clipboard |
| `Esc` / `N`         | Close the popup                     |

Every timeline row is a **lane**. A stream contributes a kernel lane, and — if it
has `gpu_user_annotation` events — an annotation lane rendered directly above it,
sharing the same time axis. All lanes behave identically: `Tab`/`Shift+Tab` move
between lanes (annotation lane, then kernel lane, then the next stream), and
`A`/`D` step through whichever lane is selected. When you switch lanes the
selection jumps to the item nearest in time, so you stay in the same region of
the timeline. Search spans both kernels and annotations.

The bottom panel shows full details for the selected item — a kernel (name,
stream, device, start/end/duration, grid/block dimensions, shared memory,
registers per thread, correlation id) or an annotation (name, stream, timing).

Zoom operates on a shared time window across **all** streams — when you zoom in
on a busy stream, other lanes show what they were doing in that same time slice
(or appear empty if they were idle then), just like Perfetto.

## Comparing traces

Open two or more traces at once to overlay them on one shared time axis. Lanes
are interleaved by row — lane 1 of every trace, then lane 2 of every trace, and
so on — and each lane is labelled with the part of its filename that differs from
the others (e.g. `baseline` vs `tuned`).

Because independently captured traces have unrelated absolute timestamps, every
trace is **zero-based** to the earliest event across all traces, so they all
start at the same left edge with their internal timing preserved.

### Two traces — diff / normal toggle

When you open **exactly two** traces they start in **diff** mode. Press `G` to
toggle between **diff** and **normal**; the header shows the current mode.

**Diff mode** aligns the traces by a `git diff`-style (Myers) comparison of each
shared stream's **kernel-name sequence**:

- The **first trace is the anchor**. It keeps its real kernel positions and the
  idle gaps between them — except that a new gap is inserted wherever a kernel
  appears only in the second trace, and everything after it shifts right by the
  width of that inserted run.
- **Matched kernels** in the second trace snap their start onto the anchor
  kernel's position, while keeping their **own duration** — so duration
  differences (exactly what you are comparing) stay visible.
- **Second-trace-only kernels** land in the inserted anchor gap; the anchor lane
  shows empty space there.
- **First-trace-only kernels** stay put; the second trace simply has nothing at
  that position.
- **Annotations** are carried along the same remap, so each `gpu_user_annotation`
  span follows the kernels it brackets.

Diff mode also **colour-codes** the kernels: **added** kernels (only in the second
trace) are green, **deleted** kernels (only in the anchor) are red, and matched
kernels are dimmed so the differences stand out.

**Normal mode** skips the diff and just zero-bases both traces to the common
start, showing each trace's raw timing side by side.

### Three or more traces

Three or more traces are always rendered in **normal** mode — zero-based to the
common start, no diff. `G` has no effect.

## Kernel sequences

Press `N` on a kernel to capture the sequence of kernels from it up to (but not
including) the next kernel with the same name. The popup lists each kernel with a
per-position **median** duration computed across every repetition of that block
in the lane, so you can see the typical cost of a repeated phase at a glance.
Press `Y` to copy the table (tab-separated, paste-ready into a spreadsheet).

## Trace format

Generate a compatible trace from PyTorch:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # ... run your model ...
    pass

prof.export_chrome_trace("my_trace.pt.trace.json.gz")
```

## MCP server

The same binary can run as a [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio, so an AI assistant (Claude Code, Claude Desktop, …) can
inspect traces directly instead of you scrubbing the TUI and pasting CSVs into
the chat:

```bash
pytorch-trace-tui --mcp
```

It exposes five read-only tools. Every list-returning tool is bounded by
`offset`/`limit` (default 100, hard cap 1000) so a multi-gigabyte trace can't
overflow the model's context:

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_traces` | — | `*.pt.trace.json.gz` in the working directory |
| `summary` | `path` | kernel/annotation counts, streams, duration, and a per-stage (prefill/decode/mixed/none) breakdown |
| `lane_kernels_csv` | `path`, `stream`, `offset?`, `limit?`, `stage?` | per-stream kernel CSV incl. the `annotation` and `stage` columns; optional `stage` (`prefill`/`decode`/`mixed`) returns only matching kernels |
| `kernel_sequence` | `path`, `stream`, `kernel_name`, `offset?`, `limit?`, `stage?` | the kernel block from a named kernel up to its next occurrence; optional `stage` filter |
| `stage_summary` | `path`, `stream` | per-stage aggregate stats for a stream (kernel count, total and median duration) |

The `stage` filters and `stage_summary` let you analyse **prefill** and
**decode** phases separately — e.g. "give me only the decode kernels on stream
4" returns a bounded, pre-filtered result.

### Claude Code

```bash
claude mcp add pytorch-trace-tui -- /absolute/path/to/pytorch-trace-tui --mcp
```

Everything after `--` is the command Claude launches. Use `-s user` to register
it for every project on the machine instead of just the current one:

```bash
claude mcp add -s user pytorch-trace-tui -- /absolute/path/to/pytorch-trace-tui --mcp
```

### Claude Desktop

Add the server to `claude_desktop_config.json`:

- Linux: `~/.config/Claude/claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pytorch-trace-tui": {
      "command": "/absolute/path/to/pytorch-trace-tui",
      "args": ["--mcp"],
      "cwd": "/path/to/your/traces"
    }
  }
}
```

Fully quit and reopen Claude Desktop after editing — the config is only read on
startup.

### Working directory

`list_traces` scans the **current directory** for `*.pt.trace.json.gz`, so run
the server from your traces folder (Claude Code launches it in the directory
where you started `claude`; Claude Desktop uses the `cwd` above). The other
tools take an explicit `path`, so they work regardless of the working directory.

## License

MIT — see [LICENSE](LICENSE).
