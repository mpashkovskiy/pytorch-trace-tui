# pytorch-trace-tui

A terminal UI for exploring **GPU kernels** in PyTorch profiler traces, rendered
Perfetto-style as horizontal timeline lanes — one lane per CUDA stream, all
sharing a single time axis.

Reads Chrome-trace JSON produced by `torch.profiler`, filters out everything
except GPU kernels (`kernel`, `gpu_memcpy`, `gpu_memset`), and lets you scrub and
zoom the timeline directly in your terminal. Handles both plain `.json` and
gzipped `.json.gz` traces via streaming deserialization, so 800 MB+ traces load
in seconds.

## Install

```bash
cargo install --path .
```

or build locally:

```bash
cargo build --release
```

## Usage

```bash
# Open a specific trace
pytorch-trace-tui my_trace.pt.trace.json.gz

# Or run with no argument to scan the current directory
# for *.pt.trace.json.gz and pick one interactively
pytorch-trace-tui
```

## Controls

| Key             | Action                          |
| --------------- | ------------------------------- |
| `A` / `←`       | Select previous kernel          |
| `D` / `→`       | Select next kernel              |
| `W` / `↑`       | Zoom in                         |
| `S` / `↓`       | Zoom out                        |
| `Tab`           | Next GPU stream                 |
| `Shift+Tab`     | Previous GPU stream             |
| `Q` / `Esc`     | Quit                            |

The bottom panel shows full details for the selected kernel: name, stream,
device, start/end/duration, grid/block dimensions, shared memory, registers per
thread, and correlation id.

Zoom operates on a shared time window across **all** streams — when you zoom in
on a busy stream, other lanes show what they were doing in that same time slice
(or appear empty if they were idle then), just like Perfetto.

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

## License

MIT — see [LICENSE](LICENSE).
