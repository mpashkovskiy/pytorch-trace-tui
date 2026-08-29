Set FontSize 13
Set Width 1024
Set Height 380
Set Padding 10

# --- launch: pick one trace from the recursive picker ---
Sleep 800ms
Type@15ms "target/x86_64-unknown-linux-musl/release/pytorch-trace-tui"
Enter
Sleep 500ms
Type "1"
Enter
Sleep 1500ms

# --- zoom into a busy region (W = in) ---
Type@90ms "wwwwwwwwwwwwwwww"
Sleep 800ms

# --- incremental name search (/) ---
Type "/"
Sleep 1s
Type@120ms "moe"
Sleep 1s
Type "_"
Sleep 1s
Type@120ms "gemm"
Sleep 1s
Escape
Sleep 500ms

# --- zoom in further, then step through kernels (A/D) ---
Type@90ms "wwww"
Sleep 800ms
Type@500ms "aa"
Sleep 500ms
Type@500ms "ddda"
Sleep 1500ms

# --- kernel sequence popup (N), scroll it, copy (Y) ---
Type "n"
Sleep 1200ms
Down@200ms 3
Sleep 800ms
Type "y"
Sleep 1500ms
Escape
Sleep 500ms

# --- quit, then preview the exported kernel-sequence CSV ---
Type "q"
Sleep 800ms
Type@40ms "head -5 kernel-sequence-*.csv"
Sleep 400ms
Enter
Sleep 2500ms
