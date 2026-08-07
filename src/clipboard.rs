use std::io::Write;

pub struct CopyOutcome {
    pub via_native: bool,
    pub via_osc52: bool,
    pub file_path: std::path::PathBuf,
}

pub struct ClipboardManager {
    native: Option<arboard::Clipboard>,
}

impl ClipboardManager {
    pub fn new() -> Self {
        Self {
            native: arboard::Clipboard::new().ok(),
        }
    }

    /// Copy `text` to the system clipboard. arboard is tried first; OSC 52 is
    /// emitted to `out` as a terminal fallback. A CSV file is ALWAYS written to
    /// the current working directory so the data is recoverable even when both
    /// clipboard paths silently no-op.
    pub fn copy<W: Write>(&mut self, text: &str, out: &mut W) -> std::io::Result<CopyOutcome> {
        let via_native = self
            .native
            .as_mut()
            .and_then(|c| c.set_text(text.to_owned()).ok())
            .is_some();

        let via_osc52 = self.emit_osc52(text, out).is_ok();

        let dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let file_path = dir.join(format!(
            "kernel-sequence-{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        std::fs::write(&file_path, text)?;

        Ok(CopyOutcome {
            via_native,
            via_osc52,
            file_path,
        })
    }

    fn emit_osc52<W: Write>(&self, text: &str, out: &mut W) -> std::io::Result<()> {
        let encoded = base64_encode(text.as_bytes());
        write!(out, "\x1b]52;c;{}\x07", encoded)?;
        out.flush()
    }
}

impl Default for ClipboardManager {
    fn default() -> Self {
        Self::new()
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        out.push(B64[((n >> 18) & 63) as usize] as char);
        out.push(B64[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn copy_always_writes_file_and_emits_osc52() {
        let mut mgr = ClipboardManager { native: None };
        let mut buf: Vec<u8> = Vec::new();
        let outcome = mgr.copy("idx,name\n1,foo\n", &mut buf).unwrap();

        assert!(outcome.via_osc52);
        assert!(outcome.file_path.exists());
        let written = std::fs::read_to_string(&outcome.file_path).unwrap();
        assert_eq!(written, "idx,name\n1,foo\n");

        let emitted = String::from_utf8(buf).unwrap();
        assert!(emitted.starts_with("\x1b]52;c;"));
        assert!(emitted.ends_with('\x07'));

        let _ = std::fs::remove_file(&outcome.file_path);
    }
}
