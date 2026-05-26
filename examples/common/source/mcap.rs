//! MCAP file as a [`FrameSource`].
//!
//! Reads a bubbaloop `mcap-recorder` session and yields grayscale frames from
//! one chosen channel. The default channel suffix is `mono_left` — already
//! grayscale at 640×400, no color conversion needed. Each MCAP message is a
//! CBOR-wrapped SDK envelope `{header, body:{width, height, encoding:"jpeg", data}}`;
//! we strip the envelope, JPEG-decode `body.data` via `kornia-io`, and (for
//! the color `compressed` channel) convert to luma8 via `kornia-imgproc`.

use std::fs;
use std::path::Path;

use ciborium::value::Value as CborValue;
use kornia_3d::camera::PinholeCamera;
use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::jpeg::{
    decode_image_jpeg_layout, decode_image_jpeg_mono8, decode_image_jpeg_rgb8,
};
use kornia_tensor::CpuAllocator;
use mcap::McapError;

use super::{FrameItem, FrameSource, SourceError};

/// A pre-decoded grayscale frame queued for `next_frame`.
struct PreparedFrame {
    timestamp_sec: f64,
    image: Image<u8, 1, CpuAllocator>,
}

/// Offline MCAP frame source.
pub struct McapSource {
    frames: std::vec::IntoIter<PreparedFrame>,
    n_total: usize,
    cursor: usize,
    camera: PinholeCamera,
}

impl McapSource {
    /// Open `path`, pick the channel whose topic ends in `/<channel_suffix>`,
    /// JPEG-decode every message to luma8, and stash the result.
    ///
    /// `start_frame` and `max_frames` mirror `EurocSource`: skip the first
    /// `start_frame`, then yield up to `max_frames` (0 = until exhausted).
    pub fn open(
        path: &Path,
        channel_suffix: &str,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        // Read the whole file into memory. mcap::MessageStream borrows from
        // a byte slice; for our recordings (a few hundred MB at most) this is
        // simpler than memory-mapping and the parsing is still streaming on
        // top of the buffer. Switch to memmap2 if the recordings grow huge.
        let bytes = fs::read(path).map_err(SourceError::Io)?;
        let mut decoded: Vec<PreparedFrame> = Vec::new();
        let mut first_size: Option<(usize, usize)> = None;

        let want = format!("/{channel_suffix}");
        let stream = mcap::MessageStream::new(&bytes).map_err(map_mcap)?;
        for msg_result in stream {
            let msg = msg_result.map_err(map_mcap)?;
            if !msg.channel.topic.ends_with(&want) {
                continue;
            }
            let envelope: CborValue =
                ciborium::from_reader(msg.data.as_ref()).map_err(SourceError::other)?;
            let body = unwrap_body(&envelope).ok_or_else(|| {
                SourceError::other(format!(
                    "MCAP message on {} is not a {{header, body}} envelope",
                    msg.channel.topic
                ))
            })?;
            let jpeg_bytes = extract_jpeg(body).ok_or_else(|| {
                SourceError::other(format!(
                    "MCAP message on {} has no JPEG payload (expected encoding=jpeg + data)",
                    msg.channel.topic
                ))
            })?;

            let image = decode_jpeg_to_luma(jpeg_bytes).map_err(SourceError::other)?;
            let (got_w, got_h) = (image.width(), image.height());
            if let Some(prev) = first_size {
                if prev != (got_w, got_h) {
                    return Err(SourceError::other(format!(
                        "frame size changed mid-stream: {prev:?} → {got_w}x{got_h}"
                    )));
                }
            } else {
                first_size = Some((got_w, got_h));
            }
            decoded.push(PreparedFrame {
                timestamp_sec: msg.log_time as f64 / 1e9,
                image,
            });
        }

        if decoded.is_empty() {
            return Err(SourceError::other(format!(
                "no messages found on a channel ending with /{channel_suffix}"
            )));
        }

        let (w, h) = first_size.expect("first_size set when decoded non-empty");

        // Trim using start_frame / max_frames.
        let start = start_frame.min(decoded.len());
        let mut trimmed: Vec<PreparedFrame> = decoded.into_iter().skip(start).collect();
        if max_frames > 0 && trimmed.len() > max_frames {
            trimmed.truncate(max_frames);
        }
        let n_total = trimmed.len();

        // Re-base timestamps to start at 0 — easier to read in logs and
        // matches what live sources (oakd/uvc) do via `Instant::elapsed`.
        if let Some(t0) = trimmed.first().map(|f| f.timestamp_sec) {
            for f in trimmed.iter_mut() {
                f.timestamp_sec -= t0;
            }
        }

        // Placeholder intrinsics. OAK-D mono is 1280×800 native @ fx≈fy≈880;
        // our recorded mono is 640×400 so scale=0.5.
        let scale = w as f64 / 1280.0;
        let camera = PinholeCamera {
            fx: 880.0 * scale,
            fy: 880.0 * scale,
            cx: w as f64 * 0.5,
            cy: h as f64 * 0.5,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        eprintln!(
            "[mcap] {}: {n_total} frames @ {w}x{h} from /{channel_suffix} \
             (placeholder intrinsics fx={:.2} fy={:.2})",
            path.display(),
            camera.fx,
            camera.fy,
        );

        Ok(Self {
            frames: trimmed.into_iter(),
            n_total,
            cursor: 0,
            camera,
        })
    }
}

impl FrameSource for McapSource {
    fn camera(&self) -> PinholeCamera {
        self.camera.clone()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.n_total)
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        let Some(prepared) = self.frames.next() else {
            return Ok(None);
        };
        let idx = self.cursor;
        self.cursor += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec: prepared.timestamp_sec,
            image: prepared.image,
        }))
    }
}

/// Decode a JPEG payload to a luma8 image. Mono JPEGs are decoded directly;
/// RGB JPEGs (used by the `compressed` channel) are decoded then converted.
fn decode_jpeg_to_luma(
    jpeg_bytes: &[u8],
) -> Result<Image<u8, 1, CpuAllocator>, Box<dyn std::error::Error + Send + Sync>> {
    let layout = decode_image_jpeg_layout(jpeg_bytes)?;
    let size = layout.image_size;
    match layout.channels {
        1 => {
            let mut dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
            decode_image_jpeg_mono8(jpeg_bytes, &mut dst)?;
            Ok(dst)
        }
        3 => {
            let mut rgb = Image::<u8, 3, _>::from_size_val(size, 0, CpuAllocator)?;
            decode_image_jpeg_rgb8(jpeg_bytes, &mut rgb)?;
            let mut gray = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
            gray_from_rgb_u8(&rgb, &mut gray)?;
            Ok(gray)
        }
        n => Err(format!("unsupported JPEG channel count: {n} (want 1 or 3)").into()),
    }
}

/// `{header, body}` envelope → body. Returns None if the value isn't a map
/// containing a `body` key.
fn unwrap_body(value: &CborValue) -> Option<&CborValue> {
    if let CborValue::Map(entries) = value {
        for (k, v) in entries {
            if let CborValue::Text(s) = k {
                if s == "body" {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Pull JPEG bytes out of either a flat `{encoding:"jpeg", data}` body or
/// the nested `{rgb:{encoding:"jpeg", data}}` shape from the `compressed`
/// channel.
fn extract_jpeg(body: &CborValue) -> Option<&[u8]> {
    let CborValue::Map(entries) = body else {
        return None;
    };
    let mut encoding: Option<&str> = None;
    let mut data: Option<&[u8]> = None;
    let mut rgb_sub: Option<&CborValue> = None;
    for (k, v) in entries {
        let CborValue::Text(key) = k else { continue };
        match key.as_str() {
            "encoding" => encoding = cbor_text(v),
            "data" => data = cbor_bytes(v),
            "rgb" => rgb_sub = Some(v),
            _ => {}
        }
    }
    if let (Some(enc), Some(d)) = (encoding, data) {
        if enc == "jpeg" {
            return Some(d);
        }
    }
    if let Some(rgb) = rgb_sub {
        return extract_jpeg(rgb);
    }
    None
}

fn cbor_text(v: &CborValue) -> Option<&str> {
    match v {
        CborValue::Text(s) => Some(s.as_str()),
        _ => None,
    }
}

fn cbor_bytes(v: &CborValue) -> Option<&[u8]> {
    match v {
        CborValue::Bytes(b) => Some(b.as_slice()),
        _ => None,
    }
}

fn map_mcap(err: McapError) -> SourceError {
    SourceError::other(err)
}
