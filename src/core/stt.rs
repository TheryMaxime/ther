use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::core::Callback;

const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// How often the worker re-transcribes the accumulated audio.
const TRANSCRIBE_INTERVAL: Duration = Duration::from_secs(2);
/// Minimum amount of audio (in 16 kHz samples) before running Whisper.
const MIN_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize; // ~1 second

/// Owns the live microphone stream and the background transcription worker.
///
/// Dropping (via [`Recorder::stop`]) signals the worker to finish and releases
/// the audio device.
pub struct Recorder {
    stop_flag: Arc<AtomicBool>,
    stream: Option<cpal::Stream>,
    worker: Option<JoinHandle<()>>,
}

impl Recorder {
    /// Start capturing the default input device and spawn the Whisper worker.
    ///
    /// Transcript updates are delivered to `on_transcript`, status changes to
    /// `on_status`, and (for the LLM worker) full transcripts on `transcript_tx`.
    pub fn start(
        model: PathBuf,
        language: String,
        on_transcript: Callback,
        on_status: Callback,
        transcript_tx: Sender<String>,
    ) -> Result<Recorder, String> {
        if !Path::new(&model).exists() {
            return Err(format!(
                "Whisper model not found at {}. Run scripts/download-model.sh",
                model.display()
            ));
        }

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| "No input (microphone) device available".to_string())?;
        let config = device
            .default_input_config()
            .map_err(|e| format!("Failed to get input config: {e}"))?;

        let input_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();

        // Mono samples at the device's native rate; resampled to 16 kHz by the worker.
        let raw: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let stop_flag = Arc::new(AtomicBool::new(false));

        let stream = build_stream(&device, &config.into(), sample_format, channels, raw.clone())?;
        stream
            .play()
            .map_err(|e| format!("Failed to start audio stream: {e}"))?;

        let worker = {
            let raw = raw.clone();
            let stop_flag = stop_flag.clone();
            std::thread::spawn(move || {
                run_worker(
                    model,
                    language,
                    raw,
                    input_rate,
                    stop_flag,
                    on_transcript,
                    on_status,
                    transcript_tx,
                );
            })
        };

        Ok(Recorder {
            stop_flag,
            stream: Some(stream),
            worker: Some(worker),
        })
    }

    /// Stop capturing and wait for the worker to emit its final transcript.
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        // Dropping the stream stops audio capture.
        self.stream.take();
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    channels: usize,
    raw: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, String> {
    let err_fn = |e| eprintln!("Audio stream error: {e}");

    let stream = match format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &_| push_mono(&raw, data, channels, |s| s),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &_| {
                push_mono(&raw, data, channels, |s| s as f32 / i16::MAX as f32)
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _: &_| {
                push_mono(&raw, data, channels, |s| {
                    (s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)
                })
            },
            err_fn,
            None,
        ),
        other => return Err(format!("Unsupported sample format: {other:?}")),
    };

    stream.map_err(|e| format!("Failed to build input stream: {e}"))
}

/// Down-mix interleaved frames to mono f32 and append to the shared buffer.
fn push_mono<T: Copy>(
    raw: &Arc<Mutex<Vec<f32>>>,
    data: &[T],
    channels: usize,
    to_f32: impl Fn(T) -> f32,
) {
    if channels == 0 {
        return;
    }
    let mut buf = match raw.lock() {
        Ok(b) => b,
        Err(_) => return,
    };
    for frame in data.chunks(channels) {
        let sum: f32 = frame.iter().map(|&s| to_f32(s)).sum();
        buf.push(sum / channels as f32);
    }
}

fn run_worker(
    model: PathBuf,
    language: String,
    raw: Arc<Mutex<Vec<f32>>>,
    input_rate: u32,
    stop_flag: Arc<AtomicBool>,
    on_transcript: Callback,
    on_status: Callback,
    transcript_tx: Sender<String>,
) {
    let ctx = match WhisperContext::new_with_params(
        &model.to_string_lossy(),
        WhisperContextParameters::default(),
    ) {
        Ok(c) => c,
        Err(e) => {
            on_status(format!("Failed to load Whisper model: {e}"));
            return;
        }
    };
    let mut state = match ctx.create_state() {
        Ok(s) => s,
        Err(e) => {
            on_status(format!("Failed to init Whisper state: {e}"));
            return;
        }
    };

    let mut last_sent = String::new();
    loop {
        let stopping = stop_flag.load(Ordering::SeqCst);

        if !stopping {
            // Sleep in short slices so Stop is responsive.
            let mut waited = Duration::ZERO;
            while waited < TRANSCRIBE_INTERVAL && !stop_flag.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
                waited += Duration::from_millis(100);
            }
        }

        let snapshot = { raw.lock().map(|b| b.clone()).unwrap_or_default() };
        let audio = resample_to_16k(&snapshot, input_rate);

        if audio.len() >= MIN_SAMPLES {
            if let Some(text) = transcribe(&mut state, &audio, &language) {
                on_transcript(text.clone());
                // Feed the LLM worker only when the transcript actually changed.
                if !text.trim().is_empty() && text != last_sent {
                    last_sent = text.clone();
                    let _ = transcript_tx.send(text);
                }
            }
        }

        if stopping {
            break;
        }
    }
}

fn transcribe(state: &mut whisper_rs::WhisperState, audio: &[f32], language: &str) -> Option<String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(language));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    if let Err(e) = state.full(params, audio) {
        eprintln!("Whisper transcription failed: {e}");
        return None;
    }

    let n = state.full_n_segments().unwrap_or(0);
    let mut out = String::new();
    for i in 0..n {
        if let Ok(seg) = state.full_get_segment_text(i) {
            out.push_str(seg.trim());
            out.push(' ');
        }
    }
    Some(out.trim().to_string())
}

/// Linear-interpolation resample from `in_rate` to 16 kHz.
fn resample_to_16k(input: &[f32], in_rate: u32) -> Vec<f32> {
    if input.is_empty() || in_rate == WHISPER_SAMPLE_RATE {
        return input.to_vec();
    }
    let ratio = WHISPER_SAMPLE_RATE as f64 / in_rate as f64;
    let out_len = (input.len() as f64 * ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}
