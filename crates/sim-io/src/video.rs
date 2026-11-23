use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VideoPipelineError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("WAV error: {0}")]
    Wav(#[from] hound::Error),
    #[error("External tool '{tool}' failed: {message}")]
    ProcessFailed { tool: String, message: String },
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, VideoPipelineError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioSlowmoMode {
    /// Physical time dilation (`asetrate = sr / div, aresample = 48000` + optional `atempo`).
    PhysicalRateScale,
    /// Pitch-preserving temporal slowdown via chained `atempo` filters + `aresample = 48000`.
    PitchPreservingAtempo,
}

#[derive(Clone, Debug)]
pub struct VideoRenderOptions {
    pub blender_script: PathBuf,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub slowmo_fps: u32,
    pub audio_mode: AudioSlowmoMode,
    pub pre_roll_s: f64,
    pub post_roll_s: f64,
    pub wav_pre_roll_s: f64,
    pub frame_step: usize,
}

impl Default for VideoRenderOptions {
    fn default() -> Self {
        Self {
            blender_script: PathBuf::from("render/blender_cycles_render.py"),
            width: 1280,
            height: 720,
            samples: 64,
            slowmo_fps: 30,
            audio_mode: AudioSlowmoMode::PhysicalRateScale,
            pre_roll_s: 0.75,
            post_roll_s: 0.35,
            wav_pre_roll_s: 0.0,
            frame_step: 1,
        }
    }
}

fn run_blender_impl(
    manifest_path: &Path,
    frames_out_dir: &Path,
    opts: &VideoRenderOptions,
    only_frames: Option<&[usize]>,
) -> Result<()> {
    fs::create_dir_all(frames_out_dir)?;
    let mut cmd = Command::new("blender");
    cmd.arg("--background")
        .arg("--factory-startup")
        .arg("--python-exit-code")
        .arg("1")
        .arg("--python")
        .arg(&opts.blender_script)
        .arg("--")
        .arg("--manifest")
        .arg(manifest_path)
        .arg("--out-frames")
        .arg(frames_out_dir)
        .arg("--width")
        .arg(opts.width.to_string())
        .arg("--height")
        .arg(opts.height.to_string())
        .arg("--samples")
        .arg(opts.samples.to_string())
        .arg("--frame-step")
        .arg(opts.frame_step.max(1).to_string());

    if let Some(frames) = only_frames {
        let joined: Vec<String> = frames.iter().map(|f| f.to_string()).collect();
        cmd.arg("--only-frames").arg(joined.join(","));
    }

    let out = cmd.output()?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        return Err(VideoPipelineError::ProcessFailed {
            tool: "blender".to_string(),
            message: format!(
                "exit {:?}\n--- stdout tail ---\n{}\n--- stderr tail ---\n{}",
                out.status.code(),
                tail_str(&stdout, 2000),
                tail_str(&stderr, 2000)
            ),
        });
    }
    Ok(())
}

pub fn render_frames_with_blender(
    manifest_path: impl AsRef<Path>,
    frames_out_dir: impl AsRef<Path>,
    opts: &VideoRenderOptions,
) -> Result<()> {
    run_blender_impl(manifest_path.as_ref(), frames_out_dir.as_ref(), opts, None)
}

pub fn render_preview_frames_with_blender(
    manifest_path: impl AsRef<Path>,
    frames_out_dir: impl AsRef<Path>,
    opts: &VideoRenderOptions,
    only_frames: &[usize],
) -> Result<()> {
    run_blender_impl(
        manifest_path.as_ref(),
        frames_out_dir.as_ref(),
        opts,
        Some(only_frames),
    )
}

fn tail_str(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        chars[chars.len() - max_chars..].iter().collect()
    }
}

fn chained_atempo_filters(target_tempo: f64) -> Vec<String> {
    let mut filters = Vec::new();
    let mut rem = target_tempo.max(1e-6);
    while rem < 0.5 {
        filters.push("atempo=0.5".to_string());
        rem *= 2.0;
    }
    while rem > 2.0 {
        filters.push("atempo=2.0".to_string());
        rem *= 0.5;
    }
    if (rem - 1.0).abs() > 1e-6 {
        filters.push(format!("atempo={rem:.8}"));
    }
    filters
}

fn build_slowmo_audio_chain(sr: u32, factor: f64, mode: AudioSlowmoMode) -> String {
    let mut stages = vec!["aformat=sample_fmts=fltp:channel_layouts=stereo".to_string()];
    match mode {
        AudioSlowmoMode::PhysicalRateScale => {
            let rate_div = factor.min(64.0).max(1.0);
            let scaled_sr = ((sr as f64) / rate_div).max(600.0);
            stages.push(format!("asetrate={scaled_sr:.4}"));
            stages.push("aresample=48000".to_string());
            let rem_factor = factor / ((sr as f64) / scaled_sr);
            if rem_factor > 1.0005 {
                stages.extend(chained_atempo_filters(1.0 / rem_factor));
            }
        }
        AudioSlowmoMode::PitchPreservingAtempo => {
            // Pad tail slightly so WSOLA atempo window never starves on short (55-120 ms) WAV clips
            stages.push("aresample=48000".to_string());
            stages.push("apad=pad_dur=0.12".to_string());
            stages.extend(chained_atempo_filters(1.0 / factor.max(1.0)));
        }
    }
    stages.join(",")
}

/// Assemble a two-part synchronized MP4:
/// - Part 1 (Real-Time 1x): `pre_roll_s` freeze + real-time playback (`duration_s`) synced to unmodified WAV + `post_roll_s` freeze.
/// - Part 2 (Slow-Mo): all `num_frames` at `slowmo_fps` synced to time-scaled ground-truth audio.
pub fn mux_video_with_audio(
    frames_dir: impl AsRef<Path>,
    wav_path: impl AsRef<Path>,
    duration_s: f64,
    num_frames: usize,
    out_mp4: impl AsRef<Path>,
    opts: &VideoRenderOptions,
) -> Result<()> {
    if num_frames < 2 || duration_s <= 0.0 || opts.slowmo_fps == 0 {
        return Err(VideoPipelineError::InvalidConfig(format!(
            "Invalid mux params: num_frames={num_frames}, duration_s={duration_s}, slowmo_fps={}",
            opts.slowmo_fps
        )));
    }
    let wav_p = wav_path.as_ref();
    let reader = hound::WavReader::open(wav_p)?;
    let wav_sr = reader.spec().sample_rate;
    drop(reader);

    if let Some(parent) = out_mp4.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }

    let out_fps = 60.0_f64;
    let lead_in = if opts.wav_pre_roll_s > 0.0 {
        opts.wav_pre_roll_s
    } else {
        opts.pre_roll_s
    };
    let active_dur = (duration_s - opts.wav_pre_roll_s).max(0.005);
    let p1_total_raw = lead_in + active_dur + opts.post_roll_s;
    let p1_dur = (p1_total_raw * out_fps).ceil() / out_fps;
    let post_adj = (p1_dur - lead_in - active_dur).max(opts.post_roll_s);
    let pre_ms = (lead_in * 1000.0).round() as u64;

    // Input 0 read at 60 fps -> base span is num_frames / 60.0; scale PTS onto exact `active_dur`
    let base_span = (num_frames as f64) / out_fps;
    let pts_scale = active_dur / base_span;

    let slowmo_dur_raw = (num_frames as f64) / (opts.slowmo_fps as f64);
    let p2_dur = (slowmo_dur_raw * out_fps).round() / out_fps;
    let slowmo_factor = p2_dur / active_dur;
    let slowmo_audio = build_slowmo_audio_chain(wav_sr, slowmo_factor, opts.audio_mode);
    let fade_out_st = (p2_dur - 0.015).max(0.0);

    let pattern = frames_dir.as_ref().join("frame_%04d.png");
    let pattern_str = pattern.to_string_lossy();
    let wav_str = wav_p.to_string_lossy();

    let (p1_audio, p2_audio_src) = if opts.wav_pre_roll_s > 0.0 {
        (
            format!("[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{p1_dur:.6},asetpts=PTS-STARTPTS[a1]"),
            format!("[1:a]atrim=start={:.6},asetpts=PTS-STARTPTS,{slowmo_audio},apad,atrim=0:{p2_dur:.6},afade=t=in:st=0:d=0.005,afade=t=out:st={fade_out_st:.6}:d=0.015,asetpts=PTS-STARTPTS[a2]", opts.wav_pre_roll_s),
        )
    } else {
        (
            format!("[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,adelay={pre_ms}|{pre_ms},apad,atrim=0:{p1_dur:.6},asetpts=PTS-STARTPTS[a1]"),
            format!("[1:a]{slowmo_audio},apad,atrim=0:{p2_dur:.6},afade=t=in:st=0:d=0.005,afade=t=out:st={fade_out_st:.6}:d=0.015,asetpts=PTS-STARTPTS[a2]"),
        )
    };

    let filter_complex = format!(
        "[0:v]setpts=PTS*{pts_scale:.8},fps=60,tpad=start_duration={lead_in:.6}:start_mode=clone:stop_duration={post_adj:.6}:stop_mode=clone,trim=duration={p1_dur:.6},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v1];\
         {p1_audio};\
         [2:v]fps=60,trim=duration={p2_dur:.6},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v2];\
         {p2_audio_src};\
         [v1][a1][v2][a2]concat=n=2:v=1:a=1[v][a]"
    );

    let out = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-framerate")
        .arg("60")
        .arg("-i")
        .arg(pattern_str.as_ref())
        .arg("-i")
        .arg(wav_str.as_ref())
        .arg("-framerate")
        .arg(opts.slowmo_fps.to_string())
        .arg("-i")
        .arg(pattern_str.as_ref())
        .arg("-filter_complex")
        .arg(&filter_complex)
        .arg("-map")
        .arg("[v]")
        .arg("-map")
        .arg("[a]")
        .arg("-c:v")
        .arg("libx264")
        .arg("-profile:v")
        .arg("high")
        .arg("-crf")
        .arg("17")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("320k")
        .arg("-ar")
        .arg("48000")
        .arg("-ac")
        .arg("2")
        .arg("-movflags")
        .arg("+faststart")
        .arg(out_mp4.as_ref())
        .output()?;

    if !out.status.success() {
        return Err(VideoPipelineError::ProcessFailed {
            tool: "ffmpeg".to_string(),
            message: format!(
                "mux_video_with_audio exit {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            ),
        });
    }
    Ok(())
}

/// Assemble a synchronized 1x real-time MP4 from pre-rolled WAV audio (`[0, pre_roll_s]` silent)
/// and active frames (`frame_0000.png`.. spanning `[pre_roll_s, total_duration_s]`).
pub fn mux_realtime_video_with_audio(
    frames_dir: impl AsRef<Path>,
    wav_path: impl AsRef<Path>,
    total_duration_s: f64,
    pre_roll_s: f64,
    active_num_frames: usize,
    out_mp4: impl AsRef<Path>,
    post_roll_s: f64,
) -> Result<()> {
    let active_dur = total_duration_s - pre_roll_s;
    if active_num_frames < 2 || active_dur <= 0.0 || pre_roll_s < 0.0 || post_roll_s < 0.0 {
        return Err(VideoPipelineError::InvalidConfig(format!(
            "Invalid realtime mux params: active_num_frames={active_num_frames}, total_duration_s={total_duration_s}, pre_roll_s={pre_roll_s}, post_roll_s={post_roll_s}"
        )));
    }
    if let Some(parent) = out_mp4.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }

    let out_fps = 60.0_f64;
    let final_dur = ((total_duration_s + post_roll_s) * out_fps).ceil() / out_fps;
    let stop_adj = (final_dur - pre_roll_s - active_dur).max(post_roll_s);
    let fps_active = (active_num_frames as f64) / active_dur;

    let pattern = frames_dir.as_ref().join("frame_%04d.png");
    let pattern_str = pattern.to_string_lossy();
    let wav_str = wav_path.as_ref().to_string_lossy();

    let filter_complex = format!(
        "[0:v]tpad=start_duration={pre_roll_s:.6}:start_mode=clone:stop_duration={stop_adj:.6}:stop_mode=clone,\
         fps=60,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,\
         setsar=1,trim=duration={final_dur:.6},setpts=PTS-STARTPTS,format=yuv420p[v];\
         [1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{final_dur:.6},asetpts=PTS-STARTPTS[a]"
    );

    let out = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-framerate")
        .arg(format!("{fps_active:.8}"))
        .arg("-i")
        .arg(pattern_str.as_ref())
        .arg("-i")
        .arg(wav_str.as_ref())
        .arg("-filter_complex")
        .arg(&filter_complex)
        .arg("-map")
        .arg("[v]")
        .arg("-map")
        .arg("[a]")
        .arg("-c:v")
        .arg("libx264")
        .arg("-profile:v")
        .arg("high")
        .arg("-crf")
        .arg("17")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("320k")
        .arg("-ar")
        .arg("48000")
        .arg("-ac")
        .arg("2")
        .arg("-movflags")
        .arg("+faststart")
        .arg(out_mp4.as_ref())
        .output()?;

    if !out.status.success() {
        return Err(VideoPipelineError::ProcessFailed {
            tool: "ffmpeg".to_string(),
            message: format!(
                "mux_realtime_video_with_audio exit {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            ),
        });
    }
    Ok(())
}

pub fn probe_media_duration(path: impl AsRef<Path>) -> Result<f64> {
    let out = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(path.as_ref())
        .output()?;
    if !out.status.success() {
        return Err(VideoPipelineError::ProcessFailed {
            tool: "ffprobe".to_string(),
            message: String::from_utf8_lossy(&out.stderr).to_string(),
        });
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    s.parse::<f64>().map_err(|_| VideoPipelineError::ProcessFailed {
        tool: "ffprobe".to_string(),
        message: format!("Could not parse duration '{s}'"),
    })
}

/// Create a `1280x720` side-by-side comparison where `left_mp4` plays in full first
/// while `right_mp4` holds frame 0, then `right_mp4` plays in full while `left_mp4` holds final frame.
pub fn compose_side_by_side_comparison(
    left_mp4: impl AsRef<Path>,
    right_mp4: impl AsRef<Path>,
    out_mp4: impl AsRef<Path>,
) -> Result<()> {
    let d_l = probe_media_duration(&left_mp4)?;
    let d_r = probe_media_duration(&right_mp4)?;
    let total = d_l + d_r;

    let filter = format!(
        "[0:v]scale=640:720:force_original_aspect_ratio=increase,crop=640:720,tpad=stop_duration={d_r:.6}:stop_mode=clone,fps=60,setsar=1,format=yuv420p[vl];\
         [1:v]scale=640:720:force_original_aspect_ratio=increase,crop=640:720,tpad=start_duration={d_l:.6}:start_mode=clone,fps=60,setsar=1,format=yuv420p[vr];\
         [vl][vr]hstack=inputs=2,trim=duration={total:.6},setpts=PTS-STARTPTS[v];\
         [0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[al];\
         [1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[ar];\
         [al][ar]concat=n=2:v=0:a=1,atrim=0:{total:.6},asetpts=PTS-STARTPTS[a]"
    );

    let out = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(left_mp4.as_ref())
        .arg("-i")
        .arg(right_mp4.as_ref())
        .arg("-filter_complex")
        .arg(&filter)
        .arg("-map")
        .arg("[v]")
        .arg("-map")
        .arg("[a]")
        .arg("-c:v")
        .arg("libx264")
        .arg("-profile:v")
        .arg("high")
        .arg("-crf")
        .arg("17")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("320k")
        .arg("-ar")
        .arg("48000")
        .arg("-ac")
        .arg("2")
        .arg("-movflags")
        .arg("+faststart")
        .arg(out_mp4.as_ref())
        .output()?;

    if !out.status.success() {
        return Err(VideoPipelineError::ProcessFailed {
            tool: "ffmpeg".to_string(),
            message: format!(
                "compose_side_by_side_comparison exit {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            ),
        });
    }
    Ok(())
}

/// Concatenate multiple `.mp4` videos into a uniform `1280x720 @ 60fps` `48kHz stereo` showcase reel.
pub fn compose_showcase_reel(mp4_paths: &[PathBuf], out_mp4: impl AsRef<Path>) -> Result<()> {
    if mp4_paths.is_empty() {
        return Err(VideoPipelineError::InvalidConfig(
            "compose_showcase_reel requires at least one MP4".to_string(),
        ));
    }
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y").arg("-v").arg("error");
    for p in mp4_paths {
        cmd.arg("-i").arg(p);
    }

    let n = mp4_paths.len();
    let mut filter = String::new();
    let mut concat_inputs = String::new();
    for idx in 0..n {
        filter.push_str(&format!(
            "[{idx}:v]fps=60,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{idx}];\
             [{idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{idx}];"
        ));
        concat_inputs.push_str(&format!("[v{idx}][a{idx}]"));
    }
    filter.push_str(&format!("{concat_inputs}concat=n={n}:v=1:a=1[v][a]"));

    let out = cmd
        .arg("-filter_complex")
        .arg(&filter)
        .arg("-map")
        .arg("[v]")
        .arg("-map")
        .arg("[a]")
        .arg("-c:v")
        .arg("libx264")
        .arg("-profile:v")
        .arg("high")
        .arg("-crf")
        .arg("17")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("320k")
        .arg("-ar")
        .arg("48000")
        .arg("-ac")
        .arg("2")
        .arg("-movflags")
        .arg("+faststart")
        .arg(out_mp4.as_ref())
        .output()?;

    if !out.status.success() {
        return Err(VideoPipelineError::ProcessFailed {
            tool: "ffmpeg".to_string(),
            message: format!(
                "compose_showcase_reel exit {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
            ),
        });
    }
    Ok(())
}
