use std::f64::consts::PI;
use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioIoError {
    #[error("WAV IO error: {0}")]
    Hound(#[from] hound::Error),
    #[error("Invalid audio parameters: {0}")]
    InvalidParam(String),
}

pub type Result<T> = std::result::Result<T, AudioIoError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WavWriteStats {
    pub path: String,
    pub sample_rate: u32,
    pub sample_count: usize,
    pub peak_pa: f64,
    pub rms_pa: f64,
    pub gain_applied_db: f64,
    pub normalized_peak: f32,
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = 0.5 * x;
    for k in 1..40 {
        let kf = k as f64;
        term *= (half_x / kf) * (half_x / kf);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Design a linear-phase Kaiser-windowed sinc low-pass FIR filter for integer decimation.
pub fn design_kaiser_lpf(decim_factor: usize, taps_per_wing: usize, beta: f64) -> Vec<f64> {
    let half = decim_factor * taps_per_wing;
    let len = 2 * half + 1;
    let fc = 0.44 / (decim_factor as f64); // Transition comfortably below Nyquist / decim_factor
    let i0_beta = bessel_i0(beta);

    let mut taps = vec![0.0_f64; len];
    let mut sum = 0.0_f64;
    for n in 0..len {
        let m = (n as isize - half as isize) as f64;
        let sinc = if m.abs() < 1e-12 {
            2.0 * fc
        } else {
            (2.0 * PI * fc * m).sin() / (PI * m)
        };
        let ratio = m / (half as f64);
        let arg = (1.0 - ratio * ratio).max(0.0).sqrt();
        let win = bessel_i0(beta * arg) / i0_beta;
        let h = sinc * win;
        taps[n] = h;
        sum += h;
    }
    for h in &mut taps {
        *h /= sum;
    }
    taps
}

/// Decimate an oversampled physical pressure signal (in Pa) by integer factor `oversample`
/// using a linear-phase Kaiser-windowed FIR anti-aliasing low-pass filter.
pub fn decimate_kaiser(input_pa: &[f32], oversample: usize) -> Vec<f64> {
    if oversample <= 1 {
        return input_pa.iter().map(|&x| x as f64).collect();
    }
    if input_pa.is_empty() {
        return Vec::new();
    }
    let taps = design_kaiser_lpf(oversample, 12, 8.5);
    let half = (taps.len() - 1) / 2;
    let n_in = input_pa.len();
    let n_out = (n_in + oversample - 1) / oversample;
    let mut out = Vec::with_capacity(n_out);

    for k in 0..n_out {
        let center = k * oversample;
        let mut acc = 0.0_f64;
        for (j, &w) in taps.iter().enumerate() {
            let idx = center as isize + j as isize - half as isize;
            let s = if idx < 0 {
                input_pa[0] as f64
            } else if idx >= n_in as isize {
                input_pa[n_in - 1] as f64
            } else {
                input_pa[idx as usize] as f64
            };
            acc += w * s;
        }
        out.push(acc);
    }
    out
}

/// Apply a 2nd-order Butterworth high-pass filter at `cutoff_hz` (default 20 Hz)
/// to remove static DC/hydrostatic offset without distorting audible transients.
pub fn apply_highpass_2nd_order(samples: &mut [f64], sample_rate: u32, cutoff_hz: f64) {
    if samples.is_empty() || cutoff_hz <= 0.0 {
        return;
    }
    let fs = sample_rate as f64;
    if cutoff_hz >= 0.45 * fs {
        return;
    }
    let k = (PI * cutoff_hz / fs).tan();
    let sqrt2 = std::f64::consts::SQRT_2;
    let norm = 1.0 / (1.0 + sqrt2 * k + k * k);
    let b0 = 1.0 * norm;
    let b1 = -2.0 * norm;
    let b2 = 1.0 * norm;
    let a1 = 2.0 * (k * k - 1.0) * norm;
    let a2 = (1.0 - sqrt2 * k + k * k) * norm;

    // Seed state at initial sample to eliminate startup step transient
    let x_init = samples[0];
    let mut x1 = x_init;
    let mut x2 = x_init;
    let mut y1 = 0.0_f64;
    let mut y2 = 0.0_f64;

    for x in samples.iter_mut() {
        let x0 = *x;
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *x = y0;
    }
}

/// Integer upsampling using linear-phase Kaiser-windowed sinc interpolation.
pub fn resample_kaiser_upsample(input: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 || input.is_empty() {
        return input.to_vec();
    }
    let taps_per_wing = 12usize;
    let taps = design_kaiser_lpf(factor, taps_per_wing, 8.5);
    let half = (taps.len() - 1) / 2;
    let gain = factor as f64;
    let n_in = input.len();
    let n_out = n_in * factor;
    let mut out = Vec::with_capacity(n_out);

    for n in 0..n_out {
        let mut acc = 0.0_f64;
        let offset = (n + half) % factor;
        let mut j = offset;
        while j < taps.len() {
            let m = ((n as isize + half as isize - j as isize) / (factor as isize))
                .clamp(0, (n_in - 1) as isize) as usize;
            acc += taps[j] * input[m];
            j += factor;
        }
        out.push(acc * gain);
    }
    out
}

/// Arbitrary-ratio windowed-sinc resampler for dry speech WAV assets.
pub fn resample_sinc_f32(input: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if input.is_empty() || src_sr == 0 || dst_sr == 0 || src_sr == dst_sr {
        return input.to_vec();
    }
    let ratio = (dst_sr as f64) / (src_sr as f64);
    let n_out = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let fc = (0.45 * ratio.min(1.0)).max(0.05);
    let half_w = 16isize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let t_src = (i as f64) / ratio;
        let center = t_src.floor() as isize;
        let mut sum = 0.0_f64;
        let mut w_sum = 0.0_f64;
        for m in (center - half_w)..=(center + half_w) {
            if m < 0 || m >= input.len() as isize {
                continue;
            }
            let x = t_src - (m as f64);
            let sinc = if x.abs() < 1e-12 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * x).sin() / (PI * x)
            };
            let u = x / (half_w as f64);
            let win = if u.abs() < 1.0 {
                0.5 * (1.0 + (PI * u).cos())
            } else {
                0.0
            };
            let w = sinc * win;
            sum += w * (input[m as usize] as f64);
            w_sum += w;
        }
        let val = if w_sum.abs() > 1e-12 { sum / w_sum } else { 0.0 };
        out.push(val as f32);
    }
    out
}

/// Pure-Rust multi-syllable formant speech phrase fallback (~4.2 s) if no external speech WAV is provided.
pub fn generate_fallback_speech_phrase(sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate.max(8000);
    let syllables: &[(f64, f64, f64, f64, f64, f64)] = &[
        // (start_s, dur_s, f0_hz, f1_hz, f2_hz, f3_hz)
        (0.05, 0.24, 128.0, 480.0, 1750.0, 2650.0),
        (0.32, 0.28, 136.0, 560.0, 1450.0, 2500.0),
        (0.66, 0.22, 124.0, 380.0, 2100.0, 2850.0),
        (0.94, 0.34, 132.0, 690.0, 1180.0, 2480.0),
        (1.42, 0.26, 138.0, 520.0, 1850.0, 2700.0),
        (1.74, 0.30, 129.0, 440.0, 1020.0, 2400.0),
        (2.12, 0.25, 134.0, 620.0, 1520.0, 2620.0),
        (2.44, 0.36, 122.0, 720.0, 1260.0, 2520.0),
        (2.96, 0.24, 127.0, 410.0, 1980.0, 2760.0),
        (3.26, 0.42, 116.0, 540.0, 1320.0, 2460.0),
    ];
    let total = (4.0 * (sr as f64)).round() as usize;
    let mut out = vec![0.0_f32; total];
    let mut rng_state: u64 = 0x5EED_C0DE_1234_5678;

    for &(st, dur, f0, f1, f2, f3) in syllables {
        let i0 = (st * sr as f64) as usize;
        let i1 = (((st + dur) * sr as f64) as usize).min(total);
        for i in i0..i1 {
            let t_loc = ((i - i0) as f64) / (sr as f64);
            let u = t_loc / dur;
            let env = (PI * u).sin().powf(0.75);
            let f0_t = f0 * (1.0 + 0.04 * (2.0 * PI * 3.0 * t_loc).sin() - 0.08 * u);
            let phi = 2.0 * PI * f0_t * t_loc;
            let mut sig = 0.0_f64;
            for h in 1..=26 {
                let fh = (h as f64) * f0_t;
                if fh >= 0.45 * (sr as f64) {
                    break;
                }
                let r1 = 1.0 / (1.0 + ((fh - f1) / 70.0).powi(2));
                let r2 = 0.65 / (1.0 + ((fh - f2) / 110.0).powi(2));
                let r3 = 0.35 / (1.0 + ((fh - f3) / 150.0).powi(2));
                sig += (r1 + r2 + r3) * (h as f64 * phi).sin() / (h as f64).sqrt();
            }
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let breath = (((rng_state >> 32) as f64) / (u32::MAX as f64) - 0.5) * 0.06;
            out[i] += (env * (0.28 * sig + breath)) as f32;
        }
    }
    let peak = out.iter().fold(0.0_f32, |a, &b| a.max(b.abs())).max(1e-6);
    for x in &mut out {
        *x /= peak;
    }
    out
}

pub fn prepare_probe_wav_signal_f64(
    raw_oversampled_pa: &[f32],
    oversample_factor: usize,
    probe_sample_rate: u32,
    target_wav_sample_rate: Option<u32>,
    highpass_hz: f64,
) -> Result<(u32, Vec<f64>, f64, f64)> {
    if probe_sample_rate == 0 {
        return Err(AudioIoError::InvalidParam(
            "sample rate must be > 0".to_string(),
        ));
    }
    let mut decimated = decimate_kaiser(raw_oversampled_pa, oversample_factor);
    apply_highpass_2nd_order(&mut decimated, probe_sample_rate, highpass_hz);

    let peak_pa = decimated
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let sum_sq: f64 = decimated.iter().map(|&v| v * v).sum();
    let rms_pa = if decimated.is_empty() {
        0.0
    } else {
        (sum_sq / decimated.len() as f64).sqrt()
    };

    let (out_sr, out_sig) = if let Some(tsr) = target_wav_sample_rate {
        if tsr > probe_sample_rate && tsr % probe_sample_rate == 0 {
            let factor = (tsr / probe_sample_rate) as usize;
            (tsr, resample_kaiser_upsample(&decimated, factor))
        } else {
            (probe_sample_rate, decimated)
        }
    } else {
        (probe_sample_rate, decimated)
    };

    Ok((out_sr, out_sig, peak_pa, rms_pa))
}

/// Write a pre-filtered physical pressure array (Pa) into a 32-bit float WAV
/// using either individual `-1.0 dBFS` auto-normalization (`fixed_gain = None`) or a shared gain factor.
pub fn write_probe_wav_f32_with_gain<P: AsRef<Path>>(
    path: P,
    signal_pa: &[f64],
    sample_rate: u32,
    peak_pa: f64,
    rms_pa: f64,
    fixed_gain: Option<f64>,
) -> Result<WavWriteStats> {
    if sample_rate == 0 {
        return Err(AudioIoError::InvalidParam(
            "sample rate must be > 0".to_string(),
        ));
    }
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let post_peak = signal_pa
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(peak_pa);
    let target_peak = 10.0_f64.powf(-1.0 / 20.0); // -1.0 dBFS ≈ 0.89125
    let (gain, gain_db) = if let Some(g) = fixed_gain {
        (g, if g > 0.0 { 20.0 * g.log10() } else { 0.0 })
    } else if post_peak > 1e-18 {
        let g = target_peak / post_peak;
        (g, 20.0 * g.log10())
    } else {
        (1.0, 0.0)
    };

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path.as_ref(), spec)?;
    let mut norm_peak = 0.0_f32;
    for &s in signal_pa {
        let v = ((s * gain).clamp(-0.9999, 0.9999)) as f32;
        norm_peak = norm_peak.max(v.abs());
        writer.write_sample(v)?;
    }
    writer.finalize()?;

    Ok(WavWriteStats {
        path: path.as_ref().display().to_string(),
        sample_rate,
        sample_count: signal_pa.len(),
        peak_pa,
        rms_pa,
        gain_applied_db: gain_db,
        normalized_peak: norm_peak,
    })
}

/// Process raw oversampled GPU probe pressure (Pa) -> decimated, DC-filtered 32-bit float WAV
/// normalized to -1.0 dBFS while reporting exact physical Pa peak & RMS.
pub fn write_probe_wav_f32<P: AsRef<Path>>(
    path: P,
    raw_oversampled_pa: &[f32],
    oversample_factor: usize,
    target_sample_rate: u32,
    highpass_hz: f64,
) -> Result<WavWriteStats> {
    let (sr, sig, peak_pa, rms_pa) = prepare_probe_wav_signal_f64(
        raw_oversampled_pa,
        oversample_factor,
        target_sample_rate,
        None,
        highpass_hz,
    )?;
    write_probe_wav_f32_with_gain(path, &sig, sr, peak_pa, rms_pa, None)
}

pub fn read_wav_f32<P: AsRef<Path>>(path: P) -> Result<(WavSpec, Vec<f32>)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<f32>().collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((spec, samples))
}

/// Read any PCM (16/24/32-bit int) or 32-bit float WAV and downmix to normalized mono `f32` in `[-1, 1]`.
pub fn read_wav_mono_f32<P: AsRef<Path>>(path: P) -> Result<(u32, Vec<f32>)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let raw_f: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        SampleFormat::Int => {
            let shift = spec.bits_per_sample.saturating_sub(1).min(31);
            let scale = 1.0_f64 / ((1u64 << shift) as f64);
            reader
                .samples::<i32>()
                .map(|r| r.map(|v| (v as f64 * scale) as f32))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let n_frames = raw_f.len() / ch;
    let mut mono = Vec::with_capacity(n_frames);
    let inv_ch = 1.0_f32 / (ch as f32);
    for f in 0..n_frames {
        let mut s = 0.0_f32;
        for c in 0..ch {
            s += raw_f[f * ch + c];
        }
        mono.push(s * inv_ch);
    }
    Ok((spec.sample_rate, mono))
}
