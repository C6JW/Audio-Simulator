use std::collections::VecDeque;
use std::f64::consts::PI;

use sim_io::{apply_highpass_2nd_order, decimate_kaiser, ProbeAcousticSummary};

use crate::units::{AIR_GAMMA, AMBIENT_P0, WATER_RHO, WATER_SIGMA};

#[derive(Clone, Copy, Debug)]
struct Complex64 {
    re: f64,
    im: f64,
}

fn fft_radix2(buf: &mut [Complex64]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / (len as f64);
        let wlen = Complex64 {
            re: ang.cos(),
            im: ang.sin(),
        };
        let half = len >> 1;
        for i in (0..n).step_by(len) {
            let mut w = Complex64 { re: 1.0, im: 0.0 };
            for k in 0..half {
                let u = buf[i + k];
                let v0 = buf[i + k + half];
                let v = Complex64 {
                    re: v0.re * w.re - v0.im * w.im,
                    im: v0.re * w.im + v0.im * w.re,
                };
                buf[i + k] = Complex64 {
                    re: u.re + v.re,
                    im: u.im + v.im,
                };
                buf[i + k + half] = Complex64 {
                    re: u.re - v.re,
                    im: u.im - v.im,
                };
                w = Complex64 {
                    re: w.re * wlen.re - w.im * wlen.im,
                    im: w.re * wlen.im + w.im * wlen.re,
                };
            }
        }
        len <<= 1;
    }
}

pub fn power_spectrum_hann(signal: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    if signal.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n_fft = signal.len().next_power_of_two().max(512);
    let mut buf = vec![Complex64 { re: 0.0, im: 0.0 }; n_fft];
    let n = signal.len();
    for i in 0..n {
        let win = if n > 1 {
            0.5 * (1.0 - (2.0 * PI * (i as f64) / ((n - 1) as f64)).cos())
        } else {
            1.0
        };
        buf[i].re = signal[i] * win;
    }
    fft_radix2(&mut buf);

    let bins = n_fft / 2 + 1;
    let mut freqs = Vec::with_capacity(bins);
    let mut power = Vec::with_capacity(bins);
    let df = sample_rate / (n_fft as f64);
    for k in 0..bins {
        freqs.push((k as f64) * df);
        let p = buf[k].re * buf[k].re + buf[k].im * buf[k].im;
        power.push(p);
    }
    (freqs, power)
}

pub fn dominant_freq_and_centroid(
    signal: &[f64],
    sample_rate: f64,
    min_hz: f64,
    max_hz: f64,
) -> (f64, f64) {
    let (freqs, power) = power_spectrum_hann(signal, sample_rate);
    let mut best_f = 0.0;
    let mut best_idx: Option<usize> = None;
    let mut best_p = -1.0;
    let mut num = 0.0;
    let mut den = 0.0;
    for (idx, (&f, &p)) in freqs.iter().zip(power.iter()).enumerate() {
        if f >= min_hz && f <= max_hz {
            if p > best_p {
                best_p = p;
                best_f = f;
                best_idx = Some(idx);
            }
            num += f * p;
            den += p;
        }
    }
    if let Some(k) = best_idx {
        if k > 0
            && k + 1 < power.len()
            && freqs.len() > 1
            && power[k - 1] > 1e-30
            && power[k] > 1e-30
            && power[k + 1] > 1e-30
        {
            let a = power[k - 1].ln();
            let b = power[k].ln();
            let c = power[k + 1].ln();
            let denom = a - 2.0 * b + c;
            if denom.abs() > 1e-12 {
                let delta = (0.5 * (a - c) / denom).clamp(-0.5, 0.5);
                let df = freqs[1] - freqs[0];
                best_f = ((k as f64) + delta) * df;
            }
        }
    }
    let centroid = if den > 1e-30 { num / den } else { 0.0 };
    (best_f, centroid)
}

pub fn analyze_probe_signal(
    name: &str,
    is_hydrophone: bool,
    position: [f32; 3],
    raw_oversampled_pa: &[f32],
    oversample_factor: usize,
    target_sample_rate: u32,
) -> ProbeAcousticSummary {
    let mut sig = decimate_kaiser(raw_oversampled_pa, oversample_factor);
    apply_highpass_2nd_order(&mut sig, target_sample_rate, 20.0);

    let n = sig.len();
    if n == 0 {
        return ProbeAcousticSummary {
            name: name.to_string(),
            is_hydrophone,
            position,
            ..Default::default()
        };
    }

    let fs = target_sample_rate as f64;
    let dt = 1.0 / fs;

    let mut peak_pa = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut mean = 0.0_f64;
    for &v in &sig {
        peak_pa = peak_pa.max(v.abs());
        sum_sq += v * v;
        mean += v;
    }
    mean /= n as f64;
    let rms_pa = (sum_sq / (n as f64)).sqrt();
    let total_energy_pa2_s = sum_sq * dt;

    // Kurtosis
    let mut m2 = 0.0_f64;
    let mut m4 = 0.0_f64;
    for &v in &sig {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n as f64;
    m4 /= n as f64;
    let kurtosis = if m2 > 1e-30 { m4 / (m2 * m2) } else { 0.0 };

    // Early vs late RMS split (after peak onset)
    let peak_idx = sig
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let post = &sig[peak_idx..];
    let split = (post.len() / 3).max(1);
    let early = &post[..split.min(post.len())];
    let late = if post.len() > split {
        &post[post.len() - split..]
    } else {
        early
    };
    let early_rms = (early.iter().map(|v| v * v).sum::<f64>() / (early.len() as f64)).sqrt();
    let late_rms = (late.iter().map(|v| v * v).sum::<f64>() / (late.len() as f64)).sqrt();
    let decay_tail_ratio = if early_rms > 1e-24 {
        late_rms / early_rms
    } else {
        0.0
    };

    let (peak_freq_hz, spectral_centroid_hz) =
        dominant_freq_and_centroid(&sig, fs, 50.0, 0.45 * fs);

    // STFT early vs late window peak frequency around active ringdown
    let (chirp_early_freq_hz, chirp_late_freq_hz) = if post.len() >= 128 {
        let half = post.len() / 2;
        let (f1, _) = dominant_freq_and_centroid(&post[..half], fs, 200.0, 0.45 * fs);
        let (f2, _) = dominant_freq_and_centroid(&post[half..], fs, 200.0, 0.45 * fs);
        (f1, f2)
    } else {
        (peak_freq_hz, peak_freq_hz)
    };

    ProbeAcousticSummary {
        name: name.to_string(),
        is_hydrophone,
        position,
        wav: None,
        peak_freq_hz,
        spectral_centroid_hz,
        rms_pa,
        peak_pa,
        total_energy_pa2_s,
        early_rms_pa: early_rms,
        late_rms_pa: late_rms,
        decay_tail_ratio,
        kurtosis,
        chirp_early_freq_hz,
        chirp_late_freq_hz,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BubbleDiagnostic {
    pub crater_min_z: f64,
    pub subsurface_air_volume: f64,
    pub bubble_radius_m: f64,
    pub minnaert_freq_hz: f64,
}

/// Read-only connected-component analyzer on a Level alpha snapshot.
/// Distinguishes open atmospheric/crater air (connected to top z boundary)
/// from pinched-off subsurface entrained air pockets.
pub fn analyze_subsurface_bubble(
    alpha_padded: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    halo: usize,
    dx: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    water_surface_z: f32,
) -> BubbleDiagnostic {
    let nxp = nx + 2 * halo;
    let nyp = ny + 2 * halo;
    let total_int = nx * ny * nz;
    if total_int == 0 {
        return BubbleDiagnostic::default();
    }

    let idx_p = |ii: usize, jj: usize, kk: usize| -> usize {
        (ii + halo) + nxp * ((jj + halo) + nyp * (kk + halo))
    };
    let idx_i = |ii: usize, jj: usize, kk: usize| -> usize { ii + nx * (jj + ny * kk) };

    let mut open_air = vec![false; total_int];
    let mut queue = VecDeque::new();

    // Seed from top interior layer where alpha < 0.5
    let k_top = nz - 1;
    for j in 0..ny {
        for i in 0..nx {
            if alpha_padded[idx_p(i, j, k_top)] < 0.5 {
                let id = idx_i(i, j, k_top);
                open_air[id] = true;
                queue.push_back((i, j, k_top));
            }
        }
    }

    while let Some((i, j, k)) = queue.pop_front() {
        let neighbors = [
            (i.wrapping_sub(1), j, k, i > 0),
            (i + 1, j, k, i + 1 < nx),
            (i, j.wrapping_sub(1), k, j > 0),
            (i, j + 1, k, j + 1 < ny),
            (i, j, k.wrapping_sub(1), k > 0),
            (i, j, k + 1, k + 1 < nz),
        ];
        for (ni, nj, nk, valid) in neighbors {
            if !valid {
                continue;
            }
            let nid = idx_i(ni, nj, nk);
            if !open_air[nid] && alpha_padded[idx_p(ni, nj, nk)] < 0.5 {
                open_air[nid] = true;
                queue.push_back((ni, nj, nk));
            }
        }
    }

    // Mark cells in open_air or 6-adjacent to open_air so subsurface bubble shell expansion
    // cannot bleed into the free-surface diffuse MTHINC transition band.
    let mut open_adj = vec![false; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let id = idx_i(i, j, k);
                if !open_air[id] {
                    continue;
                }
                open_adj[id] = true;
                let neighbors = [
                    (i.wrapping_sub(1), j, k, i > 0),
                    (i + 1, j, k, i + 1 < nx),
                    (i, j.wrapping_sub(1), k, j > 0),
                    (i, j + 1, k, j + 1 < ny),
                    (i, j, k.wrapping_sub(1), k > 0),
                    (i, j, k + 1, k + 1 < nz),
                ];
                for (ni, nj, nk, valid) in neighbors {
                    if valid {
                        open_adj[idx_i(ni, nj, nk)] = true;
                    }
                }
            }
        }
    }

    let mut crater_min_z = f64::INFINITY;
    let mut bubble_mask = vec![false; nx * ny * nz];
    queue.clear();
    let dx_f = dx as f64;
    let surf_z = water_surface_z as f64;
    let _ = (ox, oy);

    for k in 0..nz {
        let z = oz as f64 + ((k as f64) + 0.5) * dx_f;
        for j in 0..ny {
            for i in 0..nx {
                let id = idx_i(i, j, k);
                if open_air[id] && z < crater_min_z {
                    crater_min_z = z;
                }
                if i > 0
                    && i + 1 < nx
                    && j > 0
                    && j + 1 < ny
                    && k > 0
                    && k + 1 < nz
                    && !open_adj[id]
                    && z < surf_z - 1.5 * dx_f
                {
                    let a = alpha_padded[idx_p(i, j, k)] as f64;
                    if a < 0.5 {
                        bubble_mask[id] = true;
                        queue.push_back((i, j, k));
                    }
                }
            }
        }
    }

    // Expand bubble mask across adjacent subsurface interface cells (alpha < 0.98)
    while let Some((i, j, k)) = queue.pop_front() {
        let neighbors = [
            (i.wrapping_sub(1), j, k, i > 1),
            (i + 1, j, k, i + 2 < nx),
            (i, j.wrapping_sub(1), k, j > 1),
            (i, j + 1, k, j + 2 < ny),
            (i, j, k.wrapping_sub(1), k > 1),
            (i, j, k + 1, k + 2 < nz),
        ];
        for (ni, nj, nk, valid) in neighbors {
            if !valid {
                continue;
            }
            let nid = idx_i(ni, nj, nk);
            if bubble_mask[nid] || open_adj[nid] {
                continue;
            }
            let nz_pos = oz as f64 + ((nk as f64) + 0.5) * dx_f;
            if nz_pos >= surf_z - 0.5 * dx_f {
                continue;
            }
            let na = alpha_padded[idx_p(ni, nj, nk)] as f64;
            if na < 0.98 {
                bubble_mask[nid] = true;
                queue.push_back((ni, nj, nk));
            }
        }
    }

    let mut sub_air_vol = 0.0_f64;
    let mut sub_air_z_sum = 0.0_f64;
    let dv = dx_f.powi(3);

    for k in 1..nz.saturating_sub(1) {
        let z = oz as f64 + ((k as f64) + 0.5) * dx_f;
        for j in 1..ny.saturating_sub(1) {
            for i in 1..nx.saturating_sub(1) {
                let id = idx_i(i, j, k);
                if bubble_mask[id] {
                    let a = alpha_padded[idx_p(i, j, k)] as f64;
                    let air_frac = 1.0 - a.clamp(0.0, 1.0);
                    let v = air_frac * dv;
                    sub_air_vol += v;
                    sub_air_z_sum += v * z;
                }
            }
        }
    }

    if !crater_min_z.is_finite() {
        crater_min_z = water_surface_z as f64;
    }

    let (bubble_radius_m, minnaert_freq_hz) = if sub_air_vol > 1e-13 {
        let rb = (3.0 * sub_air_vol / (4.0 * PI)).cbrt();
        let z_b = sub_air_z_sum / sub_air_vol;
        let depth = ((water_surface_z as f64) - z_b).max(0.0);
        let p_abs = AMBIENT_P0 + WATER_RHO * 9.80665 * depth + 2.0 * WATER_SIGMA / rb;
        let fm_inf = (1.0 / (2.0 * PI * rb)) * (3.0 * AIR_GAMMA * p_abs / WATER_RHO).sqrt();
        // Distance to nearest pressure-release free surface directly above the bubble
        // (crater floor if depressed below water_surface_z).
        let surface_above = if crater_min_z < (water_surface_z as f64) && z_b < crater_min_z {
            crater_min_z
        } else {
            water_surface_z as f64
        };
        let h_free = (surface_above - z_b).max(rb);
        // Strasberg (1953) / Longuet-Higgins (1989) free-surface added-mass correction:
        // f(h) = f_inf / sqrt(1 - R_b / (2 h)), bounded by the tangent limit h = R_b (sqrt(2)).
        let surface_factor = (1.0 - rb / (2.0 * h_free)).max(0.5).sqrt().recip();
        let fm = fm_inf * surface_factor;
        (rb, fm)
    } else {
        (0.0, 0.0)
    };

    BubbleDiagnostic {
        crater_min_z,
        subsurface_air_volume: sub_air_vol,
        bubble_radius_m,
        minnaert_freq_hz,
    }
}
