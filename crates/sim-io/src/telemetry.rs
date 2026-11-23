use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::audio::WavWriteStats;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimeSeriesRow {
    pub time_s: f64,
    pub dt_ac_min: f64,
    pub c_max: f64,
    pub u_max: f64,
    pub acoustic_energy: f64,
    pub kinetic_energy: f64,
    pub solid_elastic_energy: f64,
    pub solid_kinetic_energy: f64,
    pub total_energy: f64,
    pub water_volume: f64,
    pub max_abs_p: f64,
    pub vram_tracked_bytes: usize,
    pub fractured_fraction: f64,
    pub crater_min_z: f64,
    pub subsurface_air_volume: f64,
    pub bubble_radius_mm: f64,
    pub minnaert_freq_hz: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProbeAcousticSummary {
    pub name: String,
    pub is_hydrophone: bool,
    pub position: [f32; 3],
    pub wav: Option<WavWriteStats>,
    pub peak_freq_hz: f64,
    pub spectral_centroid_hz: f64,
    pub rms_pa: f64,
    pub peak_pa: f64,
    pub total_energy_pa2_s: f64,
    pub early_rms_pa: f64,
    pub late_rms_pa: f64,
    pub decay_tail_ratio: f64,
    pub kurtosis: f64,
    pub chirp_early_freq_hz: f64,
    pub chirp_late_freq_hz: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunTelemetry {
    pub scenario: String,
    pub compute_capability: String,
    pub total_vram_bytes: usize,
    pub vram_cap_bytes: usize,
    pub peak_vram_tracked_bytes: usize,
    pub peak_vram_driver_used_bytes: usize,
    pub sim_duration_s: f64,
    pub wall_time_s: f64,
    pub total_substeps: u64,
    pub nan_count: u64,
    pub initial_water_volume: f64,
    pub final_water_volume: f64,
    pub water_volume_rel_error: f64,
    pub initial_total_energy: f64,
    pub final_total_energy: f64,
    pub energy_conservation_error: f64,
    pub probes: Vec<ProbeAcousticSummary>,
    pub notes: BTreeMap<String, f64>,
}

pub fn write_telemetry_json<P: AsRef<Path>>(path: P, telemetry: &RunTelemetry) -> std::io::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, telemetry)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

pub fn write_timeseries_csv<P: AsRef<Path>>(path: P, rows: &[TimeSeriesRow]) -> std::io::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "time_s,dt_ac_min,c_max,u_max,acoustic_energy,kinetic_energy,solid_elastic_energy,solid_kinetic_energy,total_energy,water_volume,max_abs_p,vram_tracked_bytes,fractured_fraction,crater_min_z,subsurface_air_volume,bubble_radius_mm,minnaert_freq_hz"
    )?;
    for r in rows {
        writeln!(
            w,
            "{:.9},{:.9e},{:.4},{:.6},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{},{:.6},{:.6},{:.9e},{:.6},{:.2}",
            r.time_s,
            r.dt_ac_min,
            r.c_max,
            r.u_max,
            r.acoustic_energy,
            r.kinetic_energy,
            r.solid_elastic_energy,
            r.solid_kinetic_energy,
            r.total_energy,
            r.water_volume,
            r.max_abs_p,
            r.vram_tracked_bytes,
            r.fractured_fraction,
            r.crater_min_z,
            r.subsurface_air_volume,
            r.bubble_radius_mm,
            r.minnaert_freq_hz
        )?;
    }
    w.flush()
}
