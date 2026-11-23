pub mod audio;
pub mod obj_loader;
pub mod render_export;
pub mod telemetry;
pub mod video;

pub use audio::{
    apply_highpass_2nd_order, decimate_kaiser, generate_fallback_speech_phrase,
    prepare_probe_wav_signal_f64, read_wav_f32, read_wav_mono_f32, resample_kaiser_upsample,
    resample_sinc_f32, write_probe_wav_f32, write_probe_wav_f32_with_gain, AudioIoError,
    WavWriteStats,
};
pub use hound::SampleFormat;
pub use obj_loader::{ObjError, ObjMesh};
pub use render_export::{
    extract_surface_nets_isosurface, write_ply_mesh, write_pressure_slice_f32,
    write_render_manifest, RenderBodyManifest, RenderExportError, RenderManifest, RenderMicManifest,
    RenderSourceManifest,
};
pub use telemetry::{
    write_telemetry_json, write_timeseries_csv, ProbeAcousticSummary, RunTelemetry, TimeSeriesRow,
};
pub use video::{
    compose_showcase_reel, compose_side_by_side_comparison, mux_realtime_video_with_audio,
    mux_video_with_audio, probe_media_duration, render_frames_with_blender,
    render_preview_frames_with_blender, AudioSlowmoMode, VideoPipelineError, VideoRenderOptions,
};
