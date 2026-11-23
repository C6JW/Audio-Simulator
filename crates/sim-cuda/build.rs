use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".to_string());
    let nvcc = Path::new(&cuda_path).join("bin/nvcc");
    let cuda_include = Path::new(&cuda_path).join("include");
    let cuda_lib64 = Path::new(&cuda_path).join("lib64");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_dir = manifest_dir.join("cuda");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

    let mut cu_files = Vec::new();
    let mut header_files = Vec::new();
    if let Ok(entries) = fs::read_dir(&cuda_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext == "cu" {
                    cu_files.push(path);
                } else if ext == "cuh" || ext == "h" {
                    header_files.push(path);
                }
            }
        }
    }
    cu_files.sort();
    header_files.sort();

    for h in &header_files {
        println!("cargo:rerun-if-changed={}", h.display());
    }

    let ccbin = env::var("NVCC_CCBIN").ok();
    let mut obj_files = Vec::new();

    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());
        let stem = cu.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.o"));

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-c")
            .arg("-arch=sm_120")
            .arg("-O3")
            .arg("--std=c++17")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-lineinfo")
            .arg(format!("-I{}", cuda_include.display()))
            .arg(format!("-I{}", cuda_dir.display()));

        if let Some(ref compiler) = ccbin {
            cmd.arg("-ccbin").arg(compiler);
        }

        cmd.arg(cu).arg("-o").arg(&obj);

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn nvcc for {}: {e}", cu.display()));
        if !status.success() {
            panic!("nvcc failed compiling {}", cu.display());
        }
        obj_files.push(obj);
    }

    if !obj_files.is_empty() {
        let archive = out_dir.join("libsim_cuda_kernels.a");
        if archive.exists() {
            let _ = fs::remove_file(&archive);
        }
        let mut ar = Command::new("ar");
        ar.arg("rcs").arg(&archive);
        for obj in &obj_files {
            ar.arg(obj);
        }
        let status = ar.status().expect("failed to invoke ar");
        if !status.success() {
            panic!("ar failed building {}", archive.display());
        }

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=sim_cuda_kernels");
    }

    println!("cargo:rustc-link-search=native={}", cuda_lib64.display());
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
