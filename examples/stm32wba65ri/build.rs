use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("active_model.rs");

    let explicit_env = env::var("MODEL_PATH").ok().map(PathBuf::from);
    let home_model = env::var("HOME").ok().map(|h| PathBuf::from(h).join("Projects/model.rs"));

    let mut candidate_paths = Vec::new();
    if let Some(p) = explicit_env {
        candidate_paths.push(p);
    }
    if let Some(p) = home_model {
        candidate_paths.push(p);
    }
    candidate_paths.extend(vec![
        PathBuf::from("model.rs"),
        PathBuf::from("models/model.rs"),
        PathBuf::from("src/model_source.rs"),
        PathBuf::from("../../model.rs"),
        PathBuf::from("../../../../model.rs"),
    ]);

    let mut valid_models = Vec::new();

    for candidate in candidate_paths {
        if candidate.exists() && candidate.is_file() {
            println!("cargo:rerun-if-changed={}", candidate.display());
            if let Ok(content) = fs::read_to_string(&candidate) {
                if content.contains("predict") && content.contains("ARENA_SIZE") {
                    let mtime = fs::metadata(&candidate)
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    valid_models.push((mtime, candidate, content));
                }
            }
        }
    }

    // Pick the most recently modified valid model file
    valid_models.sort_by(|a, b| b.0.cmp(&a.0));

    let (model_content, resolved_path_str) = if let Some((_, path, content)) = valid_models.into_iter().next() {
        (Some(content), Some(path.display().to_string()))
    } else {
        (None, None)
    };

    if let Some(content) = model_content {
        // Detect struct name from the model file
        let struct_name = content
            .lines()
            .find_map(|line| {
                let trimmed = line.trim();
                if trimmed.starts_with("pub struct ") {
                    let rest = &trimmed["pub struct ".len()..];
                    let name = rest.trim_end_matches(';').trim();
                    if !name.is_empty() && !name.contains(' ') {
                        return Some(name.to_string());
                    }
                }
                None
            })
            .unwrap_or_else(|| "GestureNeuralNet".to_string());

        let generated = format!(
            "// Model discovered from {}\n\
             {}\n\n\
             #[allow(dead_code)]\n\
             pub type ActiveModel = {};\n\
             #[allow(dead_code)]\n\
             pub type SineFc = {};\n",
            resolved_path_str.as_deref().unwrap_or("unknown"),
            content,
            struct_name,
            struct_name,
        );
        fs::write(&dest_path, generated).expect("Failed to write active_model.rs");
    } else {
        // Fallback default fixture using proc macro
        let default_content = r#"
use embedded_nn_macros::embedded_nn_model;

#[embedded_nn_model("../../crates/embedded-nn-tflite/fixtures/constructed/sine_fc_int8.tflite")]
pub struct SineFc;

#[allow(dead_code)]
pub type ActiveModel = SineFc;
"#;
        fs::write(&dest_path, default_content).expect("Failed to write fallback active_model.rs");
    }
}
