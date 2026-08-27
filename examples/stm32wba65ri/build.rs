use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn is_valid_model(content: &str) -> bool {
    content.contains("predict") && content.contains("ARENA_SIZE")
}

fn load_valid_model(path: &Path) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    is_valid_model(&content).then_some(content)
}

fn emit_active_model(dest: &Path, source_path: &str, content: &str) {
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
        "// Model discovered from {source_path}\n\
         {content}\n\n\
         #[allow(dead_code)]\n\
         pub type ActiveModel = {struct_name};\n\
         #[allow(dead_code)]\n\
         pub type SineFc = {struct_name};\n"
    );
    fs::write(dest, generated).expect("Failed to write active_model.rs");
}

fn write_sine_fallback(dest: &Path) {
    let default_content = r#"
use embedded_nn_macros::embedded_nn_model;

#[embedded_nn_model("../../crates/embedded-nn-tflite/fixtures/constructed/sine_fc_int8.tflite")]
pub struct SineFc;

#[allow(dead_code)]
pub type ActiveModel = SineFc;
"#;
    fs::write(dest, default_content).expect("Failed to write fallback active_model.rs");
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("active_model.rs");

    println!("cargo:rerun-if-env-changed=MODEL_PATH");

    if let Ok(explicit) = env::var("MODEL_PATH") {
        let path = PathBuf::from(&explicit);
        println!("cargo:rerun-if-changed={}", path.display());
        match load_valid_model(&path) {
            Some(content) => {
                println!(
                    "cargo:warning=embedded-nn: using MODEL_PATH={}",
                    path.display()
                );
                emit_active_model(&dest_path, &path.display().to_string(), &content);
            }
            None => panic!(
                "MODEL_PATH={} is missing or is not a generated model (needs `predict` and `ARENA_SIZE`)",
                path.display()
            ),
        }
        return;
    }

    let in_tree = [
        PathBuf::from("src/model_source.rs"),
        PathBuf::from("model.rs"),
        PathBuf::from("models/model.rs"),
    ];
    for candidate in &in_tree {
        println!("cargo:rerun-if-changed={}", candidate.display());
        if candidate.is_file()
            && let Some(content) = load_valid_model(candidate)
        {
            println!(
                "cargo:warning=embedded-nn: using in-tree model {}",
                candidate.display()
            );
            emit_active_model(&dest_path, &candidate.display().to_string(), &content);
            return;
        }
    }

    println!(
        "cargo:warning=embedded-nn: no in-tree model.rs found; falling back to sine_fc_int8.tflite (set MODEL_PATH to override)"
    );
    write_sine_fallback(&dest_path);
}
