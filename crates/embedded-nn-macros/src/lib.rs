use embedded_nn_codegen::RustCodeGenerator;
use embedded_nn_compiler::ir::ModelGraph;
use proc_macro::TokenStream;
use std::path::{Path, PathBuf};
use syn::{ItemStruct, LitStr, parse_macro_input};

fn parse_model_path(attr: TokenStream) -> Result<String, syn::Error> {
    let lit = syn::parse::<LitStr>(attr)?;
    Ok(lit.value())
}

fn resolve_model_path(manifest_dir: &str, relative: &str) -> PathBuf {
    Path::new(manifest_dir).join(relative)
}

fn load_model_graph(path: &Path) -> Result<ModelGraph, String> {
    let bytes = std::fs::read(path)
        .map_err(|err| format!("Failed to read model file at {:?}: {}", path, err))?;

    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("json") => serde_json::from_slice(&bytes)
            .map_err(|err| format!("Failed to parse model JSON: {}", err)),
        Some(extension)
            if extension.eq_ignore_ascii_case("tflite")
                || extension.eq_ignore_ascii_case("bin") =>
        {
            embedded_nn_tflite::import_tflite(&bytes)
                .map_err(|err| format!("Failed to import TFLite model: {}", err))
        }
        Some(extension) => Err(format!(
            "Unsupported model file extension .{}; expected .json, .tflite, or .bin",
            extension
        )),
        None => Err("Model file has no extension; expected .json, .tflite, or .bin".into()),
    }
}

fn generate_code(graph: &ModelGraph, struct_name: &str) -> String {
    let codegen = RustCodeGenerator::new(struct_name);
    codegen.generate(graph)
}

fn parse_generated_tokens(code: &str) -> Result<proc_macro2::TokenStream, String> {
    code.parse::<proc_macro2::TokenStream>()
        .map_err(|err| format!("Failed to parse generated tokens: {}", err))
}

fn dump_generated_code_to(out_dir: &Path, code: &str) -> std::io::Result<()> {
    std::fs::write(out_dir.join("embedded-nn-expansion.rs"), code)
}

fn dump_generated_code(code: &str) {
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let _ = dump_generated_code_to(Path::new(&out_dir), code);
    }
}

#[proc_macro_attribute]
pub fn embedded_nn_model(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_struct = parse_macro_input!(item as ItemStruct);
    let struct_name = input_struct.ident.to_string();

    let path_str = match parse_model_path(attr) {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let full_path = resolve_model_path(&manifest_dir, &path_str);

    let graph = match load_model_graph(&full_path) {
        Ok(g) => g,
        Err(err) => {
            return syn::Error::new_spanned(&input_struct.ident, err)
                .to_compile_error()
                .into();
        }
    };

    let generated_code = generate_code(&graph, &struct_name);
    dump_generated_code(&generated_code);

    match parse_generated_tokens(&generated_code) {
        Ok(tokens) => tokens.into(),
        Err(err) => syn::Error::new_spanned(&input_struct.ident, err)
            .to_compile_error()
            .into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_nn_compiler::ir::*;

    /// Writes `contents` to a uniquely-named file under the OS temp dir and returns its path.
    fn write_temp_file(name: &str, contents: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "embedded_nn_macros_test_{}_{}",
            std::process::id(),
            name
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn test_resolve_model_path() {
        let resolved = resolve_model_path("/workspace/my-crate", "models/gesture.json");
        assert_eq!(
            resolved,
            PathBuf::from("/workspace/my-crate/models/gesture.json")
        );
    }

    #[test]
    fn test_load_model_graph_success() {
        let graph = ModelGraph::new("tiny_net");
        let json = serde_json::to_string(&graph).unwrap();

        let path = write_temp_file("valid.json", &json);
        let loaded = load_model_graph(&path).unwrap();
        assert_eq!(loaded.name, "tiny_net");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_model_graph_malformed_json() {
        let path = write_temp_file("malformed.json", "{ not valid json");
        let err = load_model_graph(&path).unwrap_err();
        assert!(err.contains("Failed to parse model JSON"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_model_graph_missing_file() {
        let err = load_model_graph(Path::new("/nonexistent/path/model.json")).unwrap_err();
        assert!(err.contains("Failed to read model file"));
    }

    #[test]
    fn test_load_model_graph_dispatches_tflite_and_bin_to_importer() {
        for extension in ["tflite", "bin"] {
            let path = write_temp_file(&format!("invalid.{extension}"), "not a flatbuffer");
            let err = load_model_graph(&path).unwrap_err();
            assert!(err.contains("Failed to import TFLite model"), "{err}");
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn test_load_model_graph_rejects_unknown_extension() {
        let path = write_temp_file("model.txt", "{}");
        let err = load_model_graph(&path).unwrap_err();
        assert!(
            err.contains("Unsupported model file extension .txt"),
            "{err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_dump_generated_code_writes_expected_file() {
        let out_dir = std::env::temp_dir().join(format!(
            "embedded_nn_macros_dump_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&out_dir).unwrap();

        let path = out_dir.join("embedded-nn-expansion.rs");
        dump_generated_code_to(&out_dir, "pub struct Dumped;").unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "pub struct Dumped;"
        );
        let _ = std::fs::remove_dir_all(out_dir);
    }

    #[test]
    fn test_generate_code_contains_markers() {
        let mut builder = embedded_nn_compiler::builder::ModelBuilder::new("test_net");
        let in_id = builder.add_input("input", TensorShape::new_1d(4), DataType::Int8, None);
        let fc_id = builder.add_dense_layer(
            "dense1",
            in_id,
            2,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            None,
            Some(vec![0, 0]),
            ActivationType::None,
            None,
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", fc_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = generate_code(&graph, "TestNet");
        assert!(code.contains("pub struct TestNet"));
        assert!(code.contains("fully_connected_s8"));
        assert!(code.contains("softmax_s8"));
    }

    #[test]
    fn test_parse_generated_tokens_roundtrip() {
        let tokens = parse_generated_tokens("pub struct Foo;").unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_parse_generated_tokens_invalid() {
        let err = parse_generated_tokens("pub struct Foo {").unwrap_err();
        assert!(err.contains("Failed to parse generated tokens"));
    }
}
