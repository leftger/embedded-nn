use embedded_nn_codegen::RustCodeGenerator;
use embedded_nn_compiler::ir::ModelGraph;
use proc_macro::TokenStream;
use std::fs;
use std::path::Path;
use syn::{ItemStruct, LitStr, parse_macro_input};

#[proc_macro_attribute]
pub fn embedded_nn_model(attr: TokenStream, item: TokenStream) -> TokenStream {
    let model_path = parse_macro_input!(attr as LitStr);
    let input_struct = parse_macro_input!(item as ItemStruct);
    let struct_name = input_struct.ident.to_string();

    let path_str = model_path.value();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let full_path = Path::new(&manifest_dir).join(&path_str);

    let content = match fs::read_to_string(&full_path) {
        Ok(c) => c,
        Err(err) => {
            return syn::Error::new_spanned(
                model_path,
                format!("Failed to read model file at {:?}: {}", full_path, err),
            )
            .to_compile_error()
            .into();
        }
    };

    let graph: ModelGraph = match serde_json::from_str(&content) {
        Ok(g) => g,
        Err(err) => {
            return syn::Error::new_spanned(
                model_path,
                format!("Failed to parse model JSON: {}", err),
            )
            .to_compile_error()
            .into();
        }
    };

    let codegen = RustCodeGenerator::new(&struct_name);
    let generated_code = codegen.generate(&graph);

    match generated_code.parse::<proc_macro2::TokenStream>() {
        Ok(tokens) => tokens.into(),
        Err(err) => syn::Error::new_spanned(
            input_struct.ident,
            format!("Failed to parse generated tokens: {}", err),
        )
        .to_compile_error()
        .into(),
    }
}
