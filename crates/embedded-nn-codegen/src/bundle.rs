//! Multi-Target Firmware Project Exporter & Bundler.
//!
//! Packages compiled ModelGraphs into ready-to-build C/C++ CMake / Makefile packs
//! and `#![no_std]` Rust crates for seamless drop-in integration into STM32CubeIDE,
//! Keil MDK, Zephyr RTOS, and embedded Cargo firmware projects.

use crate::emit_c::CCodeGenerator;
use crate::emit_rust::RustCodeGenerator;
use embedded_nn_compiler::ir::ModelGraph;
use serde::{Deserialize, Serialize};

/// A generated file inside a project export bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleFile {
    pub path: String,
    pub content: String,
}

/// Generates a complete standalone C99 / C++ CMake project bundle.
pub fn generate_c_project_bundle(model_name: &str, graph: &ModelGraph) -> Vec<BundleFile> {
    let mut files = Vec::new();

    let c_code = CCodeGenerator::new(model_name).generate(graph);
    let name_lower = model_name.to_lowercase();
    let name_upper = model_name.to_uppercase();

    // 1. Header file
    files.push(BundleFile {
        path: format!("include/{}.h", name_lower),
        content: c_code,
    });

    // 2. Main demo driver
    let main_c = format!(
        r#"/* Auto-generated demonstration entrypoint for {model_name} */
#include <stdio.h>
#include <stdint.h>
#include "{name_lower}.h"

static uint8_t g_arena[{name_upper}_ARENA_SIZE_BYTES];
static int8_t g_input[{name_upper}_INPUT_DIM];
static int8_t g_output[{name_upper}_OUTPUT_DIM];

int main(void) {{
    printf("Starting embedded-nn inference: {model_name}\n");
    printf("Input dimension: %d, Output dimension: %d, Arena size: %d bytes\n",
           {name_upper}_INPUT_DIM, {name_upper}_OUTPUT_DIM, {name_upper}_ARENA_SIZE_BYTES);

    // Populate test input
    for (int i = 0; i < {name_upper}_INPUT_DIM; i++) {{
        g_input[i] = 0;
    }}

    int ret = {name_lower}_predict(g_input, g_output, g_arena);
    if (ret != 0) {{
        printf("Inference failed with error code: %d\n", ret);
        return ret;
    }}

    printf("Inference successful! Top output: %d\n", g_output[0]);
    return 0;
}}
"#
    );
    files.push(BundleFile {
        path: "src/main.c".into(),
        content: main_c,
    });

    // 3. CMakeLists.txt
    let cmake = format!(
        r#"cmake_minimum_required(VERSION 3.15)
project({model_name}_firmware C)

set(CMAKE_C_STANDARD 99)

include_directories(include)

add_executable({name_lower}_demo src/main.c)

# Embedded Optimization Flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options({name_lower}_demo PRIVATE -O3 -Wall -Wextra)
endif()
"#
    );
    files.push(BundleFile {
        path: "CMakeLists.txt".into(),
        content: cmake,
    });

    // 4. Makefile
    let makefile = format!(
        r#"CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Iinclude

all: {name_lower}_demo

{name_lower}_demo: src/main.c include/{name_lower}.h
	$(CC) $(CFLAGS) -o $@ src/main.c

clean:
	rm -f {name_lower}_demo
"#
    );
    files.push(BundleFile {
        path: "Makefile".into(),
        content: makefile,
    });

    files
}

/// Generates a standalone `#![no_std]` Rust crate bundle.
pub fn generate_rust_crate_bundle(model_name: &str, graph: &ModelGraph) -> Vec<BundleFile> {
    let mut files = Vec::new();
    let name_lower = model_name.to_lowercase().replace(' ', "_");

    let rust_code = RustCodeGenerator::new(model_name).generate(graph);

    // 1. Cargo.toml
    let cargo_toml = format!(
        r#"[package]
name = "{name_lower}"
version = "0.1.0"
edition = "2024"

[dependencies]
embedded-nn = {{ version = "0.2", default-features = false, features = ["alloc"] }}
"#
    );
    files.push(BundleFile {
        path: "Cargo.toml".into(),
        content: cargo_toml,
    });

    // 2. src/lib.rs
    files.push(BundleFile {
        path: "src/lib.rs".into(),
        content: rust_code,
    });

    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_nn_compiler::builder::ModelBuilder;
    use embedded_nn_compiler::ir::*;

    #[test]
    fn test_generate_c_project_bundle() {
        let mut builder = ModelBuilder::new("TinyNet");
        let in_id = builder.add_input("in", TensorShape::new_1d(16), DataType::Int8, None);
        let fc_id = builder.add_dense_layer(
            "fc",
            in_id,
            2,
            vec![1i8; 32],
            None,
            Some(vec![0i32; 2]),
            ActivationType::None,
            None,
            None,
        );
        builder.mark_output(fc_id);
        let graph = builder.build();

        let bundle = generate_c_project_bundle("TinyNet", &graph);
        assert_eq!(bundle.len(), 4);
        assert!(bundle.iter().any(|f| f.path == "include/tinynet.h"));
        assert!(bundle.iter().any(|f| f.path == "CMakeLists.txt"));
        assert!(bundle.iter().any(|f| f.path == "src/main.c"));
    }

    #[test]
    fn test_generate_rust_crate_bundle() {
        let mut builder = ModelBuilder::new("TinyNet");
        let in_id = builder.add_input("in", TensorShape::new_1d(16), DataType::Int8, None);
        let fc_id = builder.add_dense_layer(
            "fc",
            in_id,
            2,
            vec![1i8; 32],
            None,
            Some(vec![0i32; 2]),
            ActivationType::None,
            None,
            None,
        );
        builder.mark_output(fc_id);
        let graph = builder.build();

        let bundle = generate_rust_crate_bundle("TinyNet", &graph);
        assert_eq!(bundle.len(), 2);
        assert!(bundle.iter().any(|f| f.path == "Cargo.toml"));
        assert!(bundle.iter().any(|f| f.path == "src/lib.rs"));
    }
}
