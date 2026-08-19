use clap::{Parser, Subcommand};
use embedded_nn_codegen::RustCodeGenerator;
use embedded_nn_compiler::arena::ArenaScheduler;
use embedded_nn_compiler::ir::ModelGraph;
use embedded_nn_live::host::UsbBridge;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "enn")]
#[command(about = "embedded-nn TinyML Command-Line Tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate #![no_std] Rust inference code from a model graph
    Codegen {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long, default_value = "EmbeddedModel")]
        name: String,
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Inspect static memory footprint (Flash weights vs SRAM arena)
    Profile {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// List connected USB devices for live streaming
    Devices,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Codegen { model, name, out } => {
            let json_content = fs::read_to_string(&model)?;
            let graph: ModelGraph = serde_json::from_str(&json_content)?;
            let codegen = RustCodeGenerator::new(&name);
            let code = codegen.generate(&graph);

            if let Some(out_path) = out {
                fs::write(&out_path, &code)?;
                println!("Generated Rust inference code written to {:?}", out_path);
            } else {
                println!("{}", code);
            }
        }
        Commands::Profile { model } => {
            let json_content = fs::read_to_string(&model)?;
            let graph: ModelGraph = serde_json::from_str(&json_content)?;
            let plan = ArenaScheduler::schedule(&graph);

            println!("==================================================");
            println!("  embedded-nn Static Memory Profile: {}", graph.name);
            println!("==================================================");
            println!("Total Layers:          {}", graph.layers.len());
            println!(
                "Total Weights (Flash): {} bytes",
                graph.total_weights_size_bytes()
            );
            println!(
                "Peak Arena (SRAM):     {} bytes (zero heap allocation)",
                plan.total_arena_bytes
            );
            println!("--------------------------------------------------");
            println!("Layer Allocations:");
            for (id, alloc) in &plan.allocations {
                println!(
                    "  [Tensor {:02}] {:<16} offset: 0x{:04x}, size: {:>5} B, lifetime: [{}, {}]",
                    id,
                    alloc.name,
                    alloc.byte_offset,
                    alloc.byte_size,
                    alloc.lifetime.start_step,
                    alloc.lifetime.end_step
                );
            }
            println!("==================================================");
        }
        Commands::Devices => {
            println!("Scanning for USB devices...");
            let devs = UsbBridge::list_devices();
            if devs.is_empty() {
                println!("No devices detected.");
            } else {
                for (i, dev) in devs.iter().enumerate() {
                    println!("  [{}] {}", i, dev);
                }
            }
        }
    }

    Ok(())
}
