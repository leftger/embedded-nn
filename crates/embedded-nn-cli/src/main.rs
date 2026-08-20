use clap::{Parser, Subcommand};
use embedded_nn_codegen::RustCodeGenerator;
use embedded_nn_compiler::arena::ArenaScheduler;
use embedded_nn_compiler::ir::ModelGraph;
use embedded_nn_live::host::UsbBridge;
use std::collections::BTreeMap;
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
    /// Work with JSON Lines dataset interchange files
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },
}

#[derive(Subcommand)]
enum DatasetAction {
    /// Report record count, channel-shape consistency and label distribution
    Validate { path: PathBuf },
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
        Commands::Dataset {
            action: DatasetAction::Validate { path },
        } => {
            let records = embedded_nn_live::parse_jsonl(&fs::read_to_string(&path)?)
                .map_err(|e| format!("invalid dataset {}: {}", path.display(), e))?;

            let mut labels: BTreeMap<&str, usize> = BTreeMap::new();
            let mut channel_shapes: BTreeMap<String, usize> = BTreeMap::new();
            let mut ragged: Vec<&str> = Vec::new();

            for record in &records {
                *labels
                    .entry(record.label.as_deref().unwrap_or("<unlabeled>"))
                    .or_default() += 1;
                *channel_shapes
                    .entry(record.channel_names.join(","))
                    .or_default() += 1;
                if record
                    .waveform
                    .iter()
                    .any(|step| step.len() != record.channel_names.len())
                {
                    ragged.push(&record.sample_id);
                }
            }

            println!("==================================================");
            println!("  Dataset: {:?}", path);
            println!("==================================================");
            println!("Records:               {}", records.len());
            println!(
                "Time steps (min/max):  {}/{}",
                records.iter().map(|r| r.waveform.len()).min().unwrap_or(0),
                records.iter().map(|r| r.waveform.len()).max().unwrap_or(0)
            );
            println!("--------------------------------------------------");
            println!("Channel layouts:");
            for (names, count) in &channel_shapes {
                println!("  [{}] -> {} record(s)", names, count);
            }
            println!("--------------------------------------------------");
            println!("Label distribution:");
            for (label, count) in &labels {
                println!("  {:<24} {} record(s)", label, count);
            }
            if !ragged.is_empty() {
                println!("--------------------------------------------------");
                println!(
                    "WARNING: {} record(s) have time steps that do not match channel_names:",
                    ragged.len()
                );
                for id in &ragged {
                    println!("  {}", id);
                }
            }
            println!("==================================================");
        }
    }

    Ok(())
}
