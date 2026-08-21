use clap::{Parser, Subcommand};
use embedded_nn_codegen::RustCodeGenerator;
use embedded_nn_compiler::arena::ArenaScheduler;
use embedded_nn_compiler::dsp_contract::DspContract;
use embedded_nn_compiler::ir::ModelGraph;
use embedded_nn_live::host::{OwnedMsg, Transport, UsbBridge, handshake};
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
    /// Import a TensorFlow Lite (.tflite) model into embedded-nn's JSON ModelGraph format,
    /// ready for `enn codegen`
    Import {
        #[arg(short, long)]
        tflite: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// List connected embedded-nn USB-HS bulk agents (VID 0x1209 / PID 0xE612)
    Devices,
    /// Hardware-in-the-loop over the live USB protocol
    Hil {
        #[command(subcommand)]
        action: HilAction,
    },
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

#[derive(Subcommand)]
enum HilAction {
    /// Handshake and Ping/Pong with the first (or selected) agent
    Ping {
        #[arg(long)]
        device: Option<String>,
    },
    /// Run integer inference on the device
    Infer {
        #[arg(long)]
        device: Option<String>,
        /// Comma-separated i8 input tensor
        #[arg(long)]
        input: String,
        #[arg(long, default_value_t = 0)]
        model_id: u32,
    },
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
                let sidecar = out_path.with_file_name("dsp_contract.json");
                let contract = DspContract {
                    version: DspContract::VERSION,
                    window_type: "hann".into(),
                    window_size: 64,
                    num_mel_bins: 16,
                    high_pass_cutoff_hz: 10.0,
                    sample_rate_hz: 100.0,
                    frame_hop_size: 32,
                    capture_samples: 256,
                    input_scale: 1.0 / 127.0,
                    input_zero_point: 0,
                };
                fs::write(&sidecar, serde_json::to_string_pretty(&contract)?)?;
                println!("Generated Rust inference code written to {:?}", out_path);
                println!("DSP contract written to {:?}", sidecar);
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
        Commands::Import { tflite, out } => {
            let bytes = fs::read(&tflite)?;
            let graph = embedded_nn_tflite::import_tflite(&bytes)
                .map_err(|e| format!("failed to import {:?}: {}", tflite, e))?;
            let json = serde_json::to_string_pretty(&graph)?;
            fs::write(&out, json)?;
            println!(
                "Imported {:?} ({} layers) -> {:?}",
                tflite,
                graph.layers.len(),
                out
            );
        }
        Commands::Devices => {
            println!("Scanning for embedded-nn agents (1209:e612)...");
            let devs = UsbBridge::enumerate_nn_agents()?;
            if devs.is_empty() {
                println!("No agents detected.");
            } else {
                for (i, dev) in devs.iter().enumerate() {
                    println!("  [{}] {}  {}", i, dev.stable_id(), dev.display_name());
                }
            }
        }
        Commands::Hil { action } => match action {
            HilAction::Ping { device } => {
                let mut transport = open_agent(device.as_deref())?;
                let ready = handshake(&mut transport, 0, 0, 0)?;
                println!("Ready: {ready:?}");
                transport.send(&OwnedMsg::Ping)?;
                match transport.receive()? {
                    OwnedMsg::Pong => println!("Pong"),
                    other => println!("unexpected {other:?}"),
                }
            }
            HilAction::Infer {
                device,
                input,
                model_id,
            } => {
                let bytes = parse_i8_csv(&input)?;
                let mut transport = open_agent(device.as_deref())?;
                let ready = handshake(&mut transport, model_id, bytes.len() as u32, 0)?;
                println!("Ready: {ready:?}");
                transport.send(&OwnedMsg::RunInference {
                    seq: 1,
                    model_id,
                    input: bytes,
                })?;
                match transport.receive()? {
                    OwnedMsg::InferenceResult {
                        seq,
                        execution_cycles,
                        execution_time_us,
                        logits,
                        ..
                    } => {
                        println!(
                            "seq={seq} cycles={execution_cycles} time_us={execution_time_us} logits={logits:?}"
                        );
                    }
                    other => println!("unexpected {other:?}"),
                }
            }
        },
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

fn open_agent(
    device: Option<&str>,
) -> Result<embedded_nn_live::host::UsbTransport, Box<dyn std::error::Error>> {
    let agents = UsbBridge::enumerate_nn_agents()?;
    let chosen = if let Some(id) = device {
        agents
            .into_iter()
            .find(|agent| agent.stable_id() == id || agent.display_name() == id)
            .ok_or_else(|| format!("no agent matching {id}"))?
    } else {
        agents.into_iter().next().ok_or("no embedded-nn agent connected")?
    };
    println!("Opening {}", chosen.stable_id());
    Ok(chosen.open()?)
}

fn parse_i8_csv(input: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    input
        .split(',')
        .map(|part| {
            let value: i8 = part.trim().parse()?;
            Ok::<u8, Box<dyn std::error::Error>>(value as u8)
        })
        .collect()
}
