use crate::state::StudioState;
use crate::theme::configure_theme;
use crate::views::arena::ArenaView;
use crate::views::codegen::CodegenView;
use crate::views::dsp::DspView;
use crate::views::ingest::IngestView;
use crate::views::train::TrainView;
use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioTab {
    Ingest,
    Dsp,
    Train,
    Arena,
    Codegen,
}

pub struct EmbeddedNnStudioApp {
    pub current_tab: StudioTab,
    pub state: StudioState,
    pub ingest_view: IngestView,
    pub dsp_view: DspView,
    pub train_view: TrainView,
    pub arena_view: ArenaView,
    pub codegen_view: CodegenView,
}

impl Default for EmbeddedNnStudioApp {
    fn default() -> Self {
        Self {
            current_tab: StudioTab::Ingest,
            state: StudioState::default(),
            ingest_view: IngestView::new(),
            dsp_view: DspView::new(),
            train_view: TrainView::new(),
            arena_view: ArenaView::new(),
            codegen_view: CodegenView::new(),
        }
    }
}

impl eframe::App for EmbeddedNnStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("studio_top_header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("⚡ embedded-nn studio");
                ui.label("• Embedded TinyML Platform");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::widgets::global_theme_preference_buttons(ui);
                });
            });

            ui.add_space(4.0);

            // Step-by-Step Pipeline Navigation Bar with Live Status Badges
            ui.horizontal(|ui| {
                let sample_count = self.state.samples.len();
                let mel_bins = self.state.dsp.num_mel_bins;
                let latest_acc = self.state.val_acc_history.last().copied().unwrap_or(0.0);
                let arena_bytes = self
                    .state
                    .arena_plan
                    .as_ref()
                    .map(|p| p.total_arena_bytes)
                    .unwrap_or(0);

                let tab_1_label = format!("1. Ingest ({} samples)", sample_count);
                let tab_2_label = format!("2. DSP ({} bins)", mel_bins);
                let tab_3_label = format!("3. Train ({:.0}% acc)", latest_acc);
                let tab_4_label = format!("4. Arena ({} B)", arena_bytes);
                let tab_5_label = "5. Rust Codegen & HIL".to_string();

                ui.selectable_value(&mut self.current_tab, StudioTab::Ingest, tab_1_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Dsp, tab_2_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Train, tab_3_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Arena, tab_4_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Codegen, tab_5_label);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            StudioTab::Ingest => self.ingest_view.show(ui, &mut self.state),
            StudioTab::Dsp => self.dsp_view.show(ui, &mut self.state),
            StudioTab::Train => self.train_view.show(ui, &mut self.state),
            StudioTab::Arena => self.arena_view.show(ui, &mut self.state),
            StudioTab::Codegen => self.codegen_view.show(ui, &mut self.state),
        });

        // Request continuous frame redraw when recording or training is actively running
        if self.ingest_view.is_recording || self.state.is_training {
            ctx.request_repaint();
        }
    }
}

pub fn run_studio() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 720.0])
            .with_min_inner_size([850.0, 520.0])
            .with_title("embedded-nn Studio - TinyML Development Platform"),
        ..Default::default()
    };

    eframe::run_native(
        "embedded-nn Studio",
        native_options,
        Box::new(|cc| {
            configure_theme(&cc.egui_ctx);
            Ok(Box::new(EmbeddedNnStudioApp::default()))
        }),
    )
}
