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
        egui::TopBottomPanel::top("top_header_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("⚡ embedded-nn studio");
                ui.label("• Embedded TinyML Platform");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::widgets::global_theme_preference_buttons(ui);
                });
            });

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.current_tab,
                    StudioTab::Ingest,
                    "1. Ingest & Sensors",
                );
                ui.selectable_value(&mut self.current_tab, StudioTab::Dsp, "2. DSP & Features");
                ui.selectable_value(
                    &mut self.current_tab,
                    StudioTab::Train,
                    "3. Burn QAT Training",
                );
                ui.selectable_value(
                    &mut self.current_tab,
                    StudioTab::Arena,
                    "4. Memory & Hardware",
                );
                ui.selectable_value(&mut self.current_tab, StudioTab::Codegen, "5. Rust Codegen");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            StudioTab::Ingest => self.ingest_view.show(ui),
            StudioTab::Dsp => self.dsp_view.show(ui),
            StudioTab::Train => self.train_view.show(ui),
            StudioTab::Arena => self.arena_view.show(ui),
            StudioTab::Codegen => self.codegen_view.show(ui),
        });
    }
}
