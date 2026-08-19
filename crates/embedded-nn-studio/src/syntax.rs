//! Custom lightweight Rust syntax highlighter for egui text views.

use eframe::egui::{self, Color32, FontId, TextFormat};

/// Formats generated Rust code with syntax highlighting into an `egui::text::LayoutJob`.
pub fn highlight_rust(ctx: &egui::Context, text: &str) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let default_font = FontId::monospace(12.5);
    let is_dark = ctx.style().visuals.dark_mode;

    let col_keyword = if is_dark {
        Color32::from_rgb(255, 120, 160)
    } else {
        Color32::from_rgb(200, 30, 90)
    };
    let col_type = if is_dark {
        Color32::from_rgb(100, 220, 240)
    } else {
        Color32::from_rgb(10, 130, 200)
    };
    let col_fn = if is_dark {
        Color32::from_rgb(140, 200, 255)
    } else {
        Color32::from_rgb(30, 100, 230)
    };
    let col_macro = if is_dark {
        Color32::from_rgb(255, 180, 100)
    } else {
        Color32::from_rgb(200, 100, 20)
    };
    let col_string = if is_dark {
        Color32::from_rgb(150, 230, 150)
    } else {
        Color32::from_rgb(40, 150, 60)
    };
    let col_comment = if is_dark {
        Color32::from_rgb(110, 120, 135)
    } else {
        Color32::from_rgb(130, 140, 150)
    };
    let col_num = if is_dark {
        Color32::from_rgb(255, 210, 120)
    } else {
        Color32::from_rgb(210, 130, 20)
    };
    let col_bracket = if is_dark {
        Color32::from_rgb(200, 210, 225)
    } else {
        Color32::from_rgb(60, 70, 85)
    };
    let col_text = if is_dark {
        Color32::from_rgb(220, 225, 235)
    } else {
        Color32::from_rgb(30, 35, 45)
    };

    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            job.append(
                line,
                0.0,
                TextFormat {
                    font_id: default_font.clone(),
                    color: col_comment,
                    ..Default::default()
                },
            );
            job.append(
                "\n",
                0.0,
                TextFormat {
                    font_id: default_font.clone(),
                    color: col_comment,
                    ..Default::default()
                },
            );
            continue;
        }

        let mut in_string = false;
        let mut string_buf = String::new();
        let mut word_buf = String::new();
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            if in_string {
                string_buf.push(ch);
                if ch == '"' && (i == 0 || chars[i - 1] != '\\') {
                    in_string = false;
                    job.append(
                        &string_buf,
                        0.0,
                        TextFormat {
                            font_id: default_font.clone(),
                            color: col_string,
                            ..Default::default()
                        },
                    );
                    string_buf.clear();
                }
                i += 1;
                continue;
            }

            if ch == '"' {
                if !word_buf.is_empty() {
                    let color = classify_rust_word(
                        &word_buf,
                        col_keyword,
                        col_type,
                        col_fn,
                        col_macro,
                        col_num,
                        col_text,
                    );
                    job.append(
                        &word_buf,
                        0.0,
                        TextFormat {
                            font_id: default_font.clone(),
                            color,
                            ..Default::default()
                        },
                    );
                    word_buf.clear();
                }
                in_string = true;
                string_buf.push(ch);
                i += 1;
                continue;
            }

            if ch == '{'
                || ch == '}'
                || ch == '('
                || ch == ')'
                || ch == '['
                || ch == ']'
                || ch == ';'
                || ch == ','
                || ch == ':'
                || ch == '&'
                || ch == '*'
            {
                if !word_buf.is_empty() {
                    let color = classify_rust_word(
                        &word_buf,
                        col_keyword,
                        col_type,
                        col_fn,
                        col_macro,
                        col_num,
                        col_text,
                    );
                    job.append(
                        &word_buf,
                        0.0,
                        TextFormat {
                            font_id: default_font.clone(),
                            color,
                            ..Default::default()
                        },
                    );
                    word_buf.clear();
                }
                job.append(
                    &ch.to_string(),
                    0.0,
                    TextFormat {
                        font_id: default_font.clone(),
                        color: col_bracket,
                        ..Default::default()
                    },
                );
                i += 1;
                continue;
            }

            if ch.is_whitespace() {
                if !word_buf.is_empty() {
                    let color = classify_rust_word(
                        &word_buf,
                        col_keyword,
                        col_type,
                        col_fn,
                        col_macro,
                        col_num,
                        col_text,
                    );
                    job.append(
                        &word_buf,
                        0.0,
                        TextFormat {
                            font_id: default_font.clone(),
                            color,
                            ..Default::default()
                        },
                    );
                    word_buf.clear();
                }
                job.append(
                    &ch.to_string(),
                    0.0,
                    TextFormat {
                        font_id: default_font.clone(),
                        color: col_text,
                        ..Default::default()
                    },
                );
                i += 1;
                continue;
            }

            word_buf.push(ch);
            i += 1;
        }

        if in_string && !string_buf.is_empty() {
            job.append(
                &string_buf,
                0.0,
                TextFormat {
                    font_id: default_font.clone(),
                    color: col_string,
                    ..Default::default()
                },
            );
        } else if !word_buf.is_empty() {
            let color = classify_rust_word(
                &word_buf,
                col_keyword,
                col_type,
                col_fn,
                col_macro,
                col_num,
                col_text,
            );
            job.append(
                &word_buf,
                0.0,
                TextFormat {
                    font_id: default_font.clone(),
                    color,
                    ..Default::default()
                },
            );
        }

        job.append(
            "\n",
            0.0,
            TextFormat {
                font_id: default_font.clone(),
                color: col_text,
                ..Default::default()
            },
        );
    }

    job
}

fn classify_rust_word(
    word: &str,
    kw_col: Color32,
    type_col: Color32,
    fn_col: Color32,
    macro_col: Color32,
    num_col: Color32,
    text_col: Color32,
) -> Color32 {
    match word {
        "pub" | "struct" | "impl" | "fn" | "let" | "mut" | "match" | "if" | "else" | "return"
        | "use" | "self" | "Self" | "const" | "enum" | "for" | "in" | "where" | "as" | "true"
        | "false" | "mod" | "trait" | "type" | "ref" | "move" | "unsafe" | "static" => kw_col,
        "Ok"
        | "Err"
        | "Some"
        | "None"
        | "Result"
        | "Option"
        | "String"
        | "Vec"
        | "u8"
        | "u16"
        | "u32"
        | "u64"
        | "usize"
        | "i8"
        | "i16"
        | "i32"
        | "i64"
        | "isize"
        | "f32"
        | "f64"
        | "bool"
        | "char"
        | "str"
        | "Activation"
        | "Dims"
        | "FcParams"
        | "PerTensorQuantParams"
        | "GestureClassifier"
        | "GestureNeuralNet" => type_col,
        s if s.ends_with('!') => macro_col,
        s if s.chars().all(|c| {
            c.is_ascii_digit() || c == '.' || c == '_' || c == 'x' || c == 'X' || c == '-'
        }) =>
        {
            num_col
        }
        s if s.chars().next().is_some_and(|c| c.is_uppercase()) => type_col,
        "new" | "predict" | "fully_connected_s8" | "fully_connected_s4" | "softmax_s8"
        | "convolve_s8" | "unwrap" => fn_col,
        _ => text_col,
    }
}
