//! Generic dataset interchange format for importing externally captured sensor data.
//!
//! Files are JSON Lines: one [`DatasetRecord`] object per line. See
//! `docs/DATASET_IMPORT_FORMAT.md` for the specification.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub sample_id: String,
    /// `None` or empty means the record still needs human annotation.
    pub label: Option<String>,
    pub sample_rate_hz: f32,
    /// One name per channel, e.g. `["x", "y", "z"]` or `["value"]`.
    pub channel_names: Vec<String>,
    /// Outer index is the time step, inner index is the channel.
    pub waveform: Vec<Vec<f32>>,
}

impl DatasetRecord {
    /// Collapses the waveform into a single scalar channel: the vector magnitude
    /// across channels per time step, or the lone channel passed through.
    pub fn scalar_channel(&self) -> Vec<f32> {
        self.waveform
            .iter()
            .map(|step| match step.as_slice() {
                [] => 0.0,
                [single] => *single,
                multi => multi.iter().map(|v| v * v).sum::<f32>().sqrt(),
            })
            .collect()
    }

    /// The per-axis motion path, for 3-axis records only. Empty otherwise,
    /// since a scalar or 6-DOF record has no single unambiguous XYZ path.
    pub fn xyz_trajectory(&self) -> Vec<[f32; 3]> {
        if self.channel_count() != 3 {
            return Vec::new();
        }
        self.waveform
            .iter()
            .map(|step| match step.as_slice() {
                [x, y, z] => [*x, *y, *z],
                _ => [0.0; 3],
            })
            .collect()
    }

    /// Number of distinct sensor channels in this record.
    pub fn channel_count(&self) -> usize {
        self.channel_names.len()
    }

    /// Extracts a specific channel across all time steps.
    pub fn channel_slice(&self, channel_idx: usize) -> Vec<f32> {
        self.waveform
            .iter()
            .map(|step| step.get(channel_idx).copied().unwrap_or(0.0))
            .collect()
    }

    /// Computes separate Accelerometer (first 3 channels) and Gyroscope (channels 3..6)
    /// vector magnitudes for 6-DOF IMU sensor fusion.
    pub fn imu_6dof_magnitude_split(&self) -> (Vec<f32>, Vec<f32>) {
        let mut accel_mag = Vec::with_capacity(self.waveform.len());
        let mut gyro_mag = Vec::with_capacity(self.waveform.len());

        for step in &self.waveform {
            let a_sq: f32 = step.iter().take(3).map(|v| v * v).sum();
            let g_sq: f32 = step.iter().skip(3).take(3).map(|v| v * v).sum();
            accel_mag.push(a_sq.sqrt());
            gyro_mag.push(g_sq.sqrt());
        }

        (accel_mag, gyro_mag)
    }

    pub fn label_or(&self, fallback: &str) -> String {
        match self.label.as_deref().map(str::trim) {
            Some(label) if !label.is_empty() => label.to_string(),
            _ => fallback.to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("line {line}: {source}")]
pub struct DatasetParseError {
    pub line: usize,
    #[source]
    pub source: serde_json::Error,
}

/// Parses JSON Lines dataset contents, skipping blank lines.
pub fn parse_jsonl(contents: &str) -> Result<Vec<DatasetRecord>, DatasetParseError> {
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|source| DatasetParseError {
                line: i + 1,
                source,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(waveform: Vec<Vec<f32>>) -> DatasetRecord {
        DatasetRecord {
            sample_id: "s0".into(),
            label: None,
            sample_rate_hz: 100.0,
            channel_names: vec!["x".into(), "y".into(), "z".into()],
            waveform,
        }
    }

    #[test]
    fn scalar_channel_takes_magnitude_of_multi_channel_steps() {
        let r = record(vec![vec![3.0, 4.0, 0.0], vec![0.0, 0.0, 2.0]]);
        assert_eq!(r.scalar_channel(), vec![5.0, 2.0]);
    }

    #[test]
    fn scalar_channel_passes_single_channel_through() {
        let mut r = record(vec![vec![-1.5], vec![0.25]]);
        r.channel_names = vec!["value".into()];
        assert_eq!(r.scalar_channel(), vec![-1.5, 0.25]);
    }

    #[test]
    fn xyz_trajectory_kept_for_three_axis_records_only() {
        let r = record(vec![vec![3.0, 4.0, 0.0], vec![0.0, 0.0, 2.0]]);
        assert_eq!(r.xyz_trajectory(), vec![[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]);

        let mut scalar = record(vec![vec![-1.5], vec![0.25]]);
        scalar.channel_names = vec!["value".into()];
        assert!(scalar.xyz_trajectory().is_empty());
    }

    #[test]
    fn parse_jsonl_skips_blank_lines() {
        let contents = "\n{\"sample_id\":\"a\",\"label\":null,\"sample_rate_hz\":50.0,\"channel_names\":[\"value\"],\"waveform\":[[1.0]]}\n\n";
        let records = parse_jsonl(contents).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].sample_id, "a");
    }

    #[test]
    fn parse_jsonl_reports_offending_line_number() {
        let contents = "{\"sample_id\":\"a\",\"label\":null,\"sample_rate_hz\":50.0,\"channel_names\":[\"value\"],\"waveform\":[[1.0]]}\nnot json\n";
        let err = parse_jsonl(contents).unwrap_err();
        assert_eq!(err.line, 2);
    }

    #[test]
    fn test_imu_6dof_magnitude_split() {
        let r = DatasetRecord {
            sample_id: "s6dof".into(),
            label: Some("swing".into()),
            sample_rate_hz: 100.0,
            channel_names: vec![
                "ax".into(),
                "ay".into(),
                "az".into(),
                "gx".into(),
                "gy".into(),
                "gz".into(),
            ],
            waveform: vec![
                vec![3.0, 4.0, 0.0, 6.0, 8.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
            ],
        };
        let (accel, gyro) = r.imu_6dof_magnitude_split();
        assert_eq!(accel, vec![5.0, 1.0]);
        assert_eq!(gyro, vec![10.0, 2.0]);
        assert_eq!(r.channel_slice(0), vec![3.0, 0.0]);
        assert_eq!(r.channel_slice(3), vec![6.0, 0.0]);
    }

    #[test]
    fn label_or_falls_back_for_missing_and_blank_labels() {
        let mut r = record(vec![]);
        assert_eq!(r.label_or("unlabeled"), "unlabeled");
        r.label = Some("  ".into());
        assert_eq!(r.label_or("unlabeled"), "unlabeled");
        r.label = Some("normal".into());
        assert_eq!(r.label_or("unlabeled"), "normal");
    }
}
