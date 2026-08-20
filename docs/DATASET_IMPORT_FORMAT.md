# Dataset Import Format

`embedded-nn` accepts externally captured sensor data through a single, tool-agnostic
interchange file. Anything that can emit this format — a device log decoder, a Python
capture script, a CSV converter — can feed the Studio training pipeline without
`embedded-nn` knowing anything about the source.

## File format

**JSON Lines** (`.jsonl` / `.ndjson`): one JSON object per line, UTF-8, no enclosing
array. Blank lines are ignored. This keeps files streamable, appendable and diffable.

Each line is a `DatasetRecord`:

```json
{
  "sample_id": "burst_0001",
  "label": null,
  "sample_rate_hz": 400.0,
  "channel_names": ["x", "y", "z"],
  "waveform": [[0.1, 0.0, 1.0], [0.9, -0.2, 1.1], [2.4, 0.6, 0.3]]
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Stable identifier for the capture, unique within the file. |
| `label` | string or `null` | Class name. `null` or empty means the record needs human annotation. |
| `sample_rate_hz` | number | Sampling rate of the waveform. |
| `channel_names` | array of string | One name per sensor channel, e.g. `["x","y","z"]` or `["value"]`. |
| `waveform` | array of array of number | Outer index is the time step, inner index is the channel. Each inner array should have `channel_names.len()` entries. |

Records in one file may differ in length, sample rate and channel layout; the importer
handles each independently.

## Multi-channel collapse

The Studio feature pipeline operates on a single scalar channel. On import, records with
more than one channel are collapsed per time step to the vector magnitude
(`sqrt(x² + y² + …)`); single-channel records pass through unchanged. Export data in
whatever unit is natural for the sensor (g, mg, rad/s, …) — the DSP stage normalises.

## Validating a file

Before opening Studio, sanity-check a file headlessly:

```sh
enn dataset validate dataset.jsonl
```

This prints the record count, time-step range, the channel layouts present, the label
distribution, and warns about records whose time steps do not match `channel_names`.

## Importing and labelling

In Studio's **Data Ingestion** tab, use **📂 Import Dataset File(s)** and select one or
more `.jsonl` files. Imported records become dataset samples; labels not already present
are added as new classes, and unlabelled records land under `unlabeled_import`.

Review and annotate them in the **Dataset Samples Explorer** at the bottom of the same
tab — each row has a class dropdown, so a human can assign or correct the label of every
imported sample before training.
