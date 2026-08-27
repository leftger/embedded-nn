# MicroFlow TFLite fixtures

Vendored copies of MicroFlow's three public models so importer/codegen tests
run in CI instead of silently skipping when a local `microflow-rs` tree is
missing. See [NOTICE](NOTICE) for license and provenance.

| File | Role |
| --- | --- |
| `sine.tflite` | Tiny fully-connected sine approximator |
| `speech.tflite` | Keyword / micro-speech graph |
| `person_detect.tflite` | Visual wake-word person detector |
