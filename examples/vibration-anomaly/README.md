# Industrial Vibration Anomaly Detection & Functional Safety

Edge AI predictive maintenance and functional safety (ISO 26262 / IEC 61508) for industrial microcontrollers.

---

## Features

1. **Autoencoder Reconstruction Error Scoring (`ReconstructionAnomalyDetector`)**:
   - Calculates fixed-point Mean Squared Error (MSE) between raw sensor vibration input and autoencoder output.
   - Flags anomalies when the reconstruction error exceeds normal baseline distributions.
2. **Multivariate Mahalanobis Distance Scoring (`MahalanobisAnomalyDetector`)**:
   - Compares multi-channel statistical features (RMS acceleration, kurtosis, peak-to-peak, dominant frequency) against calibrated baseline distributions.
3. **ISO 26262 Boot Safety Integrity**:
   - **`verify_weights_integrity`**: Validates Flash weight tables via IEEE 802.3 CRC32 at boot time to detect bitflips.
   - **`verify_arena_integrity`**: Guard canary checks (`0xDEAD_CAFE`) catch activation buffer overruns and stack collisions.

---

## Running the Example

```bash
cargo run --package embedded-nn --example vibration_anomaly --features="libm"
```

Or from within this directory:

```bash
cd examples/vibration-anomaly
cargo run
```
