#!/usr/bin/env python3
"""
Example: Train a Gesture Recognition 1D-CNN in PyTorch and export with LiteRT (ai-edge-torch).
"""

import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not installed. Run: pip install torch")
    sys.exit(0)

class GestureClassifier(nn.Module):
    """3-axis accelerometer 1D temporal convolution model."""
    def __init__(self, in_channels=3, seq_len=128, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(8 * (seq_len // 2), num_classes)

    def forward(self, x):
        # x shape: [batch, in_channels, seq_len]
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)

def main():
    print("=== PyTorch to LiteRT / embedded-nn Export Demo ===")
    model = GestureClassifier().eval()
    dummy_input = torch.randn(1, 3, 128)

    try:
        import ai_edge_torch
        edge_model = ai_edge_torch.convert(model, (dummy_input,))
        edge_model.export("gesture_model.tflite")
        print("Successfully exported gesture_model.tflite via ai_edge_torch!")
    except ImportError:
        print("\nai_edge_torch not installed in this environment.")
        print("To install LiteRT for PyTorch:")
        print("  pip install ai-edge-torch ai-edge-quantizer")
        print("\nOnce exported, import directly into embedded-nn:")
        print("  cargo run -p embedded-nn-cli -- codegen gesture_model.tflite --name GestureNet")

if __name__ == "__main__":
    main()
