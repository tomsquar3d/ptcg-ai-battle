"""Neural layers (docs §3 Layers 3-4): state/action featurization → encoder → policy/value heads.

Featurization (`features.py`) is pure Python + numpy — the belief-engine → network interface — and
carries no torch dependency so it stays testable and importable everywhere. The encoder/heads
(torch) consume it; ONNX-on-CPU is the deployment path.
"""
