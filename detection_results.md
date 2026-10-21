# WIDER FACE Detection Experiment Results

Evaluation on WIDER FACE dataset (Train: 11,592 images, Val: 1,288 images with 18,099 face annotations).

## Comparison Table

| Run Name | Start Weights | imgsz | Epochs | Batch | Hardware | Time / Epoch | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `wider_face_test` (Partner) | Pretrained | 320 | 5 | 4 | Windows CPU | ~50 min | 0.647 | 0.290 | 0.318 | 0.153 |
| `wider_face_full_1epoch` | `best.pt` (Partner ep 5) | 320 | 1 | 16 | Apple M3 (MPS) | 3.0 min (182 s) | 0.613 | 0.271 | 0.289 | 0.135 |
| `wider_face_640_1epoch` | `best.pt` (Partner ep 5) | 640 | 1 | 8 | Apple M3 (MPS) | 8.9 min (535 s) | 0.705 | 0.445 | 0.506 | 0.240 |

---

## Notes and Observations
- **Train/Val Split Consistency**: Verified identical to partner's split. Running `yolo val` with `best.pt` directly on the local 1,288 validation images gives **mAP50 = 0.319**, matching partner's reported **0.318**.
- **Hardware Acceleration**: Apple M3 MPS achieves ~4.4 it/s at 320px (~3.0 min/epoch) and ~3.0 it/s at 640px (~8.9 min/epoch), ~15–20× faster than CPU.
- **Decision Rule Outcome**: `mAP50` jumped from **0.289** (320px) to **0.506** (640px), with Recall jumping from **27.1%** to **44.5%**. Since $0.506 \ge 0.33$, the decision rule triggers: **train 10 epochs at 640px** (estimated total time: ~89 minutes / 1.5 hours).
