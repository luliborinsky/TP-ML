# WIDER FACE Detection Experiment Results

Evaluation on WIDER FACE dataset (Train: 11,592 images, Val: 1,288 images with 18,099 face annotations).

## Comparison Table

| Run Name | Start Weights | imgsz | Epochs | Batch | Hardware | Time / Epoch | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `wider_face_test` (Partner) | Pretrained | 320 | 5 | 4 | Windows CPU | ~50 min | 0.647 | 0.290 | 0.318 | 0.153 |
| `partner_best_val640` (Partner ep 5, no train) | `best.pt` | 640 | 0 (val only) | 8 | Apple M3 (MPS) | 15.9 s | 0.740 | 0.473 | 0.537 | 0.261 |
| `wider_face_full_1epoch` | `best.pt` (Partner ep 5) | 320 | 1 | 16 | Apple M3 (MPS) | 3.0 min (182 s) | 0.613 | 0.271 | 0.289 | 0.135 |
| `wider_face_640_1epoch` | `best.pt` (Partner ep 5) | 640 | 1 | 8 | Apple M3 (MPS) | 8.9 min (535 s) | 0.705 | 0.445 | 0.506 | 0.240 |

---

## Notes and Observations
- **Train/Val Split Consistency**: Verified identical to partner's split. Running `yolo val` with `best.pt` directly on the local 1,288 validation images gives **mAP50 = 0.319** at 320px, matching partner's reported **0.318**.
- **Resolution Confounder**: The earlier 320 vs 640 comparison mixed training and evaluation resolution: the massive gain came from simply *evaluating* at 640px. The partner's 320-trained weights evaluated at 640px achieve **mAP50 = 0.537** (P: 0.740, R: 0.473). Thus, the true baseline to beat for full 640 training is **0.537** (not 0.318 / 0.289).
- **`max_det` Warning**: WIDER FACE images contain up to 883 annotated faces in a single scene (`val=883`), exceeding YOLO's default `max_det=300`. Ultralytics automatically raised `max_det=883`. Every final model must be validated using the exact same `yolo detect val ... imgsz=640 batch=8 device=mps` command so that all comparison rows use identical `max_det` settings.
- **Hardware Acceleration**: Apple M3 MPS achieves ~4.4 it/s at 320px (~3.0 min/epoch) and ~3.0 it/s at 640px (~8.9 min/epoch), ~15–20× faster than CPU.
- **Next Step**: Train 10 epochs at 640px (`wider_face_640_10epochs`) to beat the true baseline of **0.537**.
