# Phase 2 Model Provenance & Evolution

## Critical Discovery

During documentation, we discovered that `train_improved` is a failed training run with 0% validation metrics (mAP50=0, Precision=0, Recall=0). However, the fine-tuned version `phase2_clothes_v6` recovered and achieved the best performance.

## Model Comparison (Actual Metrics)

| Model | mAP50 | Precision | Recall | Status |
|-------|-------|-----------|--------|--------|
| `deepfashion2_yolo` | 0.369 | 0.491 | 0.482 | V1 - Working |
| `deepfashion2_yolo_v2_optimized` | 0.294 | 0.319 | 0.572 | V2 - Working |
| `train_improved` | **0.000** | **0.000** | **0.000** | **FAILED** |
| `finetune/phase2_clothes_v6` | **0.801** | **0.839** | **0.720** | **PRODUCTION** |

## What Went Wrong with `train_improved`

Looking at the training artifacts:
- **Training loss decreased normally** (box_loss: 1.1→0.5, cls_loss: 2.1→0.35)
- **Validation loss exploded** (val/cls_loss: 100→326 over 100 epochs)
- **All validation metrics stayed at 0** throughout training
- **Confusion matrix**: All predictions classified as "background"

**Likely cause**: Mismatch between training and validation data paths, or corrupted validation set in `deepfashion2_yolo_improved/`.

## Production Model: `phase2_clothes_v6`

Despite being fine-tuned from a broken base model, the Roboflow fine-tuning **successfully trained the model from scratch** on the new annotations.

### Location
```
runs/finetune/phase2_clothes_v6/weights/best.pt
```

### Performance
- **mAP50**: 0.801 (Best of all Phase 2 models!)
- **Precision**: 0.839
- **Recall**: 0.720

### Training Details
- **Base**: `train_improved` (broken, but weights still initialized some features)
- **Fine-tune Data**: Roboflow v6i split (clothes-only subset)
- **Epochs**: 30 with patience=10
- **Freeze**: First 10 layers

## GUI Usage (Correct)

The production GUI correctly uses the working fine-tuned model:
```python
# app_with_preprocessing.py
PATH_CLOTHES_DETECTOR = r"d:/AAI3001/runs/finetune/phase2_clothes_v6/weights/best.pt"
```

## Model Evolution Timeline

```
Phase 1: ResNet50 Classification (15 classes, 91.45% accuracy)
    ↓
Phase 2: YOLOv8 Detection (13 classes)
    │
    ├── train_improved/              [100 epochs, mAP50=0.000] FAILED (Nov 14, 2025)
    │
    └── finetune/phase2_clothes_v6/  [30 epochs, mAP50=0.801] PRODUCTION
    ↓
Phase 3: Fashionpedia Accessories (11 classes, mAP50=0.75)
    ↓
Shoe Classifier: ResNet50 (7 classes, 82.5% accuracy)
```

## Known Limitations

- **Catastrophic Forgetting**: While `phase2_clothes_v6` excels at "Shorts" and "Trousers" (due to fine-tuning), it has lost the ability to detect "Dresses" because they were absent from the fine-tuning dataset.
- **Planned Fix**: Data Mixing (Replay Training) in future iterations.
