# Nexar Crash Prediction: Contributor-Selected Wide-Crop Model

This is the contributor's selected **MobileNetV3-Small + GRU32** candidate for team comparison, **not the team's final model**. The checkpoint and recorded results are included; the video dataset is external.

Start with [MobileNetV3_GRU_Wide.ipynb](../advanced_tasks/MobileNetV3_GRU_Wide.ipynb). Read the [experiment report](../docs/experiment-report.md) for the full experimental history and limitations.

## Recorded performance

| Split | Videos | Binary-F1 | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|
| Validation | 300 | **75.25%** | 75.00% | 75.00% |
| Public test | 667 | **69.87%** | 68.94% | 68.97% |

Binary-F1 is the positive-class F1. Validation Macro-F1 rounds from 74.9975%; validation accuracy is exactly 75%. Private test has not been evaluated.

The selected checkpoint is **seed 42, epoch 7, crop 224 x 384, without EMA**. It was selected by validation Binary-F1, breaking ties with lower validation loss. Public evaluation used this fixed checkpoint and argmax, without test-threshold tuning. The earlier square-crop reference had validation Binary-F1 72.61% and public Binary-F1 70.80%. The wide model improved validation performance but did **not** improve public Binary-F1. The protected square model is not included in this contribution.

## Run the notebook

1. Use the existing `quera-gpu` conda environment and its Jupyter kernel. The verified dependencies are Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, NumPy 1.26.4, OpenCV 4.8.1, Matplotlib 3.10.9 and ipykernel. The recorded GPU environment uses CUDA 12.4. Package installation, if needed, should use conda; no environment file is included.
2. Open `advanced_tasks/MobileNetV3_GRU_Wide.ipynb` from this repository. Keep the repository folder structure intact.
3. Run the notebook cells in order. The setup cell finds `NexarCrashPrediction` from the repository root or the notebook directory. If needed, set `PROJECT_DIR` to that folder.
4. Leave `VIDEO_PATH = None` to load the saved checkpoint, display recorded metrics and plot the saved training history without processing any videos. Set it to a video path only when you want a single-video prediction.

The notebook contains its own checkpoint-loading and prediction functions and imports only `nexar_pipeline.py` from this package. No separate inference script is needed. Saved-model loading uses `weights=None` and does not download ImageNet weights. CUDA is selected when available; CPU loading is supported, although numerical probabilities may differ slightly across devices. The notebook verifies the checkpoint SHA256 and the pipeline SHA256; the latter normalizes CRLF to LF for Windows/Linux Git checkouts.

## Model and inputs

- ImageNet-initialized MobileNetV3-Small encoder, global average pooling, GRU with 32 hidden units, dropout 0.4 and a two-class linear head; 985,634 parameters.
- Four chronological frames. Inference samples the final 1.5 seconds of the supplied video; no class label or event timestamp is required.
- Resize the shorter edge to 256, then center-crop 224 x 384 with bilinear interpolation and antialiasing; apply ImageNet normalization. The wider crop retains more lateral context than the earlier square crop.
- Fixed development partition: 1,200 training and 300 validation videos, balanced by class. The shared [split](../src/video_split.csv) remains unchanged.
- Development positives use frames 2, 1.5, 1 and 0.5 seconds before the annotated event. Negatives use the final 1.5 seconds. The development/inference endpoint mismatch is a limitation.

## Recorded training protocol

Epochs 1-2 train only the GRU and classifier. From epoch 3, encoder blocks 11/12 are also fine-tuned while BatchNorm running statistics remain frozen. Training used seed 42, batch size 8, AdamW, weight decay 0.01, the archived cosine learning-rate schedule, at most 12 epochs and patience 3. Early stopping ended the completed run at epoch 10; epoch 7 was selected.

The notebook plots recorded train/validation loss and accuracy over those 10 epochs. It does not start training or evaluate a dataset. Training functions and the original settings are retained for inspection, but this contribution does not include a standalone training launcher or development manifest and does not claim turnkey training reproduction.

## Included artifacts

- `nexar_pipeline.py`: model, frame decoding, sampling and original training functions.
- `model/best_model.pt`: selected checkpoint.
- `model/metadata.json`: model identity, source run and checksums.
- `model/settings.json`: original archived training configuration. Its validation-only evaluation scope describes the training run; the subsequent public evaluation is recorded separately in `results/metrics.json`.
- `results/`: recorded metrics, training history, validation/public predictions and the dataset-relative public manifest.
- `../advanced_tasks/MobileNetV3_GRU_Wide.ipynb`: presentation, saved-model loading, optional single-video inference and recorded curves.
- `../docs/experiment-report.md`: experiment history and interpretation.

Dataset videos, caches, optimizer checkpoints, historical experiment folders and the protected BestModel package are not included. Result manifests contain dataset-relative paths and are used as provenance; they do not include videos.

Dataset: [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). Positive labels include collisions and near-misses. The dataset is distributed separately under its own license.
