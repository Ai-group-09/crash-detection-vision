# Nexar Crash Prediction: Complete Experiment and Delivery Report

**Report date:** 2026-11-06, Asia/Tehran.
**Scope:** this contributor's documented experiments, diagnostics, model preservation, and presentation work. This is not a ranking of the team's models or a declaration of the team's final model.

This report consolidates the historical experiment archive and the subsequent wide-crop experiments into one English document. Completed training, evaluations of existing checkpoints, interrupted runs, technical checks, and cancelled proposals are distinguished throughout. Creating this report did not run training or evaluate any video.

## Repository contribution scope

This contribution includes the presentation notebook, this report, the shared model pipeline, selected weights and metadata, and recorded results. The notebook contains its own checkpoint-loading and single-video prediction code; it does not require a separate inference script. Environment specifications, standalone training scripts, datasets, caches, the protected BestModel archive, and historical experiment folders are not included. Historical packaging descriptions below document earlier local work, not the contents of this contribution. References to omitted local archives are retained as plain text; their reported metrics remain in the tables and appendices.

## 1. Outcome and model identities

The contributor selected **MobileNetV3-Small + GRU32 with a 224 x 384 crop, seed 42, epoch 7** for presentation. Its validation Binary-F1 is **75.2475%** and its public-test Binary-F1 is **69.8690%**. The validation target of more than 75% was reached by this checkpoint, but the public-test target was not reached. A second wide-crop seed achieved 74.0260% validation Binary-F1, so exceeding 75% was not reproduced across both seeds.

The separately protected square-crop reference remains in `BestModel`. Its public Binary-F1 is higher, at 70.8039%. It was not replaced. The author's decision to present the wide model does not establish that it is better on every metric or that it is the team's best model.

| Model / role | Seed | Selected epoch | Validation Binary-F1 (%) | Validation Macro-F1 (%) | Validation accuracy (%) | Public Binary-F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Square crop 224 x 224 / protected reference | 42 | 7 | 72.61 | 72.33 | 72.33 | 70.80 |
| Wide crop 224 x 384 / selected presentation model | 42 | 7 | 75.25 | 75.00 | 75.00 | 69.87 |
| Wide crop 224 x 384 / seed confirmation | 43 | 6 | 74.03 | 73.31 | 73.33 | Not evaluated |

The displayed 75.00% wide validation Macro-F1 is rounded from **74.9975%**; validation accuracy is exactly 75.00%. Binary-F1, Macro-F1, and accuracy are separate metrics.

The primary presentation artifact is [MobileNetV3_GRU_Wide.ipynb](../advanced_tasks/MobileNetV3_GRU_Wide.ipynb). It contains English explanations and one plotting cell with train/validation loss and accuracy over the **10 epochs actually completed**. It currently reads the existing checkpoint and JSON files in [NexarCrashPrediction](../NexarCrashPrediction/README.md).

## 2. Data, environment, and metric definitions

### Data and fixed split

| Dataset | Negative | Positive | Total | Use in this work |
|---|---:|---:|---:|---|
| Training partition | 600 | 600 | 1,200 | Model fitting |
| Validation partition | 150 | 150 | 300 | Development and checkpoint selection |
| Public test | 333 | 334 | 667 | Several historical comparisons and one fixed wide-checkpoint evaluation |
| Private test | 339 | 338 | 677 | Reserved; not evaluated in this work |

The development partitions come from the 1,500-video training set. Positive means collision **or near-miss**; negative means normal driving. Frames from a video stay in its assigned partition. The existing [video_split.csv](../src/video_split.csv) was preserved, rather than generating a new split. Its local byte-level SHA256 is:

`bfe763d95efb4932c33375f69066345c66069d32cdafe5b74edf42ec0982ccde`

The GitHub copy was checked against the local split: all 1,500 rows matched after normalizing line endings. Different LF/CRLF representations can produce different file hashes without changing the partition.

Dataset videos are external to the project repository and are not included in the presentation package. Existing dataset documentation and download/loading work were reused. Dataset source: [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). Dataset counts and timing descriptions are also recorded in [dataset-overview.md](dataset-overview.md) and the local dataset README.

### Environment

| Component | Recorded environment |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB VRAM |
| System memory / CPU | 32 GB RAM / Intel Core i9-13980HX |
| Conda environment | `quera-gpu`; installations restricted to conda |
| Python | 3.10; delivery checks used 3.10.22 |
| PyTorch / torchvision | 2.5.1 / 0.20.1 |
| CUDA build | 12.4 |
| OpenCV / NumPy | 4.8.1 / 1.26.4 |
| Plot verification | Matplotlib 3.10.9; nbformat 5.11.1 |

### Metrics and interpretation

Unless explicitly marked otherwise, all F1 and accuracy values in tables are **percentages**, and loss is the recorded cross-entropy value. `N/R` means not recorded or not recoverable from the available evidence; it does not mean zero. `pp` means percentage points.

Confusion matrices use `[[TN, FP], [FN, TP]]`. Binary-F1 is the positive-class F1: `2TP / (2TP + FP + FN)`. Macro-F1 averages positive- and negative-class F1. Accuracy is `(TP + TN) / N`. For earlier logs without F1, these quantities were reconstructed from a recorded confusion matrix when possible. F1 cannot be recovered from accuracy alone.

Binary-F1 became the main development criterion. Early VideoMAE runs selected by validation accuracy; later temporal experiments used Binary-F1, usually with lower validation loss as the tie-breaker. These historical criteria are retained rather than retrospectively relabelling all selected checkpoints as F1-optimal.

## 3. Initial data checks and frozen ResNet18 baseline

### Data inspection and sampling checks

The initial data review checked filenames against metadata, duplicate metadata filenames, and zero-byte MP4 files. Two training metadata rows had missing weather values. The early review did not yet decode every video. Later development auditing found all 1,500 video paths present and all their reported dimensions equal to 1280 x 720.

| Videos | Sampling FPS | Frames extracted | Stored frame memory (MiB) | Mean seconds/video | Result |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 384 | 67.87 | 1.59 | Completed |
| 10 | 2 | 760 | 134.22 | 2.59 | Completed |
| 10 | 4 | 1515 | 267.70 | 2.03 | Completed |
| 50 | 2 | 3817 | 753.49 | 2.53 | Completed |
| 100 | 2 | 7572 | 1510.26 | N/R | 100/100 succeeded; zero failed |

The 10-video sampling checks used five positive and five negative videos. The 50- and 100-video checks used 2 FPS. These were data-pipeline checks, not separately trained classifiers. The 100-video extraction completed without failures; it used the beginning of sorted filenames rather than a representative random sample. Sampling time and memory values describe their specific runs and are not general hardware benchmarks.

### Baseline method

The baseline used an ImageNet-pretrained, **frozen ResNet18**, extracted 512-dimensional frame features at 2 FPS across the video, averaged them into one video representation, and trained a linear binary classifier. Images were padded to preserve aspect ratio at 224 x 224. The optimizer was Adam with learning rate 0.001 and binary cross-entropy with logits. Averaging removed explicit frame-order information, and the encoder was not fine-tuned.

| Baseline variant | Train/validation videos | Epochs | Threshold | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 videos / linear | 85/15 | 20 | 0.50 | 70.59 | 66.06 | 66.67 | `[[4, 3], [2, 6]]` |
| 500 videos / linear | 400/100 | 20 | 0.50 | 49.44 | 54.45 | 55.00 | `[[33, 17], [28, 22]]` |
| 1,500 videos / linear | 1,200/300 | 20 | 0.50 | 61.97 | 63.90 | 64.00 | `[[104, 46], [62, 88]]` |
| 1,500 videos / linear | 1,200/300 | 100 | 0.50 | 65.52 | 66.63 | 66.67 | `[[105, 45], [55, 95]]` |
| Same 100-epoch checkpoint / validation threshold | 1,200/300 | 100 | 0.45 | 69.90 | 68.97 | 69.00 | `[[99, 51], [42, 108]]` |
| Extended linear training | 1,200/300 | 200 | 0.45 | 69.71 | 68.98 | 69.00 | `[[100, 50], [43, 107]]` |
| MLP: 512 -> 64 -> 1, ReLU | 1,200/300 | 100 | 0.45 | 67.99 | 67.66 | 67.67 | `[[100, 50], [47, 103]]` |
| Normalized features / linear | 1,200/300 | 100 | 0.45 | 68.44 | 68.33 | 68.33 | `[[102, 48], [47, 103]]` |

An initial 8-training/2-validation-video exercise was only a code check; it is not a meaningful model-quality result. The 100-video experiment used saved JPEG frames and an 85/15 split (validation: eight positive and seven negative videos). The 500-video experiment used 400/100, and the full-data experiment used the fixed 1,200/300 split. These different validation populations are not directly comparable. The separately recorded epoch-20 training losses were 0.6303 for the 500-video run and 0.6368 for the 1,500-video run; validation losses were not preserved for those baseline rows.

The threshold change from 0.5 to 0.45 was selected on validation. The selected baseline reached public Binary-F1 66.17%, Macro-F1 65.81%, and accuracy 65.82%. Adding an MLP hidden layer, feature normalization, or extending training to 200 epochs did not establish a better baseline. Full-video baseline inputs are not strictly comparable with the later pre-event-only positive clips.

## 4. VideoMAE experiments

### Method and changes

The VideoMAE encoder used 16 frames at 224 x 224 with 768-dimensional representations. The first run trained the head with the encoder frozen on an 8-second window. The next run unfroze the final two encoder blocks. The corrected experiment shortened the window to 2 seconds, retained a 1-second prediction horizon, reduced encoder/head learning rates to `5e-6 / 5e-5`, used batch size 1 with accumulation over 8 batches, clip-consistent brightness/contrast augmentation, clipping at 1.0, and cosine decay. It started from the earlier frozen-model checkpoint rather than from an unrelated model.

The 8-second fine-tuning run used encoder/head learning rates `1e-5 / 1e-4`, batch size 1, weight decay 0.01, and a maximum of 8 epochs with patience 3. The corrected 2-second run allowed 4 epochs with patience 2 and selected by highest validation accuracy, then lower loss.

The source encoder was `MCG-NJU/videomae-base-finetuned-kinetics`: 12 encoder blocks and 86,228,738 parameters in the two-class model. Head-only training exposed 3,074 parameters; two-block fine-tuning exposed 14,177,282. Initial preprocessing used aspect-preserving black padding with processor resizing/cropping disabled. The original 8-second sampler omitted the endpoint, placing its last frame about 0.5 seconds earlier than intended; the corrected sampler included the last permitted frame. The corrected run restarted from the frozen checkpoint, not from the overfit 8-second fine-tuned checkpoint. Its multiple simultaneous changes prevent attribution to one factor.

| Configuration / evaluation | Selected epoch | Recorded epochs | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VideoMAE: frozen head, 8 seconds | 3 | 3 | N/R | N/R | 65.00 | 0.685174 | N/R |
| VideoMAE: final two blocks, 8 seconds | 1 | 4 | N/R | N/R | 65.33 | 1.486126 | N/R |
| VideoMAE: corrected final two blocks, 2 seconds | 3 | 4 | 67.14 | 68.90 | 69.00 | 0.645881 | `[[112, 38], [55, 95]]` |
| Corrected inputs before fine-tuning (existing frozen checkpoint) | N/A | Evaluation only | 66.89 | 66.67 | 66.67 | 0.652079 | `[[99, 51], [49, 101]]` |
| Selected corrected checkpoint: 2-second inference window | 3 | Evaluation only | 67.14 | 68.90 | 69.00 | 0.645867 | `[[112, 38], [55, 95]]` |
| Selected corrected checkpoint: 4-second inference window | 3 | Evaluation only | 62.37 | 64.83 | 65.00 | 0.677405 | `[[108, 42], [63, 87]]` |
| Selected corrected checkpoint: 8-second inference window | 3 | Evaluation only | 60.45 | 64.26 | 64.67 | 0.710910 | `[[113, 37], [69, 81]]` |
| VideoMAE frozen multi-clip / last | 3 | 9 | 67.11 | 67.00 | 67.00 | 0.664615 | `[[100, 50], [49, 101]]` |
| VideoMAE frozen multi-clip / mean | 6 | 12 | 66.67 | 67.95 | 68.00 | 0.708071 | `[[108, 42], [54, 96]]` |
| VideoMAE frozen multi-clip / gru | 7 | 13 | 68.90 | 69.00 | 69.00 | 0.815712 | `[[104, 46], [47, 103]]` |

The 2-, 4-, and 8-second inference-window rows evaluate **the same selected checkpoint** without retraining. Window comparisons are not three independent training runs. The 4- and 8-second evaluations each had one shortened clip; the 2-second evaluation had none.

The three single-clip training histories record approximately 78.83, 132.14, and 114.15 minutes respectively. The frozen run selected epoch 3 with validation loss 0.685174; the 8-second fine-tuned run selected epoch 1 with loss 1.486126 and subsequently reached training accuracy 83.75% while validation worsened. The corrected run's epoch-3 training-history loss was 0.645881; its separate 2-second inference evaluation recorded 0.645867 with the same confusion matrix. These tiny loss differences do not represent a new checkpoint. The corrected scheduler's final 0.1 multiplier was set after the last training epoch, so no later epoch used it.

Additional archived post-hoc discrimination measures were:

| Inference window | Positive precision (%) | ROC-AUC | Average precision |
|---|---:|---:|---:|
| 2 seconds | 71.43 | 0.7325 | 0.7097 |
| 4 seconds | 67.44 | 0.7107 | 0.6849 |
| 8 seconds | 68.64 | 0.6943 | 0.6510 |

These are overall validation measurements, not a grouped-by-time test challenge score. At the 1-second pre-event endpoint, 146/600 training positives and 43/150 validation positives ended before the annotated alert; five training and one validation positive lacked a full 8 seconds of earlier history. No relabelling or exclusion followed this timing analysis. These counts differ from the later 0.5-second MobileNet audit because the endpoints differ.

### Exploratory thresholds

| Window (seconds) | Validation-explored threshold | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.26076576 | 71.67 | 64.58 | 66.00 | `[[69, 81], [21, 129]]` |
| 4 | 0.32858330 | 70.41 | 66.12 | 66.67 | `[[81, 69], [31, 119]]` |
| 8 | 0.18039526 | 69.79 | 58.04 | 61.33 | `[[50, 100], [16, 134]]` |

These thresholds were explored on the same validation predictions. They are not independent test results or additional trained models. The positive-class F1 gains came with weaker Macro-F1/accuracy and did not justify treating the threshold-exploration values as a general improvement.

### Frozen multi-clip features and temporal heads

Feature extraction completed for all 1,500 videos in approximately **109.4 minutes**, producing **5,989 clips**, up to four 768-dimensional clip features per video; six videos had fewer than four valid clips. Masks handled short sequences. Feature normalization was fitted on training clips only.

The `last`, `mean`, and `GRU` heads used batch size 64, AdamW learning rate 0.001, weight decay 0.01, dropout 0.3, a maximum of 30 epochs, and patience 6. The GRU head used a 128-dimensional projection and hidden size 64. Selection used positive-class F1 and argmax without threshold tuning. The encoder remained frozen in this comparison, so these head-only experiments did not satisfy the real encoder fine-tuning requirement on their own.

There were **six VideoMAE training runs** in the recorded sequence: three single-clip training runs and three frozen-feature head runs. The GRU head selected epoch 7, completed 13 epochs, and eventually reached approximately 97.50% training F1 while validation F1 remained around 68%; this supported the overfitting diagnosis. No public/private result is recorded for VideoMAE in this work. The route was archived rather than repeatedly retried.

The last/mean heads each had 1,538 parameters; the projected GRU head had 135,810. All cached feature tensors were finite. Relative to the single-clip checkpoint, the selected GRU head gained eight true positives but lost eight true negatives, retaining 207/300 correct predictions. The empty `finetune_20261102_185909` directory did not contain a completed experiment.

## 5. R(2+1)D-18 and ResNet18 + GRU

### R(2+1)D-18

R(2+1)D-18 used Kinetics-400 pretrained weights, 16 frames from a 2-second window ending 1 second before the event for positives, a 112 x 112 input, batch size 2 with accumulation 4, dropout 0.3, weight decay 0.01, fixed BatchNorm running statistics, and argmax decisions. The standard partial fine-tuning setup trained `layer3`, `layer4`, and the classifier at backbone/head learning rates `1e-5 / 1e-4`, with up to 10 epochs and patience 3.

| Configuration | Selected epoch | Recorded epochs | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R(2+1)D: layer3 + layer4, standard LR | 4 | 7 | 71.34 | 68.39 | 68.67 | 0.987251 | `[[89, 61], [33, 117]]` |
| R(2+1)D: lower LR 2e-6 / 2e-5 | 2 | 5 | 64.29 | 63.31 | 63.33 | 0.628960 | `[[91, 59], [51, 99]]` |
| R(2+1)D: repeated lower-LR run | 2 | 5 | 64.29 | 63.31 | 63.33 | 0.628960 | `[[91, 59], [51, 99]]` |
| R(2+1)D: full-frame black padding | 4 | 7 | 71.27 | 63.79 | 65.33 | 1.077505 | `[[67, 83], [21, 129]]` |
| R(2+1)D: layer4 only | 7 | 10 | 70.22 | 68.21 | 68.33 | 1.557450 | `[[93, 57], [38, 112]]` |
| R(2+1)D: stratified temporal jitter | 4 | 7 | 72.00 | 69.45 | 69.67 | 0.902945 | `[[92, 58], [33, 117]]` |
| R(2+1)D: progressive unfreezing | 7 | 10 | 70.62 | 68.53 | 68.67 | 0.838754 | `[[93, 57], [37, 113]]` |

The two lower-learning-rate runs have numerically identical epoch metrics despite different timings; they must not be counted as independent seed confirmation. Full-frame padding preserved the complete frame at 112 x 112 but reduced object scale and did not improve the result. Training only `layer4` also did not beat the original setup.

Temporal jitter sampled 16 frames from stratified temporal bins while keeping the first/last frames fixed. Validation frame indices remained unchanged. The completed jitter run reached 72.00% Binary-F1. A similarly named earlier directory contains preparation/settings only, with no completed training history.

Progressive unfreezing trained the head in epochs 1-2, `layer4` plus head in epochs 3-4, and `layer3/layer4` plus head from epoch 5. It preserved weights and optimizer state across stages. Checkpoints became eligible at epoch 3; early stopping operated in the final stage. This did not beat the jitter control. No public/private result is recorded for these R(2+1)D runs.

The jitter comparison corrected seven reference errors and regressed four, leaving TP unchanged at 117 and increasing TN from 89 to 92. Its seven recorded epochs took approximately 23.21 minutes. Overfitting persisted: final training F1 reached about 98.42% for the initial reference, 98.83% for full-frame padding, 99.58% for layer4-only, 97.85% for jitter, and 95.51% for progressive unfreezing. Four principal R(2+1)D configurations shared 52 incorrect validation videos (13 positives and 39 negatives). Layer4-only corrected 16 reference errors but introduced 17. In the reference's 43 positive clips ending before annotated alert, 30 were TP and 13 FN; among the other 107, 87 were TP and 20 FN. These were exploratory error analyses, not proof of causal factors or label errors. A few user-selected clip inspections did not reveal an obvious input issue, but were not a full visual audit.

### ResNet18 + GRU64

This experiment used an ImageNet-pretrained ResNet18, a single GRU64 with final-hidden-state readout, and a two-class classifier. It retained 16-frame temporal jitter, 224 x 224 ImageNet preprocessing, frozen BatchNorm statistics, an FP32 GRU under mixed precision, and a two-epoch temporal-head warm-up before fine-tuning `encoder.layer4`. The warm-up head learning rate was 0.001, then backbone/head rates were `1e-5 / 1e-4`; batch size 2 and accumulation 4 were retained.

| Configuration | Selected epoch | Recorded epochs | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 + GRU64, 16 frames | 3 | 6 | 71.39 | 67.11 | 67.67 | 0.629891 | `[[82, 68], [29, 121]]` |

Only fine-tuned epochs were eligible for selection. The selected epoch was 3; six epochs were recorded. Training F1 later rose to approximately 84.72% while validation F1 was approximately 70.22%, indicating overfitting. This was a **16-frame unidirectional ResNet18+GRU** experiment, not the unexecuted four-frame ResNet18+BiGRU proposal.

## 6. Four-frame MobileNetV3-Small + GRU32

### Shared architecture and real fine-tuning

The simpler mentor-aligned model uses MobileNetV3-Small features, global average pooling, GRU32, dropout 0.4, and a two-class linear output: **985,634 total parameters**. The initial two epochs train the temporal head, then `encoder.11`, `encoder.12`, the GRU, and classifier are trained. Trainable parameter counts are 58,626 during warm-up and 409,170 during fine-tuning. BatchNorm running statistics remain fixed.

AdamW uses weight decay 0.01. Head warm-up learning rate is 0.001; fine-tuning starts at encoder/head rates `1e-5 / 1e-4`, with cosine decay to 10% over the scheduled epochs. Batch size is 8, at most 12 epochs, patience 3, clipping norm 1.0, and argmax decisions. The encoder uses CUDA FP16 autocast while the GRU computation uses FP32. Fine-tuning eligibility starts at epoch 3, preventing a head-only warm-up model from being selected as the final trained checkpoint.

### Sampling correction and AMP repair

Positive development frames are selected at event minus 2.0, 1.5, 1.0, and 0.5 seconds, rounded down to real chronological frames. Initially, negative development and test inputs were uniform across the whole video. They were later changed to the final 1.5 seconds. At inference, the same final-window rule is applied to both classes without using event/alert metadata.

| Configuration / evaluation | Selected epoch | Recorded epochs | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNet + GRU32: uniform negative/test sampling | 3 | 6 | 74.00 | 74.00 | 74.00 | 0.554713 | `[[111, 39], [39, 111]]` |
| Old uniform-sampler checkpoint on corrected validation | 3 | Evaluation only | 72.31 | 71.65 | 71.67 | 0.615684 | `[[104, 46], [39, 111]]` |
| MobileNet + GRU32: final-window sampler, first attempt | 4 | 4 | 72.24 | 72.33 | 72.33 | 0.590316 | `[[109, 41], [42, 108]]` |
| MobileNet + GRU32: final-window sampler, AMP repair | 7 | 10 | 72.61 | 72.33 | 72.33 | 0.587492 | `[[107, 43], [40, 110]]` |

The original 74.00% validation score used a different negative-input protocol. Re-evaluating that old checkpoint on the corrected validation inputs produced 72.31% Binary-F1; this was an evaluation, not new training. Its repeated appearance in two run directories is not a second independent control.

The first corrected run was interrupted by an AMP gradient failure in epoch 5; only four completed epochs are preserved. Its epoch-4 result is not a completed final-run outcome. The repaired implementation unscales and checks gradients before clipping, lets GradScaler skip non-finite updates, reduces its scale, and stops after repeated overflow. The completed protected reference recorded zero skipped updates throughout its ten epochs.

### Multi-horizon training

Positive training endpoints were sampled at 0.5, 1.0, or 1.5 seconds before the event, retaining half-second frame spacing. Negatives retained the final-window sampler. Selection used the arithmetic mean of validation F1 across the three horizons, with six epochs recorded and epoch 3 selected.

| Checkpoint | Lead time | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- |
| Existing single-horizon control | 0.5s | 72.61 | 72.33 | 72.33 | 0.587492 | `[[107, 43], [40, 110]]` |
| Existing single-horizon control | 1.0s | 69.15 | 69.66 | 69.67 | 0.618365 | `[[107, 43], [48, 102]]` |
| Existing single-horizon control | 1.5s | 65.97 | 67.28 | 67.33 | 0.656554 | `[[107, 43], [55, 95]]` |
| Existing single-horizon control | Mean of 3 horizons | 69.24 | 69.76 | 69.78 | 0.620804 | N/R |
| Multi-horizon candidate, epoch 3 | 0.5s | 69.90 | 68.97 | 69.00 | 0.604986 | `[[99, 51], [42, 108]]` |
| Multi-horizon candidate, epoch 3 | 1.0s | 67.33 | 67.00 | 67.00 | 0.623333 | `[[99, 51], [48, 102]]` |
| Multi-horizon candidate, epoch 3 | 1.5s | 65.10 | 65.33 | 65.33 | 0.645877 | `[[99, 51], [53, 97]]` |
| Multi-horizon candidate, epoch 3 | Mean of 3 horizons | 67.44 | 67.10 | 67.11 | 0.624732 | N/R |

These evaluations use the **same 300 validation videos** at three horizons, not 900 independent videos. The mean is an arithmetic mean of per-horizon metrics, not F1 computed by pooling all predictions. The experiment reduced mean Binary-F1 from 69.24% to 67.44% and was not adopted. The control's decline at longer horizons establishes timing sensitivity, but does not justify repeating the same failed augmentation without a new rationale.

### Seed stability and progressive fine-tuning

| Seed | Selected epoch | Recorded epochs | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 7 | 10 | 72.61 | 72.33 | 72.33 | 0.587492 | `[[107, 43], [40, 110]]` |
| 43 | 6 | 9 | 72.31 | 71.65 | 71.67 | 0.591029 | `[[104, 46], [39, 111]]` |
| 44 | 5 | 8 | 72.85 | 72.67 | 72.67 | 0.596499 | `[[108, 42], [40, 110]]` |

Seed 42 was the existing protected checkpoint, not a newly trained replicate. Seeds 43 and 44 were trained separately. The three-seed mean Binary-F1 was 72.5892% with sample SD 0.2679 pp on the **same validation set**. Macro-F1 averaged 72.2158% with sample SD 0.5168 pp; accuracy averaged 72.2222% with sample SD 0.5092 pp. This describes limited optimization variability, not a confidence interval or three independent test datasets.

| Seed | Checkpoint type | Epoch | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Matching-seed narrow control | 7 | 72.61 | 72.33 | 72.33 | 0.587492 | `[[107, 43], [40, 110]]` |
| 42 | Before block10 / same progressive run | 4 | 72.24 | 72.33 | 72.33 | 0.590307 | `[[109, 41], [42, 108]]` |
| 42 | Selected final-stage progressive checkpoint | 10 | 72.91 | 73.00 | 73.00 | 0.589563 | `[[110, 40], [41, 109]]` |
| 43 | Matching-seed narrow control | 6 | 72.31 | 71.65 | 71.67 | 0.591029 | `[[104, 46], [39, 111]]` |
| 43 | Before block10 / same progressive run | 4 | 70.47 | 70.67 | 70.67 | 0.595737 | `[[107, 43], [45, 105]]` |
| 43 | Selected final-stage progressive checkpoint | 8 | 72.55 | 71.99 | 72.00 | 0.589872 | `[[105, 45], [39, 111]]` |

MobileNet ProgressiveFT added `encoder.10` after the head and `encoder.11/12` stages, with an initial learning rate of `2e-6` for the newly opened block. The final-stage results improved over matching-seed controls by approximately 0.30 pp and 0.24 pp. The before-block10 rows are intermediate checkpoints within those progressive runs, not independent controls. Final-stage eligibility and stopping rules also changed, so this is not a perfectly isolated one-parameter ablation. No public result is recorded for these candidates.

### Bidirectional GRU trial

| Configuration | Selected epoch | Recorded epochs | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matching-seed GRU32 control | 7 | 10 | 72.61 | 72.33 | 72.33 | 0.587492 | `[[107, 43], [40, 110]]` |
| MobileNet + BiGRU16 per direction: seed 42 | 3 | 6 | 70.97 | 69.97 | 70.00 | 0.590633 | `[[100, 50], [40, 110]]` |

The MobileNet BiGRU used 16 hidden units per direction, retaining a 32-dimensional output and approximately the same total capacity: 984,098 parameters versus 985,634 for GRU32. It preserved the four-frame protocol and other training settings. Six epochs were completed, epoch 3 was selected, and zero optimizer updates were skipped. The validation result was worse than the matching GRU32 control; it was not adopted and was not evaluated on public/private.

## 7. Plateau diagnosis before the wide-crop experiment

The subsequent review inspected archived code, settings, training curves, saved predictions, split identity, metadata, video headers, and small technical examples before proposing a new experiment.

- **No obvious broken-training explanation:** the encoder tail was actually updated; BN statistics stayed frozen; the FP32 temporal path preserved gradients; the completed reference had no skipped AMP updates.
- **Mild plateau/overfitting in the compact reference:** at epoch 7, train Binary-F1 was 75.16% versus validation 72.61%; at epoch 9, 76.82% versus 71.81%. Training metrics include augmentation/dropout, so this is not an exactly matched generalization-gap measurement.
- **Spatial context loss:** all 1,500 development videos had dimensions 1280 x 720. Resizing the short edge to 256 then center-cropping 224 x 224 retains approximately 49.2% of the original width and 87.5% of the height. A 224 x 384 crop at the same resize scale retains approximately 84.4% of the width. Loss of lateral context was a plausible, unproven bottleneck.
- **Threshold-only changes were not the chosen explanation:** the reference had 83 validation errors; 14 had probabilities in [0.4, 0.6], and 32 were confidently wrong (negative with probability >0.8, or positive with probability <0.2). No new threshold search was conducted.
- **Timing ambiguity:** every sampled frame preceded the annotated alert in 36/600 training positives and 13/150 validation positives. Ten of those 13 validation positives were false negatives, compared with 30 false negatives among the other 137. Median event-minus-alert time was 1.436 seconds in training and 1.3805 seconds in validation. These small exploratory groups do not prove incorrect labels or invisible hazards.
- **Development/inference mismatch remains:** positive development endpoints use an event-relative 0.5-second lead, while test clips can terminate at 0.5/1.0/1.5-second leads. Event time is not an input feature, but selecting development clips using it still shapes the input distribution.

The choice was to test **crop width only**, retaining image scale, vertical crop, model, chronological frame indices, split, optimizer, augmentation ranges, learning rates, and checkpoint criterion. The earlier R(2+1)D full-frame-padding failure was not the same experiment: it used a different model, 112-square output, and reduced object scale.

## 8. Wide-crop experiment and seed confirmation

### Controlled change

The crop changed from **224 x 224 to 224 x 384**, with short-edge resize 256, bilinear interpolation, antialiasing, and ImageNet normalization unchanged. There was no EMA, extra temporal branch, threshold adjustment, or larger model. Initial weights came from fresh cached ImageNet weights; the protected trained checkpoint was not used to initialize training.

The middle 224 columns of the unaugmented wide crop were verified to match the original square crop exactly. Contrast augmentation uses the crop's mean, so augmented central pixels need not remain identical after the context changes. The new crop still removes some borders; it is not full-frame processing.

### Validation results

| Configuration | Selected epoch | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix | Binary-F1 delta vs same-seed narrow (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Narrow, seed 42 | 7 | 72.61 | 72.33 | 72.33 | 0.587492 | `[[107, 43], [40, 110]]` | Control |
| Wide, seed 42 | 7 | 75.25 | 75.00 | 75.00 | 0.555595 | `[[111, 39], [36, 114]]` | +2.6403 |
| Narrow, seed 43 | 6 | 72.31 | 71.65 | 71.67 | 0.591029 | `[[104, 46], [39, 111]]` | Control |
| Wide, seed 43 | 6 | 74.03 | 73.31 | 73.33 | 0.560930 | `[[106, 44], [36, 114]]` | +1.7133 |

| Wide-crop seeds 42 and 43 | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) |
| --- | --- | --- | --- |
| Mean | 74.6367 | 74.1559 | 74.1667 |
| Sample SD (pp) | 0.8638 | 1.1902 | 1.1785 |

Both seeds use the same 300 validation videos; these are not independent validation sets. The seed-42 to seed-43 Binary-F1 decrease is **1.2216 pp**.

The printed comparison for the seed-43 run still referenced the narrow seed-42 checkpoint. Its displayed +1.4187 pp F1 and 20 corrections/17 regressions are therefore a cross-seed comparison, not an isolated width effect. The appropriate matching-seed narrow control has Binary-F1 72.3127%, giving a wide-crop seed-43 improvement of approximately 1.7133 pp.

For seed 42, 21 reference errors were corrected and 13 new errors appeared: eight more correct predictions, with both FP and FN reduced by four. Negative examples had 11 corrections and 7 regressions; positive examples had 10 corrections and 6 regressions. Validation loss decreased from 0.587492 to 0.555595.

Seed 42 completed ten epochs and selected epoch 7. Validation Binary-F1 over epochs 6-10 was 74.83%, 75.25%, 75.08%, 74.25%, and 75.08%. Seed 43 completed nine epochs and selected epoch 6. All recorded skipped optimizer updates in both runs were zero. Multiple nearby epochs are not independent statistical replications.

The seed-42 run took approximately eight minutes in the recorded epoch timings, including about 6.77 minutes in the first epoch for decoding/cache population. Subsequent cached epochs were much faster. These timings do not indicate skipped training data: each epoch processed all 1,200 training and 300 validation videos.

### Fixed-checkpoint public evaluation

After the user explicitly requested public checking before promotion, the seed-42, epoch-7 wide checkpoint was evaluated once on all 667 public videos. The checkpoint hash, crop, seed, argmax rule, and same video/frame-index signature were fixed before evaluation. Both classes used the final 1.5 seconds; no test event, alert, or time-to-accident metadata was used.

| Model evaluated on public test (667 videos) | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| Frozen ResNet18 linear baseline / 100 epochs / threshold 0.45 | 66.17 | 65.81 | 65.82 | N/R | `[[216, 117], [111, 223]]` |
| Initial MobileNet + GRU32 / uniform test sampling | 68.65 | 68.36 | 68.37 | 0.616806 | `[[225, 108], [103, 231]]` |
| Protected narrow MobileNet + GRU32 / final-window sampling | 70.80 | 68.84 | 68.97 | 0.582431 | `[[209, 124], [83, 251]]` |
| Selected wide MobileNet + GRU32 / final-window sampling | 69.87 | 68.94 | 68.97 | 0.594884 | `[[220, 113], [94, 240]]` |

Wide minus protected reference: **Binary-F1 -0.9350 pp; Macro-F1 +0.0956 pp; accuracy 0.0000 pp**. Earlier baseline/uniform-sampler rows used different input protocols and are not controlled crop-width ablations.

For the wide model versus the protected reference, FP decreased by 11 while FN increased by 11. Both correctly classified 460/667 videos. The paired comparison contained 43 corrections and 43 regressions. The gain on validation did not transfer to public Binary-F1; the small Macro-F1 increase did not outweigh the decrease in the primary metric for the preservation decision.

No further threshold, seed, or checkpoint was selected using these public results. The protected `BestModel/current.json` pointer remained unchanged. The contributor later chose the wide model for presentation anyway, with its actual public score disclosed. Private test remained untouched.

## 9. Technical validation and reproducibility work

Technical checks were kept separate from full training and performance reporting:

1. Historical pipelines checked sample indices, no post-event positive frames, successful decoding, tensor shape, finite loss/gradients, real encoder updates, and unchanged frozen weights/BN buffers. R(2+1)D, ResNet+GRU, and progressive schedules had stage-specific checks.
2. The protected model bundle stored a complete checkpoint, independent model/sampling/training code, settings, manifests, metrics, per-video predictions, source provenance, and checksums. Registration tests covered preservation of old versions, rejection of incomplete/non-improving candidates, and detection of changed files. A ZIP snapshot was also verified historically.
3. The wide-input smoke test used eight balanced training videos and two optimizer steps: one warm-up and one fine-tuning. It verified active CNN/GRU updates, frozen BN and weights, uint8-cache integrity, the exact central crop, and restoration of temporary weights. Reversing frame order changed logits; this verifies temporal connectivity, not predictive superiority.
4. Four saved reference validation predictions were reproduced exactly in the initial CUDA check. The wide deployment pipeline later matched source pixels and CPU logits exactly on four development videos. Its CUDA comparison with archived predictions had maximum probability difference about 0.000363 and identical class decisions. CPU-versus-CUDA-AMP probability differences were recorded separately, not treated as changes in the model.
5. The wide smoke recorded approximately 0.103 GiB peak allocated and 0.148 GiB reserved by PyTorch. These short-run allocator readings exclude some driver/runtime memory. Full uint8 pixel caching was estimated at approximately 1.442 GiB. Input pixels increased by a factor of 384/224, without increasing model parameters.
6. The packaged checkpoint was loaded independently; saved validation/public metrics were reconstructed from prediction files; four development examples and one label-free video path were checked. No main training or private evaluation was part of packaging.
7. The English notebook was schema-validated and executed with default settings. It has four code cells, exactly one plotting cell, and an embedded figure. Plot x/y data were checked against the saved ten-epoch history; no smoothing, fabricated epochs, or substitution of F1 for accuracy was used. Existing repository files were unchanged when that notebook was added.

The final report itself is based on stored evidence only. It does not produce a new evaluation or change any checkpoint.

## 10. Proposed, interrupted, and cancelled work

| Item | Actual status | How to interpret it |
|---|---|---|
| R(2+1)D jitter directory ending `014622_825976` | Settings/preparation recorded; no completed history | Not a trained-model result |
| First corrected four-frame run ending `173909_764244` | Four completed epochs, failure during epoch 5 | Interrupted run; not the final corrected reference |
| Four-frame ResNet18 + BiGRU | Suggested only | Distinct from the completed 16-frame ResNet18 + GRU |
| MobileNet fine-tuning of `encoder.12` only | Suggested only | No completed score |
| Frozen DINOv2 features | Discussed, not run | Did not satisfy the requested real fine-tuning direction |
| YOLO plus a temporal head | Concept discussed only | No training result |
| Full-encoder VideoMAE fine-tuning | Concept discussed only | No training result |
| 30-FPS / 60-frame variant | Concept discussed only | No training result |
| VideoMAE directory ending `185909` | Empty/incomplete preparation | Not an additional completed experiment |
| Wide 224 x 384 crop | Originally a proposal; subsequently completed for seeds 42 and 43 | Earlier archive statements saying “not run” are superseded by Section 8 |
| EMA with decay 0.99 | Proposed as a final experiment; preparation started, then cancelled for time | No Nexar EMA training result or selected EMA checkpoint |
| Private-test evaluation | Reserved, not run | No private score should be reported |
| Team-wide model selection | Not performed in this contributor's work | Another contributor's model may be stronger |

EMA was designed to start at epoch 3 and update only after successful optimizer steps, with frozen BN statistics, and to compare seeds 42/43 against their same-seed wide controls. The mechanism was checked on synthetic tensors, including skipped AMP-overflow steps. Manual source edits and a draft `MentorEMA` notebook/README exist, but the independent implementation was not completed before cancellation. In particular, the draft's ready-to-run wording is stale: no `MentorEMA/ema_experiment.py` or completed EMA outputs were found. These files are preparation remnants, not evidence of a successful experiment. The selected presentation checkpoint remains **non-EMA**.

## 11. Preservation, packaging, and team-repository review

### What was preserved

The initial reference remains in `BestModel/versions/v_20261105_222803_896915`. Its recorded checkpoint SHA256 is:

`07183cd3da74c33defa40cfb56e2078c37dba1535cd805394bbaff8f240091e8`

The wide seed-42 selected checkpoint SHA256 is:

`fb83cf3a8fcb166f46eb7ee4107367458f00d1b571ad1e72277321bd78a7f1bd`

The wide seed-43 checkpoint SHA256 is:

`e0b4dc7a213d4bd64f0283f8bbfec04e043c13c38fd66bb8cef2f0df4b28aa8e`

The original experiment directories remain the authoritative run evidence when present. The historical Markdown preserves numerical results for directories that were subsequently removed; it is not a substitute for missing executable code or weights. The older `ALL_EXPERIMENTS.md` was not overwritten while preparing this report.

### Delivery package and subsequent simplification review

A self-contained `NexarCrashPrediction` package was assembled with the selected wide checkpoint, inference entry point, optional reproduction entry point, model/sampling code, fixed split, portable manifests, recorded results, documentation, and integrity checks. Dataset videos and the optimizer/last-epoch checkpoint were not included. Loading the selected checkpoint does not require downloading ImageNet weights; fresh training requires the original ImageNet initialization separately.

The user then requested a less cluttered contribution inside the existing team repository. The review found a duplicate split, repeated metadata/settings already partly present in the checkpoint, and substantial bookkeeping in the approximately 456-line pipeline; the actual model class is only about 20 lines. Removing source-code bookkeeping would improve readability, not materially change inference compute.

The team's GitHub tree was reviewed read-only at commit `f21704608865de409b10617fbb07786d594a3aea`. It contained shared data preparation/documentation and newer TensorFlow training notebooks. Dataset videos/NPZ caches were external. The existing baseline and other team sampling pipelines were not interchangeable with this model's four-frame, wide-crop input. No teammate's checkpoint or score was evaluated or adopted here. GitHub source: [team repository](https://github.com/Ai-group-09/crash-detection-vision).

A possible integration layout was proposed: an architecture-named notebook and a compact helper beside existing work, shared `src/video_split.csv`, and common model/results/docs directories. **That full refactor was not implemented.** The user narrowed the immediate task to the English notebook and one curve-plotting cell, leaving other files unchanged. The delivered notebook still depends on `NexarCrashPrediction`; it must not be described as a self-contained, weight-bearing notebook or as an already completed compact-helper migration.

No GitHub commit or push was performed by the assistant. The notebook was deliberately named `MobileNetV3_GRU_Wide.ipynb`, avoiding a claim that it is the team's final model. This report closes the contributor's current experiment documentation without selecting a winner among team members.

## 12. Conclusions and limits of the evidence

- The project progressed from a frozen, order-agnostic baseline to a small temporal model with verified encoder fine-tuning.
- More complex temporal architectures, broader unfreezing, and BiGRU did not consistently outperform the compact GRU32 configuration under the documented protocols.
- Changing spatial crop width produced the strongest selected validation score in this contributor's recorded experiments: 75.2475%. Seed 43 also improved over its narrow control, but the two-seed mean remained below 75%.
- The wide model's public Binary-F1 was 69.8690%, below the protected reference's 70.8039%. The target of 75% on public was not achieved, and no private score exists.
- Repeated validation use, changing input protocols across model families, small sample sizes, endpoint-distribution differences, and previously viewed public results limit generalization claims. A few seed runs do not resolve validation-selection bias.
- The selected notebook presents the contributor's chosen checkpoint and its recorded results. Selecting the team's best model remains outside the scope of this contributor's experiments.

## 13. Evidence map and coverage

| Evidence | Repository file |
| --- | --- |
| Historical archive: baseline through BiGRU, raw JSON and earlier work | ALL_EXPERIMENTS.md (local archive reference; not included in this contribution) |
| Early sampling and frozen-baseline summary | [docs/experiments-overview.md](../docs/experiments-overview.md) |
| Dataset review and split context | [docs/dataset-overview.md](../docs/dataset-overview.md) |
| Fixed development split | [src/video_split.csv](../src/video_split.csv) |
| Protected-version pointer | BestModel/current.json (local archive reference; not included in this contribution) |
| Protected reference selected metrics | BestModel/versions/v_20261105_222803_896915/result.json (local archive reference; not included in this contribution) |
| Protected reference public metrics | BestModel/versions/v_20261105_222803_896915/test-public_metrics.json (local archive reference; not included in this contribution) |
| Pre-wide diagnosis | MentorWideContext/audit_report.json (local archive reference; not included in this contribution) |
| Wide seed-42 selected metrics | MentorWideContext/outputs/wide_seed42_20261106_021409_950300/result.json (local archive reference; not included in this contribution) |
| Wide seed-43 selected metrics | MentorWideContext/outputs/wide_seed43_20261106_120506_83018720261106_120506_830187/result.json (local archive reference; not included in this contribution) |
| Wide seed-42 paired validation comparison | MentorWideContext/outputs/wide_seed42_20261106_021409_950300/paired_validation_comparison.json (local archive reference; not included in this contribution) |
| Fixed wide public evaluation plan | MentorWideContext/outputs/wide_seed42_20261106_021409_950300/public_evaluation_plan.json (local archive reference; not included in this contribution) |
| Wide public metrics and reference deltas | MentorWideContext/outputs/wide_seed42_20261106_021409_950300/test-public_metrics.json (local archive reference; not included in this contribution) |
| Completed-wide-run verification | MentorWideContext/completed_run_verification.json (local archive reference; not included in this contribution) |
| Public comparison and preservation review | MentorWideContext/public_result_review.json (local archive reference; not included in this contribution) |
| Independent deployment pipeline verification | MentorWideContext/packaging/pipeline_verification.json (local archive reference; not included in this contribution) |
| Presentation package and entry points | [NexarCrashPrediction/README.md](../NexarCrashPrediction/README.md) |
| English presentation notebook and actual training curves | [advanced_tasks/MobileNetV3_GRU_Wide.ipynb](../advanced_tasks/MobileNetV3_GRU_Wide.ipynb) |

Where an original run directory no longer exists, its exact original identifier is listed below and its preserved payload is available in `ALL_EXPERIMENTS.md`. Copies of the protected reference and presentation package are not additional training runs. Public results live in separate `test-public_metrics.json` records.

The historical `<details>` blocks contain 71 JSON records, including model/preprocessor configurations, settings, histories, selected results, and study summaries. This report includes every archived JSON training history plus both newer wide-crop histories in Appendix A. Evaluation rows, stage snapshots, and duplicate controls are not counted as independent training runs. Missing metrics remain explicitly marked rather than inferred from unrelated observations.

### Archived run identifiers

| Appendix | Experiment | Original run directory | Epochs recorded | Selected epoch | Status |
| --- | --- | --- | --- | --- | --- |
| [A01](#history-a01) | VideoMAE: frozen head, 8 seconds | `VideoMAE/outputs/run_20261102_010748` | 3 | 3 | Completed |
| [A02](#history-a02) | VideoMAE: final two blocks, 8 seconds | `VideoMAE/outputs/finetune_20261102_023340` | 4 | 1 | Completed |
| [A03](#history-a03) | VideoMAE: corrected final two blocks, 2 seconds | `VideoMAE/outputs/finetune_20261102_190028` | 4 | 3 | Completed |
| [A04](#history-a04) | VideoMAE frozen multi-clip / last | `VideoMAE/outputs/head_comparison_20261103_002458/last` | 9 | 3 | Completed |
| [A05](#history-a05) | VideoMAE frozen multi-clip / mean | `VideoMAE/outputs/head_comparison_20261103_002458/mean` | 12 | 6 | Completed |
| [A06](#history-a06) | VideoMAE frozen multi-clip / gru | `VideoMAE/outputs/head_comparison_20261103_002458/gru` | 13 | 7 | Completed |
| [A07](#history-a07) | R(2+1)D: layer3 + layer4, standard LR | `R2Plus1D/outputs/finetune_20261103_010824_486004` | 7 | 4 | Completed |
| [A08](#history-a08) | R(2+1)D: lower LR 2e-6 / 2e-5 | `R2Plus1D/outputs/finetune_20261103_103252_662277` | 5 | 2 | Completed |
| [A09](#history-a09) | R(2+1)D: repeated lower-LR run | `R2Plus1D/outputs/finetune_20261103_182149_227723` | 5 | 2 | Completed; metrics duplicate earlier run |
| [A10](#history-a10) | R(2+1)D: full-frame black padding | `R2Plus1D/outputs/finetune_20261103_202727_633455` | 7 | 4 | Completed |
| [A11](#history-a11) | R(2+1)D: layer4 only | `R2Plus1D/outputs/finetune_20261103_205431_199155` | 10 | 7 | Completed |
| [A12](#history-a12) | R(2+1)D: stratified temporal jitter | `R2Plus1D/outputs/temporal_jitter_20261104_014632_431431` | 7 | 4 | Completed |
| [A13](#history-a13) | R(2+1)D: progressive unfreezing | `R2Plus1D/outputs/progressive_20261104_024634_830954` | 10 | 7 | Completed |
| [A14](#history-a14) | ResNet18 + GRU64, 16 frames | `ResNet18_GRU/outputs/resnet18_gru_20261105_121809_438598` | 6 | 3 | Completed |
| [A15](#history-a15) | MobileNet + GRU32: uniform negative/test sampling | `Mentor4Frames/outputs/four_frames_20261105_145438_341658` | 6 | 3 | Completed |
| [A16](#history-a16) | MobileNet + GRU32: final-window sampler, first attempt | `Mentor4Frames/outputs/four_frames_last15_20261105_173909_764244` | 4 | 4 | Interrupted during epoch 5 |
| [A17](#history-a17) | MobileNet + GRU32: final-window sampler, AMP repair | `Mentor4Frames/outputs/four_frames_last15_20261105_181840_706976` | 10 | 7 | Completed |
| [A18](#history-a18) | MobileNet + GRU32: multi-horizon training | `MentorMultiHorizon/outputs/multihorizon_20261105_202422_856567` | 6 | 3 | Completed |
| [A19](#history-a19) | MobileNet narrow: seed 43 | `MentorSeedStability/outputs/seed_study_20261105_211409_690778/seed_43` | 9 | 6 | Completed |
| [A20](#history-a20) | MobileNet narrow: seed 44 | `MentorSeedStability/outputs/seed_study_20261105_211409_690778/seed_44` | 8 | 5 | Completed |
| [A21](#history-a21) | MobileNet progressive block10: seed 42 | `MentorProgressiveFT/outputs/block10_20261105_214536_024567` | 12 | 10 | Completed |
| [A22](#history-a22) | MobileNet progressive block10: seed 43 | `MentorProgressiveFT/outputs/block10_seed43_20261105_220831_274817` | 11 | 8 | Completed |
| [A23](#history-a23) | MobileNet + BiGRU16 per direction: seed 42 | `MentorBiGRU/outputs/bigru_seed42_20261106_013426_981097/training` | 6 | 3 | Completed |
| [A24](#history-a24) | MobileNet wide crop: seed 42 | `MentorWideContext/outputs/wide_seed42_20261106_021409_950300` | 10 | 7 | Completed |
| [A25](#history-a25) | MobileNet wide crop: seed 43 | `MentorWideContext/outputs/wide_seed43_20261106_120506_83018720261106_120506_830187` | 9 | 6 | Completed |

The earlier frozen-ResNet18 baseline runs have narrative/notebook-output evidence rather than preserved epoch-by-epoch JSON histories; all their recoverable selected results are in Section 3. The preparation-only directories and cancelled ideas are in Section 10.

## Appendix A. All available per-epoch JSON training histories

Tables below retain all recorded epochs rather than only selected peaks. F1 columns are the positive-class Binary-F1. Loss is not a percentage. F1 is reconstructed from a recorded confusion matrix where needed. `N/R` indicates absent data. MultiHorizon validation uses the explicitly labelled mean of the three horizons; per-horizon values are also retained below its table. These tables do not turn repeated epochs into independent experiments.

<a id="history-a01"></a>
<details>
<summary>A01. VideoMAE: frozen head, 8 seconds (3 recorded epochs; selected epoch 3)</summary>

Original source: `VideoMAE/outputs/run_20261102_010748/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.661068 | N/R | N/R | 59.17 | 0.672892 | N/R | N/R | 64.00 | N/R | N/R | 26.477 |
| 2 | N/R | 0.610659 | N/R | N/R | 66.58 | 0.703973 | N/R | N/R | 63.33 | N/R | N/R | 26.530 |
| 3 | N/R | 0.582880 | N/R | N/R | 69.83 | 0.685174 | N/R | N/R | 65.00 | N/R | N/R | 25.826 |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a02"></a>
<details>
<summary>A02. VideoMAE: final two blocks, 8 seconds (4 recorded epochs; selected epoch 1)</summary>

Original source: `VideoMAE/outputs/finetune_20261102_023340/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.869623 | N/R | N/R | 68.42 | 1.486126 | N/R | N/R | 65.33 | N/R | N/R | 30.975 |
| 2 | N/R | 0.963347 | N/R | N/R | 75.50 | 1.808213 | N/R | N/R | 65.33 | N/R | N/R | 33.692 |
| 3 | N/R | 0.825011 | N/R | N/R | 78.58 | 1.769166 | N/R | N/R | 64.67 | N/R | N/R | 33.779 |
| 4 | N/R | 0.605440 | N/R | N/R | 83.75 | 1.838329 | N/R | N/R | 63.00 | N/R | N/R | 33.692 |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a03"></a>
<details>
<summary>A03. VideoMAE: corrected final two blocks, 2 seconds (4 recorded epochs; selected epoch 3)</summary>

Original source: `VideoMAE/outputs/finetune_20261102_190028/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.524039 | N/R | N/R | 73.42 | 0.644944 | 65.70 | 68.15 | 68.33 | `[[114, 36], [59, 91]]` | N/R | 40.910 |
| 2 | N/R | 0.499907 | N/R | N/R | 76.00 | 0.646144 | 65.96 | 67.88 | 68.00 | `[[111, 39], [57, 93]]` | N/R | 25.500 |
| 3 | N/R | 0.481880 | N/R | N/R | 77.67 | 0.645881 | 67.14 | 68.90 | 69.00 | `[[112, 38], [55, 95]]` | N/R | 24.505 |
| 4 | N/R | 0.469769 | N/R | N/R | 77.92 | 0.648511 | 67.14 | 68.90 | 69.00 | `[[112, 38], [55, 95]]` | N/R | 23.231 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `[5e-06, 5e-05]` | N/R |
| 2 | `[4.340990257669732e-06, 4.340990257669732e-05]` | N/R |
| 3 | `[2.7500000000000004e-06, 2.7500000000000004e-05]` | N/R |
| 4 | `[1.1590097423302683e-06, 1.1590097423302684e-05]` | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a04"></a>
<details>
<summary>A04. VideoMAE frozen multi-clip / last (9 recorded epochs; selected epoch 3)</summary>

Original source: `VideoMAE/outputs/head_comparison_20261103_002458/last/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.648822 | 63.46 | 62.65 | 62.67 | 0.676502 | 64.67 | 64.67 | 64.67 | `[[97, 53], [53, 97]]` | N/R | N/R |
| 2 | N/R | 0.531434 | 74.48 | 73.09 | 73.17 | 0.674995 | 64.00 | 64.00 | 64.00 | `[[96, 54], [54, 96]]` | N/R | N/R |
| 3 | N/R | 0.498880 | 77.00 | 75.68 | 75.75 | 0.664615 | 67.11 | 67.00 | 67.00 | `[[100, 50], [49, 101]]` | N/R | N/R |
| 4 | N/R | 0.479357 | 76.78 | 75.70 | 75.75 | 0.677876 | 65.78 | 65.67 | 65.67 | `[[98, 52], [51, 99]]` | N/R | N/R |
| 5 | N/R | 0.466548 | 77.95 | 76.59 | 76.67 | 0.685952 | 64.88 | 65.00 | 65.00 | `[[98, 52], [53, 97]]` | N/R | N/R |
| 6 | N/R | 0.458641 | 79.01 | 78.13 | 78.17 | 0.692389 | 65.31 | 65.99 | 66.00 | `[[102, 48], [54, 96]]` | N/R | N/R |
| 7 | N/R | 0.442090 | 80.35 | 79.64 | 79.67 | 0.685839 | 63.73 | 64.32 | 64.33 | `[[99, 51], [56, 94]]` | N/R | N/R |
| 8 | N/R | 0.438045 | 79.97 | 79.22 | 79.25 | 0.685485 | 65.10 | 65.33 | 65.33 | `[[99, 51], [53, 97]]` | N/R | N/R |
| 9 | N/R | 0.435238 | 80.16 | 78.83 | 78.92 | 0.703153 | 63.48 | 64.31 | 64.33 | `[[100, 50], [57, 93]]` | N/R | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a05"></a>
<details>
<summary>A05. VideoMAE frozen multi-clip / mean (12 recorded epochs; selected epoch 6)</summary>

Original source: `VideoMAE/outputs/head_comparison_20261103_002458/mean/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.660933 | 60.34 | 60.66 | 60.67 | 0.727280 | 60.96 | 61.97 | 62.00 | `[[97, 53], [61, 89]]` | N/R | N/R |
| 2 | N/R | 0.597039 | 69.12 | 69.25 | 69.25 | 0.732032 | 60.93 | 63.49 | 63.67 | `[[106, 44], [65, 85]]` | N/R | N/R |
| 3 | N/R | 0.566353 | 72.50 | 72.50 | 72.50 | 0.719560 | 61.54 | 63.25 | 63.33 | `[[102, 48], [62, 88]]` | N/R | N/R |
| 4 | N/R | 0.554748 | 72.10 | 72.33 | 72.33 | 0.700297 | 63.16 | 64.91 | 65.00 | `[[105, 45], [60, 90]]` | N/R | N/R |
| 5 | N/R | 0.540247 | 72.64 | 72.50 | 72.50 | 0.728533 | 62.90 | 64.89 | 65.00 | `[[106, 44], [61, 89]]` | N/R | N/R |
| 6 | N/R | 0.543698 | 72.96 | 73.25 | 73.25 | 0.708071 | 66.67 | 67.95 | 68.00 | `[[108, 42], [54, 96]]` | N/R | N/R |
| 7 | N/R | 0.529047 | 74.31 | 74.42 | 74.42 | 0.707316 | 63.41 | 64.93 | 65.00 | `[[104, 46], [59, 91]]` | N/R | N/R |
| 8 | N/R | 0.527384 | 73.16 | 73.58 | 73.58 | 0.719795 | 60.69 | 61.96 | 62.00 | `[[98, 52], [62, 88]]` | N/R | N/R |
| 9 | N/R | 0.523849 | 74.92 | 74.32 | 74.33 | 0.717479 | 62.24 | 63.92 | 64.00 | `[[103, 47], [61, 89]]` | N/R | N/R |
| 10 | N/R | 0.504015 | 75.00 | 75.17 | 75.17 | 0.737038 | 60.27 | 61.31 | 61.33 | `[[96, 54], [62, 88]]` | N/R | N/R |
| 11 | N/R | 0.525265 | 74.23 | 74.25 | 74.25 | 0.713878 | 65.26 | 66.92 | 67.00 | `[[108, 42], [57, 93]]` | N/R | N/R |
| 12 | N/R | 0.509526 | 75.75 | 75.67 | 75.67 | 0.730377 | 62.24 | 63.92 | 64.00 | `[[103, 47], [61, 89]]` | N/R | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a06"></a>
<details>
<summary>A06. VideoMAE frozen multi-clip / gru (13 recorded epochs; selected epoch 7)</summary>

Original source: `VideoMAE/outputs/head_comparison_20261103_002458/gru/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.593679 | 67.89 | 67.92 | 67.92 | 0.656602 | 63.16 | 64.91 | 65.00 | `[[105, 45], [60, 90]]` | N/R | N/R |
| 2 | N/R | 0.499653 | 74.50 | 74.67 | 74.67 | 0.663361 | 61.86 | 62.97 | 63.00 | `[[99, 51], [60, 90]]` | N/R | N/R |
| 3 | N/R | 0.451326 | 78.31 | 78.17 | 78.17 | 0.634137 | 66.67 | 67.00 | 67.00 | `[[102, 48], [51, 99]]` | N/R | N/R |
| 4 | N/R | 0.387536 | 83.82 | 83.75 | 83.75 | 0.698410 | 64.79 | 66.57 | 66.67 | `[[108, 42], [58, 92]]` | N/R | N/R |
| 5 | N/R | 0.340741 | 84.72 | 84.67 | 84.67 | 0.729853 | 66.22 | 66.66 | 66.67 | `[[102, 48], [52, 98]]` | N/R | N/R |
| 6 | N/R | 0.294517 | 88.06 | 88.00 | 88.00 | 0.743894 | 64.26 | 66.80 | 67.00 | `[[112, 38], [61, 89]]` | N/R | N/R |
| 7 | N/R | 0.241071 | 90.01 | 90.08 | 90.08 | 0.815712 | 68.90 | 69.00 | 69.00 | `[[104, 46], [47, 103]]` | N/R | N/R |
| 8 | N/R | 0.187761 | 93.83 | 93.83 | 93.83 | 0.923869 | 66.90 | 67.96 | 68.00 | `[[107, 43], [53, 97]]` | N/R | N/R |
| 9 | N/R | 0.156960 | 94.04 | 94.00 | 94.00 | 1.037135 | 67.12 | 67.98 | 68.00 | `[[106, 44], [52, 98]]` | N/R | N/R |
| 10 | N/R | 0.121941 | 95.59 | 95.58 | 95.58 | 1.120041 | 62.73 | 66.02 | 66.33 | `[[114, 36], [65, 85]]` | N/R | N/R |
| 11 | N/R | 0.087831 | 97.07 | 97.08 | 97.08 | 1.264151 | 66.43 | 68.23 | 68.33 | `[[111, 39], [56, 94]]` | N/R | N/R |
| 12 | N/R | 0.090224 | 96.23 | 96.25 | 96.25 | 1.228994 | 65.05 | 66.29 | 66.33 | `[[105, 45], [56, 94]]` | N/R | N/R |
| 13 | N/R | 0.061069 | 97.50 | 97.50 | 97.50 | 1.350967 | 68.06 | 69.28 | 69.33 | `[[110, 40], [52, 98]]` | N/R | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a07"></a>
<details>
<summary>A07. R(2+1)D: layer3 + layer4, standard LR (7 recorded epochs; selected epoch 4)</summary>

Original source: `R2Plus1D/outputs/finetune_20261103_010824_486004/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.662031 | 62.60 | 62.16 | 62.17 | 0.626030 | 65.52 | 66.63 | 66.67 | `[[105, 45], [55, 95]]` | N/R | 6.563 |
| 2 | N/R | 0.502258 | 74.90 | 74.41 | 74.42 | 0.663450 | 67.09 | 65.23 | 65.33 | `[[90, 60], [44, 106]]` | N/R | 0.381 |
| 3 | N/R | 0.378190 | 84.52 | 84.25 | 84.25 | 0.816704 | 66.43 | 67.93 | 68.00 | `[[109, 41], [55, 95]]` | N/R | 0.450 |
| 4 | N/R | 0.253063 | 89.61 | 89.58 | 89.58 | 0.987251 | 71.34 | 68.39 | 68.67 | `[[89, 61], [33, 117]]` | N/R | 0.450 |
| 5 | N/R | 0.149910 | 94.64 | 94.58 | 94.58 | 1.259939 | 66.45 | 66.33 | 66.33 | `[[99, 51], [50, 100]]` | N/R | 0.453 |
| 6 | N/R | 0.069359 | 97.18 | 97.17 | 97.17 | 1.805563 | 65.10 | 65.33 | 65.33 | `[[99, 51], [53, 97]]` | N/R | 0.454 |
| 7 | N/R | 0.036422 | 98.42 | 98.42 | 98.42 | 1.947627 | 63.64 | 65.26 | 65.33 | `[[105, 45], [59, 91]]` | N/R | 0.451 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `[1e-05, 0.0001]` | N/R |
| 2 | `[9.779754323328192e-06, 9.779754323328192e-05]` | N/R |
| 3 | `[9.140576474687263e-06, 9.140576474687264e-05]` | N/R |
| 4 | `[8.14503363531613e-06, 8.14503363531613e-05]` | N/R |
| 5 | `[6.890576474687264e-06, 6.890576474687263e-05]` | N/R |
| 6 | `[5.500000000000001e-06, 5.500000000000001e-05]` | N/R |
| 7 | `[4.109423525312738e-06, 4.109423525312738e-05]` | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a08"></a>
<details>
<summary>A08. R(2+1)D: lower LR 2e-6 / 2e-5 (5 recorded epochs; selected epoch 2)</summary>

Original source: `R2Plus1D/outputs/finetune_20261103_103252_662277/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.774287 | 58.60 | 50.51 | 51.83 | 0.659838 | 61.27 | 63.23 | 63.33 | `[[103, 47], [63, 87]]` | N/R | 0.302 |
| 2 | N/R | 0.630575 | 63.54 | 63.75 | 63.75 | 0.628960 | 64.29 | 63.31 | 63.33 | `[[91, 59], [51, 99]]` | N/R | 0.754 |
| 3 | N/R | 0.571780 | 71.65 | 71.58 | 71.58 | 0.625265 | 64.10 | 62.61 | 62.67 | `[[88, 62], [50, 100]]` | N/R | 1.420 |
| 4 | N/R | 0.525480 | 72.21 | 72.42 | 72.42 | 0.633927 | 62.94 | 64.59 | 64.67 | `[[104, 46], [60, 90]]` | N/R | 1.453 |
| 5 | N/R | 0.487169 | 77.42 | 77.25 | 77.25 | 0.639003 | 63.97 | 64.33 | 64.33 | `[[98, 52], [55, 95]]` | N/R | 1.332 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `[2e-06, 2e-05]` | N/R |
| 2 | `[1.9559508646656384e-06, 1.9559508646656384e-05]` | N/R |
| 3 | `[1.8281152949374526e-06, 1.8281152949374527e-05]` | N/R |
| 4 | `[1.6290067270632256e-06, 1.629006727063226e-05]` | N/R |
| 5 | `[1.3781152949374526e-06, 1.3781152949374527e-05]` | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a09"></a>
<details>
<summary>A09. R(2+1)D: repeated lower-LR run (5 recorded epochs; selected epoch 2)</summary>

Original source: `R2Plus1D/outputs/finetune_20261103_182149_227723/history.json`. Status: Completed; metrics duplicate earlier run.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.774287 | 58.60 | 50.51 | 51.83 | 0.659838 | 61.27 | 63.23 | 63.33 | `[[103, 47], [63, 87]]` | N/R | 7.423 |
| 2 | N/R | 0.630575 | 63.54 | 63.75 | 63.75 | 0.628960 | 64.29 | 63.31 | 63.33 | `[[91, 59], [51, 99]]` | N/R | 0.340 |
| 3 | N/R | 0.571780 | 71.65 | 71.58 | 71.58 | 0.625265 | 64.10 | 62.61 | 62.67 | `[[88, 62], [50, 100]]` | N/R | 0.337 |
| 4 | N/R | 0.525480 | 72.21 | 72.42 | 72.42 | 0.633927 | 62.94 | 64.59 | 64.67 | `[[104, 46], [60, 90]]` | N/R | 0.337 |
| 5 | N/R | 0.487169 | 77.42 | 77.25 | 77.25 | 0.639003 | 63.97 | 64.33 | 64.33 | `[[98, 52], [55, 95]]` | N/R | 0.342 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `[2e-06, 2e-05]` | N/R |
| 2 | `[1.9559508646656384e-06, 1.9559508646656384e-05]` | N/R |
| 3 | `[1.8281152949374526e-06, 1.8281152949374527e-05]` | N/R |
| 4 | `[1.6290067270632256e-06, 1.629006727063226e-05]` | N/R |
| 5 | `[1.3781152949374526e-06, 1.3781152949374527e-05]` | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a10"></a>
<details>
<summary>A10. R(2+1)D: full-frame black padding (7 recorded epochs; selected epoch 4)</summary>

Original source: `R2Plus1D/outputs/finetune_20261103_202727_633455/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.655791 | 64.30 | 63.04 | 63.08 | 0.621465 | 66.00 | 66.00 | 66.00 | `[[99, 51], [51, 99]]` | N/R | 11.358 |
| 2 | N/R | 0.511316 | 75.85 | 75.15 | 75.17 | 0.652923 | 69.48 | 68.64 | 68.67 | `[[99, 51], [43, 107]]` | N/R | 1.585 |
| 3 | N/R | 0.380239 | 84.48 | 84.16 | 84.17 | 0.782779 | 66.90 | 68.58 | 68.67 | `[[111, 39], [55, 95]]` | N/R | 1.637 |
| 4 | N/R | 0.275005 | 89.00 | 88.92 | 88.92 | 1.077505 | 71.27 | 63.79 | 65.33 | `[[67, 83], [21, 129]]` | N/R | 0.638 |
| 5 | N/R | 0.154976 | 94.02 | 94.00 | 94.00 | 1.312830 | 66.45 | 66.33 | 66.33 | `[[99, 51], [50, 100]]` | N/R | 0.727 |
| 6 | N/R | 0.076125 | 97.27 | 97.25 | 97.25 | 1.758818 | 67.38 | 69.52 | 69.67 | `[[115, 35], [56, 94]]` | N/R | 0.821 |
| 7 | N/R | 0.037670 | 98.83 | 98.83 | 98.83 | 1.989909 | 69.28 | 68.65 | 68.67 | `[[100, 50], [44, 106]]` | N/R | 0.769 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `[1e-05, 0.0001]` | N/R |
| 2 | `[9.779754323328192e-06, 9.779754323328192e-05]` | N/R |
| 3 | `[9.140576474687263e-06, 9.140576474687264e-05]` | N/R |
| 4 | `[8.14503363531613e-06, 8.14503363531613e-05]` | N/R |
| 5 | `[6.890576474687264e-06, 6.890576474687263e-05]` | N/R |
| 6 | `[5.500000000000001e-06, 5.500000000000001e-05]` | N/R |
| 7 | `[4.109423525312738e-06, 4.109423525312738e-05]` | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a11"></a>
<details>
<summary>A11. R(2+1)D: layer4 only (10 recorded epochs; selected epoch 7)</summary>

Original source: `R2Plus1D/outputs/finetune_20261103_205431_199155/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.671158 | 61.24 | 60.31 | 60.33 | 0.628000 | 63.92 | 64.97 | 65.00 | `[[102, 48], [57, 93]]` | N/R | 11.415 |
| 2 | N/R | 0.514051 | 75.00 | 74.66 | 74.67 | 0.656191 | 67.09 | 65.23 | 65.33 | `[[90, 60], [44, 106]]` | N/R | 0.599 |
| 3 | N/R | 0.404559 | 82.85 | 82.32 | 82.33 | 0.783798 | 65.99 | 66.33 | 66.33 | `[[101, 49], [52, 98]]` | N/R | 0.586 |
| 4 | N/R | 0.291096 | 88.00 | 88.00 | 88.00 | 0.889841 | 69.70 | 66.33 | 66.67 | `[[85, 65], [35, 115]]` | N/R | 0.603 |
| 5 | N/R | 0.196345 | 92.33 | 92.16 | 92.17 | 1.021878 | 65.99 | 66.33 | 66.33 | `[[101, 49], [52, 98]]` | N/R | 0.645 |
| 6 | N/R | 0.113465 | 96.47 | 96.42 | 96.42 | 1.364154 | 68.35 | 66.57 | 66.67 | `[[92, 58], [42, 108]]` | N/R | 0.481 |
| 7 | N/R | 0.073603 | 97.20 | 97.17 | 97.17 | 1.557450 | 70.22 | 68.21 | 68.33 | `[[93, 57], [38, 112]]` | N/R | 0.886 |
| 8 | N/R | 0.046699 | 98.59 | 98.58 | 98.58 | 1.667560 | 63.41 | 64.93 | 65.00 | `[[104, 46], [59, 91]]` | N/R | 1.610 |
| 9 | N/R | 0.037624 | 99.25 | 99.25 | 99.25 | 1.824058 | 64.34 | 65.93 | 66.00 | `[[106, 44], [58, 92]]` | N/R | 0.517 |
| 10 | N/R | 0.019400 | 99.58 | 99.58 | 99.58 | 1.890354 | 64.63 | 65.32 | 65.33 | `[[101, 49], [55, 95]]` | N/R | 0.613 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `[1e-05, 0.0001]` | N/R |
| 2 | `[9.779754323328192e-06, 9.779754323328192e-05]` | N/R |
| 3 | `[9.140576474687263e-06, 9.140576474687264e-05]` | N/R |
| 4 | `[8.14503363531613e-06, 8.14503363531613e-05]` | N/R |
| 5 | `[6.890576474687264e-06, 6.890576474687263e-05]` | N/R |
| 6 | `[5.500000000000001e-06, 5.500000000000001e-05]` | N/R |
| 7 | `[4.109423525312738e-06, 4.109423525312738e-05]` | N/R |
| 8 | `[2.854966364683872e-06, 2.854966364683872e-05]` | N/R |
| 9 | `[1.8594235253127373e-06, 1.8594235253127375e-05]` | N/R |
| 10 | `[1.2202456766718092e-06, 1.2202456766718093e-05]` | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a12"></a>
<details>
<summary>A12. R(2+1)D: stratified temporal jitter (7 recorded epochs; selected epoch 4)</summary>

Original source: `R2Plus1D/outputs/temporal_jitter_20261104_014632_431431/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N/R | 0.660513 | 62.84 | 62.24 | 62.25 | 0.630986 | 65.03 | 66.59 | 66.67 | `[[107, 43], [57, 93]]` | N/R | 14.822 |
| 2 | N/R | 0.510934 | 75.08 | 74.31 | 74.33 | 0.657406 | 66.88 | 65.26 | 65.33 | `[[91, 59], [45, 105]]` | N/R | 0.755 |
| 3 | N/R | 0.399507 | 82.85 | 82.32 | 82.33 | 0.804473 | 65.23 | 67.51 | 67.67 | `[[112, 38], [59, 91]]` | N/R | 0.931 |
| 4 | N/R | 0.288143 | 87.80 | 87.75 | 87.75 | 0.902945 | 72.00 | 69.45 | 69.67 | `[[92, 58], [33, 117]]` | N/R | 0.934 |
| 5 | N/R | 0.169760 | 93.52 | 93.42 | 93.42 | 1.085686 | 62.55 | 65.43 | 65.67 | `[[111, 39], [64, 86]]` | N/R | 2.252 |
| 6 | N/R | 0.104318 | 95.77 | 95.75 | 95.75 | 1.523993 | 68.87 | 68.67 | 68.67 | `[[102, 48], [46, 104]]` | N/R | 2.047 |
| 7 | N/R | 0.057910 | 97.85 | 97.83 | 97.83 | 1.680355 | 69.06 | 68.32 | 68.33 | `[[99, 51], [44, 106]]` | N/R | 1.466 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `[1e-05, 0.0001]` | N/R |
| 2 | `[9.779754323328192e-06, 9.779754323328192e-05]` | N/R |
| 3 | `[9.140576474687263e-06, 9.140576474687264e-05]` | N/R |
| 4 | `[8.14503363531613e-06, 8.14503363531613e-05]` | N/R |
| 5 | `[6.890576474687264e-06, 6.890576474687263e-05]` | N/R |
| 6 | `[5.500000000000001e-06, 5.500000000000001e-05]` | N/R |
| 7 | `[4.109423525312738e-06, 4.109423525312738e-05]` | N/R |

`N/R` stage means the history did not store a stage label; it does not imply that training was absent.

</details>

<a id="history-a13"></a>
<details>
<summary>A13. R(2+1)D: progressive unfreezing (10 recorded epochs; selected epoch 7)</summary>

Original source: `R2Plus1D/outputs/progressive_20261104_024634_830954/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | head | 0.815307 | 51.94 | 46.74 | 47.25 | 0.691169 | 42.11 | 50.80 | 52.33 | `[[105, 45], [98, 52]]` | N/R | 7.836 |
| 2 | head | 0.719693 | 51.21 | 51.08 | 51.08 | 0.669679 | 63.06 | 58.50 | 59.00 | `[[72, 78], [45, 105]]` | N/R | 0.238 |
| 3 | layer4 | 0.637404 | 63.97 | 64.41 | 64.42 | 0.630849 | 67.52 | 66.29 | 66.33 | `[[94, 56], [45, 105]]` | N/R | 0.359 |
| 4 | layer4 | 0.536343 | 74.24 | 73.91 | 73.92 | 0.635496 | 66.89 | 66.67 | 66.67 | `[[99, 51], [49, 101]]` | N/R | 0.362 |
| 5 | layer3_layer4 | 0.449236 | 79.45 | 78.90 | 78.92 | 0.678713 | 69.30 | 66.02 | 66.33 | `[[85, 65], [36, 114]]` | N/R | 0.402 |
| 6 | layer3_layer4 | 0.333214 | 85.88 | 85.83 | 85.83 | 0.800409 | 67.11 | 67.33 | 67.33 | `[[102, 48], [50, 100]]` | N/R | 0.401 |
| 7 | layer3_layer4 | 0.266772 | 88.91 | 88.66 | 88.67 | 0.838754 | 70.62 | 68.53 | 68.67 | `[[93, 57], [37, 113]]` | N/R | 0.402 |
| 8 | layer3_layer4 | 0.195800 | 92.50 | 92.50 | 92.50 | 0.913209 | 65.97 | 67.28 | 67.33 | `[[107, 43], [55, 95]]` | N/R | 0.403 |
| 9 | layer3_layer4 | 0.165556 | 93.62 | 93.58 | 93.58 | 1.028912 | 66.90 | 68.58 | 68.67 | `[[111, 39], [55, 95]]` | N/R | 0.405 |
| 10 | layer3_layer4 | 0.120824 | 95.51 | 95.50 | 95.50 | 1.089852 | 67.55 | 67.33 | 67.33 | `[[100, 50], [48, 102]]` | N/R | 0.403 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"layer3": 0.0, "layer4": 0.0, "fc": 0.0001}` | 1026 |
| 2 | `{"layer3": 0.0, "layer4": 0.0, "fc": 9.779754323328192e-05}` | 1026 |
| 3 | `{"layer3": 0.0, "layer4": 9.140576474687263e-06, "fc": 9.140576474687264e-05}` | 23496550 |
| 4 | `{"layer3": 0.0, "layer4": 8.14503363531613e-06, "fc": 8.14503363531613e-05}` | 23496550 |
| 5 | `{"layer3": 6.890576474687264e-06, "layer4": 6.890576474687264e-06, "fc": 6.890576474687263e-05}` | 29371286 |
| 6 | `{"layer3": 5.500000000000001e-06, "layer4": 5.500000000000001e-06, "fc": 5.500000000000001e-05}` | 29371286 |
| 7 | `{"layer3": 4.109423525312738e-06, "layer4": 4.109423525312738e-06, "fc": 4.109423525312738e-05}` | 29371286 |
| 8 | `{"layer3": 2.854966364683872e-06, "layer4": 2.854966364683872e-06, "fc": 2.854966364683872e-05}` | 29371286 |
| 9 | `{"layer3": 1.8594235253127373e-06, "layer4": 1.8594235253127373e-06, "fc": 1.8594235253127375e-05}` | 29371286 |
| 10 | `{"layer3": 1.2202456766718092e-06, "layer4": 1.2202456766718092e-06, "fc": 1.2202456766718093e-05}` | 29371286 |

</details>

<a id="history-a14"></a>
<details>
<summary>A14. ResNet18 + GRU64, 16 frames (6 recorded epochs; selected epoch 3)</summary>

Original source: `ResNet18_GRU/outputs/resnet18_gru_20261105_121809_438598/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.671856 | 60.81 | 60.24 | 60.25 | 0.626989 | 67.52 | 66.29 | 66.33 | `[[94, 56], [45, 105]]` | N/R | 9.269 |
| 2 | temporal_head | 0.622684 | 64.88 | 65.00 | 65.00 | 0.627544 | 71.39 | 67.11 | 67.67 | `[[82, 68], [29, 121]]` | N/R | 0.235 |
| 3 | encoder_layer4 | 0.589634 | 69.90 | 69.49 | 69.50 | 0.629891 | 71.39 | 67.11 | 67.67 | `[[82, 68], [29, 121]]` | N/R | 0.257 |
| 4 | encoder_layer4 | 0.525514 | 74.75 | 74.67 | 74.67 | 0.610112 | 71.11 | 69.59 | 69.67 | `[[97, 53], [38, 112]]` | N/R | 0.262 |
| 5 | encoder_layer4 | 0.445400 | 81.34 | 81.42 | 81.42 | 0.607439 | 70.10 | 68.96 | 69.00 | `[[98, 52], [41, 109]]` | N/R | 0.257 |
| 6 | encoder_layer4 | 0.392680 | 84.72 | 84.67 | 84.67 | 0.629403 | 70.22 | 68.21 | 68.33 | `[[93, 57], [38, 112]]` | N/R | 0.263 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder.layer4": 0.0, "gru": 0.001, "classifier": 0.001}` | 111106 |
| 2 | `{"encoder.layer4": 0.0, "gru": 0.001, "classifier": 0.001}` | 111106 |
| 3 | `{"encoder.layer4": 1e-05, "gru": 0.0001, "classifier": 0.0001}` | 8504834 |
| 4 | `{"encoder.layer4": 9.554359905560887e-06, "gru": 9.554359905560886e-05, "classifier": 9.554359905560886e-05}` | 8504834 |
| 5 | `{"encoder.layer4": 8.305704108364301e-06, "gru": 8.305704108364303e-05, "classifier": 8.305704108364303e-05}` | 8504834 |
| 6 | `{"encoder.layer4": 6.501344202803415e-06, "gru": 6.501344202803415e-05, "classifier": 6.501344202803415e-05}` | 8504834 |

</details>

<a id="history-a15"></a>
<details>
<summary>A15. MobileNet + GRU32: uniform negative/test sampling (6 recorded epochs; selected epoch 3)</summary>

Original source: `Mentor4Frames/outputs/four_frames_20261105_145438_341658/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.621333 | 63.50 | 64.15 | 64.17 | 0.609969 | 61.00 | 65.69 | 66.33 | `[[120, 30], [71, 79]]` | N/R | 4.775 |
| 2 | temporal_head | 0.555564 | 72.13 | 71.92 | 71.92 | 0.563053 | 72.16 | 72.98 | 73.00 | `[[114, 36], [45, 105]]` | N/R | 0.074 |
| 3 | encoder_tail | 0.511629 | 73.59 | 74.15 | 74.17 | 0.554713 | 74.00 | 74.00 | 74.00 | `[[111, 39], [39, 111]]` | N/R | 0.069 |
| 4 | encoder_tail | 0.497490 | 76.55 | 76.42 | 76.42 | 0.553355 | 73.22 | 73.66 | 73.67 | `[[113, 37], [42, 108]]` | N/R | 0.065 |
| 5 | encoder_tail | 0.490148 | 77.83 | 77.49 | 77.50 | 0.562667 | 71.22 | 73.19 | 73.33 | `[[121, 29], [51, 99]]` | N/R | 0.065 |
| 6 | encoder_tail | 0.486436 | 75.94 | 76.49 | 76.50 | 0.548632 | 73.90 | 74.33 | 74.33 | `[[114, 36], [41, 109]]` | N/R | 0.064 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 409170 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 409170 |

</details>

<a id="history-a16"></a>
<details>
<summary>A16. MobileNet + GRU32: final-window sampler, first attempt (4 recorded epochs; selected epoch 4)</summary>

Original source: `Mentor4Frames/outputs/four_frames_last15_20261105_173909_764244/history.json`. Status: Interrupted during epoch 5.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.627922 | 63.99 | 64.17 | 64.17 | 0.625902 | 61.24 | 66.00 | 66.67 | `[[121, 29], [71, 79]]` | N/R | 7.806 |
| 2 | temporal_head | 0.576372 | 70.65 | 69.72 | 69.75 | 0.589695 | 70.82 | 70.33 | 70.33 | `[[103, 47], [42, 108]]` | N/R | 0.125 |
| 3 | encoder_tail | 0.541494 | 71.46 | 71.25 | 71.25 | 0.589392 | 70.82 | 70.33 | 70.33 | `[[103, 47], [42, 108]]` | N/R | 0.300 |
| 4 | encoder_tail | 0.524964 | 74.75 | 74.33 | 74.33 | 0.590316 | 72.24 | 72.33 | 72.33 | `[[109, 41], [42, 108]]` | N/R | 0.346 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |

</details>

<a id="history-a17"></a>
<details>
<summary>A17. MobileNet + GRU32: final-window sampler, AMP repair (10 recorded epochs; selected epoch 7)</summary>

Original source: `Mentor4Frames/outputs/four_frames_last15_20261105_181840_706976/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.627922 | 63.99 | 64.17 | 64.17 | 0.625902 | 61.24 | 66.00 | 66.67 | `[[121, 29], [71, 79]]` | 0 | 8.475 |
| 2 | temporal_head | 0.576372 | 70.65 | 69.72 | 69.75 | 0.589695 | 70.82 | 70.33 | 70.33 | `[[103, 47], [42, 108]]` | 0 | 0.194 |
| 3 | encoder_tail | 0.541485 | 71.46 | 71.25 | 71.25 | 0.589396 | 70.82 | 70.33 | 70.33 | `[[103, 47], [42, 108]]` | 0 | 0.178 |
| 4 | encoder_tail | 0.524958 | 74.75 | 74.33 | 74.33 | 0.590307 | 72.24 | 72.33 | 72.33 | `[[109, 41], [42, 108]]` | 0 | 0.170 |
| 5 | encoder_tail | 0.518423 | 75.93 | 74.69 | 74.75 | 0.595537 | 70.55 | 71.31 | 71.33 | `[[111, 39], [47, 103]]` | 0 | 0.149 |
| 6 | encoder_tail | 0.519396 | 74.90 | 74.58 | 74.58 | 0.589768 | 71.62 | 72.00 | 72.00 | `[[110, 40], [44, 106]]` | 0 | 0.204 |
| 7 | encoder_tail | 0.512500 | 75.16 | 74.83 | 74.83 | 0.587492 | 72.61 | 72.33 | 72.33 | `[[107, 43], [40, 110]]` | 0 | 0.208 |
| 8 | encoder_tail | 0.504154 | 76.40 | 75.73 | 75.75 | 0.589720 | 71.81 | 72.00 | 72.00 | `[[109, 41], [43, 107]]` | 0 | 0.205 |
| 9 | encoder_tail | 0.491904 | 76.82 | 76.41 | 76.42 | 0.589215 | 71.81 | 72.00 | 72.00 | `[[109, 41], [43, 107]]` | 0 | 0.204 |
| 10 | encoder_tail | 0.503700 | 75.71 | 75.07 | 75.08 | 0.589803 | 71.81 | 72.00 | 72.00 | `[[109, 41], [43, 107]]` | 0 | 0.200 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 409170 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 409170 |
| 7 | `{"encoder_tail": 6.281416799501188e-06, "temporal_head": 6.281416799501188e-05}` | 409170 |
| 8 | `{"encoder_tail": 4.718583200498814e-06, "temporal_head": 4.718583200498814e-05}` | 409170 |
| 9 | `{"encoder_tail": 3.2500000000000015e-06, "temporal_head": 3.250000000000001e-05}` | 409170 |
| 10 | `{"encoder_tail": 2.0528000059646e-06, "temporal_head": 2.0528000059645997e-05}` | 409170 |

</details>

<a id="history-a18"></a>
<details>
<summary>A18. MobileNet + GRU32: multi-horizon training (6 recorded epochs; selected epoch 3)</summary>

Original source: `MentorMultiHorizon/outputs/multihorizon_20261105_202422_856567/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

Validation columns in the first table are means across the three horizons.

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.639433 | 62.67 | 63.16 | 63.17 | 0.644319 | 58.60 | 63.46 | 64.11 | N/R | 0 | 8.596 |
| 2 | temporal_head | 0.595398 | 67.78 | 67.83 | 67.83 | 0.623123 | 67.44 | 66.76 | 66.78 | N/R | 0 | 2.595 |
| 3 | encoder_tail | 0.571878 | 71.67 | 71.07 | 71.08 | 0.624732 | 67.44 | 67.10 | 67.11 | N/R | 0 | 1.797 |
| 4 | encoder_tail | 0.551277 | 72.47 | 72.33 | 72.33 | 0.625989 | 65.96 | 66.32 | 66.33 | N/R | 0 | 1.500 |
| 5 | encoder_tail | 0.557869 | 72.89 | 72.41 | 72.42 | 0.624819 | 66.85 | 66.99 | 67.00 | N/R | 0 | 0.970 |
| 6 | encoder_tail | 0.546369 | 73.55 | 73.08 | 73.08 | 0.627364 | 65.81 | 66.21 | 66.22 | N/R | 0 | 0.687 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 409170 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 409170 |

Per-horizon validation details:

| Epoch | Horizon | Binary-F1 (%) | Macro-F1 (%) | Accuracy (%) | Loss | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5s | 61.13 | 65.19 | 65.67 | 0.629638 | `[[116, 34], [69, 81]]` |
| 1 | 1.0s | 59.54 | 64.09 | 64.67 | 0.645317 | `[[116, 34], [72, 78]]` |
| 1 | 1.5s | 55.12 | 61.09 | 62.00 | 0.658001 | `[[116, 34], [80, 70]]` |
| 2 | 0.5s | 69.87 | 68.62 | 68.67 | 0.604941 | `[[97, 53], [41, 109]]` |
| 2 | 1.0s | 67.32 | 66.65 | 66.67 | 0.622063 | `[[97, 53], [47, 103]]` |
| 2 | 1.5s | 65.12 | 65.00 | 65.00 | 0.642365 | `[[97, 53], [52, 98]]` |
| 3 | 0.5s | 69.90 | 68.97 | 69.00 | 0.604986 | `[[99, 51], [42, 108]]` |
| 3 | 1.0s | 67.33 | 67.00 | 67.00 | 0.623333 | `[[99, 51], [48, 102]]` |
| 3 | 1.5s | 65.10 | 65.33 | 65.33 | 0.645877 | `[[99, 51], [53, 97]]` |
| 4 | 0.5s | 68.65 | 68.33 | 68.33 | 0.604292 | `[[101, 49], [46, 104]]` |
| 4 | 1.0s | 65.99 | 66.33 | 66.33 | 0.624703 | `[[101, 49], [52, 98]]` |
| 4 | 1.5s | 63.23 | 64.30 | 64.33 | 0.648972 | `[[101, 49], [58, 92]]` |
| 5 | 0.5s | 69.93 | 69.32 | 69.33 | 0.603329 | `[[101, 49], [43, 107]]` |
| 5 | 1.0s | 66.44 | 66.67 | 66.67 | 0.623572 | `[[101, 49], [51, 99]]` |
| 5 | 1.5s | 64.16 | 64.98 | 65.00 | 0.647555 | `[[101, 49], [56, 94]]` |
| 6 | 0.5s | 68.65 | 68.33 | 68.33 | 0.604353 | `[[101, 49], [46, 104]]` |
| 6 | 1.0s | 65.08 | 65.66 | 65.67 | 0.625999 | `[[101, 49], [54, 96]]` |
| 6 | 1.5s | 63.70 | 64.64 | 64.67 | 0.651741 | `[[101, 49], [57, 93]]` |

Training positive counts by horizon:

| Epoch | 0.5 seconds | 1.0 seconds | 1.5 seconds |
| --- | --- | --- | --- |
| 1 | 196 | 202 | 202 |
| 2 | 206 | 200 | 194 |
| 3 | 194 | 208 | 198 |
| 4 | 212 | 195 | 193 |
| 5 | 203 | 186 | 211 |
| 6 | 207 | 223 | 170 |

</details>

<a id="history-a19"></a>
<details>
<summary>A19. MobileNet narrow: seed 43 (9 recorded epochs; selected epoch 6)</summary>

Original source: `MentorSeedStability/outputs/seed_study_20261105_211409_690778/seed_43/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.636889 | 63.03 | 62.75 | 62.75 | 0.630417 | 70.24 | 66.18 | 66.67 | `[[82, 68], [32, 118]]` | 0 | 6.231 |
| 2 | temporal_head | 0.574412 | 70.27 | 70.25 | 70.25 | 0.614460 | 63.04 | 67.67 | 68.33 | `[[124, 26], [69, 81]]` | 0 | 0.094 |
| 3 | encoder_tail | 0.540383 | 71.61 | 72.24 | 72.25 | 0.591424 | 70.20 | 70.00 | 70.00 | `[[104, 46], [44, 106]]` | 0 | 0.101 |
| 4 | encoder_tail | 0.525508 | 73.29 | 72.91 | 72.92 | 0.595737 | 70.47 | 70.67 | 70.67 | `[[107, 43], [45, 105]]` | 0 | 0.097 |
| 5 | encoder_tail | 0.525497 | 73.96 | 73.83 | 73.83 | 0.592290 | 71.48 | 70.99 | 71.00 | `[[104, 46], [41, 109]]` | 0 | 0.089 |
| 6 | encoder_tail | 0.512414 | 74.86 | 74.58 | 74.58 | 0.591029 | 72.31 | 71.65 | 71.67 | `[[104, 46], [39, 111]]` | 0 | 0.081 |
| 7 | encoder_tail | 0.506458 | 76.59 | 75.45 | 75.50 | 0.592087 | 70.71 | 71.00 | 71.00 | `[[108, 42], [45, 105]]` | 0 | 0.098 |
| 8 | encoder_tail | 0.507500 | 75.36 | 75.58 | 75.58 | 0.589512 | 72.13 | 71.66 | 71.67 | `[[105, 45], [40, 110]]` | 0 | 0.097 |
| 9 | encoder_tail | 0.495252 | 75.97 | 75.75 | 75.75 | 0.590206 | 72.19 | 72.00 | 72.00 | `[[107, 43], [41, 109]]` | 0 | 0.100 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 409170 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 409170 |
| 7 | `{"encoder_tail": 6.281416799501188e-06, "temporal_head": 6.281416799501188e-05}` | 409170 |
| 8 | `{"encoder_tail": 4.718583200498814e-06, "temporal_head": 4.718583200498814e-05}` | 409170 |
| 9 | `{"encoder_tail": 3.2500000000000015e-06, "temporal_head": 3.250000000000001e-05}` | 409170 |

</details>

<a id="history-a20"></a>
<details>
<summary>A20. MobileNet narrow: seed 44 (8 recorded epochs; selected epoch 5)</summary>

Original source: `MentorSeedStability/outputs/seed_study_20261105_211409_690778/seed_44/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.631453 | 65.33 | 63.86 | 63.92 | 0.634173 | 62.41 | 66.23 | 66.67 | `[[117, 33], [67, 83]]` | 0 | 0.094 |
| 2 | temporal_head | 0.575707 | 70.45 | 70.50 | 70.50 | 0.590426 | 72.20 | 70.95 | 71.00 | `[[100, 50], [37, 113]]` | 0 | 0.099 |
| 3 | encoder_tail | 0.539705 | 73.44 | 72.73 | 72.75 | 0.592071 | 71.66 | 70.98 | 71.00 | `[[103, 47], [40, 110]]` | 0 | 0.110 |
| 4 | encoder_tail | 0.522243 | 75.49 | 74.99 | 75.00 | 0.595656 | 72.13 | 71.66 | 71.67 | `[[105, 45], [40, 110]]` | 0 | 0.113 |
| 5 | encoder_tail | 0.513074 | 75.34 | 75.82 | 75.83 | 0.596499 | 72.85 | 72.67 | 72.67 | `[[108, 42], [40, 110]]` | 0 | 0.109 |
| 6 | encoder_tail | 0.511234 | 75.66 | 75.50 | 75.50 | 0.596413 | 72.13 | 71.66 | 71.67 | `[[105, 45], [40, 110]]` | 0 | 0.095 |
| 7 | encoder_tail | 0.501566 | 76.03 | 75.83 | 75.83 | 0.595192 | 72.13 | 71.66 | 71.67 | `[[105, 45], [40, 110]]` | 0 | 0.096 |
| 8 | encoder_tail | 0.496923 | 76.31 | 76.25 | 76.25 | 0.596636 | 71.71 | 71.33 | 71.33 | `[[105, 45], [41, 109]]` | 0 | 0.102 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 409170 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 409170 |
| 7 | `{"encoder_tail": 6.281416799501188e-06, "temporal_head": 6.281416799501188e-05}` | 409170 |
| 8 | `{"encoder_tail": 4.718583200498814e-06, "temporal_head": 4.718583200498814e-05}` | 409170 |

</details>

<a id="history-a21"></a>
<details>
<summary>A21. MobileNet progressive block10: seed 42 (12 recorded epochs; selected epoch 10)</summary>

Original source: `MentorProgressiveFT/outputs/block10_20261105_214536_024567/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.627922 | 63.99 | 64.17 | 64.17 | 0.625902 | 61.24 | 66.00 | 66.67 | `[[121, 29], [71, 79]]` | 0 | 6.003 |
| 2 | temporal_head | 0.576372 | 70.65 | 69.72 | 69.75 | 0.589695 | 70.82 | 70.33 | 70.33 | `[[103, 47], [42, 108]]` | 0 | 0.080 |
| 3 | encoder_11_12 | 0.541485 | 71.46 | 71.25 | 71.25 | 0.589396 | 70.82 | 70.33 | 70.33 | `[[103, 47], [42, 108]]` | 0 | 0.082 |
| 4 | encoder_11_12 | 0.524958 | 74.75 | 74.33 | 74.33 | 0.590307 | 72.24 | 72.33 | 72.33 | `[[109, 41], [42, 108]]` | 0 | 0.080 |
| 5 | encoder_10_11_12 | 0.518454 | 75.71 | 74.44 | 74.50 | 0.594871 | 70.99 | 71.65 | 71.67 | `[[111, 39], [46, 104]]` | 0 | 0.085 |
| 6 | encoder_10_11_12 | 0.519384 | 74.90 | 74.58 | 74.58 | 0.590128 | 72.05 | 72.33 | 72.33 | `[[110, 40], [43, 107]]` | 0 | 0.085 |
| 7 | encoder_10_11_12 | 0.513550 | 75.29 | 74.82 | 74.83 | 0.587851 | 72.43 | 72.33 | 72.33 | `[[108, 42], [41, 109]]` | 0 | 0.087 |
| 8 | encoder_10_11_12 | 0.505048 | 76.52 | 75.90 | 75.92 | 0.589657 | 71.81 | 72.00 | 72.00 | `[[109, 41], [43, 107]]` | 0 | 0.084 |
| 9 | encoder_10_11_12 | 0.492562 | 76.78 | 76.41 | 76.42 | 0.589021 | 71.38 | 71.66 | 71.67 | `[[109, 41], [44, 106]]` | 0 | 0.084 |
| 10 | encoder_10_11_12 | 0.504397 | 75.51 | 74.90 | 74.92 | 0.589563 | 72.91 | 73.00 | 73.00 | `[[110, 40], [41, 109]]` | 0 | 0.084 |
| 11 | encoder_10_11_12 | 0.495590 | 76.50 | 75.81 | 75.83 | 0.589843 | 72.91 | 73.00 | 73.00 | `[[110, 40], [41, 109]]` | 0 | 0.088 |
| 12 | encoder_10_11_12 | 0.494578 | 76.75 | 76.41 | 76.42 | 0.590047 | 72.48 | 72.67 | 72.67 | `[[110, 40], [42, 108]]` | 0 | 0.085 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_10": 0.0, "encoder_11_12": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_10": 0.0, "encoder_11_12": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_10": 0.0, "encoder_11_12": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_10": 0.0, "encoder_11_12": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_10": 2e-06, "encoder_11_12": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 703266 |
| 6 | `{"encoder_10": 1.910871981112177e-06, "encoder_11_12": 7.75e-06, "temporal_head": 7.75e-05}` | 703266 |
| 7 | `{"encoder_10": 1.6611408216728601e-06, "encoder_11_12": 6.281416799501188e-06, "temporal_head": 6.281416799501188e-05}` | 703266 |
| 8 | `{"encoder_10": 1.3002688405606828e-06, "encoder_11_12": 4.718583200498814e-06, "temporal_head": 4.718583200498814e-05}` | 703266 |
| 9 | `{"encoder_10": 8.997311594393171e-07, "encoder_11_12": 3.2500000000000015e-06, "temporal_head": 3.250000000000001e-05}` | 703266 |
| 10 | `{"encoder_10": 5.388591783271399e-07, "encoder_11_12": 2.0528000059646e-06, "temporal_head": 2.0528000059645997e-05}` | 703266 |
| 11 | `{"encoder_10": 2.891280188878229e-07, "encoder_11_12": 1.2713832064634127e-06, "temporal_head": 1.2713832064634126e-05}` | 703266 |
| 12 | `{"encoder_10": 2e-07, "encoder_11_12": 1.0000000000000002e-06, "temporal_head": 1e-05}` | 703266 |

</details>

<a id="history-a22"></a>
<details>
<summary>A22. MobileNet progressive block10: seed 43 (11 recorded epochs; selected epoch 8)</summary>

Original source: `MentorProgressiveFT/outputs/block10_seed43_20261105_220831_274817/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.636889 | 63.03 | 62.75 | 62.75 | 0.630417 | 70.24 | 66.18 | 66.67 | `[[82, 68], [32, 118]]` | 0 | 5.939 |
| 2 | temporal_head | 0.574412 | 70.27 | 70.25 | 70.25 | 0.614460 | 63.04 | 67.67 | 68.33 | `[[124, 26], [69, 81]]` | 0 | 0.081 |
| 3 | encoder_11_12 | 0.540383 | 71.61 | 72.24 | 72.25 | 0.591424 | 70.20 | 70.00 | 70.00 | `[[104, 46], [44, 106]]` | 0 | 0.083 |
| 4 | encoder_11_12 | 0.525508 | 73.29 | 72.91 | 72.92 | 0.595737 | 70.47 | 70.67 | 70.67 | `[[107, 43], [45, 105]]` | 0 | 0.080 |
| 5 | encoder_10_11_12 | 0.525133 | 73.80 | 73.67 | 73.67 | 0.592502 | 71.71 | 71.33 | 71.33 | `[[105, 45], [41, 109]]` | 0 | 0.083 |
| 6 | encoder_10_11_12 | 0.512888 | 75.12 | 74.83 | 74.83 | 0.591186 | 71.90 | 71.32 | 71.33 | `[[104, 46], [40, 110]]` | 0 | 0.083 |
| 7 | encoder_10_11_12 | 0.507439 | 76.46 | 75.37 | 75.42 | 0.592153 | 70.90 | 71.00 | 71.00 | `[[107, 43], [44, 106]]` | 0 | 0.087 |
| 8 | encoder_10_11_12 | 0.508364 | 75.29 | 75.50 | 75.50 | 0.589872 | 72.55 | 71.99 | 72.00 | `[[105, 45], [39, 111]]` | 0 | 0.086 |
| 9 | encoder_10_11_12 | 0.496163 | 75.97 | 75.75 | 75.75 | 0.590625 | 71.33 | 71.33 | 71.33 | `[[107, 43], [43, 107]]` | 0 | 0.097 |
| 10 | encoder_10_11_12 | 0.493156 | 76.85 | 76.75 | 76.75 | 0.591186 | 71.76 | 71.67 | 71.67 | `[[107, 43], [42, 108]]` | 0 | 0.102 |
| 11 | encoder_10_11_12 | 0.492427 | 76.42 | 76.08 | 76.08 | 0.590835 | 72.19 | 72.00 | 72.00 | `[[107, 43], [41, 109]]` | 0 | 0.098 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_10": 0.0, "encoder_11_12": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_10": 0.0, "encoder_11_12": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_10": 0.0, "encoder_11_12": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_10": 0.0, "encoder_11_12": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_10": 2e-06, "encoder_11_12": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 703266 |
| 6 | `{"encoder_10": 1.910871981112177e-06, "encoder_11_12": 7.75e-06, "temporal_head": 7.75e-05}` | 703266 |
| 7 | `{"encoder_10": 1.6611408216728601e-06, "encoder_11_12": 6.281416799501188e-06, "temporal_head": 6.281416799501188e-05}` | 703266 |
| 8 | `{"encoder_10": 1.3002688405606828e-06, "encoder_11_12": 4.718583200498814e-06, "temporal_head": 4.718583200498814e-05}` | 703266 |
| 9 | `{"encoder_10": 8.997311594393171e-07, "encoder_11_12": 3.2500000000000015e-06, "temporal_head": 3.250000000000001e-05}` | 703266 |
| 10 | `{"encoder_10": 5.388591783271399e-07, "encoder_11_12": 2.0528000059646e-06, "temporal_head": 2.0528000059645997e-05}` | 703266 |
| 11 | `{"encoder_10": 2.891280188878229e-07, "encoder_11_12": 1.2713832064634127e-06, "temporal_head": 1.2713832064634126e-05}` | 703266 |

</details>

<a id="history-a23"></a>
<details>
<summary>A23. MobileNet + BiGRU16 per direction: seed 42 (6 recorded epochs; selected epoch 3)</summary>

Original source: `MentorBiGRU/outputs/bigru_seed42_20261106_013426_981097/training/history.json`. Status: Completed.

Preserved in ALL_EXPERIMENTS.md (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.657115 | 63.43 | 62.30 | 62.33 | 0.625769 | 62.77 | 65.74 | 66.00 | `[[112, 38], [64, 86]]` | 0 | 6.336 |
| 2 | temporal_head | 0.593144 | 68.67 | 68.67 | 68.67 | 0.593646 | 70.59 | 69.99 | 70.00 | `[[102, 48], [42, 108]]` | 0 | 0.073 |
| 3 | encoder_tail | 0.560806 | 72.36 | 71.65 | 71.67 | 0.590633 | 70.97 | 69.97 | 70.00 | `[[100, 50], [40, 110]]` | 0 | 0.075 |
| 4 | encoder_tail | 0.542174 | 73.55 | 73.08 | 73.08 | 0.591373 | 69.93 | 69.32 | 69.33 | `[[101, 49], [43, 107]]` | 0 | 0.073 |
| 5 | encoder_tail | 0.533290 | 73.98 | 72.87 | 72.92 | 0.597694 | 69.83 | 70.33 | 70.33 | `[[108, 42], [47, 103]]` | 0 | 0.073 |
| 6 | encoder_tail | 0.541887 | 73.21 | 73.17 | 73.17 | 0.594285 | 69.90 | 68.97 | 69.00 | `[[99, 51], [42, 108]]` | 0 | 0.075 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 57090 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 57090 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 407634 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 407634 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 407634 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 407634 |

</details>

<a id="history-a24"></a>
<details>
<summary>A24. MobileNet wide crop: seed 42 (10 recorded epochs; selected epoch 7)</summary>

Original source: `MentorWideContext/outputs/wide_seed42_20261106_021409_950300/history.json`. Status: Completed.

Current source: history.json (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.631040 | 64.39 | 64.33 | 64.33 | 0.617600 | 58.17 | 64.04 | 65.00 | `[[122, 28], [77, 73]]` | 0 | 6.772 |
| 2 | temporal_head | 0.573772 | 72.52 | 71.82 | 71.83 | 0.564549 | 72.31 | 71.65 | 71.67 | `[[104, 46], [39, 111]]` | 0 | 0.150 |
| 3 | encoder_tail | 0.536555 | 73.58 | 73.25 | 73.25 | 0.561179 | 73.68 | 73.33 | 73.33 | `[[108, 42], [38, 112]]` | 0 | 0.138 |
| 4 | encoder_tail | 0.525310 | 74.71 | 74.15 | 74.17 | 0.561817 | 72.73 | 73.00 | 73.00 | `[[111, 39], [42, 108]]` | 0 | 0.134 |
| 5 | encoder_tail | 0.518087 | 75.44 | 74.46 | 74.50 | 0.567729 | 69.69 | 70.95 | 71.00 | `[[113, 37], [50, 100]]` | 0 | 0.135 |
| 6 | encoder_tail | 0.516426 | 75.21 | 75.17 | 75.17 | 0.557768 | 74.83 | 74.67 | 74.67 | `[[111, 39], [37, 113]]` | 0 | 0.138 |
| 7 | encoder_tail | 0.511331 | 75.02 | 74.75 | 74.75 | 0.555595 | 75.25 | 75.00 | 75.00 | `[[111, 39], [36, 114]]` | 0 | 0.132 |
| 8 | encoder_tail | 0.503423 | 77.29 | 77.08 | 77.08 | 0.556189 | 75.08 | 75.00 | 75.00 | `[[112, 38], [37, 113]]` | 0 | 0.132 |
| 9 | encoder_tail | 0.489563 | 76.68 | 76.33 | 76.33 | 0.556581 | 74.25 | 74.33 | 74.33 | `[[112, 38], [39, 111]]` | 0 | 0.132 |
| 10 | encoder_tail | 0.501129 | 76.61 | 76.33 | 76.33 | 0.556624 | 75.08 | 75.00 | 75.00 | `[[112, 38], [37, 113]]` | 0 | 0.133 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 409170 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 409170 |
| 7 | `{"encoder_tail": 6.281416799501188e-06, "temporal_head": 6.281416799501188e-05}` | 409170 |
| 8 | `{"encoder_tail": 4.718583200498814e-06, "temporal_head": 4.718583200498814e-05}` | 409170 |
| 9 | `{"encoder_tail": 3.2500000000000015e-06, "temporal_head": 3.250000000000001e-05}` | 409170 |
| 10 | `{"encoder_tail": 2.0528000059646e-06, "temporal_head": 2.0528000059645997e-05}` | 409170 |

</details>

<a id="history-a25"></a>
<details>
<summary>A25. MobileNet wide crop: seed 43 (9 recorded epochs; selected epoch 6)</summary>

Original source: `MentorWideContext/outputs/wide_seed43_20261106_120506_83018720261106_120506_830187/history.json`. Status: Completed.

Current source: history.json (local archive reference; not included in this contribution).

| Epoch | Stage | Train loss | Train F1 (%) | Train Macro-F1 (%) | Train acc. (%) | Val. loss | Val. F1 (%) | Val. Macro-F1 (%) | Val. acc. (%) | Val. confusion matrix | Skipped updates | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temporal_head | 0.641322 | 62.10 | 62.66 | 62.67 | 0.605573 | 71.81 | 67.84 | 68.33 | `[[84, 66], [29, 121]]` | 0 | 7.747 |
| 2 | temporal_head | 0.566785 | 71.35 | 71.08 | 71.08 | 0.599932 | 60.32 | 65.79 | 66.67 | `[[124, 26], [74, 76]]` | 0 | 0.145 |
| 3 | encoder_tail | 0.536722 | 73.74 | 74.00 | 74.00 | 0.564383 | 72.61 | 72.33 | 72.33 | `[[107, 43], [40, 110]]` | 0 | 0.142 |
| 4 | encoder_tail | 0.519201 | 75.60 | 75.25 | 75.25 | 0.565221 | 72.73 | 73.00 | 73.00 | `[[111, 39], [42, 108]]` | 0 | 0.140 |
| 5 | encoder_tail | 0.510848 | 75.67 | 75.07 | 75.08 | 0.562000 | 73.95 | 72.96 | 73.00 | `[[104, 46], [35, 115]]` | 0 | 0.143 |
| 6 | encoder_tail | 0.506521 | 76.43 | 76.25 | 76.25 | 0.560930 | 74.03 | 73.31 | 73.33 | `[[106, 44], [36, 114]]` | 0 | 0.140 |
| 7 | encoder_tail | 0.500625 | 76.67 | 75.89 | 75.92 | 0.565198 | 72.11 | 72.66 | 72.67 | `[[112, 38], [44, 106]]` | 0 | 0.141 |
| 8 | encoder_tail | 0.502297 | 76.36 | 76.42 | 76.42 | 0.561031 | 74.03 | 73.31 | 73.33 | `[[106, 44], [36, 114]]` | 0 | 0.136 |
| 9 | encoder_tail | 0.492210 | 76.88 | 76.75 | 76.75 | 0.561316 | 73.68 | 73.33 | 73.33 | `[[108, 42], [38, 112]]` | 0 | 0.138 |

Recorded optimizer schedule (list-valued rates retain original group order):

| Epoch | Learning rates | Trainable parameters |
| --- | --- | --- |
| 1 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 2 | `{"encoder_tail": 0.0, "temporal_head": 0.001}` | 58626 |
| 3 | `{"encoder_tail": 1e-05, "temporal_head": 0.0001}` | 409170 |
| 4 | `{"encoder_tail": 9.728616793536588e-06, "temporal_head": 9.728616793536588e-05}` | 409170 |
| 5 | `{"encoder_tail": 8.947199994035402e-06, "temporal_head": 8.947199994035401e-05}` | 409170 |
| 6 | `{"encoder_tail": 7.75e-06, "temporal_head": 7.75e-05}` | 409170 |
| 7 | `{"encoder_tail": 6.281416799501188e-06, "temporal_head": 6.281416799501188e-05}` | 409170 |
| 8 | `{"encoder_tail": 4.718583200498814e-06, "temporal_head": 4.718583200498814e-05}` | 409170 |
| 9 | `{"encoder_tail": 3.2500000000000015e-06, "temporal_head": 3.250000000000001e-05}` | 409170 |

</details>

## Appendix B. Recorded numerical configuration snapshots

The following compact snapshots preserve available training/evaluation settings without machine-specific absolute paths, large manifests, or repeated checkpoint bookkeeping. Model and preprocessing configs without training results are included as configuration evidence, not experiments. Full original payloads remain in the historical archive. Unknown or absent settings have not been invented.

<details>
<summary>B01. VideoMAE/outputs/finetune_20261102_023340/best_model/config.json</summary>

```json
{
  "architectures": [
    "VideoMAEForVideoClassification"
  ],
  "attention_probs_dropout_prob": 0.0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 768,
  "id2label": {
    "0": "negative",
    "1": "positive"
  },
  "image_size": 224,
  "intermediate_size": 3072,
  "label2id": {
    "negative": 0,
    "positive": 1
  },
  "layer_norm_eps": 1e-12,
  "model_type": "videomae",
  "num_attention_heads": 12,
  "num_channels": 3,
  "num_frames": 16,
  "num_hidden_layers": 12,
  "patch_size": 16,
  "qkv_bias": true,
  "tubelet_size": 2,
  "use_mean_pooling": true
}
```

</details>

<details>
<summary>B02. VideoMAE/outputs/finetune_20261102_023340/best_model/preprocessor_config.json</summary>

```json
{
  "crop_size": {
    "height": 224,
    "width": 224
  },
  "do_center_crop": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.485,
    0.456,
    0.406
  ],
  "image_processor_type": "VideoMAEImageProcessor",
  "image_std": [
    0.229,
    0.224,
    0.225
  ],
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 224
  }
}
```

</details>

<details>
<summary>B03. VideoMAE/outputs/finetune_20261102_023340/settings.json</summary>

```json
{
  "unfrozen_encoder_blocks": 2,
  "encoder_learning_rate": 1e-05,
  "head_learning_rate": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 8,
  "patience": 3,
  "batch_size": 1,
  "window_seconds": 8.0,
  "prediction_horizon": 1.0,
  "starting_validation_accuracy": 0.65,
  "train_samples": 1200,
  "validation_samples": 300,
  "seed": 42
}
```

</details>

<details>
<summary>B04. VideoMAE/outputs/finetune_20261102_190028/settings.json</summary>

```json
{
  "seed": 42,
  "train_samples": 1200,
  "validation_samples": 300,
  "window_seconds": 2.0,
  "prediction_horizon": 1.0,
  "negative_reference_window_seconds": 8.0,
  "sampling": "uniform frame indices including the last allowed frame",
  "num_frames": 16,
  "image_size": 224,
  "unfrozen_encoder_blocks": 2,
  "batch_size": 1,
  "accumulation_steps": 8,
  "encoder_learning_rate": 5e-06,
  "head_learning_rate": 5e-05,
  "weight_decay": 0.01,
  "gradient_clip_norm": 1.0,
  "scheduler": "epoch cosine decay to 10 percent of initial learning rates",
  "augmentation": "clip-consistent brightness and contrast, each 0.9 to 1.1",
  "max_epochs": 4,
  "patience": 2,
  "selection": "highest accuracy, then lowest loss on ties"
}
```

</details>

<details>
<summary>B05. VideoMAE/outputs/head_comparison_20261103_002458/settings.json</summary>

```json
{
  "seed": 42,
  "models": [
    "last",
    "mean",
    "gru"
  ],
  "selection_metric": "f1_positive",
  "decision_rule": "argmax of two logits; no threshold tuning",
  "max_epochs": 30,
  "patience": 6,
  "batch_size": 64,
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "dropout": 0.3,
  "gru_projection_size": 128,
  "gru_hidden_size": 64,
  "normalization": "per-feature mean/std fitted on training clips only"
}
```

</details>

<details>
<summary>B06. R2Plus1D/outputs/finetune_20261103_010824_486004/settings.json</summary>

```json
{
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "trainable": [
    "layer3",
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 1e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "VideoClassification(\n    crop_size=[112, 112]\n    resize_size=[128, 171]\n    mean=[0.43216, 0.394666, 0.37645]\n    std=[0.22803, 0.22145, 0.216989]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B07. R2Plus1D/outputs/finetune_20261103_103252_662277/settings.json</summary>

```json
{
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "trainable": [
    "layer3",
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 2e-06,
  "head_lr": 2e-05,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "VideoClassification(\n    crop_size=[112, 112]\n    resize_size=[128, 171]\n    mean=[0.43216, 0.394666, 0.37645]\n    std=[0.22803, 0.22145, 0.216989]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B08. R2Plus1D/outputs/finetune_20261103_182149_227723/settings.json</summary>

```json
{
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "trainable": [
    "layer3",
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 2e-06,
  "head_lr": 2e-05,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "VideoClassification(\n    crop_size=[112, 112]\n    resize_size=[128, 171]\n    mean=[0.43216, 0.394666, 0.37645]\n    std=[0.22803, 0.22145, 0.216989]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B09. R2Plus1D/outputs/finetune_20261103_202727_633455/settings.json</summary>

```json
{
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "trainable": [
    "layer3",
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 1e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "FullFramePreprocess(size=112, keep_aspect_ratio=True, center_crop=False, padding=black, interpolation=bilinear, antialias=False, mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B10. R2Plus1D/outputs/finetune_20261103_205431_199155/settings.json</summary>

```json
{
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "trainable": [
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 1e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "VideoClassification(\n    crop_size=[112, 112]\n    resize_size=[128, 171]\n    mean=[0.43216, 0.394666, 0.37645]\n    std=[0.22803, 0.22145, 0.216989]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B11. R2Plus1D/outputs/progressive_20261104_024634_830954/settings.json</summary>

```json
{
  "experiment": "progressive_unfreezing_v1",
  "fine_tuning_strategy": "progressive: fc -> layer4+fc -> layer3+layer4+fc",
  "stage_plan": [
    {
      "name": "head",
      "start_epoch": 1,
      "end_epoch": 2,
      "trainable": [
        "fc"
      ]
    },
    {
      "name": "layer4",
      "start_epoch": 3,
      "end_epoch": 4,
      "trainable": [
        "layer4",
        "fc"
      ]
    },
    {
      "name": "layer3_layer4",
      "start_epoch": 5,
      "end_epoch": 10,
      "trainable": [
        "layer3",
        "layer4",
        "fc"
      ]
    }
  ],
  "optimizer_transition": "single optimizer; preserve weights and Adam state",
  "scheduler": "same global 10-epoch cosine multiplier as temporal-jitter control",
  "early_stopping": "only in final stage; compare with final-stage best F1/loss",
  "checkpoint_eligibility": "epoch >= 3; backbone must have been fine-tuned",
  "training_sampling": "16 frames; stratified temporal jitter; fixed first/last frame",
  "validation_sampling": "reference uniform indices; unchanged",
  "cache_format": "uint8 spatially transformed frame pools; resample on every visit",
  "train_cache_limit_gib": 6.0,
  "validation_cache_limit_gib": 1.0,
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "final_trainable": [
    "layer3",
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 1e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "VideoClassification(\n    crop_size=[112, 112]\n    resize_size=[128, 171]\n    mean=[0.43216, 0.394666, 0.37645]\n    std=[0.22803, 0.22145, 0.216989]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B12. R2Plus1D/outputs/temporal_jitter_20261104_014622_825976/settings.json</summary>

```json
{
  "experiment": "temporal_jitter_v1",
  "fine_tuning_strategy": "partial; layer3/layer4/fc trainable from epoch 1",
  "training_sampling": "16 frames; stratified temporal jitter; fixed first/last frame",
  "validation_sampling": "reference uniform indices; unchanged",
  "cache_format": "uint8 spatially transformed frame pools; resample on every visit",
  "train_cache_limit_gib": 6.0,
  "validation_cache_limit_gib": 1.0,
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "trainable": [
    "layer3",
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 1e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "VideoClassification(\n    crop_size=[112, 112]\n    resize_size=[128, 171]\n    mean=[0.43216, 0.394666, 0.37645]\n    std=[0.22803, 0.22145, 0.216989]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B13. R2Plus1D/outputs/temporal_jitter_20261104_014632_431431/settings.json</summary>

```json
{
  "experiment": "temporal_jitter_v1",
  "fine_tuning_strategy": "partial; layer3/layer4/fc trainable from epoch 1",
  "training_sampling": "16 frames; stratified temporal jitter; fixed first/last frame",
  "validation_sampling": "reference uniform indices; unchanged",
  "cache_format": "uint8 spatially transformed frame pools; resample on every visit",
  "train_cache_limit_gib": 6.0,
  "validation_cache_limit_gib": 1.0,
  "model": "r2plus1d_18",
  "weights": "KINETICS400_V1",
  "seed": 42,
  "trainable": [
    "layer3",
    "layer4",
    "fc"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 1e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "VideoClassification(\n    crop_size=[112, 112]\n    resize_size=[128, 171]\n    mean=[0.43216, 0.394666, 0.37645]\n    std=[0.22803, 0.22145, 0.216989]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B14. ResNet18_GRU/outputs/resnet18_gru_20261105_121809_438598/settings.json</summary>

```json
{
  "experiment": "resnet18_gru_v1",
  "fine_tuning_strategy": "gru+classifier warm-up -> encoder.layer4+gru+classifier",
  "stage_plan": [
    {
      "name": "temporal_head",
      "start_epoch": 1,
      "end_epoch": 2,
      "trainable": [
        "gru",
        "classifier"
      ]
    },
    {
      "name": "encoder_layer4",
      "start_epoch": 3,
      "end_epoch": 10,
      "trainable": [
        "encoder.layer4",
        "gru",
        "classifier"
      ]
    }
  ],
  "optimizer_transition": "single optimizer; preserve weights and Adam state",
  "scheduler": "2 head warm-up epochs, then cosine decay over epochs 3-10",
  "early_stopping": "only in final stage; compare with final-stage best F1/loss",
  "checkpoint_eligibility": "epoch >= 3; backbone must have been fine-tuned",
  "training_sampling": "16 frames; stratified temporal jitter; fixed first/last frame",
  "validation_sampling": "reference uniform indices; unchanged",
  "cache_format": "uint8 spatially transformed frame pools; resample on every visit",
  "train_cache_limit_gib": 12.0,
  "validation_cache_limit_gib": 1.0,
  "model": "resnet18_gru",
  "gru_hidden_size": 64,
  "gru_layers": 1,
  "temporal_readout": "final hidden state",
  "gru_precision": "float32; gradient preserved through encoder feature cast",
  "image_size": 224,
  "spatial_antialias": true,
  "head_warmup_lr": 0.001,
  "weights": "IMAGENET1K_V1",
  "seed": 42,
  "final_trainable": [
    "encoder.layer4",
    "gru",
    "classifier"
  ],
  "batchnorm_running_stats": "frozen",
  "dropout": 0.3,
  "batch_size": 2,
  "accumulation": 4,
  "backbone_lr": 1e-05,
  "head_lr": 0.0001,
  "weight_decay": 0.01,
  "max_epochs": 10,
  "patience": 3,
  "selection": "highest binary_f1; tie: lowest validation loss",
  "decision": "argmax; positive class = 1",
  "frames": 16,
  "window_seconds": 2.0,
  "prediction_horizon_seconds": 1.0,
  "preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "augmentation": "clip-consistent brightness and contrast, 0.9 to 1.1",
  "cache_in_ram": true
}
```

</details>

<details>
<summary>B15. Mentor4Frames/outputs/four_frames_20261105_145438_341658/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "uniform_whole_video",
  "test_sampling": "uniform_whole_video_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 42,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie: min validation loss; eligible epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634
}
```

</details>

<details>
<summary>B16. Mentor4Frames/outputs/four_frames_last15_20261105_181840_706976/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 42,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie: min validation loss; eligible epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634
}
```

</details>

<details>
<summary>B17. MentorMultiHorizon/outputs/multihorizon_20261105_202422_856567/settings.json</summary>

```json
{
  "experiment": "multihorizon_train",
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "training_positive_horizons": [
    0.5,
    1.0,
    1.5
  ],
  "frame_spacing_seconds": 0.5,
  "horizon_selection": "uniform independent choice per positive video per epoch; local RNG",
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 42,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max mean of 3 validation binary_f1 scores; tie: min mean loss; epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_unique_count": 300,
  "validation_horizons": [
    0.5,
    1.0,
    1.5
  ],
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634
}
```

</details>

<details>
<summary>B18. MentorSeedStability/outputs/seed_study_20261105_211409_690778/seed_43/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 43,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie: min validation loss; eligible epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634,
  "experiment": "fixed_window_seed_stability"
}
```

</details>

<details>
<summary>B19. MentorSeedStability/outputs/seed_study_20261105_211409_690778/seed_44/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 44,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie: min validation loss; eligible epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634,
  "experiment": "fixed_window_seed_stability"
}
```

</details>

<details>
<summary>B20. MentorSeedStability/outputs/seed_study_20261105_211409_690778/study_settings.json</summary>

```json
{
  "new_seeds": [
    43,
    44
  ],
  "existing_seed": 42,
  "purpose": "Estimate training seed variability on the same 300 validation videos",
  "selection": "Per run: validation Binary-F1, tie lower loss, epoch >=3; no best-seed selection",
  "standard_deviation": "sample SD, ddof=1; not a confidence interval"
}
```

</details>

<details>
<summary>B21. MentorProgressiveFT/outputs/block10_20261105_214536_024567/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 42,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.10",
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie min loss; final stage epoch >=5",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634,
  "experiment": "progressive_block10",
  "stages": [
    {
      "start_epoch": 1,
      "name": "temporal_head",
      "prefixes": [
        "gru.",
        "classifier."
      ]
    },
    {
      "start_epoch": 3,
      "name": "encoder_11_12",
      "prefixes": [
        "encoder.11.",
        "encoder.12.",
        "gru.",
        "classifier."
      ]
    },
    {
      "start_epoch": 5,
      "name": "encoder_10_11_12",
      "prefixes": [
        "encoder.10.",
        "encoder.11.",
        "encoder.12.",
        "gru.",
        "classifier."
      ]
    }
  ],
  "new_block_lr": 2e-06,
  "early_stopping": "fresh final-stage counter from epoch 5",
  "final_trainable_parameters": 703266
}
```

</details>

<details>
<summary>B22. MentorProgressiveFT/outputs/block10_seed43_20261105_220831_274817/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 43,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.10",
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie min loss; final stage epoch >=5",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634,
  "experiment": "progressive_block10",
  "stages": [
    {
      "start_epoch": 1,
      "name": "temporal_head",
      "prefixes": [
        "gru.",
        "classifier."
      ]
    },
    {
      "start_epoch": 3,
      "name": "encoder_11_12",
      "prefixes": [
        "encoder.11.",
        "encoder.12.",
        "gru.",
        "classifier."
      ]
    },
    {
      "start_epoch": 5,
      "name": "encoder_10_11_12",
      "prefixes": [
        "encoder.10.",
        "encoder.11.",
        "encoder.12.",
        "gru.",
        "classifier."
      ]
    }
  ],
  "new_block_lr": 2e-06,
  "early_stopping": "fresh final-stage counter from epoch 5",
  "final_trainable_parameters": 703266
}
```

</details>

<details>
<summary>B23. MentorBiGRU/outputs/bigru_seed42_20261106_013426_981097/training/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_BiGRU16",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)",
  "seed": 42,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie: min validation loss; eligible epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 984098,
  "experiment": "mobilenet_bigru16",
  "gru_hidden_size_per_direction": 16,
  "bidirectional": true,
  "fine_tune_trainable_parameters": 407634
}
```

</details>

<details>
<summary>B24. MentorWideContext/outputs/wide_seed42_20261106_021409_950300/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "resize shorter edge 256, bilinear antialias; center crop 224x384; ImageNet normalization",
  "seed": 42,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie: min validation loss; eligible epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634,
  "experiment": "wide_context_crop_width_only",
  "crop_height": 224,
  "crop_width": 384,
  "resize_short_edge": 256,
  "single_changed_factor": "center crop width: 224 -> 384",
  "evaluation_scope": "fixed development validation only; no test evaluation",
  "initialization": "fresh cached ImageNet weights; never warm-start BestModel"
}
```

</details>

<details>
<summary>B25. MentorWideContext/outputs/wide_seed43_20261106_120506_83018720261106_120506_830187/settings.json</summary>

```json
{
  "model": "MobileNetV3Small_features_GRU32",
  "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
  "precision": "cuda_fp16_gru_fp32",
  "amp_init_scale": 1024.0,
  "frames": 4,
  "positive_offsets_seconds": [
    2,
    1.5,
    1,
    0.5
  ],
  "negative_sampling": "last_1p5_seconds",
  "test_sampling": "last_1p5_seconds_both_classes",
  "spatial_preprocessing": "resize shorter edge 256, bilinear antialias; center crop 224x384; ImageNet normalization",
  "seed": 43,
  "batch_size": 8,
  "max_epochs": 12,
  "patience": 3,
  "warmup_epochs": 2,
  "fine_tuned_modules": [
    "encoder.11",
    "encoder.12"
  ],
  "selection": "max validation binary_f1; tie: min validation loss; eligible epoch >= 3",
  "decision": "argmax, no threshold tuning",
  "train_count": 1200,
  "validation_count": 300,
  "weight_decay": 0.01,
  "dropout": 0.4,
  "backbone_lr": 1e-05,
  "warmup_head_lr": 0.001,
  "fine_tune_head_lr": 0.0001,
  "total_parameters": 985634,
  "experiment": "wide_context_crop_width_only",
  "crop_height": 224,
  "crop_width": 384,
  "resize_short_edge": 256,
  "single_changed_factor": "center crop width: 224 -> 384",
  "evaluation_scope": "fixed development validation only; no test evaluation",
  "initialization": "fresh cached ImageNet weights; never warm-start BestModel"
}
```

</details>

The wide-run setting `evaluation_scope: fixed development validation only; no test evaluation` records the original training plan. The separately authorized public evaluation occurred later and is documented in Section 8; it does not change the saved training configuration.

## Appendix C. Integrity anchors for this report

| Source file | SHA256 of bytes read for this report |
| --- | --- |
| `ALL_EXPERIMENTS.md` | `1d26ce003d55de14793fe7e2df85222468cea66bd98d97dea6363e9519ac2bb1` |
| `src/video_split.csv` | `bfe763d95efb4932c33375f69066345c66069d32cdafe5b74edf42ec0982ccde` |
| `BestModel/current.json` | `9fb1ad6379e003e881b721cd92b602d9323a6205dd2d70f0adc7cc2d3e0b95bd` |
| `BestModel/versions/v_20261105_222803_896915/result.json` | `0d367317403ecbc78dec6bd7b3814a0998c71fc3e123a8ed7162238b5641834c` |
| `BestModel/versions/v_20261105_222803_896915/history.json` | `46e8964417a535d0a1e8cfde40c2967fb906215bf5016b89783374e33c52c5ea` |
| `BestModel/versions/v_20261105_222803_896915/test-public_metrics.json` | `69dc0b83e341da6421fdc20ce7b8072dbf5e2ffe3348ab53fde586b2154c65b0` |
| `MentorWideContext/outputs/wide_seed42_20261106_021409_950300/result.json` | `0aa3921caeaefedd73031ff2091701f333219410ab5b58d2729e18ed7e657f61` |
| `MentorWideContext/outputs/wide_seed42_20261106_021409_950300/history.json` | `19388c3ba38d8be497247a89e580f801872f8d4bf64d24677312503769ebb828` |
| `MentorWideContext/outputs/wide_seed42_20261106_021409_950300/test-public_metrics.json` | `472421d86f36a5107cbfe89cb32991abf1d49c33f0f811160c7f54db15eb7211` |
| `MentorWideContext/outputs/wide_seed43_20261106_120506_83018720261106_120506_830187/result.json` | `312c26c6f3fa3263c838d62d4293a88b9169eb3c21cf8d1a77c21e4401b4d096` |
| `MentorWideContext/outputs/wide_seed43_20261106_120506_83018720261106_120506_830187/history.json` | `f8d48838f37310d59fa48743240bc6c9ba4b42d6c6fce985c5f607412680c8db` |
| `advanced_tasks/MobileNetV3_GRU_Wide.ipynb` | `2a81e91be5e22fb643832da58801529afcb8c66f1f05ec1b5965ce077c83f3c3` |

Coverage checks: **71 archived JSON records parsed**, **23 archived histories plus 2 wide histories**, and **193 recorded epochs** retained in Appendix A. Baseline summaries without complete histories remain in the main results table. Confusion-matrix-derived Binary-F1, Macro-F1, and accuracy were checked against stored metrics wherever both were available.

This documentation task wrote only `docs/experiment-report.md`. It did not edit code, notebooks, model weights, split files, or prior reports, and did not run training or test-set inference.
