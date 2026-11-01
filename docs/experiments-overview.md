# Experiments overview
This file describes our sampling and baseline experiments.

## Sampling: 10 videos
We tested five positive and five negative videos. We saved the frames as
JPEG files at their original size.

| Sampling rate | Total frames | Average frames per video | Average time per video | Total frame size |
|---|---:|---:|---:|---:|
| 1 FPS | 384 | 38.4 | 1.59 s | 67.87 MiB |
| 2 FPS | 760 | 76.0 | 2.59 s | 134.22 MiB |
| 4 FPS | 1515 | 151.5 | 2.03 s | 267.70 MiB |

The processing times are from earlier runs and can change on another run.
These results do not tell us which rate is best for the model.

## Sampling: 50 videos
We tested an earlier version of the sampling code on 25 positive and
25 negative videos at 2 FPS.

- Total frames: 3,817
- Average frames per video: 76.3
- Total frame size: 753.49 MiB
- Average processing time per video: 2.53 seconds

## Sampling: 100 videos
We sampled 50 positive and 50 negative training videos at 2 FPS.

- Completed videos: 100
- Failed videos: 0
- Total frames: 7,572
- Total frame size: 1,510.26 MiB

We selected videos from the start of the sorted lists, so this sample may
not represent the full dataset.

## Baseline model
We use a pretrained ResNet18 to extract 512 features from each frame.
We average the frame features to make one feature vector for each video.
We then train a linear classifier to predict whether a video is positive
or negative.

- ResNet18: pretrained on ImageNet and frozen
- Input: 224 x 224, with padding to preserve the aspect ratio
- Optimizer: Adam
- Learning rate: 0.001
- Loss: binary cross-entropy with logits
- Initial epochs: 20
- Initial decision threshold: 0.5
- Selected model: 100 epochs and a decision threshold of 0.45

### Initial check
The first tests used 8 training videos and 2 validation videos. We used
them to check the code. This set is too small to judge model performance.

### 100-video experiment
The existing video split placed the 100 sampled videos in these sets:

- Training: 85 videos, including 42 positive and 43 negative
- Validation: 15 videos, including 8 positive and 7 negative

| Validation result | Count |
|---|---:|
| Correct positive predictions | 6 |
| Correct negative predictions | 4 |
| False positives | 3 |
| False negatives | 2 |

- Validation accuracy: 66.7% (10 out of 15 videos)
- Precision: 66.7%
- Recall: 75%

The validation set is small, so these are only initial results. Averaging
frame features does not use the order of the frames.

### 500-video experiment

We read videos directly and sampled frames at 2 FPS. We used the existing
split: 400 training videos and 100 validation videos. Each set had the same
number of positive and negative videos.

| Validation result | Count |
|---|---:|
| Correct positive predictions | 22 |
| Correct negative predictions | 33 |
| False positives | 17 |
| False negatives | 28 |

- Validation accuracy: 55% (55 out of 100 videos)
- Precision: 56.4% (22 out of 39 predicted positive videos)
- Recall: 44% (22 out of 50 positive videos)
- Training loss after 20 epochs: 0.6303

The 100-video experiment used saved JPEG frames and a different set of
videos. We cannot compare the two results directly. These results are
from validation.

### 1500-video experiment

We read videos directly at 2 FPS. We used 1,200 training videos and 300
validation videos from the existing split. Each set had the same number of
positive and negative videos.

| Validation result | Count |
|---|---:|
| Correct positive predictions | 88 |
| Correct negative predictions | 104 |
| False positives | 46 |
| False negatives | 62 |

- Validation accuracy: 64% (192 out of 300 videos)
- Precision: 65.7% (88 out of 134 predicted positive videos)
- Recall: 58.7% (88 out of 150 positive videos)
- Training loss after 20 epochs: 0.6368

The 500- and 1500-video experiments used different validation sets, so
we cannot directly compare their scores.

### 1500 videos: 100 training epochs

We trained the same classifier for 80 more epochs. We used the same
features and validation videos as the 20-epoch experiment.

| Validation result | Count |
|---|---:|
| Correct positive predictions | 95 |
| Correct negative predictions | 105 |
| False positives | 45 |
| False negatives | 55 |

- Validation accuracy: 66.7% (200 out of 300 videos)
- Precision: 67.9% (95 out of 140 predicted positive videos)
- Recall: 63.3% (95 out of 150 positive videos)

These are validation results, not test results.

### 1500 videos: threshold 0.45

We used the same 100-epoch model and validation predictions. We tested a
threshold of 0.45 instead of 0.5 without training the model again.

| Validation result | Count |
|---|---:|
| Correct positive predictions | 108 |
| Correct negative predictions | 99 |
| False positives | 51 |
| False negatives | 42 |

- Validation accuracy: 69% (207 out of 300 videos)
- Precision: 67.9% (108 out of 159 predicted positive videos)
- Recall: 72% (108 out of 150 positive videos)

We chose the threshold using the validation results. The public test
results are listed below.

### 1500 videos: 200 training epochs

We trained a copy of the 100-epoch classifier for 100 more epochs. We
checked it on the same validation videos with a threshold of 0.45.

| Validation result | Count |
|---|---:|
| Correct positive predictions | 107 |
| Correct negative predictions | 100 |
| False positives | 50 |
| False negatives | 43 |

- Validation accuracy: 69% (207 out of 300 videos)
- Precision: 68.2% (107 out of 157 predicted positive videos)
- Recall: 71.3% (107 out of 150 positive videos)

The validation accuracy is the same as the 100-epoch result at a threshold
of 0.45.

### 1500 videos: classifier with a hidden layer

We tested a classifier with a hidden layer of 64 units and ReLU. We trained
it for 100 epochs on the same features and used a threshold of 0.45.

| Validation result | Count |
|---|---:|
| Correct positive predictions | 103 |
| Correct negative predictions | 100 |
| False positives | 50 |
| False negatives | 47 |

- Validation accuracy: 67.7% (203 out of 300 videos)
- Precision: 67.3% (103 out of 153 predicted positive videos)
- Recall: 68.7% (103 out of 150 positive videos)

We kept the linear classifier for the baseline.

### 1500 videos: normalized features

We normalized the 512 features using the mean and standard deviation from
the training set. We trained a new linear classifier for 100 epochs and
used the same normalization for validation. The threshold was 0.45.

| Validation result | Count |
|---|---:|
| Correct positive predictions | 103 |
| Correct negative predictions | 102 |
| False positives | 48 |
| False negatives | 47 |

- Validation accuracy: 68.3% (205 out of 300 videos)
- Precision: 68.2% (103 out of 151 predicted positive videos)
- Recall: 68.7% (103 out of 150 positive videos)

The linear classifier without normalization had 69% validation accuracy
and 72% recall on the same validation set.


### Public test results

We evaluated the 100-epoch linear classifier on the public test set
using a threshold of 0.45.

- Test videos: 667
- Accuracy: 65.8% (439 out of 667)
- Recall: 66.8% (223 out of 334 positive videos)
- Correct positive predictions: 223
- Correct negative predictions: 216
- False positives: 117
- False negatives: 111

We have not evaluated the private test set yet.
