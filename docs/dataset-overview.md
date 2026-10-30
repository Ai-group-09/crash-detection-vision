# Dataset overview

## Source
This project uses the Nexar Collision Prediction Dataset:
https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction

The positive class includes both collisions and near-miss situations.
The negative class contains regular driving videos.


## Dataset Structure
| Split | Positive | Negative | Total |
|---|---:|---:|---:|
| Train | 750 | 750 | 1500 |
| Test public | 334 | 333 | 667 |
| Test private | 338 | 339 | 677 |
Each class directory contains MP4 videos and a metadata.csv file and,
The file_name column connects each metadata row to its video.


## Metadata
The metadata includes event and alert times, light conditions,
weather, scene, and time_to_accident.

In the training set, time_of_event and time_of_alert are available
for positive videos and empty for negative videos.
The time_to_accident field is empty for the training set.


## Checking
1. Video filenames matched the metadata in all six class directories.
2. No duplicate metadata filenames or zero-byte MP4 files were found.
3. Two training records have missing weather values.
4. Video decoding and frame rates have not yet been checked for every file.


## Sampling decisions
1. Use fixed-interval sampling, initially at 2 frames per second.
2. Apply the same sampling method to positive and negative videos.
3. Keep frames in chronological order and preserve the aspect ratio.
4. Use event time only for analysis, not for selecting model input frames.
5. Keep all frames from one video in the same data split.
