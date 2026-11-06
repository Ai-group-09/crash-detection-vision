"""Four-frame mentor experiment: sampling, small temporal model, and epoch runner."""
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import functional as TF
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
OFFSETS = np.array([2.0, 1.5, 1.0, 0.5])


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def select_indices(frame_count, fps, event_time=None):
    """
    None means last_1p5_seconds sampling, regardless of the class.
    Event targets are rounded DOWN to a real frame, never toward the event.
    Missing/invalid timestamps fail explicitly; no clipping or dropped videos.
    """
    if frame_count < 4 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video: {frame_count=}, {fps=}")
    if event_time is None:
        # Four frames ending at the last available frame.
        offsets = np.array([1.5, 1.0, 0.5, 0.0])
        last_index = frame_count - 1
        end_seconds = last_index / fps
        requested = end_seconds - offsets
        if requested[0] < 0:
            raise ValueError("Video is too short for a 1.5-second window")
        # Round toward earlier frames; keep the final frame exact.
        indices = last_index - np.ceil(offsets * fps).astype(np.int64)
    else:
        event_time = float(event_time)
        requested = event_time - OFFSETS
        if not math.isfinite(event_time) or requested[0] < 0:
            raise ValueError(f"Not enough pre-event history: {event_time=}")
        indices = np.floor(requested * fps).astype(np.int64)
        if indices[-1] >= frame_count:
            raise ValueError("Requested pre-event frames fall outside the video")
    if indices[0] < 0 or not np.all(np.diff(indices) > 0):
        raise ValueError("Four distinct, chronological frames are required")
    return indices.tolist(), requested.tolist()


def make_record(path, video_id, label, split, event_time=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    indices, requested = select_indices(count, fps, event_time)
    return dict(video_id=video_id, label=int(label), split=split, path=str(path),
                fps=fps, frame_count=count, indices=indices,
                requested_seconds=requested, actual_seconds=[i / fps for i in indices],
                sampling="event_offsets" if event_time is not None else "last_1p5_seconds",
                event_time=float(event_time) if event_time is not None else None)


def build_train_manifest(dataset_root, split_path):
    root = Path(dataset_root)
    with Path(split_path).open(encoding="utf-8-sig", newline="") as f:
        split_rows = list(csv.DictReader(f))
    counts = Counter((r["split"], int(r["label"])) for r in split_rows)
    expected = {("train", 0): 600, ("train", 1): 600,
                ("validation", 0): 150, ("validation", 1): 150}
    if counts != expected or len({r["video_id"] for r in split_rows}) != 1500:
        raise ValueError(f"Unexpected fixed split: {counts}")
    with (root / "train/positive/metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        events = {Path(r["file_name"]).stem: float(r["time_of_event"]) for r in csv.DictReader(f)}
    records = []
    for row in split_rows:
        vid, y = row["video_id"], int(row["label"])
        path = root / "train" / ("positive" if y else "negative") / f"{vid}.mp4"
        records.append(make_record(path, vid, y, row["split"], events[vid] if y else None))
        if len(records) % 300 == 0:
            print(f"Checked sampling: {len(records)}/1500 videos", flush=True)
    return records


def build_test_manifest(dataset_root, split):
    """Both classes use the SAME sampler. Never read test event/alert/TTA metadata.

    Folder labels are retained only as targets for scoring AFTER predictions.
    """
    if split not in {"test-public", "test-private"}:
        raise ValueError(split)
    records = []
    for folder, label in [("negative", 0), ("positive", 1)]:
        for path in sorted((Path(dataset_root) / split / folder).glob("*.mp4")):
            records.append(make_record(path, path.stem, label, split, event_time=None))
    expected = {0: 333, 1: 334} if split == "test-public" else {0: 339, 1: 338}
    if Counter(r["label"] for r in records) != expected:
        raise ValueError(f"Incomplete {split}")
    if len({r["video_id"] for r in records}) != len(records):
        raise ValueError(f"Duplicate IDs in {split}")
    return sorted(records, key=lambda r: r["video_id"])


def read_frames(record, transform):
    """Decode exactly the four recorded frames; spatial cache remains uint8."""
    cap = cv2.VideoCapture(record["path"])
    frames = []
    try:
        for index in record["indices"]:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError(f"Seek failed: {record['path']} frame {index}")
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Decode failed: {record['path']} frame {index}")
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > 0.5:
                raise RuntimeError(f"Unexpected decoder position in {record['path']}")
            rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            rgb = TF.resize(rgb, transform.resize_size, interpolation=transform.interpolation,
                            antialias=transform.antialias)
            frames.append(TF.center_crop(rgb, transform.crop_size))
    finally:
        cap.release()
    return torch.stack(frames)  # T,C,H,W


class FourFrameDataset(Dataset):
    def __init__(self, records, training=False, cache=True):
        self.records = records
        self.training = training
        self.use_cache = cache
        self.cache = {}
        self.transform = WEIGHTS.transforms()
        self.transform.crop_size = [224, 384]

    def __len__(self):
        return len(self.records)

    def pixels(self, index):
        if index not in self.cache:
            pixels = read_frames(self.records[index], self.transform)
            if self.use_cache:
                self.cache[index] = pixels
            return pixels
        return self.cache[index]

    def __getitem__(self, index):
        # convert_image_dtype allocates a float tensor; augmentation never alters cache.
        x = TF.convert_image_dtype(self.pixels(index), torch.float32)
        if self.training:
            x = TF.adjust_brightness(x, random.uniform(0.9, 1.1))
            x = TF.adjust_contrast(x, random.uniform(0.9, 1.1))
        x = TF.normalize(x, self.transform.mean, self.transform.std)
        return x, self.records[index]["label"]  # T,C,H,W, exactly four frames


class SmallTemporalModel(nn.Module):
    """One small pretrained image encoder, GRU(32), dropout, linear(2)."""
    def __init__(self, weights=WEIGHTS):
        super().__init__()
        self.encoder = mobilenet_v3_small(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gru = nn.GRU(576, 32, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(32, 2))

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:3] != (4, 3):
            raise ValueError("Expected B,4,3,H,W")
        if tuple(x.shape[-2:]) != (224, 384):
            raise ValueError("This archived model requires 224x384 inputs")
        b, t, c, h, w = x.shape
        features = self.pool(self.encoder(x.reshape(b * t, c, h, w))).flatten(1)
        features = features.reshape(b, t, 576)
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, hidden = self.gru(features.float())
        return self.classifier(hidden[-1])


def set_stage(model, fine_tune):
    # Last inverted-residual block and final 1x1 convolution, not the entire CNN.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("gru.", "classifier.")) or
                                 (fine_tune and name.startswith(("encoder.11.", "encoder.12."))))
        if not parameter.requires_grad:
            parameter.grad = None


def training_mode(model):
    model.train()
    # Fixed running statistics in both stages, even in trainable CNN blocks.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def make_optimizer(model):
    return torch.optim.AdamW([
        {"params": list(model.encoder[11:].parameters()), "lr": 1e-5, "name": "encoder_tail"},
        {"params": list(model.gru.parameters()) + list(model.classifier.parameters()),
         "lr": 1e-3, "name": "temporal_head"}], weight_decay=0.01)


def epoch_lrs(optimizer, epoch, max_epochs=12):
    if epoch <= 2:
        values = (0.0, 1e-3)
    else:
        fraction = (epoch - 3) / max(1, max_epochs - 3)
        multiplier = 0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2
        values = (1e-5 * multiplier, 1e-4 * multiplier)
    for group, value in zip(optimizer.param_groups, values):
        group["lr"] = value
    return {g["name"]: g["lr"] for g in optimizer.param_groups}


def metrics_from_predictions(labels, predicted):
    matrix = np.zeros((2, 2), dtype=np.int64)
    for y, p in zip(labels, predicted):
        matrix[int(y), int(p)] += 1
    tn, fp, fn, tp = matrix.ravel()
    pos = 2 * tp / max(1, 2 * tp + fp + fn)
    neg = 2 * tn / max(1, 2 * tn + fp + fn)
    return dict(accuracy=float((tn + tp) / max(1, matrix.sum())), binary_f1=float(pos),
                macro_f1=float((pos + neg) / 2), confusion_matrix=matrix.tolist())


def optimizer_step(model, optimizer, scaler):
    """Clip finite unscaled gradients; let GradScaler safely skip AMP overflow."""
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        raise RuntimeError("No gradients were produced")
    finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().item()
    if not finite:
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite gradients without AMP scaling")
        previous_scale = scaler.get_scale()
        # unscale_ recorded inf/nan. step skips the optimizer entirely; update
        # reduces the scale for the next batch. Do NOT clip non-finite gradients.
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            raise RuntimeError("GradScaler did not reduce its scale after overflow")
        optimizer.zero_grad(set_to_none=True)
        return False
    nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    training_mode(model) if training else model.eval()
    labels, predictions, probabilities = [], [], []
    loss_sum = 0.0
    seen = 0
    skipped_updates = 0
    consecutive_overflows = 0
    for batch_index, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                scaler.scale(loss).backward()
                updated = optimizer_step(model, optimizer, scaler)
                if updated:
                    consecutive_overflows = 0
                else:
                    skipped_updates += 1
                    consecutive_overflows += 1
                    print(f"  AMP overflow at batch {batch_index}: update skipped; "
                          f"new scale={scaler.get_scale():g}", flush=True)
                    if consecutive_overflows >= 3:
                        raise FloatingPointError("Three consecutive AMP overflows; stop for diagnosis")
        loss_sum += loss.item() * len(y)
        seen += len(y)
        labels.extend(y.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
        probabilities.extend(logits.detach().float().softmax(1)[:, 1].cpu().tolist())
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"  {'train' if training else 'evaluation'}: {batch_index}/{len(loader)} batches"
                  f" | {seen}/{len(loader.dataset)} videos", flush=True)
    if seen != len(loader.dataset):
        raise RuntimeError("Not all videos were processed")
    metrics = metrics_from_predictions(labels, predictions)
    metrics["loss"] = loss_sum / seen
    if training:
        metrics["skipped_optimizer_updates"] = skipped_updates
    return metrics, dict(labels=labels, predicted=predictions, probability_positive=probabilities)


def prediction_rows(records, result):
    if len(records) != len(result["predicted"]):
        raise ValueError("Prediction length mismatch")
    return [dict(video_id=r["video_id"], label=y, prediction=p, probability_positive=prob)
            for r, y, p, prob in zip(records, result["labels"], result["predicted"],
                                     result["probability_positive"])]


def smoke_test(model, batch, device):
    """Two technical optimizer steps, then restore ALL weights and RNG seed."""
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.0)
    x, y = (tensor.to(device) for tensor in batch)
    try:
        for step, epoch in enumerate((1, 3), 1):
            set_stage(model, fine_tune=epoch >= 3)
            epoch_lrs(optimizer, epoch)
            training_mode(model)
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            assert logits.shape == (len(y), 2) and torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                else:
                    assert p.grad is None, name
            active = ["gru", "classifier"] + (["encoder.11", "encoder.12"] if epoch >= 3 else [])
            for prefix in active:
                assert any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.get_submodule(prefix).parameters()), prefix
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    assert torch.equal(before[name], p.detach().cpu()), name
            for name, buffer in model.named_buffers():
                assert torch.equal(before[name], buffer.detach().cpu()), name
            for prefix in active:
                assert any(not torch.equal(before[name], p.detach().cpu())
                           for name, p in model.named_parameters() if name.startswith(prefix + ".")), prefix
            assert int(optimizer.state[model.classifier[1].weight]["step"]) == step
            print(f"Smoke {'fine-tuning' if epoch >= 3 else 'warm-up'} passed", flush=True)
    finally:
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        set_stage(model, False)
        seed_all()
    return True


# Combined with the immutable local model/sampler code when a version is registered.
import copy
import hashlib
import time
from torch.utils.data import DataLoader

def train_saved_model(seed, train_ds, val_ds, reference_settings, output_dir, device):
    """Fresh ImageNet initialization and fresh optimizer for each seed.

    Dataset caches are reused as uint8 pixels only. RNG is reset after model
    initialization exactly as in the seed-42 notebook before its training loop.
    """
    if reference_settings.get('crop_height') != 224 or reference_settings.get('crop_width') != 384:
        raise ValueError('Reproduction requires the archived 224x384 settings')
    for dataset in (train_ds, val_ds):
        if list(dataset.transform.crop_size) != [224, 384] or list(dataset.transform.resize_size) != [256]:
            raise ValueError('Dataset preprocessing differs from the archived wide protocol')
    if seed != reference_settings['seed']:
        raise ValueError('Reproduction must use the archived seed')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(reference_settings)
    config['seed'] = int(seed)
    config['experiment'] = 'best_model_reproduction'
    save_json(out / 'settings.json', config)
    batch_size, max_epochs, patience = config['batch_size'], config['max_epochs'], config['patience']
    seed_all(seed)
    model = SmallTemporalModel().to(device)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda', drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda', drop_last=False)
    seed_all(seed)
    history, best_score, bad_epochs = [], None, 0
    print(f'\nSeed {seed} | Output directory: {out}', flush=True)
    try:
        for epoch in range(1, max_epochs + 1):
            fine_tune = epoch >= 3
            set_stage(model, fine_tune)
            lrs = epoch_lrs(optimizer, epoch, max_epochs)
            stage = 'encoder_tail' if fine_tune else 'temporal_head'
            print(f'\nSeed {seed} | Epoch {epoch}/{max_epochs} | {stage} | LRs={lrs}', flush=True)
            start = time.perf_counter()
            train_metrics, _ = run_epoch(model, train_loader, device, optimizer, scaler)
            val_metrics, predictions = run_epoch(model, val_loader, device)
            row = dict(epoch=epoch, stage=stage, lrs=lrs,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       train=train_metrics, validation=val_metrics, minutes=(time.perf_counter()-start)/60)
            history.append(row)
            score = (val_metrics['binary_f1'], -val_metrics['loss'])
            if fine_tune:
                if best_score is None or score > best_score:
                    best_score, bad_epochs = score, 0
                    best_epoch, best_metrics = epoch, val_metrics.copy()
                    torch.save(dict(model=model.state_dict(), epoch=epoch, metrics=best_metrics,
                                    settings=config), out / 'best_model.pt')
                    save_json(out / 'validation_predictions.json', prediction_rows(val_ds.records, predictions))
                    print('Best fine-tuned model saved.', flush=True)
                else:
                    bad_epochs += 1
            else:
                print('Warm-up only; selection starts at epoch 3.', flush=True)
            save_json(out / 'history.json', history)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch, settings=config), out / 'last_model.pt')
            print(f"Train F1={train_metrics['binary_f1']:.2%} | Val Binary-F1={val_metrics['binary_f1']:.2%} | "
                  f"Macro-F1={val_metrics['macro_f1']:.2%} | Accuracy={val_metrics['accuracy']:.2%} | "
                  f"Val loss={val_metrics['loss']:.4f} | {row['minutes']:.1f} min", flush=True)
            if fine_tune and bad_epochs >= patience:
                print('Early stopping in fine-tuning stage.', flush=True)
                break
        result = dict(seed=int(seed), best_epoch=best_epoch, validation=best_metrics,
                      source_run=str(out.resolve()),
                      checkpoint_sha256=hashlib.sha256((out/'best_model.pt').read_bytes()).hexdigest())
        save_json(out / 'result.json', result)
        print('Completed seed:', seed, json.dumps(best_metrics, indent=2), flush=True)
        return result
    finally:
        del model, optimizer, scaler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
