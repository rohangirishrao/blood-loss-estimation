import argparse
import os

tmp_dir = "/home/r.rohangirish/mt_ble/tmp"
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir

import glob
import random
import numpy as np
import pandas as pd
from rich.console import Console
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.models.video import r3d_18, R3D_18_Weights
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import time
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from collections import defaultdict


class VideoBleedingDetector(nn.Module):
    def __init__(self, num_classes=2, severity_levels=4):
        super().__init__()
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/pytorchvideo", "r3d_18", pretrained=True
            )
            in_features = 512
            print("Using facebookresearch r3d_18 model")
        except:
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)
        self.volume_regressor = nn.Linear(in_features, 1)

    def forward(self, x):  # x: [B, C, F, H, W] - batch of clips
        # Process batch of clips efficiently
        features = self.backbone(x)  # [B, 512]
        clip_pred = self.classifier(features)  # [B, num_classes]
        delta_volume_pred = self.volume_regressor(features).squeeze(-1)  # [B]
        return clip_pred, delta_volume_pred


class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        video_paths,
        annotation_paths,
        volume_csv_path,
        clip_length=6,
        stride=3,
        transform=None,
    ):
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.volume_data = self._load_volume_csv(volume_csv_path)
        self.video_clips = self._prepare_video_clips(video_paths, annotation_paths)

    def _load_volume_csv(self, path):
        df = pd.read_csv(path)
        df = df.dropna(subset=["video_name", "bl_loss", "measurement_frame"])
        data = defaultdict(list)
        for _, row in df.iterrows():
            data[str(row["video_name"])].append(
                (int(row["measurement_frame"]), float(row["bl_loss"]))
            )
        return data

    def _prepare_video_clips(self, video_paths, annotation_paths):
        """Prepare video clips with bleeding labels and checkpoints., NO SAMPLING."""
        video_clips = []

        for vid_path, anno_path in zip(video_paths, annotation_paths):
            video_id = os.path.basename(vid_path).split(".")[0]
            cap = cv2.VideoCapture(vid_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= self.clip_length:
                continue

            bleeding_labels = self._parse_bleeding_csv(anno_path, frame_count)

            clips = []
            for start in range(0, frame_count - self.clip_length + 1, self.stride):
                end = start + self.clip_length
                center = (start + end) // 2
                label = int(np.max(bleeding_labels[start:end]))
                clips.append({
                    "start": start,
                    "end": end,
                    "center": center,
                    "label": label,
                })

            video_clips.append({
                "video_id": video_id,
                "video_path": vid_path,
                "clips": clips,
                "bl_checkpoints": sorted(self.volume_data.get(video_id, [])),
                "frame_count": frame_count,
            })

        return video_clips

    # def _prepare_video_clips(self, video_paths, annotation_paths):
    #     """Prepare video clips with bleeding labels and checkpoints, balancing on number of bleeding clips."""
    #     video_clips = []

    #     for vid_path, anno_path in zip(video_paths, annotation_paths):
    #         video_id = os.path.basename(vid_path).split(".")[0]
    #         cap = cv2.VideoCapture(vid_path)
    #         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         cap.release()

    #         if frame_count <= self.clip_length:
    #             continue

    #         bleeding_labels = self._parse_bleeding_csv(anno_path, frame_count)

    #         # Separate bleeding and non-bleeding clips
    #         bleeding_clips = []
    #         non_bleeding_clips = []

    #         for start in range(0, frame_count - self.clip_length + 1, self.stride):
    #             end = start + self.clip_length
    #             center = (start + end) // 2
    #             label = int(np.max(bleeding_labels[start:end]))

    #             clip_info = {
    #                 "start": start,
    #                 "end": end,
    #                 "center": center,
    #                 "label": label,
    #             }

    #             if label > 0:
    #                 bleeding_clips.append(clip_info)
    #             else:
    #                 non_bleeding_clips.append(clip_info)

    #         # BALANCE THE CLIPS (like train_det/train_seg)
    #         total_bleeding = len(bleeding_clips)
    #         total_non_bleeding = len(non_bleeding_clips)

    #         print(f"Video {video_id}: {total_bleeding} bleeding, {total_non_bleeding} non-bleeding clips")

    #         if total_bleeding < total_non_bleeding:
    #             # Sample non-bleeding clips to match bleeding clips
    #             random.shuffle(non_bleeding_clips)
    #             sampled_non_bleeding = non_bleeding_clips[:total_bleeding]
    #         else:
    #             sampled_non_bleeding = non_bleeding_clips

    #         # Combine and sort by temporal order (important for cumulative sum!)
    #         all_clips = bleeding_clips + sampled_non_bleeding
    #         all_clips.sort(key=lambda x: x["start"])  # Keep temporal order

    #         print(f"After balancing: {len(all_clips)} total clips")

    #         video_clips.append({
    #             "video_id": video_id,
    #             "video_path": vid_path,
    #             "clips": all_clips,  # Now balanced!
    #             "bl_checkpoints": sorted(self.volume_data.get(video_id, [])),
    #             "frame_count": frame_count,
    #         })

    #     return video_clips

    def _parse_bleeding_csv(self, csv_path, total_frames):
        df = pd.read_csv(csv_path)
        severity_map = {"BL_Low": 1, "BL_Medium": 2, "BL_High": 3}
        labels = np.zeros(total_frames, dtype=np.int32)
        for _, row in df.iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            sev = severity_map.get(row["label"], 1)
            labels[max(0, start) : min(end + 1, total_frames)] = sev
        return labels

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        """Return video metadata without loading all clips into memory"""
        info = self.video_clips[idx]

        # Don't load all clips at once - just return metadata
        return {
            "video_id": info["video_id"],
            "video_path": info["video_path"],
            "clips": info["clips"],  # Just metadata, not actual tensors
            "bl_checkpoints": info["bl_checkpoints"],
            "frame_count": info["frame_count"],
        }

    def load_clip_batch(self, video_idx, clip_indices):
        """Load specific clips for a video - called by training loop"""
        info = self.video_clips[video_idx]
        clip_tensors = []
        clip_labels = []

        for clip_idx in clip_indices:
            if clip_idx >= len(info["clips"]):
                continue

            clip_info = info["clips"][clip_idx]
            frames = self._load_frames(
                info["video_path"],
                clip_info["start"],
                clip_info["end"]
            )

            if len(frames) < self.clip_length:
                # Pad with last frame if needed
                last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
                frames.extend([last_frame] * (self.clip_length - len(frames)))

            if self.transform:
                frames = [self.transform(f) for f in frames]

            clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, F, H, W]
            clip_tensors.append(clip_tensor)
            clip_labels.append(1 if clip_info["label"] > 0 else 0)

        if not clip_tensors:
            return None, None

        return torch.stack(clip_tensors), torch.tensor(clip_labels)

    def _load_frames(self, video_path, start, end):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames


def load_clips_raw_fast(video_path, clips_info, clip_length):
    """Load clips without transforms (apply on GPU later)"""
    import decord

    print(f"Loading video {os.path.basename(video_path)}...")
    start_time = time.time()

    vr = decord.VideoReader(video_path, num_threads=8)
    total_frames = len(vr)

    # Read entire video
    all_frames = vr.get_batch(range(total_frames)).asnumpy()
    load_time = time.time() - start_time
    print(f"Video loaded in {load_time:.1f}s")

    # Extract clips (fast array slicing)
    all_clips = []
    for i, clip_info in enumerate(clips_info):
        start_idx = clip_info["start"]
        end_idx = min(clip_info["end"], total_frames)

        clip_frames = all_frames[start_idx:end_idx]
        if len(clip_frames) < clip_length:
            last_frame = clip_frames[-1] if len(clip_frames) > 0 else np.zeros((224, 224, 3), dtype=np.uint8)
            padding = np.repeat(last_frame[np.newaxis], clip_length - len(clip_frames), axis=0)
            clip_frames = np.concatenate([clip_frames, padding])

        # Convert to tensor (no CPU transforms!)
        clip_tensor = torch.from_numpy(clip_frames[:clip_length]).permute(3, 0, 1, 2).float() / 255.0
        all_clips.append(clip_tensor)

        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"  Extracted {i}/{len(clips_info)} clips in {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"✅ Extracted {len(clips_info)} clips in {total_time:.1f}s")
    return all_clips

def apply_transforms_on_gpu(clip_batch, device):
    """Apply transforms on GPU (much faster than CPU)"""
    clip_batch = clip_batch.to(device)

    # Resize on GPU
    B, C, T, H, W = clip_batch.shape
    # Reshape to process all frames at once: [B*T, C, H, W]
    reshaped = clip_batch.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
    resized = F.interpolate(reshaped, size=(328, 512), mode='bilinear', align_corners=False)
    # Reshape back: [B, T, C, H, W] -> [B, C, T, H, W]
    clip_batch = resized.reshape(B, T, C, 328, 512).permute(0, 2, 1, 3, 4)

    # Normalize on GPU
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    clip_batch = (clip_batch - mean) / std

    return clip_batch


def train_model(
    video_dir,
    annotation_dir,
    volume_csv_path,
    output_dir="./models_quant",
    epochs=20,
    clip_batch_size=8,
    clip_length=6,
    stride=3,
    learning_rate=1e-4,
    dev="cuda:0",
    model_name="video_quant_model",
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    input_size=(328, 512),
):
    console = Console()
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(dev if torch.cuda.is_available() else "cpu")

    # Header panel
    console.print(
        Panel.fit(
            f"Video-Level Blood Loss Quantification\n"
            f"Device: {device} | Epochs: {epochs} | Clip Batch: {clip_batch_size}\n"
            f"Clip Length: {clip_length} | Stride: {stride} | LR: {learning_rate}",
            title="Training Configuration",
            border_style="cyan",
        )
    )

    # Collect video paths
    console.print("\n📂 Collecting videos and annotations...")
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    annotation_paths = []

    for vp in video_paths:
        base = os.path.basename(vp).split(".")[0]
        anno_path = os.path.join(annotation_dir, f"{base}_annotations.csv")
        if os.path.exists(anno_path):
            annotation_paths.append(anno_path)
        else:
            video_paths.remove(vp)

    console.print(f"✅ Found {len(video_paths)} videos with matching annotations")

    # Video-level splitting
    unique_ids = list(set(os.path.basename(p).split(".")[0] for p in video_paths))

    random.shuffle(unique_ids)

    n_test = max(1, int(test_split * len(unique_ids)))
    n_val = max(1, int(val_split * len(unique_ids)))
    test_ids = unique_ids[:n_test]
    val_ids = unique_ids[n_test:n_test + n_val]
    train_ids = unique_ids[n_test + n_val:]

    def split_paths(paths, ids):
        vids, annos = [], []
        for vp, ap in zip(video_paths, annotation_paths):
            if os.path.basename(vp).split(".")[0] in ids:
                vids.append(vp)
                annos.append(ap)
        return vids, annos

    train_videos, train_annos = split_paths(video_paths, train_ids)
    val_videos, val_annos = split_paths(video_paths, val_ids)
    test_videos, test_annos = split_paths(video_paths, test_ids)

    # Dataset split table
    split_table = Table(title="📊 Video Split")
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Videos", justify="right", style="green")
    split_table.add_column("Percentage", justify="right", style="yellow")
    split_table.add_column("Video IDs", style="dim")

    split_table.add_row(
        "Train",
        str(len(train_videos)),
        f"{len(train_videos)/len(unique_ids)*100:.1f}%",
        ", ".join(train_ids[:3]) + ("..." if len(train_ids) > 3 else "")
    )
    split_table.add_row(
        "Validation",
        str(len(val_videos)),
        f"{len(val_videos)/len(unique_ids)*100:.1f}%",
        ", ".join(val_ids)
    )
    split_table.add_row(
        "Test",
        str(len(test_videos)),
        f"{len(test_videos)/len(unique_ids)*100:.1f}%",
        ", ".join(test_ids)
    )

    console.print(split_table)

    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create datasets with progress
    console.print("\n🔄 Creating datasets...")

    with console.status("[bold green]Processing training videos..."):
        train_dataset = VideoSequenceDataset(
            train_videos, train_annos, volume_csv_path, clip_length, stride, transform
        )

    with console.status("[bold yellow]Processing validation videos..."):
        val_dataset = VideoSequenceDataset(
            val_videos, val_annos, volume_csv_path, clip_length, stride, transform
        )

    with console.status("[bold blue]Processing test videos..."):
        test_dataset = VideoSequenceDataset(
            test_videos, test_annos, volume_csv_path, clip_length, stride, transform
        )

    # Dataset statistics table
    def get_dataset_stats(dataset):
        total_clips = sum(len(dataset.video_clips[i]["clips"]) for i in range(len(dataset)))
        bleeding_clips = sum(
            sum(1 for clip in dataset.video_clips[i]["clips"] if clip["label"] > 0)
            for i in range(len(dataset))
        )
        checkpoints = sum(len(dataset.video_clips[i]["bl_checkpoints"]) for i in range(len(dataset)))
        return total_clips, bleeding_clips, checkpoints

    train_clips, train_bleeding, train_checkpoints = get_dataset_stats(train_dataset)
    val_clips, val_bleeding, val_checkpoints = get_dataset_stats(val_dataset)
    test_clips, test_bleeding, test_checkpoints = get_dataset_stats(test_dataset)

    stats_table = Table(title="📈 Dataset Statistics")
    stats_table.add_column("Split", style="cyan")
    stats_table.add_column("Videos", justify="right", style="white")
    stats_table.add_column("Total Clips", justify="right", style="green")
    stats_table.add_column("Bleeding Clips", justify="right", style="red")
    stats_table.add_column("Checkpoints", justify="right", style="yellow")
    stats_table.add_column("Bleeding %", justify="right", style="magenta")

    stats_table.add_row(
        "Train", str(len(train_dataset)), str(train_clips), str(train_bleeding),
        str(train_checkpoints), f"{train_bleeding/train_clips*100:.1f}%"
    )
    stats_table.add_row(
        "Validation", str(len(val_dataset)), str(val_clips), str(val_bleeding),
        str(val_checkpoints), f"{val_bleeding/val_clips*100:.1f}%"
    )
    stats_table.add_row(
        "Test", str(len(test_dataset)), str(test_clips), str(test_bleeding),
        str(test_checkpoints), f"{test_bleeding/test_clips*100:.1f}%"
    )

    console.print(stats_table)

    # Initialize model
    model = VideoBleedingDetector(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n🧠 Model parameters: {total_params:,}")

    # Training history
    history = {
        "train_loss": [], "train_clf_loss": [], "train_vol_loss": [],
        "val_loss": [], "val_clf_loss": [], "val_vol_loss": [],
        "val_accuracy": [], "val_f1": []
    }

    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")

    # Training loop
    console.print(f"\n🚀 Starting training...")

    for epoch in range(epochs):
        console.print(f"\n[bold blue]Epoch {epoch+1}/{epochs}[/bold blue]")

        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_clf_loss = 0.0
        epoch_vol_loss = 0.0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} videos"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
            console=console,
            refresh_per_second=4,
        ) as progress:

            train_task = progress.add_task("🏋️ Training", total=len(train_dataset))

            for video_idx in range(len(train_dataset)):
                video_sample = train_dataset[video_idx]
                console.print(f"\nProcessing video {video_idx + 1}/{len(train_dataset)}: {video_sample['video_id']}")

                loss, clf_loss, vol_loss = train_video(
                    model, video_sample, train_dataset, optimizer, device, clip_batch_size
                )
                # loss, clf_loss, vol_loss = train_video(
                #     model, video_sample, train_dataset, optimizer, device
                # )

                epoch_train_loss += loss
                epoch_clf_loss += clf_loss
                epoch_vol_loss += vol_loss

                progress.update(
                    train_task,
                    advance=1,
                )
                console.print(
                    f"Video {video_idx + 1}/{len(train_dataset)}: "
                    f"Loss={loss:.3f} | Vol Loss={vol_loss:.3f}"
                )

        # ===== VALIDATION PHASE =====
        model.eval()
        console.print("\n[bold]Validating...[/bold]")
        val_metrics = evaluate_model(model, val_dataset, device)
        val_metrics = evaluate_model(model, val_dataset, device, clip_batch_size)

        # Calculate averages
        avg_train_loss = epoch_train_loss / len(train_dataset)
        avg_clf_loss = epoch_clf_loss / len(train_dataset)
        avg_vol_loss = epoch_vol_loss / len(train_dataset)

        # Store metrics
        history["train_loss"].append(avg_train_loss)
        history["train_clf_loss"].append(avg_clf_loss)
        history["train_vol_loss"].append(avg_vol_loss)
        history["val_loss"].append(val_metrics["total_loss"])
        history["val_clf_loss"].append(val_metrics["clf_loss"])
        history["val_vol_loss"].append(val_metrics["vol_loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        # Results table
        results_table = Table(title=f"📊 Epoch {epoch+1} Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Train", justify="right", style="green")
        results_table.add_column("Validation", justify="right", style="yellow")

        results_table.add_row("Total Loss", f"{avg_train_loss:.4f}", f"{val_metrics['total_loss']:.4f}")
        results_table.add_row("Classification Loss", f"{avg_clf_loss:.4f}", f"{val_metrics['clf_loss']:.4f}")
        results_table.add_row("Volume Loss", f"{avg_vol_loss:.4f}", f"{val_metrics['vol_loss']:.4f}")
        results_table.add_row("Accuracy", "—", f"{val_metrics['accuracy']:.4f}")
        results_table.add_row("Precision", "—", f"{val_metrics['precision']:.4f}")
        results_table.add_row("Recall", "—", f"{val_metrics['recall']:.4f}")
        results_table.add_row("F1 Score", "—", f"{val_metrics['f1']:.4f}")

        console.print(results_table)

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            console.print(f"New best model saved! F1: {best_f1:.4f}", style="bold green")

        # Learning rate scheduling
        scheduler.step(val_metrics["total_loss"])

    # Final test evaluation
    console.print("\n🧪 Final test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate_model(model, test_dataset, device)

    # Final results
    final_table = Table(title="Final Test Results")
    final_table.add_column("Metric", style="cyan")
    final_table.add_column("Score", justify="right", style="green")

    final_table.add_row("Accuracy", f"{test_metrics['accuracy']:.4f}")
    final_table.add_row("Precision", f"{test_metrics['precision']:.4f}")
    final_table.add_row("Recall", f"{test_metrics['recall']:.4f}")
    final_table.add_row("F1 Score", f"{test_metrics['f1']:.4f}")
    final_table.add_row("Volume Loss", f"{test_metrics['vol_loss']:.4f}")

    console.print(final_table)

    # Save final model and history
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_final.pth"))

    import json
    with open(os.path.join(output_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    console.print(
        Panel.fit(
            f"✅ Training completed!\n"
            f"Best F1: {best_f1:.4f}\n"
            f"Models saved to: {output_dir}",
            title="🎉 Success",
            border_style="green",
        )
    )

    return model, history, test_metrics

# def train_video(model, video_sample, train_dataset, optimizer, device):
#     """Load entire video to GPU, train, then clear - simple and efficient"""

#     clips_info = video_sample["clips"]
#     checkpoints = video_sample["bl_checkpoints"]
#     video_path = video_sample["video_path"]

#     total_clips = len(clips_info)
#     if total_clips == 0:
#         return 0.0, 0.0, 0.0

#     # Load ALL clips from this video to CPU first
#     print(f"Loading {total_clips} clips from video...")
#     all_clips_cpu = []
#     all_labels_cpu = []

#     for clip_info in clips_info:
#         frames = train_dataset._load_frames(video_path, clip_info["start"], clip_info["end"])

#         if len(frames) < train_dataset.clip_length:
#             last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
#             frames.extend([last_frame] * (train_dataset.clip_length - len(frames)))

#         if train_dataset.transform:
#             frames = [train_dataset.transform(f) for f in frames]

#         clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, F, H, W]
#         all_clips_cpu.append(clip_tensor)
#         all_labels_cpu.append(1 if clip_info["label"] > 0 else 0)

#     # Move ALL clips to GPU at once
#     print(f"Moving {total_clips} clips to GPU...")
#     all_clips_gpu = torch.stack(all_clips_cpu).to(device)      # [T, C, F, H, W]
#     all_labels_gpu = torch.tensor(all_labels_cpu).to(device)   # [T]

#     # Clear CPU memory
#     del all_clips_cpu, all_labels_cpu

#     # Forward pass through entire video
#     print("Forward pass...")
#     clip_preds, delta_preds = model(all_clips_gpu)  # [T, num_classes], [T]

#     # Calculate losses
#     clf_loss = F.cross_entropy(clip_preds, all_labels_gpu)

#     # Volume loss with cumulative sum
#     cumsum_preds = torch.cumsum(delta_preds, dim=0)
#     volume_loss = torch.tensor(0.0, device=device)

#     if checkpoints:
#         frame_centers = torch.tensor([clip["center"] for clip in clips_info], device=device)
#         target_vals, pred_vals = [], []

#         for check_frame, true_volume in checkpoints:
#             distances = torch.abs(frame_centers.float() - check_frame)
#             closest_idx = torch.argmin(distances)
#             pred_vals.append(cumsum_preds[closest_idx])
#             target_vals.append(true_volume)

#         if target_vals:
#             volume_loss = F.mse_loss(
#                 torch.stack(pred_vals),
#                 torch.tensor(target_vals, device=device, dtype=torch.float32)
#             )

#     total_loss = clf_loss + 0.2 * volume_loss

#     # Backpropagation
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()

#     # Clear GPU memory for next video
#     del all_clips_gpu, all_labels_gpu, clip_preds, delta_preds
#     torch.cuda.empty_cache()

#     print(f"Video processed: Loss={total_loss.item():.4f}")

#     return total_loss.item(), clf_loss.item(), volume_loss.item()

def train_video(model, video_sample, train_dataset, optimizer, device, clip_batch_size=16):
    """Train on single video with GPU transforms"""

    clips_info = video_sample["clips"]
    checkpoints = video_sample["bl_checkpoints"]
    video_path = video_sample["video_path"]

    total_clips = len(clips_info)
    if total_clips == 0:
        return 0.0, 0.0, 0.0

    # Load clips without transforms (much faster)
    all_clips_cpu = load_clips_raw_fast(video_path, clips_info, train_dataset.clip_length)
    all_labels = [1 if clip["label"] > 0 else 0 for clip in clips_info]

    # Store predictions
    all_clip_preds = []
    all_delta_preds = []

    model.train()

    # Process in batches with GPU transforms
    for i in range(0, len(all_clips_cpu), clip_batch_size):
        end_idx = min(i + clip_batch_size, len(all_clips_cpu))

        # Stack batch (still on CPU)
        clip_batch = torch.stack(all_clips_cpu[i:end_idx])  # [B, C, T, H, W]

        # Apply transforms on GPU (fast!)
        clip_batch = apply_transforms_on_gpu(clip_batch, device)

        # Forward pass
        with torch.no_grad():
            clip_preds, delta_preds = model(clip_batch)

        # Store predictions
        all_clip_preds.append(clip_preds.detach().cpu())
        all_delta_preds.append(delta_preds.detach().cpu())

        # Clear GPU
        del clip_batch, clip_preds, delta_preds
        torch.cuda.empty_cache()

    # Rest of function stays the same...
    full_clip_preds = torch.cat(all_clip_preds, dim=0).to(device)
    full_delta_preds = torch.cat(all_delta_preds, dim=0).to(device)
    full_labels = torch.tensor(all_labels).to(device)

    full_clip_preds.requires_grad_(True)
    full_delta_preds.requires_grad_(True)

    # Calculate losses (same as before)
    clf_loss = F.cross_entropy(full_clip_preds, full_labels)

    cumsum_preds = torch.cumsum(full_delta_preds, dim=0)
    volume_loss = torch.tensor(0.0, device=device)

    if checkpoints:
        frame_centers = torch.tensor([clip["center"] for clip in clips_info], device=device)
        target_vals, pred_vals = [], []

        for check_frame, true_volume in checkpoints:
            distances = torch.abs(frame_centers.float() - check_frame)
            closest_idx = torch.argmin(distances)
            pred_vals.append(cumsum_preds[closest_idx])
            target_vals.append(true_volume)

        if target_vals:
            volume_loss = F.mse_loss(
                torch.stack(pred_vals),
                torch.tensor(target_vals, device=device, dtype=torch.float32)
            )

    total_loss = clf_loss + 0.2 * volume_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), clf_loss.item(), volume_loss.item()


def evaluate_model(model, dataset, device, clip_batch_size=16):
    """Evaluate using same efficient batching approach"""
    model.eval()
    all_preds = []
    all_labels = []
    total_clf_loss = 0.0
    total_vol_loss = 0.0

    with torch.no_grad():
        with Progress(
            TextColumn("🔍 Evaluating"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} videos"),
            transient=True,
        ) as progress:

            eval_task = progress.add_task("Evaluating", total=len(dataset))

            for video_idx in range(len(dataset)):
                video_sample = dataset[video_idx]
                clips_info = video_sample["clips"]
                checkpoints = video_sample["bl_checkpoints"]
                video_path = video_sample["video_path"]

                if len(clips_info) == 0:
                    progress.update(eval_task, advance=1)
                    continue

                # Load clips efficiently
                all_clips_cpu = load_clips_raw_fast(video_path, clips_info, dataset.clip_length)
                all_labels_cpu = [1 if clip["label"] > 0 else 0 for clip in clips_info]

                # Process in batches
                all_clip_preds = []
                all_delta_preds = []

                for i in range(0, len(all_clips_cpu), clip_batch_size):
                    end_idx = min(i + clip_batch_size, len(all_clips_cpu))
                    clip_batch = torch.stack(all_clips_cpu[i:end_idx])

                    # Apply GPU transforms
                    clip_batch = apply_transforms_on_gpu(clip_batch, device)

                    clip_preds, delta_preds = model(clip_batch)
                    all_clip_preds.append(clip_preds.cpu())
                    all_delta_preds.append(delta_preds.cpu())

                    del clip_batch, clip_preds, delta_preds

                # Reconstruct sequences
                full_clip_preds = torch.cat(all_clip_preds, dim=0).to(device)
                full_delta_preds = torch.cat(all_delta_preds, dim=0).to(device)
                full_labels = torch.tensor(all_labels_cpu).to(device)

                # Metrics
                _, predicted = torch.max(full_clip_preds, dim=1)
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(full_labels.cpu().tolist())

                # Losses
                clf_loss = F.cross_entropy(full_clip_preds, full_labels)
                total_clf_loss += clf_loss.item()

                # Volume loss calculation (same as training)
                if checkpoints:
                    cumsum_preds = torch.cumsum(full_delta_preds, dim=0)
                    frame_centers = torch.tensor([clip["center"] for clip in clips_info], device=device)

                    target_vals, pred_vals = [], []
                    for check_frame, true_volume in checkpoints:
                        distances = torch.abs(frame_centers.float() - check_frame)
                        closest_idx = torch.argmin(distances)
                        pred_vals.append(cumsum_preds[closest_idx])
                        target_vals.append(true_volume)

                    if target_vals:
                        vol_loss = F.mse_loss(
                            torch.stack(pred_vals),
                            torch.tensor(target_vals, device=device, dtype=torch.float32)
                        )
                        total_vol_loss += vol_loss.item()

                progress.update(eval_task, advance=1)

    # Calculate final metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clf_loss": total_clf_loss / len(dataset),
        "vol_loss": total_vol_loss / len(dataset),
        "total_loss": (total_clf_loss + 0.2 * total_vol_loss) / len(dataset),
    }

# def evaluate_model(model, dataset, device, clip_batch_size=8):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     total_clf_loss = 0.0
#     total_vol_loss = 0.0

#     with torch.no_grad():
#         with Progress(
#             TextColumn("🔍 Evaluating"),
#             BarColumn(),
#             TextColumn("{task.completed}/{task.total} videos"),
#             transient=True,
#         ) as progress:

#             eval_task = progress.add_task("Evaluating", total=len(dataset))

#             for video_idx in range(len(dataset)):
#                 video_sample = dataset[video_idx]
#                 clips = video_sample["clips"]
#                 labels = video_sample["clip_labels"]
#                 checkpoints = video_sample["bl_checkpoints"]
#                 frame_centers = video_sample["frame_centers"]

#                 T = clips.shape[0]

#                 # Process in batches
#                 all_clip_preds = []
#                 all_delta_preds = []

#                 for i in range(0, T, clip_batch_size):
#                     end_idx = min(i + clip_batch_size, T)
#                     clip_batch = clips[i:end_idx].to(device)

#                     clip_preds, delta_preds = model(clip_batch)
#                     all_clip_preds.append(clip_preds.cpu())
#                     all_delta_preds.append(delta_preds.cpu())

#                 full_clip_preds = torch.cat(all_clip_preds, dim=0).to(device)
#                 full_delta_preds = torch.cat(all_delta_preds, dim=0).to(device)
#                 labels = labels.to(device)

#                 # Classification metrics
#                 _, predicted = torch.max(full_clip_preds, dim=1)
#                 all_preds.extend(predicted.cpu().tolist())
#                 all_labels.extend(labels.cpu().tolist())

#                 # Losses
#                 clf_loss = F.cross_entropy(full_clip_preds, labels)
#                 total_clf_loss += clf_loss.item()

#                 if checkpoints:
#                     cumsum_preds = torch.cumsum(full_delta_preds, dim=0)
#                     target_vals, pred_vals = [], []
#                     for check_frame, true_volume in checkpoints:
#                         distances = torch.abs(frame_centers.float() - check_frame)
#                         closest_idx = torch.argmin(distances)
#                         pred_vals.append(cumsum_preds[closest_idx])
#                         target_vals.append(true_volume)

#                     if target_vals:
#                         vol_loss = F.mse_loss(
#                             torch.stack(pred_vals),
#                             torch.tensor(target_vals, device=device, dtype=torch.float32)
#                         )
#                         total_vol_loss += vol_loss.item()

#                 progress.update(eval_task, advance=1)

#     # Calculate metrics
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, zero_division=0)
#     recall = recall_score(all_labels, all_preds, zero_division=0)
#     f1 = f1_score(all_labels, all_preds, zero_division=0)

#     avg_clf_loss = total_clf_loss / len(dataset)
#     avg_vol_loss = total_vol_loss / len(dataset)

#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "clf_loss": avg_clf_loss,
#         "vol_loss": avg_vol_loss,
#         "total_loss": avg_clf_loss + 0.2 * avg_vol_loss,
#     }

def main():
    # VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/videos"
    # ANNO_FOLDER = "/home/r.rohangirish/mt_ble/data/labels_csv"
    VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/test_videos"
    ANNO_FOLDER = "/home/r.rohangirish/mt_ble/data/test_labels"

    parser = argparse.ArgumentParser(description="Video Bleeding Detector")

    parser.add_argument(
        "--dev", type=int, default=0, help="CUDA device number (default: 0)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    device_num = args.dev
    seed = args.seed if hasattr(args, 'seed') else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_model(
        video_dir=VIDEO_DIR,
        annotation_dir=ANNO_FOLDER,
        volume_csv_path="/home/r.rohangirish/mt_ble/data/labels_quantification/BL_data_combined.csv",
        dev=f"cuda:{device_num}",
        model_name="./models_volume_loss/video_quant_v1",
        clip_batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
