# This is a simple addition to the 3D CNN model for clip classificatio, adding a simple regression head
# for volume prediction using the sparse annotations.

import argparse
import os

tmp_dir = "/home/r.rohangirish/mt_ble/tmp"
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir

import glob
import random
import numpy as np
import pandas as pd
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# import evaluate_video

cv2.setNumThreads(6)
NUM_THREADS = 6
VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/videos"
ANNO_FOLDER = "/home/r.rohangirish/mt_ble/data/labels_csv"

TEST_VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/test_videos"
TEST_LABEL_DIR = "/home/r.rohangirish/mt_ble/data/test_labels"


import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoBleedingDetector(nn.Module):
    def __init__(self, num_classes=2, severity_levels=4):
        super().__init__()

        # Load 3D CNN backbone
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

        # Heads
        self.classifier = nn.Linear(in_features, num_classes)
        self.severity_classifier = nn.Linear(in_features, severity_levels)
        self.volume_regressor = nn.Linear(in_features, 1)  # Cumulative blood loss

    def forward(self, x):
        features = self.backbone(x)  # [B, in_features]
        clip_pred = self.classifier(features)
        severity_pred = self.severity_classifier(features)
        volume_pred = self.volume_regressor(features)  # [B, 1] cumulative BL

        return clip_pred, severity_pred, volume_pred


class SurgicalVideoDataset(Dataset):
    """
    Dataset for loading surgical video clips with bleeding annotations.
    """

    def __init__(
        self,
        video_paths,
        annotation_paths,
        clip_length=6,
        stride=3,
        transform=None,
        volume_csv_path=None,
    ):
        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.clip_length = clip_length
        self.stride = stride
        self.volume_data = (
            self._load_volume_csv(volume_csv_path) if volume_csv_path else {}
        )

        self.transform = transform

        # Create clips with annotations
        self.clips = self._prepare_clips()

    def _load_volume_csv(self, path):
        """Load cumulative blood loss data from CSV and return a mapping"""

        df = pd.read_csv(path)
        df = df.dropna(subset=["video_name", "bl_loss", "measurement_frame"])

        volume_map = {}

        for video_name, group in df.groupby("video_name"):
            entries = []
            for _, row in group.iterrows():
                frame = int(row["measurement_frame"])
                value = float(row["bl_loss"])
                entries.append((frame, value))
            entries.sort()  # ascending by frame

            total_bl = (
                float(group["a_e_bl"].dropna().iloc[0])
                if not group["a_e_bl"].dropna().empty
                else None
            )

            volume_map[str(video_name)] = {"measured": entries, "total": total_bl}

        return volume_map

    def _prepare_clips(self):
        bleeding_clips = {1: [], 2: [], 3: []}
        non_bleeding_clips = []

        for video_path, anno_path in zip(self.video_paths, self.annotation_paths):
            video_id = os.path.basename(video_path).split(".")[0]

            # Get video info
            cap = cv2.VideoCapture(video_path)
            params = [cv2.CAP_PROP_N_THREADS, NUM_THREADS]
            cap.open(video_path, apiPreference=cv2.CAP_FFMPEG, params=params)

            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                print(
                    f"Warning: Video {video_path} has invalid frame count: {frame_count}"
                )
                continue

            try:
                bleeding_frames = self._parse_bleeding_csv(anno_path, frame_count)
            except Exception as e:
                print(f"Error parsing annotations for {video_path}: {e}")
                continue

            volume_info = self.volume_data.get(video_id, {})
            measurements = sorted(volume_info.get("measured", []), key=lambda x: x[0])

            # Create overlapping clips
            for start_idx in range(0, frame_count - self.clip_length + 1, self.stride):
                end_idx = start_idx + self.clip_length
                clip_bleeding = bleeding_frames[start_idx:end_idx]

                if len(clip_bleeding) != self.clip_length:
                    continue

                has_bleeding = np.max(clip_bleeding) > 0
                max_severity = int(np.max(clip_bleeding))
                frame_center = (start_idx + end_idx) // 2

                # Match closest cumulative measurement at or after clip center
                clip_volume = None
                for frame_num, bl_val in measurements:
                    if frame_num >= frame_center:
                        clip_volume = bl_val
                        break

                clip_info = {
                    "video_path": video_path,
                    "video_id": video_id,
                    "start_frame": start_idx,
                    "end_frame": end_idx - 1,
                    "has_bleeding": has_bleeding,
                    "bleeding_frames": clip_bleeding,
                    "max_severity": max_severity,
                    "volume_label": clip_volume if clip_volume is not None else -1.0,
                    "volume_supervised": clip_volume is not None,
                }

                if has_bleeding:
                    bleeding_clips[max_severity].append(clip_info)
                else:
                    non_bleeding_clips.append(clip_info)

        # Balance non-bleeding to match bleeding
        severity_counts = {s: len(c) for s, c in bleeding_clips.items()}
        total_bleeding = sum(severity_counts.values())
        total_non_bleeding = len(non_bleeding_clips)

        if total_bleeding < total_non_bleeding:
            import random

            random.shuffle(non_bleeding_clips)
            non_bleeding_clips = non_bleeding_clips[:total_bleeding]

        all_clips = non_bleeding_clips.copy()
        for clips in bleeding_clips.values():
            all_clips.extend(clips)

        return all_clips

    @staticmethod
    def _parse_bleeding_csv(csv_path, total_frames):
        """
        Parse CSV with start_frame, end_frame, label format and
        convert to per-frame bleeding labels
        """
        frame_labels = np.zeros(total_frames, dtype=np.int32)

        df = pd.read_csv(csv_path)

        # Map bleeding severity to numerical values
        severity_map = {"BL_Low": 1, "BL_Medium": 2, "BL_High": 3}

        # Fill in bleeding frames based on start and end frames
        for _, row in df.iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            label = row["label"]

            # Convert label to severity value
            severity = severity_map.get(label, 1)  # Default to 1 if unknown

            # Mark all frames in range with this severity
            if start < total_frames and end >= 0:
                valid_start = max(0, start)
                valid_end = min(total_frames - 1, end)

                frame_labels[valid_start : valid_end + 1] = severity

        return frame_labels

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]

        # Load video clip frames
        frames = self._load_clip(
            clip_info["video_path"],
            clip_info["start_frame"],
            clip_info["end_frame"] + 1,  # +1 because end_frame is inclusive
        )

        if len(frames) < self.clip_length:
            # Duplicate last frame if we couldn't get enough frames
            last_frame = (
                frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            )
            frames.extend([last_frame] * (self.clip_length - len(frames)))

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into tensor [T, C, H, W]
        clip_tensor = torch.stack(frames)

        # Reshape to [C, T, H, W] as expected by 3D CNNs
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)

        # Get labels
        clip_label = 1 if clip_info["has_bleeding"] else 0
        frame_labels = torch.tensor(clip_info["bleeding_frames"], dtype=torch.float32)
        severity = torch.tensor(clip_info["max_severity"], dtype=torch.long)

        # Volume label and mask
        volume_val = clip_info.get("volume_label")
        volume = torch.tensor(volume_val if volume_val is not None else -1.0, dtype=torch.float32)

        volume_mask = torch.tensor(
            clip_info.get("volume_supervised", False), dtype=torch.bool
        )

        return clip_tensor, clip_label, frame_labels, severity, volume, volume_mask

    def _load_clip(self, video_path, start_frame, end_frame):
        frames = []

        cap = cv2.VideoCapture(video_path)
        params = [cv2.CAP_PROP_N_THREADS, NUM_THREADS]
        cap.open(video_path, apiPreference=cv2.CAP_FFMPEG, params=params)

        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize early to reduce usage
            # frame = cv2.resize(frame, (224, 224))

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames


# TODO: check num of severity values
def train_bleeding_detector(
    video_dir,
    annotation_dir,
    output_dir="./models",
    epochs=20,
    batch_size=4,
    clip_length=6,
    stride=3,
    learning_rate=0.0001,
    dev="cuda:3",
    model_name="final_model",
    volume_csv_path=None,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
):
    """
    Train the 3D CNN model for bleeding detection with existing balanced dataset.
    """
    console = Console()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(dev if torch.cuda.is_available() else "cpu")

    # Welcome panel
    console.print(
        Panel.fit(
            f"[bold cyan]🩸 Bleeding Classification + Volume Prediction[/bold cyan]\n"
            f"Device: [yellow]{device}[/yellow]\n"
            f"Output: [green]{output_dir}[/green]",
            title="🚀 Starting Training",
        )
    )

    # Get video and annotation paths
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    annotation_paths = []

    for vid_path in video_paths:
        vid_name = os.path.basename(vid_path).split(".")[0]
        anno_path = os.path.join(annotation_dir, f"{vid_name}_annotations.csv")
        if os.path.exists(anno_path):
            annotation_paths.append(anno_path)
        else:
            console.print(
                f"[yellow]Warning: No annotation file found for {vid_name}[/yellow]"
            )
            video_paths.remove(vid_path)

    console.print(
        f"[green]Found {len(video_paths)} videos with matching annotations[/green]"
    )

    # ----- VIDEO-LEVEL SPLITTING -----
    unique_videos = list(
        set(os.path.basename(path).split(".")[0] for path in video_paths)
    )

    # Simple random split by video
    random.seed(42)
    random.shuffle(unique_videos)

    # Split videos, atleast 1 in each
    n_test = max(1, int(test_split * len(unique_videos)))
    n_val = max(1, int(val_split * len(unique_videos)))

    test_video_ids = unique_videos[:n_test]
    val_video_ids = unique_videos[n_test : n_test + n_val]
    train_video_ids = unique_videos[n_test + n_val :]

    # Create split info table
    split_table = Table(title="📂 Video Split Information")
    split_table.add_column("Split", style="cyan", justify="center")
    split_table.add_column("Videos", style="green", justify="center")
    split_table.add_column("Video IDs", style="yellow")

    split_table.add_row(
        "Train",
        str(len(train_video_ids)),
        (
            ", ".join(train_video_ids[:3]) + "..."
            if len(train_video_ids) > 3
            else ", ".join(train_video_ids)
        ),
    )
    split_table.add_row("Validation", str(len(val_video_ids)), ", ".join(val_video_ids))
    split_table.add_row("Test", str(len(test_video_ids)), ", ".join(test_video_ids))

    console.print(split_table)

    def split_paths_by_video_ids(video_paths, annotation_paths, video_ids):
        """Split video and annotation paths based on a list of video IDs."""
        videos = []
        annos = []
        for vp, ap in zip(video_paths, annotation_paths):
            if os.path.basename(vp).split(".")[0] in video_ids:
                videos.append(vp)
                annos.append(ap)
        return videos, annos

    # Split paths by video IDs
    train_videos, train_annos = split_paths_by_video_ids(
        video_paths, annotation_paths, train_video_ids
    )
    val_videos, val_annos = split_paths_by_video_ids(
        video_paths, annotation_paths, val_video_ids
    )
    test_videos, test_annos = split_paths_by_video_ids(
        video_paths, annotation_paths, test_video_ids
    )

    # Create transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((512, 328)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
            ),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((512, 328)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    console.print("[yellow]🔄 Creating datasets...[/yellow]")

    train_dataset = SurgicalVideoDataset(
        train_videos,
        train_annos,
        clip_length=clip_length,
        stride=stride,
        transform=train_transform,
        volume_csv_path=volume_csv_path,
    )
    val_dataset = SurgicalVideoDataset(
        val_videos,
        val_annos,
        clip_length=clip_length,
        stride=stride,
        transform=val_test_transform,
        volume_csv_path=volume_csv_path,
    )
    test_dataset = SurgicalVideoDataset(
        test_videos,
        test_annos,
        clip_length=clip_length,
        stride=stride,
        transform=val_test_transform,
        volume_csv_path=volume_csv_path,
    )

    # Dataset info table
    dataset_table = Table(title="📊 Dataset Information")
    dataset_table.add_column("Split", style="cyan")
    dataset_table.add_column("Clips", style="green", justify="right")
    dataset_table.add_column("Transform", style="yellow")

    dataset_table.add_row("Train", str(len(train_dataset)), "Augmented")
    dataset_table.add_row("Validation", str(len(val_dataset)), "Clean")
    dataset_table.add_row("Test", str(len(test_dataset)), "Clean")

    console.print(dataset_table)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model
    model = VideoBleedingDetector(num_classes=2, severity_levels=4)
    model = model.to(device)

    # Loss functions and optimizer
    criterion_clf = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Initialize history tracking
    history = {
        "train_loss": [],
        "train_clf_loss": [],
        "train_sev_loss": [],
        "train_volume_loss": [],
        "val_loss": [],
        "val_clf_loss": [],
        "val_sev_loss": [],
        "val_volume_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_f1 = 0.0
    best_model_path = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[green]{task.percentage:>3.0f}%[/green]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as progress:

        # Main epoch progress
        # epoch_task = progress.add_task("[cyan]🏋️ Training Progress", total=epochs)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_clf_loss = 0.0
            train_sev_loss = 0.0

            # Batch progress for this epoch
            batch_task = progress.add_task(
                f"[green]Epoch {epoch + 1}/{epochs} - Training", total=len(train_loader)
            )

            for (
                clips,
                clip_labels,
                frame_labels,
                severity,
                volume,
                volume_mask,
            ) in train_loader:
                clips = clips.to(device)
                clip_labels = clip_labels.to(device)
                severity = severity.to(device)
                volume = volume.to(device)
                volume_mask = volume_mask.to(device)

                # Forward pass
                clip_preds, severity_preds, volume_preds = model(clips)
                clip_labels = clip_labels.long()
                severity = severity.clamp(0, 3).long()

                # Calculate losses
                loss_clf = criterion_clf(clip_preds, clip_labels)
                loss_sev = criterion_sev(severity_preds, severity)

                if volume_mask.any():
                    volume_loss = F.mse_loss(
                        volume_preds[volume_mask].squeeze(), volume[volume_mask]
                    )
                else:
                    volume_loss = torch.tensor(0.0, device=device)

                loss = (
                    loss_clf + 0.1 * loss_sev + 0.3 * volume_loss
                )  # FIXME: adjust this

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_clf_loss += loss_clf.item()
                train_sev_loss += loss_sev.item()

                # Update batch progress
                progress.update(batch_task, advance=1)

            # Remove batch progress bar
            progress.remove_task(batch_task)

            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_train_clf_loss = train_clf_loss / len(train_loader)
            avg_train_sev_loss = train_sev_loss / len(train_loader)

            # Store training metrics
            history["train_loss"].append(avg_train_loss)
            history["train_clf_loss"].append(avg_train_clf_loss)
            history["train_sev_loss"].append(avg_train_sev_loss)
            history["train_volume_loss"].append(volume_loss.item())

            # Validation phase with progress
            val_task = progress.add_task(
                f"[yellow]Epoch {epoch + 1}/{epochs} - Validation",
                total=len(val_loader),
            )

            val_metrics = evaluate_model(
                model, val_loader, criterion_clf, criterion_sev, device
            )

            progress.remove_task(val_task)

            # Store validation metrics
            history["val_loss"].append(val_metrics["loss"])
            history["val_clf_loss"].append(val_metrics["clf_loss"])
            history["val_sev_loss"].append(val_metrics["sev_loss"])
            history["val_volume_loss"].append(val_metrics["volume_loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])

            # Create beautiful results table
            results_table = Table(title=f"📈 Epoch {epoch + 1}/{epochs} Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Train", style="green", justify="right")
            results_table.add_column("Validation", style="yellow", justify="right")

            results_table.add_row(
                "Total Loss", f"{avg_train_loss:.4f}", f"{val_metrics['loss']:.4f}"
            )
            results_table.add_row(
                "Classification Loss",
                f"{avg_train_clf_loss:.4f}",
                f"{val_metrics['clf_loss']:.4f}",
            )
            results_table.add_row(
                "Severity Loss",
                f"{avg_train_sev_loss:.4f}",
                f"{val_metrics['sev_loss']:.4f}",
            )
            results_table.add_row("Accuracy", "—", f"{val_metrics['accuracy']:.4f}")
            results_table.add_row("Precision", "—", f"{val_metrics['precision']:.4f}")
            results_table.add_row("Recall", "—", f"{val_metrics['recall']:.4f}")
            results_table.add_row("F1 Score", "—", f"{val_metrics['f1']:.4f}")

            console.print(results_table)

            # Update learning rate
            scheduler.step(val_metrics["f1"])

            # Save best model based on F1 score
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")
                torch.save(model.state_dict(), best_model_path)
                console.print(
                    f"[bold green]🎯 New best model saved: F1 = {best_f1:.4f}[/bold green]"
                )

            # Update epoch progress
            progress.update(epoch_task, advance=1)

    # Final test evaluation
    console.print(
        Panel.fit(
            "[bold yellow]🧪 Running Final Test Evaluation...[/bold yellow]",
            title="📊 Testing Phase",
        )
    )

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate_model(
        model, test_loader, criterion_clf, criterion_sev, device
    )

    # Final results table
    final_table = Table(title="🏆 Final Test Results")
    final_table.add_column("Metric", style="cyan")
    final_table.add_column("Score", style="green", justify="right")

    final_table.add_row("Accuracy", f"{test_metrics['accuracy']:.4f}")
    final_table.add_row("Precision", f"{test_metrics['precision']:.4f}")
    final_table.add_row("Recall", f"{test_metrics['recall']:.4f}")
    final_table.add_row("F1 Score", f"{test_metrics['f1']:.4f}")

    console.print(final_table)

    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)

    # Success message
    console.print(
        Panel.fit(
            f"[bold green]✅ Training Complete![/bold green]\n"
            f"Best F1 Score: [yellow]{best_f1:.4f}[/yellow]\n"
            f"Best Model: [blue]{best_model_path}[/blue]\n"
            f"Final Model: [blue]{final_model_path}[/blue]",
            title="🎉 Success",
        )
    )

    return model, history, test_metrics, test_loader


def evaluate_model(
    model, data_loader, criterion_clf, criterion_sev, device, volume_loss_weight=0.3
):
    """
    Evaluate model on given data loader, including optional volume prediction.
    """

    model.eval()
    total_loss = 0.0
    total_clf_loss = 0.0
    total_sev_loss = 0.0
    total_volume_loss = 0.0
    volume_supervised_count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (
            clips,
            clip_labels,
            frame_labels,
            severity,
            volume,
            volume_mask,
        ) in data_loader:
            clips = clips.to(device)
            clip_labels = clip_labels.to(device)
            severity = severity.to(device)
            volume = volume.to(device)
            volume_mask = volume_mask.to(device)

            # Forward pass
            clip_preds, severity_preds, volume_preds = model(clips)

            # Losses
            loss_clf = criterion_clf(clip_preds, clip_labels)
            loss_sev = criterion_sev(severity_preds, severity)

            # Volume loss only on supervised samples
            if volume_mask.any():
                loss_volume = nn.functional.mse_loss(
                    volume_preds[volume_mask].squeeze(), volume[volume_mask]
                )
                total_volume_loss += loss_volume.item()
                volume_supervised_count += 1
            else:
                loss_volume = torch.tensor(0.0, device=device)

            # Combine losses
            loss = loss_clf + 0.3 * loss_sev + volume_loss_weight * loss_volume

            # Accumulate
            total_loss += loss.item()
            total_clf_loss += loss_clf.item()
            total_sev_loss += loss_sev.item()

            # Get predictions for classification metrics
            _, predicted = torch.max(clip_preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(clip_labels.cpu().numpy())

    # Averages
    avg_loss = total_loss / len(data_loader)
    avg_clf_loss = total_clf_loss / len(data_loader)
    avg_sev_loss = total_sev_loss / len(data_loader)
    avg_volume_loss = (
        total_volume_loss / volume_supervised_count
        if volume_supervised_count > 0
        else 0.0
    )

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": avg_loss,
        "clf_loss": avg_clf_loss,
        "sev_loss": avg_sev_loss,
        "volume_loss": avg_volume_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot(history, output_dir):
    """
    Plot key training metrics including classification, severity, and volume loss.
    """
    import matplotlib.pyplot as plt
    import os

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(5, 1, figsize=(10, 15))  # Increased to 5 plots

    # 1. Training Loss
    tl = history.get("train_loss", [])
    if tl:
        axes[0].plot(epochs, tl, "b-", linewidth=2)
        axes[0].set_title("Training Loss")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

    # 2. Validation Loss
    vl = history.get("val_loss", [])
    if vl:
        axes[1].plot(epochs, vl, "r-", linewidth=2)
        axes[1].set_title("Validation Loss")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, alpha=0.3)

    # 3. Validation Severity Loss
    vsl = history.get("val_sev_loss", [])
    if vsl:
        axes[2].plot(epochs, vsl, "g-", linewidth=2)
        axes[2].set_title("Validation Severity Loss")
        axes[2].set_ylabel("Loss")
        axes[2].grid(True, alpha=0.3)

    # 4. Validation Bleeding Accuracy
    vba = history.get("val_bleeding_accuracy", []) or history.get("val_accuracy", [])
    if vba:
        axes[3].plot(epochs, vba, "purple", linewidth=2)
        axes[3].set_title("Validation Bleeding Accuracy")
        axes[3].set_ylabel("Accuracy")
        axes[3].set_xlabel("Epochs")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 1)

    # 5. Validation Volume Loss
    vvl = history.get("val_volume_loss", [])
    if vvl:
        axes[4].plot(epochs, vvl, "orange", linewidth=2)
        axes[4].set_title("Validation Volume Loss")
        axes[4].set_ylabel("Loss (MSE)")
        axes[4].set_xlabel("Epochs")
        axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "training_metrics.png"), dpi=400, bbox_inches="tight"
    )
    # plt.show()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Video Bleeding Detector")
    parser.add_argument(
        "--dev", type=int, default=5, help="CUDA device number (default: 5)"
    )
    parser.add_argument(
        "--train",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to train the model (default: false)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )

    # Path to volume CSV
    volume_csv_path =  "/home/r.rohangirish/mt_ble/data/labels_quantification/BL_data_combined.csv"
    # Parse arguments
    args = parser.parse_args()
    device_num = args.dev
    do_train = args.train.lower() == "true"
    output_dir = "./models_volume_loss"
    batch_size = args.batch_size

    device = torch.device(f"cuda:{device_num}")
    model_name = "clf_sev_vol_v1"

    # Train model if specified
    if do_train:
        model, history, _, test_loader = train_bleeding_detector(
            # TEST_VIDEO_DIR,
            # TEST_LABEL_DIR,
            VIDEO_DIR,
            ANNO_FOLDER,
            output_dir=output_dir,
            epochs=30,
            batch_size=batch_size,
            clip_length=6,
            stride=3,
            volume_csv_path=volume_csv_path,
            dev=f"cuda:{device_num}",
            model_name=model_name,
        )
        plot(history, output_dir)

        # Run final test evaluation (after best model was saved)
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{model_name}_best.pth")))
        model.to(device)

        test_metrics = evaluate_model(
            model,
            test_loader,
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            device,
            volume_loss_weight=0.2
        )
        print("\n📊 Final Test Set Evaluation:")
        for key, value in test_metrics.items():
            print(f"{key:15s}: {value:.4f}")
    else:
        pass
    # TODO finish this for inference only
        # model = VideoBleedingDetector(num_classes=2, severity_levels=4)
        # model.load_state_dict(torch.load(os.path.join(output_dir, f"{model_name}_best.pth")))
        # model.to(device)

        # # Prepare test set again
        # test_dataset = SurgicalVideoDataset(
        #     test_videos, test_annos,
        #     clip_length=6,
        #     stride=3,
        #     transform=val_test_transform,
        #     volume_csv_path=volume_csv_path
        # )
        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=16,
        #     shuffle=False,
        #     num_workers=2,
        #     pin_memory=True
        # )

        # test_metrics = evaluate_model(
        #     model,
        #     test_loader,
        #     nn.CrossEntropyLoss(),
        #     nn.CrossEntropyLoss(),
        #     device,
        #     volume_loss_weight=0.2
        # )

        # print("\n📊 Final Test Set Evaluation:")
        # for key, value in test_metrics.items():
        #     print(f"{key:15s}: {value:.4f}")


if __name__ == "__main__":
    main()
