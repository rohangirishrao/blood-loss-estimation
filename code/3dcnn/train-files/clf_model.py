import argparse
import os
import xml.etree.ElementTree as ET
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
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_grad_cam not available. Install with: pip install grad-cam")
    GRADCAM_AVAILABLE = False
from typing import List, Dict, Tuple, Optional
# import evaluate_video

cv2.setNumThreads(8)
NUM_THREADS = 8
SEED = 42


VIDEO_DIR_1FPS= "/home/r.rohangirish/mt_ble/data/videos"
VIDEO_DIR_2FPS = "/raid/dsl/users/r.rohangirish/data/videos_2_fps"

ANNO_FOLDER_CSV = "/home/r.rohangirish/mt_ble/data/labels_csv"
ANNO_FOLDER_XML = "/home/r.rohangirish/mt_ble/data/labels_xml"

TEST_VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/test_videos"
TEST_LABEL_DIR = "/home/r.rohangirish/mt_ble/data/test_labels"


class VideoBleedingDetector(nn.Module):
    def __init__(self, num_classes=2, severity_levels=4):
        super().__init__()

        # Use R2Plus1D as backbone
        try:
            # Try loading from pytorchvideo hub first
            self.backbone = torch.hub.load(
                "facebookresearch/pytorchvideo", "r2plus1d_r50_16x4x1", pretrained=True
            )
            in_features = 512
            print("Using facebookresearch r2plus1d model from hub")
        except:
            # Fallback to torchvision implementation
            full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(full_model.children())[:-1])
            self.backbone.add_module('flatten', nn.Flatten())
            in_features = 512
            print("Using torchvision r2plus1d_18 model")

        # Classification head
        self.classifier = nn.Linear(in_features, num_classes)

        # Severity prediction
        self.severity_classifier = nn.Linear(in_features, severity_levels)

    def forward(self, x):
        # x: [batch_size, channels, frames, height, width]
        batch_size, _, num_frames, _, _ = x.shape

        # Get features from backbone
        features = self.backbone(x)  # [batch_size, feature_dim]

        # Classification
        clip_pred = self.classifier(features)
        severity_pred = self.severity_classifier(features)

        return clip_pred, severity_pred


class SurgicalVideoDataset(Dataset):
    """
    Dataset for loading surgical video clips with bleeding annotations from XML.
    """

    def __init__(
        self, video_paths, annotation_paths, clip_length=6, stride=3, transform=None, fps_2=False
    ):
        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.fps_2 = fps_2  # Whether to handle 2fps interpolation

        # Create clips with annotations
        self.clips = self._prepare_clips()

    def _prepare_clips(self):
        bleeding_clips = {1: [], 2: [], 3: []}
        non_bleeding_clips = []

        for video_path, anno_path in zip(self.video_paths, self.annotation_paths):
            video_id = os.path.basename(video_path).split(".")[0]

            cap = cv2.VideoCapture(video_path)
            params = [cv2.CAP_PROP_N_THREADS, NUM_THREADS]
            cap.open(video_path, apiPreference=cv2.CAP_FFMPEG, params=params)

            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                print(f"Warning: Video {video_path} has invalid frame count: {frame_count}")
                continue

            # Parse XML annotations instead of CSV
            try:
                bleeding_frames = self._xml_to_frame_labels(self.fps_2, anno_path, frame_count)
                # bleeding_frames = self._parse_bleeding_csv(anno_path, frame_count, fps_2=self.fps_2)
            except Exception as e:
                print(f"Error parsing annotations for {video_path}: {e}")
                continue

            print(f"[DEBUG] {video_id}: bleeding frames found = {np.sum(bleeding_frames > 0)}")

            # Create overlapping clips
            for start_idx in range(0, frame_count - self.clip_length + 1, self.stride):
                end_idx = start_idx + self.clip_length

                # Get bleeding status for this clip
                clip_bleeding = bleeding_frames[start_idx:end_idx]

                if len(clip_bleeding) != self.clip_length:
                    continue

                # Label clip as bleeding if at least 50% of frames have bleeding
                bleeding_frame_count = np.sum(clip_bleeding > 0)
                has_bleeding = bleeding_frame_count >= (self.clip_length * 0.5)

                # Get max severity from the clip
                max_severity = int(np.max(clip_bleeding))

                clip_info = {
                    "video_path": video_path,
                    "video_id": video_id,
                    "start_frame": start_idx,
                    "end_frame": end_idx - 1,
                    "has_bleeding": has_bleeding,
                    "bleeding_frames": clip_bleeding,
                    "max_severity": max_severity,
                }

                if has_bleeding:
                    bleeding_clips[max_severity].append(clip_info)
                else:
                    non_bleeding_clips.append(clip_info)

        # Calculate statistics
        severity_counts = {
            severity: len(clips) for severity, clips in bleeding_clips.items()
        }
        total_bleeding = sum(severity_counts.values())
        total_non_bleeding = len(non_bleeding_clips)

        print("")
        for severity, count in severity_counts.items():
            print(f"  - Severity {severity}: {count} clips")

        # Balance the dataset
        if total_bleeding < total_non_bleeding:
            import random
            random.shuffle(non_bleeding_clips)
            non_bleeding_clips = non_bleeding_clips[:total_bleeding]

        # Combine all clips
        all_clips = non_bleeding_clips.copy()
        for severity, clips in bleeding_clips.items():
            all_clips.extend(clips)

        return all_clips

    @staticmethod
    def _parse_bleeding_csv(csv_path, total_frames, fps_2=True):
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

            # Apply 2fps conversion if needed
            if fps_2:
                # Stretch the frame range to cover 2fps
                # 1fps range [1-6] becomes 2fps range [2-12] (doubled and stretched)
                video_start = start * 2
                video_end = (end * 2) + 1  # Extend to cover the interpolated frame
            else:
                video_start = start
                video_end = end

            # Mark all frames in range with this severity
            if video_start < total_frames and video_end >= 0:
                valid_start = max(0, video_start)
                valid_end = min(total_frames - 1, video_end)

                frame_labels[valid_start : valid_end + 1] = severity

        return frame_labels

    def _parse_xml_annotations(self, xml_path: str) -> List[Dict]:
        """Parse CVAT XML file into a flat list of bleeding frame annotations.
        Does NOT adjust for FPS — raw annotation frame numbers are returned.
        """
        if not os.path.exists(xml_path):
            print(f"DEBUG: XML file does not exist: {xml_path}")
            return []

        annotations = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            meta = root.find("meta")
            if meta is not None:
                original_size = meta.find("original_size")
                if original_size is not None:
                    orig_width = int(original_size.find("width").text)
                    orig_height = int(original_size.find("height").text)
                else:
                    orig_width = orig_height = 1920
            else:
                orig_width = orig_height = 1920

            track_count = 0
            total_boxes = 0
            valid_boxes = 0

            for track in root.findall("track"):
                track_label = track.get("label")
                track_id = track.get("id")
                track_count += 1

                if track_label not in ["BL_Low", "BL_Medium", "BL_High"]:
                    continue

                boxes_in_track = track.findall("box")
                for box in boxes_in_track:
                    total_boxes += 1
                    if int(box.get("outside", "0")) == 1:
                        continue

                    frame_num = int(box.get("frame"))

                    annotations.append({
                        "frame": frame_num,
                        "label": track_label,
                        "original_width": orig_width,
                        "original_height": orig_height,
                    })
                    valid_boxes += 1

            print(f"DEBUG: Tracks={track_count}  Boxes={total_boxes}  Bleeding boxes={valid_boxes}")

        except Exception as e:
            print(f"ERROR parsing {xml_path}: {e}")
            import traceback
            traceback.print_exc()
            return []

        return annotations



    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]

        frames = self._load_clip(
            clip_info["video_path"],
            clip_info["start_frame"],
            clip_info["end_frame"] + 1,
        )

        if len(frames) < self.clip_length:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames.extend([last_frame] * (self.clip_length - len(frames)))

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        clip_tensor = torch.stack(frames)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]

        clip_label = 1 if clip_info["has_bleeding"] else 0
        frame_labels = torch.tensor(clip_info["bleeding_frames"], dtype=torch.float32)
        severity = torch.tensor(clip_info["max_severity"], dtype=torch.long)

        return clip_tensor, clip_label, frame_labels, severity

    def _xml_to_frame_labels(self, fps_2: bool, xml_path: str, total_frames: int) -> np.ndarray:
        """Convert XML annotations to per-frame bleeding labels."""
        print(f"DEBUG: Converting XML to frame labels (fps_2={fps_2})")
        print(f"DEBUG: Total frames in video: {total_frames}")

        frame_labels = np.zeros(total_frames, dtype=np.int32)
        severity_map = {"BL_Low": 1, "BL_Medium": 2, "BL_High": 3}

        annotations = self._parse_xml_annotations(xml_path)
        print(f"DEBUG: Processing {len(annotations)} raw annotations from XML")

        annotations_applied = 0
        out_of_bounds = 0

        for anno in annotations:
            anno_frame = anno["frame"]
            severity = severity_map.get(anno["label"], 1)

            if not fps_2:
                # 1 FPS → use frame index directly
                if 0 <= anno_frame < total_frames:
                    frame_labels[anno_frame] = max(frame_labels[anno_frame], severity)
                    annotations_applied += 1
                else:
                    out_of_bounds += 1
            else:
                # 2 FPS → expand each annotation to two consecutive frames
                frame_a = anno_frame * 2
                frame_b = frame_a + 1

                if 0 <= frame_a < total_frames:
                    frame_labels[frame_a] = max(frame_labels[frame_a], severity)
                    annotations_applied += 1
                else:
                    out_of_bounds += 1

                if 0 <= frame_b < total_frames:
                    frame_labels[frame_b] = max(frame_labels[frame_b], severity)
                    annotations_applied += 1
                else:
                    out_of_bounds += 1

        print(f"DEBUG: Applied {annotations_applied} frame labels")
        print(f"DEBUG: {out_of_bounds} were out of bounds")

        # Summary
        unique, counts = np.unique(frame_labels, return_counts=True)
        severity_names = {0: "No bleeding", 1: "BL_Low", 2: "BL_Medium", 3: "BL_High"}
        for val, count in zip(unique, counts):
            pct = (count / len(frame_labels)) * 100
            print(f"  {severity_names.get(val, str(val))}: {count} frames ({pct:.1f}%)")

        return frame_labels

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
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    input_size=(200,320)
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
            f"[bold cyan]🩸 Bleeding Detection Training[/bold cyan]\n"
            f"Device: [yellow]{device}[/yellow]\n"
            f"Output: [green]{output_dir}[/green]",
            title="🚀 Starting Training",
        )
    )

    # Get video and annotation paths
    all_video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    # Build matched pairs only
    video_paths = []
    annotation_paths = []

    for vid_path in all_video_paths:
        vid_name = os.path.basename(vid_path).split(".")[0]
        # anno_path = os.path.join(annotation_dir, f"{vid_name}.xml")
        anno_path = os.path.join(annotation_dir, f"{vid_name}_annotations.csv")
        if os.path.exists(anno_path):
            video_paths.append(vid_path)
            annotation_paths.append(anno_path)
            console.print(f"[green]✓[/green] Matched: {vid_name}")
        else:
            console.print(
                f"[yellow]⚠ Skipping {vid_name} - no annotation file found[/yellow]"
            )

    console.print(
        f"[green]Found {len(video_paths)} videos with matching annotations[/green]"
    )

    # Verify alignment
    assert len(video_paths) == len(annotation_paths), "Mismatch in video and annotation counts!"

    # Debug: Print first few pairs to verify alignment
    console.print("\n[cyan]First few video-annotation pairs:[/cyan]")
    for i in range(min(3, len(video_paths))):
        vid_name = os.path.basename(video_paths[i]).split(".")[0]
        anno_name = os.path.basename(annotation_paths[i]).split(".")[0]
        if vid_name == anno_name:
            console.print(f"  ✓ {vid_name} <-> {anno_name}")
        else:
            console.print(f"  [red]✗ MISMATCH: {vid_name} <-> {anno_name}[/red]")

    # ----- VIDEO-LEVEL SPLITTING -----
    unique_videos = list(
        set(os.path.basename(path).split(".")[0] for path in video_paths)
    )

    # Simple random split by video
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
            transforms.Resize(input_size),
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
            transforms.Resize(input_size),
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
        fps_2=False,
    )
    val_dataset = SurgicalVideoDataset(
        val_videos,
        val_annos,
        clip_length=clip_length,
        stride=stride,
        transform=val_test_transform,
        fps_2=False,
    )
    test_dataset = SurgicalVideoDataset(
        test_videos,
        test_annos,
        clip_length=clip_length,
        stride=stride,
        transform=val_test_transform,
        fps_2=False
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
        "val_loss": [],
        "val_clf_loss": [],
        "val_sev_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_f1 = 0.0
    best_model_path = None

    # Training with Rich progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    ) as progress:

        # Main epoch progress
        epoch_task = progress.add_task("[cyan]🏋️ Training Progress", total=epochs)

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

            for clips, clip_labels, frame_labels, severity in train_loader:
                clips = clips.to(device)
                clip_labels = clip_labels.to(device)
                severity = severity.to(device)

                # Forward pass
                clip_preds, severity_preds = model(clips)
                clip_labels = clip_labels.long()
                severity = severity.clamp(0, 3).long()

                # Calculate losses
                loss_clf = criterion_clf(clip_preds, clip_labels)
                loss_sev = criterion_sev(severity_preds, severity)
                loss = loss_clf + 0.3 * loss_sev

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

    return model, history, test_metrics


def evaluate_model(model, data_loader, criterion_clf, criterion_sev, device):
    """
    Evaluate model on given data loader
    """

    model.eval()
    total_loss = 0.0
    total_clf_loss = 0.0
    total_sev_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clips, clip_labels, frame_labels, severity in data_loader:
            clips = clips.to(device)
            clip_labels = clip_labels.to(device)
            severity = severity.to(device)

            # Forward pass
            clip_preds, severity_preds = model(clips)

            # Calculate losses
            loss_clf = criterion_clf(clip_preds, clip_labels)
            loss_sev = criterion_sev(severity_preds, severity)
            loss = loss_clf + 0.3 * loss_sev

            total_loss += loss.item()
            total_clf_loss += loss_clf.item()
            total_sev_loss += loss_sev.item()

            # Get predictions
            _, predicted = torch.max(clip_preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(clip_labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    avg_clf_loss = total_clf_loss / len(data_loader)
    avg_sev_loss = total_sev_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": avg_loss,
        "clf_loss": avg_clf_loss,
        "sev_loss": avg_sev_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot(history, output_dir):
    """
    Simple plot with 4 key metrics in subplots
    """
    import matplotlib.pyplot as plt
    import os

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

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

    # 3. Severity Loss (validation)
    vsl = history.get("val_sev_loss", [])
    if vsl:
        axes[2].plot(epochs, vsl, "g-", linewidth=2)
        axes[2].set_title("Validation Severity Loss")
        axes[2].set_ylabel("Loss")
        axes[2].grid(True, alpha=0.3)

    # 4. Accuracy (bleeding detection)
    vba = history.get("val_bleeding_accuracy", [])
    if vba:
        axes[3].plot(epochs, vba, "purple", linewidth=2)
        axes[3].set_title("Validation Bleeding Accuracy")
        axes[3].set_ylabel("Accuracy")
        axes[3].set_xlabel("Epochs")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "training_metrics.png"), dpi=400, bbox_inches="tight"
    )
    plt.show()


def infer_videos(
    model_path,
    video_dir,
    annotation_dir,
    video_names,
    clip_length=12,
    batch_size=8,
    device="cuda:0",
    fps_2=False,
    viz_dir="./gradcam_videos"
):
    console = Console()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    os.makedirs(viz_dir, exist_ok=True)

    # Load model
    model = VideoBleedingDetector(num_classes=2, severity_levels=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Wrap model so GradCAM sees only classification logits
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            return self.base_model(x)[0]  # only logits

    gradcam_model = ModelWrapper(model)

    # Fixed: Find the correct target layer for GradCAM
    target_layers = []

    # Method 1: Try to find the last conv layer automatically
    def find_last_conv_layer(module):
        last_conv = None
        for name, child in module.named_modules():
            if isinstance(child, (nn.Conv3d, nn.Conv2d)):
                last_conv = child
        return last_conv

    # Try different approaches to find target layer
    if hasattr(model.backbone, 'layer4'):
        # Standard ResNet structure
        target_layers = [model.backbone.layer4[-1]]
    elif hasattr(model.backbone, 'blocks'):
        # Some other backbone structures
        target_layers = [model.backbone.blocks[-1]]
    else:
        # Sequential backbone - find last conv layer
        last_conv = find_last_conv_layer(model.backbone)
        if last_conv is not None:
            target_layers = [last_conv]
            console.print(f"[yellow]Found target layer: {last_conv}[/yellow]")
        else:
            console.print("[red]Could not find suitable conv layer for GradCAM![/red]")
            return {}

    if not target_layers:
        console.print("[red]No target layers found for GradCAM![/red]")
        return {}

    try:
        cam = GradCAM(model=gradcam_model, target_layers=target_layers)
        console.print(f"[green]GradCAM initialized successfully[/green]")
    except Exception as e:
        console.print(f"[red]Failed to initialize GradCAM: {e}[/red]")
        console.print("[yellow]Continuing without GradCAM visualizations...[/yellow]")
        cam = None

    # For metrics
    results = {}
    all_preds_all = []
    all_labels_all = []
    vid_num = 0

    # Table setup
    table = Table(title="Per-Video Evaluation")
    table.add_column("Video", style="cyan")
    table.add_column("Acc", justify="right", style="green")
    table.add_column("Prec", justify="right", style="yellow")
    table.add_column("Rec", justify="right", style="blue")
    table.add_column("F1", justify="right", style="magenta")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("TN", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("BL Clips", justify="right")
    table.add_column("Total Clips", justify="right")

    for video_name in video_names:
        console.print(f"\n[bold cyan]=== Testing video: {video_name} ===[/bold cyan]")

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        anno_path = os.path.join(annotation_dir, f"{video_name}.xml")

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((140, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        dataset = SurgicalVideoDataset(
            [video_path],
            [anno_path],
            clip_length=clip_length,
            stride=clip_length,
            fps_2=fps_2,
            transform=test_transform,
        )

        total_bleeding_clips = sum(1 for c in dataset.clips if c["has_bleeding"])

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        all_preds, all_labels = [], []
        clip_data = []
        clip_idx = 0

        with torch.no_grad(), Progress(
            TextColumn("[bold blue]Inferencing..."),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} clips"),
            console=console
        ) as progress:
            task = progress.add_task("test", total=len(dataset))

            for clips, clip_labels, _, _ in loader:
                clips = clips.to(device)
                clip_labels = clip_labels.to(device)

                preds, _ = model(clips)
                pred_classes = torch.argmax(preds, dim=1)

                all_preds.extend(pred_classes.cpu().numpy())
                all_labels.extend(clip_labels.cpu().numpy())

                for i in range(clips.size(0)):
                    clip_data.append({
                        "clip_tensor": clips[i:i+1],  # keep batch dim for GradCAM
                        "pred": pred_classes[i].item(),
                        "label": clip_labels[i].item(),
                        "confidence": torch.softmax(preds[i], dim=0)[1].item(),
                        "clip_info": dataset.clips[clip_idx+ i]
                    })
                clip_idx += clips.size(0)
                progress.update(task, advance=clips.size(0))

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        tp = sum((p == 1) and (l == 1) for p, l in zip(all_preds, all_labels))
        fp = sum((p == 1) and (l == 0) for p, l in zip(all_preds, all_labels))
        tn = sum((p == 0) and (l == 0) for p, l in zip(all_preds, all_labels))
        fn = sum((p == 0) and (l == 1) for p, l in zip(all_preds, all_labels))

        table.add_row(video_name, f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}",
                      str(tp), str(fp), str(tn), str(fn),
                      str(total_bleeding_clips), str(len(dataset)))

        results[video_name] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total_bleeding_clips": total_bleeding_clips,
            "total_clips": len(dataset)
        }

        all_preds_all.extend(all_preds)
        all_labels_all.extend(all_labels)

        # --- GradCAM Clips (only if cam is available) ---
        if cam is not None and vid_num == 0:
            tp_clips = [c for c in clip_data if c["pred"] == 1 and c["label"] == 1]
            fp_clips = [c for c in clip_data if c["pred"] == 1 and c["label"] == 0]
            tn_clips = [c for c in clip_data if c["pred"] == 0 and c["label"] == 0]

            def save_gradcam_videos(clips_subset, label):
                clips_subset = sorted(clips_subset, key=lambda x: x["confidence"], reverse=True)[:5]
                for idx, c in enumerate(clips_subset):
                    try:
                        # Generate GradCAM on the model input
                        grayscale_cam = cam(input_tensor=c["clip_tensor"], targets=None)  # (B,T,H,W)
                        grayscale_cam = grayscale_cam[0]  # remove batch dim -> (T,H,W)

                        # Get the clip info to find which frames in the original video
                        clip_info = c.get('clip_info')  # You'll need to pass this in clip_data
                        if not clip_info:
                            console.print(f"[yellow]No clip info for {label}_{idx}, skipping[/yellow]")
                            continue

                        video_path = clip_info['video_path']
                        start_frame = clip_info['start_frame']
                        end_frame = clip_info['end_frame']

                        # console.print(f"[yellow]Loading original frames from {video_path}, frames {start_frame}-{end_frame}[/yellow]")

                        # Load original high-res frames from video
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            console.print(f"[red]Could not open video: {video_path}[/red]")
                            continue

                        # Get video properties
                        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        # console.print(f"[yellow]Original video: {original_width}x{original_height} @ {fps}fps[/yellow]")

                        # Set to start frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                        # Read the actual frames
                        original_frames = []
                        for frame_idx in range(clip_length):
                            ret, frame = cap.read()
                            if not ret:
                                console.print(f"[yellow]Could not read frame {start_frame + frame_idx}[/yellow]")
                                break

                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            original_frames.append(frame_rgb)

                        cap.release()

                        if len(original_frames) != clip_length:
                            console.print(f"[yellow]Only got {len(original_frames)}/{clip_length} frames[/yellow]")
                            continue

                        # Create output video with original resolution
                        out_path = os.path.join(viz_dir, f"{video_name}_{label}_{idx}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(out_path, fourcc, 3.0, (original_width, original_height))

                        if not writer.isOpened():
                            console.print(f"[red]Failed to open video writer for {out_path}[/red]")
                            continue

                        for t in range(len(original_frames)):
                            # Get original high-res frame
                            original_frame = original_frames[t]  # (H, W, C) in RGB, uint8

                            # Get corresponding CAM (low-res from model)
                            if t < grayscale_cam.shape[0]:
                                cam_mask = grayscale_cam[t]  # (140, 256) typically
                            else:
                                # Use last available CAM if we have more frames than CAM outputs
                                cam_mask = grayscale_cam[-1]

                            # Upscale CAM to match original video resolution
                            cam_upscaled = cv2.resize(
                                cam_mask,
                                (original_width, original_height),
                                interpolation=cv2.INTER_CUBIC
                            )

                            # Normalize CAM to [0,1] range
                            cam_upscaled = (cam_upscaled - cam_upscaled.min()) / (cam_upscaled.max() - cam_upscaled.min() + 1e-8)

                            # Create heatmap using colormap
                            try:
                                colormap = plt.colormaps['plasma']  # or 'jet', 'hot', 'plasma'
                            except AttributeError:
                                colormap = cm.get_cmap('plasma')

                            heatmap_colored = colormap(cam_upscaled)[:, :, :3]  # Remove alpha
                            heatmap_rgb = (heatmap_colored * 255).astype(np.uint8)

                            # Create transparent overlay
                            alpha = 0.4  # Adjust transparency (0.2-0.5 works well)
                            overlay = cv2.addWeighted(original_frame, 1-alpha, heatmap_rgb, alpha, 0)

                            # Add text overlays with better positioning for high-res
                            font_scale = max(0.8, original_width / 1000)  # Scale font with resolution
                            thickness = max(2, int(original_width / 640))

                            def add_text_with_outline(img, text, pos, scale=font_scale, thick=thickness):
                                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick+2)
                                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)

                            y_offset = int(40 * (original_height / 400))  # Scale with height
                            line_spacing = int(40 * (original_height / 400))

                            # add_text_with_outline(overlay, f"{label} - Frame {t+1}/{len(original_frames)}",
                            #                     (20, y_offset))
                            # add_text_with_outline(overlay, f"Confidence: {c['confidence']:.3f}",
                            #                     (20, y_offset + line_spacing))
                            # add_text_with_outline(overlay, f"Pred: {'1' if c['pred'] == 1 else '0'}",
                            #                     (20, y_offset + 2*line_spacing))
                            # add_text_with_outline(overlay, f"Truth: {'1' if c['label'] == 1 else '0'}",
                            #                     (20, y_offset + 3*line_spacing))

                            # Add intensity scale bar (scaled for resolution)
                            bar_width = int(200 * (original_width / 1280))
                            bar_height = int(20 * (original_height / 800))
                            bar_x = original_width - bar_width - 20
                            bar_y = 20

                            # Create and add intensity bar
                            intensity_bar = np.linspace(0, 1, bar_width).reshape(1, -1)
                            intensity_bar = np.repeat(intensity_bar, bar_height, axis=0)
                            intensity_colored = colormap(intensity_bar)[:, :, :3]
                            intensity_rgb = (intensity_colored * 255).astype(np.uint8)

                            overlay[bar_y:bar_y+bar_height, bar_x:bar_x+bar_width] = intensity_rgb

                            # Bar labels
                            cv2.putText(overlay, "Low", (bar_x-60, bar_y+15),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.6, (255, 255, 255), thickness)
                            cv2.putText(overlay, "High", (bar_x+bar_width+10, bar_y+15),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.6, (255, 255, 255), thickness)

                            # Convert to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                            writer.write(frame_bgr)

                        writer.release()
                        # console.print(f"[green]Saved high-res GradCAM: {out_path} ({original_width}x{original_height})[/green]")

                    except Exception as e:
                        console.print(f"[red]Error creating high-res GradCAM for {label}_{idx}: {e}[/red]")
                        import traceback
                        traceback.print_exc()



            console.print(f"\n[bold green]Saving GradCAM clips to {viz_dir}[/bold green]")
            save_gradcam_videos(tp_clips, "TP")
            save_gradcam_videos(fp_clips, "FP")
            save_gradcam_videos(tn_clips, "TN")

        vid_num += 1

    console.print(table)

    # Overall metrics
    overall_acc = accuracy_score(all_labels_all, all_preds_all)
    overall_prec = precision_score(all_labels_all, all_preds_all, zero_division=0)
    overall_rec = recall_score(all_labels_all, all_preds_all, zero_division=0)
    overall_f1 = f1_score(all_labels_all, all_preds_all, zero_division=0)

    console.print(f"\n[bold green]Overall Accuracy:[/bold green] {overall_acc:.3f}")
    console.print(f"[bold green]Overall Precision:[/bold green] {overall_prec:.3f}")
    console.print(f"[bold green]Overall Recall:[/bold green] {overall_rec:.3f}")
    console.print(f"[bold green]Overall F1:[/bold green] {overall_f1:.3f}")

    return results



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
        help="Whether to train the model (default: true)",
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Parse arguments
    args = parser.parse_args()
    device_num = args.dev
    do_train = args.train.lower() == "true"
    output_dir = "./clf_models"
    device = torch.device(f"cuda:{device_num}")
    model_name = "2d1_clf_1fps"

    # Train model if specified
    if do_train:
        model, history, _ = train_bleeding_detector(
            # TEST_VIDEO_DIR,
            # TEST_LABEL_DIR,
            VIDEO_DIR_1FPS,
            ANNO_FOLDER_XML,
            output_dir=output_dir,
            epochs=20,
            batch_size=32,
            clip_length=6,
            stride=3,
            dev=f"cuda:{device_num}",
            model_name=model_name,
        )
        plot(history, output_dir)
    else:
        # Test specific videos
        model_path = os.path.join(output_dir, f"{model_name}_best.pth")
        video_names = [
            "OwRK",
        ]

        infer_videos(
            model_path=model_path,
            video_dir=VIDEO_DIR_1FPS,
            annotation_dir=ANNO_FOLDER_XML,
            video_names=video_names,
            clip_length=6,
            batch_size=16,
            device=f"cuda:{device_num}",
            fps_2=False
        )

if __name__ == "__main__":
    main()
