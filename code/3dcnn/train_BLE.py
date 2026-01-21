import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
import numpy as np
from rich.console import Console
import cv2
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import glob
from rich.table import Table
from rich.panel import Panel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import json
from rich.text import Text
from rich import print as rprint
from models import VolumeSequenceModel, MultiTaskVolumeSequenceModel

# from train_volume_sequence import test_sequence_model, test_trained_model

console = Console()
# SEED = 21

# Global variables set from config
FPS_2 = False
LR_BB = 3e-5
LR_LSTM = 1e-4


# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class VolumeSequenceDataset_NoVolumeWeights(Dataset):
    """
    Dataset that groups clips into sequences for temporal modeling.
    Simple binary weighting: bleeding vs non-bleeding sequences only.
    """

    def __init__(
        self,
        video_paths: List[str],
        annotation_paths: List[str],
        volume_csv_path: str,
        clip_length: int = 12,
        clips_per_sequence: int = 10,
        stride: int = 6,
        transform=None,
        max_sequences_per_video: Optional[int] = None,
    ):
        self.clip_length = clip_length
        self.clips_per_sequence = clips_per_sequence
        self.stride = stride
        self.transform = transform
        self.max_sequences_per_video = max_sequences_per_video

        # Load volume data
        self.volume_data = self._load_volume_csv(volume_csv_path)

        # Phase 1: Create all sequences with basic info
        self.all_sequences = self._prepare_all_sequences(video_paths, annotation_paths)

        # Phase 2: Assign volumes using checkpoint uniform distribution
        self._assign_all_volumes_by_video()

        # Phase 3: Assign simple binary weights
        self._assign_binary_weights()

        # Display summary
        self._print_summary()

    def _load_volume_csv(self, path: str) -> Dict:
        """Load volume measurements from CSV"""
        df = pd.read_csv(path)
        df = df.dropna(subset=["video_name"])

        data = defaultdict(lambda: {"checkpoints": [], "total_volume": 0.0})

        for _, row in df.iterrows():
            video_name = str(row["video_name"])

            if pd.notna(row.get("a_e_bl")):
                total_volume = float(row["a_e_bl"])
                if total_volume > 0.0:
                    data[video_name]["total_volume"] = total_volume

            if pd.notna(row.get("measurement_frame")) and pd.notna(row.get("bl_loss")):
                measurement_frame = int(row["measurement_frame"])
                if FPS_2:
                    measurement_frame *= 2
                cumulative_volume = float(row["bl_loss"])
                data[video_name]["checkpoints"].append(
                    (measurement_frame, cumulative_volume)
                )

        for video_name in data:
            data[video_name]["checkpoints"].sort(key=lambda x: x[0])

        return data

    def _parse_xml_annotations(self, xml_path: str) -> List[Dict]:
        """Parse CVAT XML file"""
        if not os.path.exists(xml_path):
            return []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotations = []

            for track in root.findall("track"):
                track_label = track.get("label")
                if track_label in ["BL_Low", "BL_Medium", "BL_High"]:
                    last_frame = None
                    for box in track.findall("box"):
                        if int(box.get("outside", "0")) == 1:
                            continue

                        frame_num = int(box.get("frame"))
                        if FPS_2:
                            frame_num *= 2

                        annotations.append({"frame": frame_num, "label": track_label})

                        if (
                            FPS_2
                            and last_frame is not None
                            and frame_num - last_frame == 2
                        ):
                            mid_frame = last_frame + 1
                            annotations.append(
                                {"frame": mid_frame, "label": track_label}
                            )

                        last_frame = frame_num

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return []

        return annotations

    def _xml_to_frame_labels(self, xml_path: str, total_frames: int) -> np.ndarray:
        """Convert XML annotations to per-frame bleeding labels"""
        frame_count = total_frames * 2 if FPS_2 else total_frames
        frame_labels = np.zeros(frame_count, dtype=np.int32)

        severity_map = {"BL_Low": 1, "BL_Medium": 2, "BL_High": 3}
        annotations = self._parse_xml_annotations(xml_path)

        for anno in annotations:
            frame_num = anno["frame"]
            severity = severity_map.get(anno["label"], 1)
            if 0 <= frame_num < len(frame_labels):
                frame_labels[frame_num] = max(frame_labels[frame_num], severity)

        return frame_labels

    def _prepare_all_sequences(
        self, video_paths: List[str], annotation_paths: List[str]
    ) -> List[Dict]:
        """Create all sequences without volume assignments"""
        all_sequences = []

        for video_path, anno_path in zip(video_paths, annotation_paths):
            video_id = os.path.basename(video_path).split(".")[0]

            if video_id not in self.volume_data:
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            bleeding_labels = self._xml_to_frame_labels(anno_path, frame_count)

            # Create clips
            all_clips = []
            for start_idx in range(
                0, frame_count - self.clip_length + 1, self.clip_length
            ):
                end_idx = start_idx + self.clip_length
                center_frame = (start_idx + end_idx) // 2
                clip_bleeding = bleeding_labels[start_idx:end_idx]

                bleeding_ratio = np.sum(clip_bleeding > 0) / len(clip_bleeding)
                has_bleeding = bleeding_ratio >= 0.5

                clip_info = {
                    "video_id": video_id,
                    "video_path": video_path,
                    "start_frame": start_idx,
                    "end_frame": end_idx - 1,
                    "center_frame": center_frame,
                    "has_bleeding": has_bleeding,
                    "max_severity": int(np.max(clip_bleeding)),
                    "bleeding_frames": clip_bleeding.tolist(),
                }
                all_clips.append(clip_info)

            # Create sequences
            for seq_start_idx in range(
                0, len(all_clips) - self.clips_per_sequence + 1, self.clips_per_sequence
            ):
                sequence_clips = all_clips[
                    seq_start_idx : seq_start_idx + self.clips_per_sequence
                ]

                bleeding_clip_count = sum(
                    1 for clip in sequence_clips if clip["has_bleeding"]
                )
                bleeding_ratio = bleeding_clip_count / len(sequence_clips)
                has_bleeding_sequence = bleeding_ratio >= 0.30

                sequence_info = {
                    "video_id": video_id,
                    "sequence_clips": sequence_clips,
                    "sequence_start_frame": sequence_clips[0]["start_frame"],
                    "sequence_end_frame": sequence_clips[-1]["end_frame"],
                    "sequence_center_frame": (
                        sequence_clips[0]["start_frame"]
                        + sequence_clips[-1]["end_frame"]
                    )
                    // 2,
                    "has_bleeding": has_bleeding_sequence,
                    "target_sequence_volume": 0.0,
                    "video_total_volume": self.volume_data[video_id]["total_volume"],
                    "training_weight": 1.0,
                }

                all_sequences.append(sequence_info)

        return all_sequences

    def _assign_all_volumes_by_video(self):
        """Assign volumes to all sequences using the checkpoint uniform distribution method"""
        sequences_by_video = defaultdict(list)
        for seq in self.all_sequences:
            sequences_by_video[seq["video_id"]].append(seq)

        for video_id, video_sequences in sequences_by_video.items():
            volume_info = self.volume_data.get(
                video_id, {"checkpoints": [], "total_volume": 0.0}
            )
            checkpoints = volume_info["checkpoints"].copy()
            total_volume = volume_info["total_volume"]

            # Handle final segment
            sum_incremental = sum(v for _, v in checkpoints)
            if abs(total_volume - sum_incremental) > 0.1:
                if checkpoints:
                    last_frame = max(f for f, _ in checkpoints)
                    max_frame = max(
                        seq["sequence_end_frame"] for seq in video_sequences
                    )
                    if max_frame > last_frame:
                        remaining = total_volume - sum_incremental
                        checkpoints.append((max_frame, remaining))
                else:
                    max_frame = max(
                        seq["sequence_end_frame"] for seq in video_sequences
                    )
                    checkpoints.append((max_frame, total_volume))

            self._assign_sequence_volume_uniform(
                video_sequences, checkpoints, total_volume, video_id
            )

    def _assign_sequence_volume_uniform(
        self,
        video_sequences: List[Dict],
        checkpoints: List[Tuple[int, float]],
        total_volume: float,
        video_id: str,
    ):
        """Assign volume based on checkpoint segments with uniform distribution"""
        for seq in video_sequences:
            seq["target_sequence_volume"] = 0.0

        if not checkpoints:
            bleeding_sequences = [seq for seq in video_sequences if seq["has_bleeding"]]
            if bleeding_sequences:
                volume_per_sequence = total_volume / len(bleeding_sequences)
                for seq in bleeding_sequences:
                    seq["target_sequence_volume"] = volume_per_sequence
            return

        checkpoints = sorted(checkpoints, key=lambda x: x[0])
        segments = [(0, checkpoints[0][0], checkpoints[0][1])]

        for i in range(1, len(checkpoints)):
            start_frame = checkpoints[i - 1][0]
            end_frame = checkpoints[i][0]
            incremental_volume = checkpoints[i][1]
            segments.append((start_frame, end_frame, incremental_volume))

        for seg_start, seg_end, seg_volume in segments:
            bleeding_sequences_in_segment = []
            for seq in video_sequences:
                seq_center = seq["sequence_center_frame"]
                if seg_start <= seq_center < seg_end and seq["has_bleeding"]:
                    bleeding_sequences_in_segment.append(seq)

            if bleeding_sequences_in_segment:
                volume_per_sequence = seg_volume / len(bleeding_sequences_in_segment)
                for seq in bleeding_sequences_in_segment:
                    seq["target_sequence_volume"] += volume_per_sequence

    def _assign_binary_weights(self):
        """Assign simple binary weights: bleeding vs non-bleeding sequences"""
        bleeding_count = sum(1 for seq in self.all_sequences if seq["has_bleeding"])
        non_bleeding_count = len(self.all_sequences) - bleeding_count

        if bleeding_count == 0 or non_bleeding_count == 0:
            # If only one class, use uniform weights
            for seq in self.all_sequences:
                seq["training_weight"] = 1.0
            return

        # Calculate inverse frequency weights
        total_sequences = len(self.all_sequences)
        bleeding_weight = total_sequences / (2.0 * bleeding_count)
        non_bleeding_weight = total_sequences / (2.0 * non_bleeding_count)

        # Assign weights
        for seq in self.all_sequences:
            if seq["has_bleeding"]:
                seq["training_weight"] = bleeding_weight
            else:
                seq["training_weight"] = non_bleeding_weight

        console.print(f"\n Binary Class Weights:")
        console.print(
            f"  Bleeding sequences: {bleeding_count:,} → {bleeding_weight:.2f}x weight"
        )
        console.print(
            f"  Non-bleeding sequences: {non_bleeding_count:,} → {non_bleeding_weight:.2f}x weight"
        )

    def _print_summary(self):
        """Print concise dataset summary"""
        videos = set(seq["video_id"] for seq in self.all_sequences)
        bleeding_sequences = [seq for seq in self.all_sequences if seq["has_bleeding"]]
        non_bleeding_sequences = [
            seq for seq in self.all_sequences if not seq["has_bleeding"]
        ]

        volumes = [seq["target_sequence_volume"] for seq in bleeding_sequences]

        console.print(f"\n📊 Dataset Summary:")
        console.print(
            f"  Videos: {len(videos)} | Sequences: {len(self.all_sequences):,}"
        )
        console.print(
            f"  Bleeding: {len(bleeding_sequences):,} ({len(bleeding_sequences)/len(self.all_sequences)*100:.1f}%)"
        )
        console.print(
            f"  Non-bleeding: {len(non_bleeding_sequences):,} ({len(non_bleeding_sequences)/len(self.all_sequences)*100:.1f}%)"
        )

        if volumes:
            console.print(f"  Volume range: {min(volumes):.2f} - {max(volumes):.2f} ml")

        console.print("-" * 50)

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, torch.Tensor, float, torch.Tensor]:
        """Get a sequence of clips with all labels"""
        sequence_info = self.all_sequences[idx]
        sequence_clips = sequence_info["sequence_clips"]

        clip_tensors = []
        clip_labels = []

        for clip_info in sequence_clips:
            frames = self._load_clip_frames(
                clip_info["video_path"],
                clip_info["start_frame"],
                clip_info["end_frame"] + 1,
            )

            if len(frames) < self.clip_length:
                last_frame = (
                    frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
                )
                frames.extend([last_frame] * (self.clip_length - len(frames)))

            if self.transform:
                frames = [self.transform(frame) for frame in frames]

            clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)
            clip_tensors.append(clip_tensor)
            clip_labels.append(1 if clip_info["has_bleeding"] else 0)

        sequence_tensor = torch.stack(clip_tensors)
        clip_labels_tensor = torch.tensor(clip_labels, dtype=torch.long)

        has_bleeding = 1 if sequence_info["has_bleeding"] else 0
        target_volume = torch.tensor(
            sequence_info["target_sequence_volume"], dtype=torch.float32
        )
        training_weight = sequence_info["training_weight"]

        return (
            sequence_tensor,
            has_bleeding,
            target_volume,
            training_weight,
            clip_labels_tensor,
        )

    def _load_clip_frames(
        self, video_path: str, start_frame: int, end_frame: int
    ) -> List[np.ndarray]:
        """Load specific frames from video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
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


def compute_multitask_loss(
    clip_preds,
    volume_pred,
    sequence_cls_pred,
    clip_labels,
    volume_targets,
    sequence_labels,
    alpha=1.0,
    beta=0.5,
    gamma=0.3,
):
    B, S = clip_labels.shape

    # Clip-level classification loss
    clip_preds_flat = clip_preds.view(B * S, -1)  # [B*S, num_classes]
    clip_labels_flat = clip_labels.view(-1)  # [B*S]
    clip_loss = F.cross_entropy(clip_preds_flat, clip_labels_flat)

    # Volume regression loss (unweighted)
    volume_loss = F.smooth_l1_loss(volume_pred, volume_targets)

    # Sequence classification loss
    sequence_loss = F.cross_entropy(sequence_cls_pred, sequence_labels)

    # Combined loss
    total_loss = alpha * clip_loss + beta * volume_loss + gamma * sequence_loss

    return total_loss, {
        "clip_loss": clip_loss.item(),
        "volume_loss": volume_loss.item(),
        "sequence_loss": sequence_loss.item(),
        "total_loss": total_loss.item(),
    }


def get_dataset_class_weights(dataset):
    """
    Extract class distribution from your existing dataset to calculate balanced weights
    """
    bleeding_sequences = [seq for seq in dataset.all_sequences if seq["has_bleeding"]]
    non_bleeding_sequences = [
        seq for seq in dataset.all_sequences if not seq["has_bleeding"]
    ]

    total_bleeding = len(bleeding_sequences)
    total_non_bleeding = len(non_bleeding_sequences)
    total_samples = total_bleeding + total_non_bleeding

    # Calculate inverse frequency weights
    if total_bleeding > 0 and total_non_bleeding > 0:
        bleeding_weight = total_samples / (2.0 * total_bleeding)
        non_bleeding_weight = total_samples / (2.0 * total_non_bleeding)
    else:
        # Fallback if only one class exists
        bleeding_weight = non_bleeding_weight = 1.0

    # Create nice display table
    table = Table(title="Dataset Class Distribution & Weights")
    table.add_column("Class", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    table.add_column("Weight", justify="right", style="magenta")

    bleeding_ratio = total_bleeding / total_samples * 100
    non_bleeding_ratio = total_non_bleeding / total_samples * 100

    table.add_row(
        "Bleeding",
        f"{total_bleeding:,}",
        f"{bleeding_ratio:.1f}%",
        f"{bleeding_weight:.3f}",
    )
    table.add_row(
        "Non-bleeding",
        f"{total_non_bleeding:,}",
        f"{non_bleeding_ratio:.1f}%",
        f"{non_bleeding_weight:.3f}",
    )
    table.add_row("Total", f"{total_samples:,}", "100.0%", "-")

    console.print(table)
    console.print(
        f"[bold]Weight Ratio (bleeding/non-bleeding): {bleeding_weight/non_bleeding_weight:.2f}x[/bold]"
    )

    return {
        "bleeding_weight": bleeding_weight,
        "non_bleeding_weight": non_bleeding_weight,
        "total_bleeding": total_bleeding,
        "total_non_bleeding": total_non_bleeding,
        "bleeding_ratio": bleeding_ratio,
        "weight_ratio": bleeding_weight / non_bleeding_weight,
    }


def create_dataloaders(
    video_dir: str,
    annotation_dir: str,
    volume_csv_path: str,
    batch_size: int = 8,  # Smaller batch size due to larger sequences
    num_workers: int = 4,
    clip_length: int = 12,  # frames per clip (6 seconds at 2fps)
    clips_per_sequence: int = 10,  # clips per sequence (60 seconds)
    stride: int = 5,  # stride between sequences
    train_split: float = 0.7,
    val_split: float = 0.15,
    input_size: Tuple[int, int] = (328, 512),
    max_sequences_per_video: Optional[int] = None,
    testing=False,
    specific_test_video: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for sequence-based training.
    """

    console = Console()

    # Collect video paths
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    console.print(f"Found {len(video_paths)} videos in folder.")

    # Match only videos that have corresponding XML
    filtered_video_paths = []
    annotation_paths = []

    for vp in video_paths:
        base = os.path.basename(vp).split(".")[0]
        xml_path = os.path.join(annotation_dir, f"{base}.xml")
        if os.path.exists(xml_path):
            filtered_video_paths.append(vp)
            annotation_paths.append(xml_path)

    chosen_video_paths = [
        "BMtu",
        "DCja",
        "DOOn",
        "DrOJ",
        "Dvpb",
        "Krni",
        "LoGf",
        # "MBhG",
        "MUSV",
        # "ZwMW",
        "MBHK",
        "OwRK",
        "QmhB",
    ]
    # Video-level split (maintain video integrity)
    unique_ids = sorted(
        list(set(os.path.basename(p).split(".")[0] for p in filtered_video_paths))
    )
    # chosen_set = set(chosen_video_paths)
    # unique_ids = [vid for vid in unique_ids if vid in chosen_set]
    console.print(f"[bold green]Video IDs:[/bold green] {', '.join(unique_ids)}")

    console.print(
        f"Using {len(annotation_paths)} videos that have corresponding XML annotations"
    )

    if specific_test_video and specific_test_video in unique_ids:
        console.print(
            f"[bold green]Using ONLY specific test video:[/bold green] {specific_test_video}"
        )
        test_ids = [specific_test_video]
        train_ids = [vid for vid in unique_ids if vid != specific_test_video]
        val_ids = []
    else:
        random.shuffle(unique_ids)
        n_test = 2
        n_val = 2
        test_ids = unique_ids[:n_test]
        val_ids = unique_ids[n_test : n_test + n_val]
        train_ids = unique_ids[n_test + n_val :]

    def split_paths(ids):
        vids, annos = [], []
        for vp, ap in zip(filtered_video_paths, annotation_paths):
            if os.path.basename(vp).split(".")[0] in ids:
                vids.append(vp)
                annos.append(ap)
        return vids, annos

    train_vids, train_annos = split_paths(train_ids)
    val_vids, val_annos = split_paths(val_ids)
    test_vids, test_annos = split_paths(test_ids)
    test_videos = ["BMtu"]
    # test_vids = [vid for vid in test_vids if vid in test_videos]

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )

    # Create datasets
    console.print("Creating sequence datasets...", style="cyan")

    train_dataset = VolumeSequenceDataset_NoVolumeWeights(
        train_vids,
        train_annos,
        volume_csv_path,
        clip_length=clip_length,
        clips_per_sequence=clips_per_sequence,
        stride=stride,
        transform=train_transform,
        max_sequences_per_video=max_sequences_per_video,
    )

    val_dataset = VolumeSequenceDataset_NoVolumeWeights(
        val_vids,
        val_annos,
        volume_csv_path,
        clip_length=clip_length,
        clips_per_sequence=clips_per_sequence,
        stride=stride * 2,  # Larger stride for validation
        transform=val_transform,
        max_sequences_per_video=(
            max_sequences_per_video // 2 if max_sequences_per_video else None
        ),
    )

    test_dataset = VolumeSequenceDataset_NoVolumeWeights(
        test_vids,
        test_annos,
        volume_csv_path,
        clip_length=clip_length,
        clips_per_sequence=clips_per_sequence,
        stride=stride * 2,
        transform=val_transform,
        max_sequences_per_video=(
            max_sequences_per_video // 2 if max_sequences_per_video else None
        ),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=False,
    )

    # Create summary table
    table = Table(title="Sequence Dataset Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Videos", justify="right", style="green")
    table.add_column("Sequences", justify="right", style="yellow")
    table.add_column("Video IDs", justify="right", style="yellow")

    def get_dataset_video_ids(dataset):
        return sorted(list(set(seq["video_id"] for seq in dataset.all_sequences)))

    train_ids_final = get_dataset_video_ids(train_dataset)
    val_ids_final = get_dataset_video_ids(val_dataset)
    test_ids_final = get_dataset_video_ids(test_dataset)

    table.add_row(
        "Train",
        str(len(train_ids_final)),
        f"{len(train_dataset):,}",
        ", ".join(train_ids_final),
    )
    table.add_row(
        "Validation",
        str(len(val_ids_final)),
        f"{len(val_dataset):,}",
        ", ".join(val_ids_final),
    )
    table.add_row(
        "Test",
        str(len(test_ids_final)),
        f"{len(test_dataset):,}",
        ", ".join(test_ids_final),
    )

    console.print(table)

    return train_loader, val_loader, test_loader


def train_sequence_model(
    video_dir,
    annotations_dir,
    volume_csv_path,
    device,
    clip_length=12,
    clips_per_sequence=10,
    stride=5,
    epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    output_dir="./models_sequence",
    model_name="sequence_bl_model",
    patience=10,
    small_dataset=False,
    input_size=(328, 512),
    viz_dir="./visualizations",
    # NEW: Multi-task loss weights
    alpha=0.5,  # clip classification weight
    beta=1.0,  # volume regression weight
    gamma=0.3,  # sequence classification weight
):
    """
    Train sequence-based bleeding detection model with multi-task learning.
    """
    os.makedirs(output_dir, exist_ok=True)

    # NEW: Use multi-task model instead
    model = MultiTaskVolumeSequenceModel(num_classes=2, hidden_dim=256, dropout=0.5)
    console.print(
        "[bold]Using Multi-Task R2Plus1D + LSTM Sequence Model [/bold]", style="cyan"
    )

    model = model.to(device)

    # ===== DIFFERENT LEARNING RATES SETUP =====
    backbone_params = list(model.backbone.parameters()) + list(
        model.spatial_pool.parameters()
    )
    lstm_params = list(model.lstm.parameters()) + list(
        model.volume_regressor.parameters()
    )
    # NEW: Add classifier parameters
    classifier_params = list(model.clip_classifier.parameters()) + list(
        model.sequence_classifier.parameters()
    )

    # Create optimizer with different learning rates
    optimizer = torch.optim.Adam(
        [
            {
                "params": backbone_params,
                "lr": LR_BB,  # 10x lower for pretrained backbone
                "weight_decay": 1e-4,
            },
            {
                "params": lstm_params
                + classifier_params,  # Combined LSTM and classifiers
                "lr": LR_LSTM,  # Full LR for LSTM and classifiers
                "weight_decay": 1e-4,
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=0.005,
        cooldown=1,
        verbose=True,
        min_lr=1e-7,
    )

    # NEW: Updated tracking for multi-task metrics
    history = {
        "train_loss": [],
        "train_clip_loss": [],
        "train_volume_loss": [],
        "train_sequence_loss": [],
        "val_loss": [],
        "val_clip_loss": [],
        "val_volume_loss": [],
        "val_sequence_loss": [],
        "val_volume_mae": [],
        "val_volume_rmse": [],
        "val_clip_accuracy": [],
        "val_sequence_accuracy": [],
        "val_volume_mae_bleeding": [],
        "val_volume_mae_nonzero": [],
    }

    best_mae = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")

    console.print(
        Panel.fit(
            f"Multi-Task Sequence Bleeding Volume Prediction\n"
            f"Device: {device}\n"
            f"Epochs: {epochs} | Batch Size: {batch_size}\n"
            f"Learning Rate: {learning_rate}\n"
            f"Clips per sequence: {clips_per_sequence}\n"
            f"Frames per clip: {clip_length}\n"
            f"Loss weights: α={alpha}, β={beta}, γ={gamma}\n"
            f"Total frames per sequence: {clips_per_sequence * clip_length}\n",
            title="Training Configuration",
            border_style="white",
        )
    )

    # Create DataLoaders
    console.print(f"Creating Sequence DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        video_dir=video_dir,
        annotation_dir=annotations_dir,
        volume_csv_path=volume_csv_path,
        batch_size=batch_size,
        num_workers=4,
        clip_length=clip_length,
        clips_per_sequence=clips_per_sequence,
        stride=stride,
        train_split=0.7,
        val_split=0.15,
        input_size=input_size,
        max_sequences_per_video=None,
        testing=small_dataset,
    )
    weights = get_dataset_class_weights(train_loader.dataset)

    # Training loop
    for epoch in range(epochs):
        console.print(f"\n[bold blue]Epoch {epoch + 1}/{epochs}[/bold blue]")

        # Training
        model.train()
        train_metrics = _train_sequence_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights=weights,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        # Validation
        model.eval()
        val_metrics = _validate_sequence_epoch(
            model,
            val_loader,
            device,
            class_weights=weights,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        # Update history
        for key in train_metrics:
            history[f"train_{key}"].append(train_metrics[key])
        for key in val_metrics:
            if f"val_{key}" in history:
                history[f"val_{key}"].append(val_metrics[key])

        # Update scheduler
        sel = val_metrics.get("volume_mae_bleeding")
        if sel is None or (isinstance(sel, float) and np.isnan(sel)):
            sel = val_metrics.get("volume_mae_nonzero", val_metrics["volume_mae"])

        scheduler.step(sel)

        # NEW: Enhanced results table with multi-task metrics
        table = Table(title=f"Epoch {epoch+1}/{epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", justify="right", style="green")
        table.add_column("Validation", justify="right", style="yellow")

        table.add_row(
            "Total Loss", f"{train_metrics['loss']:.4f}", f"{val_metrics['loss']:.4f}"
        )
        table.add_row(
            "Clip Loss",
            f"{train_metrics['clip_loss']:.4f}",
            f"{val_metrics['clip_loss']:.4f}",
        )
        table.add_row(
            "Volume Loss",
            f"{train_metrics['volume_loss']:.4f}",
            f"{val_metrics['volume_loss']:.4f}",
        )
        table.add_row(
            "Sequence Loss",
            f"{train_metrics['sequence_loss']:.4f}",
            f"{val_metrics['sequence_loss']:.4f}",
        )
        table.add_row("Volume MAE", "-", f"{val_metrics['volume_mae']:.4f}")
        table.add_row("Clip Accuracy", "-", f"{val_metrics['clip_accuracy']:.4f}")
        table.add_row(
            "Sequence Accuracy", "-", f"{val_metrics['sequence_accuracy']:.4f}"
        )

        console.print(table)

        # Save best model based on volume MAE (bleeding)
        sel = val_metrics.get("volume_mae_bleeding")
        if sel is None or (isinstance(sel, float) and np.isnan(sel)):
            sel = val_metrics.get("volume_mae_nonzero", val_metrics["volume_mae"])

        if sel < best_mae:
            best_mae = sel
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            console.print(
                f"[green]New best masked MAE: {best_mae:.4f} - Model saved![/green]"
            )
        else:
            patience_counter += 1

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)

    # Save history
    with open(os.path.join(output_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    console.print(
        Panel.fit(
            f"[bold green]Training completed[/bold green]\n"
            f"Best MAE: {best_mae:.4f}\n"
            f"Best model: {best_model_path}\n"
            f"Final model: {final_model_path}",
            title="Training Complete",
            border_style="green",
        )
    )

    # Test evaluation
    test_metrics = test_model_NEW(
        model,
        test_loader,
        device,
        viz_dir=viz_dir,
        cache_dir=os.path.join(viz_dir, "cache"),
    )

    return model, history, test_metrics, test_loader


def _train_sequence_epoch(
    model,
    train_loader,
    optimizer,
    device,
    class_weights=None,
    alpha=1.0,
    beta=0.5,
    gamma=0.3,
):
    """Training epoch for multi-task sequence model with weighted loss"""
    total_loss = 0.0
    total_clip_loss = 0.0
    total_volume_loss = 0.0
    total_sequence_loss = 0.0

    model.train()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=5,
    ) as progress:

        task = progress.add_task("Training Sequences", total=len(train_loader))

        for batch_idx, (
            sequence_clips,
            sequence_labels,
            target_volumes,
            training_weights,
            clip_labels,
        ) in enumerate(train_loader):

            sequence_clips = sequence_clips.to(device, dtype=torch.float32)
            sequence_labels = sequence_labels.to(device)
            target_volumes = target_volumes.to(device, dtype=torch.float32)
            training_weights = training_weights.to(device, dtype=torch.float32)
            clip_labels = clip_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            clip_preds, volume_pred, sequence_cls_pred = model(sequence_clips)

            # Compute multi-task loss (unweighted base loss)
            base_loss, loss_components = compute_multitask_loss(
                clip_preds,
                volume_pred,
                sequence_cls_pred,
                clip_labels,
                target_volumes,
                sequence_labels,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            # Apply training weights to the total loss
            weighted_loss = base_loss * training_weights.mean()

            # Backward pass
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # Accumulate losses (store weighted loss for training metrics)
            total_loss += weighted_loss.item()
            total_clip_loss += loss_components["clip_loss"]
            total_volume_loss += loss_components["volume_loss"]
            total_sequence_loss += loss_components["sequence_loss"]

            # Debug logging
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    bleeding_mask = sequence_labels == 1
                    if bleeding_mask.any():
                        bleeding_preds = volume_pred[bleeding_mask]
                        bleeding_targets = target_volumes[bleeding_mask]
                        console.print(
                            f"Batch {batch_idx}: Vol preds: {bleeding_preds.mean():.4f}±{bleeding_preds.std():.4f}, "
                            f"targets: {bleeding_targets.mean():.4f}, Loss: {weighted_loss.item():.4f}",
                            style="dim",
                        )

            progress.update(task, advance=1)

    num_batches = len(train_loader)
    return {
        "loss": total_loss / num_batches,
        "clip_loss": total_clip_loss / num_batches,
        "volume_loss": total_volume_loss / num_batches,
        "sequence_loss": total_sequence_loss / num_batches,
    }


def _validate_sequence_epoch(
    model, val_loader, device, class_weights=None, alpha=1.0, beta=0.5, gamma=0.3
):
    """Validation epoch for multi-task sequence model with unweighted loss"""
    total_loss = 0.0
    total_clip_loss = 0.0
    total_volume_loss = 0.0
    total_sequence_loss = 0.0

    # For computing metrics
    all_clip_preds = []
    all_clip_labels = []
    all_volume_preds = []
    all_volume_targets = []
    all_seq_preds = []
    all_seq_labels = []

    model.eval()
    with torch.no_grad():
        for (
            sequence_clips,
            sequence_labels,
            target_volumes,
            training_weights,  # We load this but don't use it for validation
            clip_labels,
        ) in val_loader:

            sequence_clips = sequence_clips.to(
                device, dtype=torch.float32, non_blocking=True
            )
            sequence_labels = sequence_labels.to(device)
            target_volumes = target_volumes.to(
                device, dtype=torch.float32, non_blocking=True
            )
            clip_labels = clip_labels.to(device)

            # Forward pass
            clip_preds, volume_pred, sequence_cls_pred = model(sequence_clips)

            # Compute multi-task loss WITHOUT any weighting
            loss, loss_components = compute_multitask_loss(
                clip_preds,
                volume_pred,
                sequence_cls_pred,
                clip_labels,
                target_volumes,
                sequence_labels,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            # Accumulate unweighted losses
            total_loss += loss.item()
            total_clip_loss += loss_components["clip_loss"]
            total_volume_loss += loss_components["volume_loss"]
            total_sequence_loss += loss_components["sequence_loss"]

            # Collect predictions for metrics
            all_clip_preds.append(clip_preds.detach().cpu())
            all_clip_labels.append(clip_labels.detach().cpu())
            all_volume_preds.append(volume_pred.detach().cpu())
            all_volume_targets.append(target_volumes.detach().cpu())
            all_seq_preds.append(sequence_cls_pred.detach().cpu())
            all_seq_labels.append(sequence_labels.detach().cpu())

    # Compute metrics
    clip_preds_all = torch.cat(all_clip_preds).view(-1, 2)  # [B*S, 2]
    clip_labels_all = torch.cat(all_clip_labels).view(-1)  # [B*S]

    volume_preds_all = torch.cat(all_volume_preds).numpy()
    volume_targets_all = torch.cat(all_volume_targets).numpy()

    seq_preds_all = torch.cat(all_seq_preds)
    seq_labels_all = torch.cat(all_seq_labels)

    # Clip accuracy
    _, clip_predicted = torch.max(clip_preds_all, 1)
    clip_accuracy = (clip_predicted == clip_labels_all).float().mean().item()

    # Volume metrics
    volume_mae = float(np.mean(np.abs(volume_preds_all - volume_targets_all)))
    volume_rmse = float(np.sqrt(np.mean((volume_preds_all - volume_targets_all) ** 2)))

    # Sequence accuracy
    _, seq_predicted = torch.max(seq_preds_all, 1)
    seq_accuracy = (seq_predicted == seq_labels_all).float().mean().item()

    # Masked volume metrics (bleeding-only and non-zero targets)
    seq_labels_np = seq_labels_all.numpy()
    ble_mask = seq_labels_np == 1
    if ble_mask.any():
        volume_mae_bleeding = float(
            np.mean(np.abs(volume_preds_all[ble_mask] - volume_targets_all[ble_mask]))
        )
    else:
        volume_mae_bleeding = float("nan")

    nz_mask = volume_targets_all > 0
    if nz_mask.any():
        volume_mae_nonzero = float(
            np.mean(np.abs(volume_preds_all[nz_mask] - volume_targets_all[nz_mask]))
        )
    else:
        volume_mae_nonzero = float("nan")

    num_batches = len(val_loader)
    return {
        "loss": total_loss / num_batches,
        "clip_loss": total_clip_loss / num_batches,
        "volume_loss": total_volume_loss / num_batches,
        "sequence_loss": total_sequence_loss / num_batches,
        "volume_mae": volume_mae,
        "volume_rmse": volume_rmse,
        "clip_accuracy": clip_accuracy,
        "sequence_accuracy": seq_accuracy,
        "volume_mae_bleeding": volume_mae_bleeding,
        "volume_mae_nonzero": volume_mae_nonzero,
    }


def plot_training_history(history_or_path, save_dir=None, model_name="model"):
    """
    Plot training and validation loss with a clean dark theme.

    Args:
        history_or_path: history dict or JSON path
        save_dir: if provided, saves plots there
        model_name: name to use in titles and saved file name
    """
    # Load from file or dict
    if isinstance(history_or_path, str):
        with open(history_or_path, "r") as f:
            history = json.load(f)
        print(f"Loaded history from {history_or_path}")
    elif isinstance(history_or_path, dict):
        history = history_or_path
    else:
        raise ValueError("Input must be a history dict or a path to JSON")

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # === Dark Theme ===
    plt.style.use("ggplot")
    sns.set_palette("bright")
    sns.set_context("talk", font_scale=0.8)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("#6E6B6B")

    for ax in axs:
        ax.set_facecolor("#4C4747")
        ax.tick_params(colors="black")
        ax.yaxis.label.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.title.set_color("black")
        ax.grid(True, alpha=0.3)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Training Loss
    axs[0].plot(
        epochs,
        history["train_loss"],
        "o-",
        label="Train Loss",
        color="#4571e9",
        linewidth=2,
    )
    axs[0].set_title(f"{model_name} - Training Loss", fontsize=14)
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Validation Loss
    axs[1].plot(
        epochs,
        history["val_loss"],
        "o-",
        label="Validation Loss",
        color="#fc8d62",
        linewidth=2,
    )
    axs[1].set_title(f"{model_name} - Validation Loss", fontsize=14)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    if save_dir:
        plot_path = os.path.join(save_dir, f"loss_plot_{model_name}.png")
        plt.savefig(
            plot_path, dpi=400, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"Saved loss plot to {plot_path}")

    plt.show()


def test_model_NEW(
    model,
    test_loader,
    device,
    viz_dir="./test_sequence_viz",
    cache_dir="./test_sequence_cache",
):
    """
    Test sequence model with checkpoint validation, light-theme visualization,
    and caching of all series for later plotting (avoids re-running inference).

    Saves per-video PNGs in `viz_dir` and per-video .npz caches in `cache_dir`.
    Also prints rich tables with summary stats.

    UPDATED: Now handles multi-task model outputs
    """

    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # ---- Matplotlib light theme (thesis-friendly) ----
    plt.rcParams.update(
        {
            "figure.figsize": (16, 8),
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "grid.color": "#CCCCCC",
            "grid.alpha": 0.6,
            "lines.linewidth": 2.5,
            "savefig.bbox": "tight",
        }
    )

    console = Console()
    model.eval()
    test_dataset = test_loader.dataset

    # Group sequences by video
    video_sequences = defaultdict(list)
    for i, sequence_info in enumerate(test_dataset.all_sequences):
        video_sequences[sequence_info["video_id"]].append((i, sequence_info))

    console.print(f"Testing on {len(video_sequences)} videos")

    video_results = {}
    all_checkpoint_errors = []

    # NEW: Track multi-task metrics
    all_clip_predictions = []
    all_clip_labels = []
    all_sequence_predictions = []
    all_sequence_labels = []

    def _to_python(x):
        # helper to make JSON-friendly
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    with torch.no_grad():
        with Progress(
            TextColumn("Testing videos"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:

            task = progress.add_task("Processing", total=len(video_sequences))

            for video_id, sequences_info in video_sequences.items():
                # Sort sequences by temporal order (important!)
                sequences_info.sort(key=lambda x: x[1]["sequence_start_frame"])

                # Collect per-sequence data (respect order)
                all_sequence_volume_predictions = []
                all_sequence_targets = []
                sequence_centers = []
                sequence_start_frames = []
                sequence_end_frames = []
                bleeding_sequences = []

                for seq_idx, sequence_info in sequences_info:
                    # UPDATED: Handle new dataset return format
                    dataset_output = test_dataset[seq_idx]
                    if len(dataset_output) == 5:  # Multi-task dataset
                        sequence_tensor, has_bleeding, target_volume, _, clip_labels = (
                            dataset_output
                        )
                    else:  # Old dataset format
                        sequence_tensor, has_bleeding, target_volume, _ = dataset_output
                        clip_labels = None

                    sequence_tensor = sequence_tensor.unsqueeze(0).to(
                        device
                    )  # Add batch dim

                    # UPDATED: Handle multi-task model output
                    model_output = model(sequence_tensor)

                    if isinstance(model_output, tuple) and len(model_output) == 3:
                        # Multi-task model: (clip_preds, volume_pred, sequence_cls_pred)
                        clip_preds, volume_pred, sequence_cls_pred = model_output

                        # Store multi-task predictions for later analysis
                        if clip_labels is not None:
                            _, clip_predicted = torch.max(
                                clip_preds[0], 1
                            )  # [0] removes batch dim
                            all_clip_predictions.extend(clip_predicted.cpu().numpy())
                            all_clip_labels.extend(clip_labels.numpy())

                        _, seq_predicted = torch.max(sequence_cls_pred, 1)
                        all_sequence_predictions.append(seq_predicted.item())
                        all_sequence_labels.append(has_bleeding)

                    else:
                        # Single-task model: just volume prediction
                        volume_pred = model_output

                    all_sequence_volume_predictions.append(float(volume_pred.item()))
                    all_sequence_targets.append(float(target_volume.item()))

                    seq_data = test_dataset.all_sequences[seq_idx]
                    sequence_centers.append(int(seq_data["sequence_center_frame"]))
                    sequence_start_frames.append(int(seq_data["sequence_start_frame"]))
                    sequence_end_frames.append(int(seq_data["sequence_end_frame"]))
                    bleeding_sequences.append(bool(seq_data["has_bleeding"]))

                # Cumulative curves
                predicted_cumulative = (
                    np.cumsum(all_sequence_volume_predictions)
                    if all_sequence_volume_predictions
                    else np.array([0.0])
                )
                target_cumulative = (
                    np.cumsum(all_sequence_targets)
                    if all_sequence_targets
                    else np.array([0.0])
                )

                # Ground-truth totals & checkpoints
                volume_info = test_dataset.volume_data.get(
                    video_id, {"checkpoints": [], "total_volume": 0.0}
                )
                csv_ground_truth = float(volume_info.get("total_volume", 0.0))
                checkpoints = volume_info.get(
                    "checkpoints", []
                )  # list of (frame, incremental_volume)

                checkpoint_frames = []
                checkpoint_cumulative = []
                checkpoint_errors = []
                checkpoint_relative_errors = []

                if checkpoints:
                    checkpoints = sorted(checkpoints, key=lambda x: x[0])
                    cumulative_gt = 0.0
                    for cp_idx, (checkpoint_frame, incremental_volume) in enumerate(
                        checkpoints
                    ):
                        cumulative_gt += float(incremental_volume)
                        checkpoint_frames.append(int(checkpoint_frame))
                        checkpoint_cumulative.append(float(cumulative_gt))

                        # nearest center to checkpoint
                        if sequence_centers:
                            closest_seq_idx = int(
                                np.argmin(
                                    [
                                        abs(center - checkpoint_frame)
                                        for center in sequence_centers
                                    ]
                                )
                            )
                            predicted_at_checkpoint = float(
                                predicted_cumulative[closest_seq_idx]
                            )
                        else:
                            predicted_at_checkpoint = 0.0

                        cp_err = abs(predicted_at_checkpoint - cumulative_gt)
                        cp_rel = cp_err / max(cumulative_gt, 1e-6)

                        checkpoint_errors.append(float(cp_err))
                        checkpoint_relative_errors.append(float(cp_rel))
                        all_checkpoint_errors.append(
                            {
                                "video_id": video_id,
                                "checkpoint_idx": int(cp_idx),
                                "checkpoint_frame": int(checkpoint_frame),
                                "predicted": float(predicted_at_checkpoint),
                                "ground_truth": float(cumulative_gt),
                                "error": float(cp_err),
                                "relative_error": float(cp_rel),
                            }
                        )

                # Final video-level metrics
                final_predicted_volume = (
                    float(predicted_cumulative[-1])
                    if predicted_cumulative.size > 0
                    else 0.0
                )
                final_error = abs(final_predicted_volume - csv_ground_truth)
                final_relative_error = final_error / max(csv_ground_truth, 1e-6)

                video_results[video_id] = {
                    "predicted_volume": final_predicted_volume,
                    "csv_ground_truth": csv_ground_truth,
                    "checkpoints_sum": (
                        float(sum(vol for _, vol in checkpoints))
                        if checkpoints
                        else 0.0
                    ),
                    "final_error": float(final_error),
                    "final_relative_error": float(final_relative_error),
                    "num_sequences": int(len(sequences_info)),
                    "checkpoint_errors": checkpoint_errors,
                    "checkpoint_relative_errors": checkpoint_relative_errors,
                }

                # ====== SAVE CACHE (.npz) FOR RE-PLOTTING ======
                np.savez_compressed(
                    os.path.join(cache_dir, f"{video_id}.npz"),
                    video_id=video_id,
                    sequence_centers=np.array(sequence_centers, dtype=np.int32),
                    sequence_start_frames=np.array(
                        sequence_start_frames, dtype=np.int32
                    ),
                    sequence_end_frames=np.array(sequence_end_frames, dtype=np.int32),
                    bleeding_sequences=np.array(bleeding_sequences, dtype=bool),
                    pred_seq=np.array(
                        all_sequence_volume_predictions, dtype=np.float32
                    ),
                    tgt_seq=np.array(all_sequence_targets, dtype=np.float32),
                    pred_cum=np.array(predicted_cumulative, dtype=np.float32),
                    tgt_cum=np.array(target_cumulative, dtype=np.float32),
                    checkpoints=np.array(
                        checkpoints, dtype=object
                    ),  # list of (frame, vol)
                    checkpoint_frames=np.array(checkpoint_frames, dtype=np.int32),
                    checkpoint_cumulative=np.array(
                        checkpoint_cumulative, dtype=np.float32
                    ),
                    checkpoint_errors=np.array(checkpoint_errors, dtype=np.float32),
                    checkpoint_relative_errors=np.array(
                        checkpoint_relative_errors, dtype=np.float32
                    ),
                    csv_ground_truth=np.array(csv_ground_truth, dtype=np.float32),
                    final_predicted_volume=np.array(
                        final_predicted_volume, dtype=np.float32
                    ),
                    final_error=np.array(final_error, dtype=np.float32),
                    final_relative_error=np.array(
                        final_relative_error, dtype=np.float32
                    ),
                )

                # ====== VISUALIZATION (light theme) ======
                fig, ax = plt.subplots()

                # shade bleeding spans
                for start_frame, end_frame, is_bleeding in zip(
                    sequence_start_frames, sequence_end_frames, bleeding_sequences
                ):
                    if is_bleeding:
                        ax.axvspan(
                            start_frame,
                            end_frame,
                            alpha=0.15,
                            color="#FFC107",
                            zorder=1,
                        )  # amber

                if any(bleeding_sequences):
                    ax.axvspan(
                        0, 0, alpha=0.15, color="#FFC107", label="Bleeding sequences"
                    )  # legend handle

                # plot predicted cumulative
                ax.plot(
                    sequence_centers,
                    predicted_cumulative,
                    label=f"Predicted cumulative ({final_predicted_volume:.1f} ml)",
                    color="#1f77b4",
                    marker="o",
                    markersize=5,
                    markerfacecolor="#ffffff",
                    markeredgecolor="#1f77b4",
                )

                # plot interpolated targets
                if len(target_cumulative) == len(sequence_centers):
                    ax.plot(
                        sequence_centers,
                        target_cumulative,
                        label=f"Uniformly Distributed Targets",
                        color="#2ca02c",
                        linestyle="--",
                        marker="x",
                        markersize=6,
                    )

                # checkpoints
                if checkpoint_frames:
                    ax.scatter(
                        checkpoint_frames,
                        checkpoint_cumulative,
                        color="#d62728",
                        s=80,
                        label=f"Checkpoints",
                        zorder=5,
                        marker="^",
                        edgecolors="#222222",
                        linewidth=0.8,
                    )
                    # labels
                    for frame, cum_vol, error in zip(
                        checkpoint_frames, checkpoint_cumulative, checkpoint_errors
                    ):
                        ax.annotate(
                            f"{cum_vol:.1f} ml\n(±{error:.1f})",
                            xy=(frame, cum_vol),
                            xytext=(6, 12),
                            textcoords="offset points",
                            fontsize=11,
                            color="#222222",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="#ffe6e6",
                                edgecolor="#d62728",
                                alpha=0.9,
                            ),
                        )

                # CSV ground truth line
                ax.axhline(
                    y=csv_ground_truth,
                    color="#444444",
                    linestyle=":",
                    linewidth=2,
                    label=f"CSV ground truth: {csv_ground_truth:.1f} ml",
                )

                ax.set_title(
                    f"Video: {video_id}  |  Final error: {final_error:.1f} ml ({final_relative_error*100:.1f}%)",
                    fontsize=18,
                    color="#222222",
                    pad=14,
                    fontweight="bold",
                )
                ax.set_xlabel(
                    "Frame number", fontsize=16, color="#222222", fontweight="bold"
                )
                ax.set_ylabel(
                    "Cumulative blood loss (ml)",
                    fontsize=16,
                    color="#222222",
                    fontweight="bold",
                )

                ax.grid(True)
                leg = ax.legend(
                    loc="upper left",
                    frameon=True,
                    fancybox=True,
                    shadow=False,
                    facecolor="white",
                    edgecolor="#CCCCCC",
                    framealpha=0.9,
                )
                for txt in leg.get_texts():
                    txt.set_color("#222222")

                for spine in ax.spines.values():
                    spine.set_color("#444444")
                    spine.set_linewidth(1.0)

                plt.tight_layout()
                save_path = os.path.join(viz_dir, f"sequence_pred_{video_id}.png")
                plt.savefig(save_path)
                plt.close()

                # Print detailed info (console)
                console.print(f"\n[cyan]Video: {video_id}[/cyan]")
                console.print(f"  Predicted Final: {final_predicted_volume:.2f} ml")
                console.print(f"  CSV Ground Truth: {csv_ground_truth:.2f} ml")
                console.print(
                    f"  Checkpoints Sum: {float(sum(vol for _, vol in checkpoints)) if checkpoints else 0.0:.2f} ml"
                )
                console.print(
                    f"  Final Error: {final_error:.2f} ml ({final_relative_error*100:.1f}%)"
                )
                if checkpoint_errors:
                    avg_checkpoint_error = float(np.mean(checkpoint_errors))
                    console.print(
                        f"  Avg Checkpoint Error: {avg_checkpoint_error:.2f} ml"
                    )
                    for i, (frame, error, rel_error) in enumerate(
                        zip(
                            checkpoint_frames,
                            checkpoint_errors,
                            checkpoint_relative_errors,
                        )
                    ):
                        console.print(
                            f"    CP{i+1} @frame{frame}: ±{error:.1f} ml ({rel_error*100:.1f}%)"
                        )

                progress.update(task, advance=1)

    # ====== SUMMARY STATISTICS ======
    console.print(f"\n[bold green]📊 Test Results Summary[/bold green]")

    # Video-level table
    table = Table(title="Video-Level Results")
    table.add_column("Video ID", style="cyan")
    table.add_column("Predicted (ml)", justify="right", style="green")
    table.add_column("CSV GT (ml)", justify="right", style="yellow")
    table.add_column("Checkpoints Sum (ml)", justify="right", style="blue")
    table.add_column("Final Error (ml)", justify="right", style="red")
    table.add_column("Rel Error (%)", justify="right", style="magenta")

    for video_id, results in sorted(video_results.items()):
        table.add_row(
            video_id,
            f"{results['predicted_volume']:.1f}",
            f"{results['csv_ground_truth']:.1f}",
            f"{results['checkpoints_sum']:.1f}",
            f"{results['final_error']:.1f}",
            f"{results['final_relative_error']*100:.1f}%",
        )
    console.print(table)

    # NEW: Multi-task classification metrics
    if all_clip_predictions and all_clip_labels:

        clip_accuracy = accuracy_score(all_clip_labels, all_clip_predictions)
        clip_precision = precision_score(
            all_clip_labels, all_clip_predictions, zero_division=0
        )
        clip_recall = recall_score(
            all_clip_labels, all_clip_predictions, zero_division=0
        )
        clip_f1 = f1_score(all_clip_labels, all_clip_predictions, zero_division=0)

        clip_table = Table(title="Clip-Level Classification Performance")
        clip_table.add_column("Metric", style="cyan")
        clip_table.add_column("Value", justify="right", style="green")

        clip_table.add_row("Accuracy", f"{clip_accuracy:.4f}")
        clip_table.add_row("Precision", f"{clip_precision:.4f}")
        clip_table.add_row("Recall", f"{clip_recall:.4f}")
        clip_table.add_row("F1-Score", f"{clip_f1:.4f}")

        console.print(clip_table)

    if all_sequence_predictions and all_sequence_labels:
        seq_accuracy = accuracy_score(all_sequence_labels, all_sequence_predictions)
        console.print(
            f"[bold]Sequence-Level Classification Accuracy: {seq_accuracy:.4f}[/bold]"
        )

    # Overall statistics
    final_errors = [r["final_error"] for r in video_results.values()]
    final_relative_errors = [r["final_relative_error"] for r in video_results.values()]

    summary = {
        "videos_tested": len(video_results),
        "final_mae": float(np.mean(final_errors)) if final_errors else 0.0,
        "final_mean_relative_error": (
            float(np.mean(final_relative_errors)) if final_relative_errors else 0.0
        ),
        "checkpoint_mae": 0.0,
        "checkpoint_mean_relative_error": 0.0,
    }

    # NEW: Add classification metrics to summary
    if all_clip_predictions and all_clip_labels:
        summary["clip_accuracy"] = float(clip_accuracy)
        summary["clip_f1"] = float(clip_f1)

    if all_sequence_predictions and all_sequence_labels:
        summary["sequence_accuracy"] = float(seq_accuracy)

    if all_checkpoint_errors:
        checkpoint_error_values = [cp["error"] for cp in all_checkpoint_errors]
        checkpoint_rel_errors = [cp["relative_error"] for cp in all_checkpoint_errors]
        summary["checkpoint_mae"] = float(np.mean(checkpoint_error_values))
        summary["checkpoint_mean_relative_error"] = float(
            np.mean(checkpoint_rel_errors)
        )

        summary_table = Table(title="Overall Performance")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Videos Tested", str(summary["videos_tested"]))
        summary_table.add_row("Final Volume MAE", f"{summary['final_mae']:.2f} ml")
        summary_table.add_row(
            "Final Volume Mean Rel Error",
            f"{summary['final_mean_relative_error']*100:.1f}%",
        )
        summary_table.add_row("Checkpoint MAE", f"{summary['checkpoint_mae']:.2f} ml")
        summary_table.add_row(
            "Checkpoint Mean Rel Error",
            f"{summary['checkpoint_mean_relative_error']*100:.1f}%",
        )
        console.print(summary_table)

    # Save summary JSON (video_results + summary + checkpoint rows index)
    summary_path = os.path.join(cache_dir, "summary.json")
    payload = {
        "video_results": {
            k: {kk: _to_python(vv) for kk, vv in v.items()}
            for k, v in video_results.items()
        },
        "summary": summary,
        "all_checkpoint_errors": [
            {k: _to_python(v) for k, v in row.items()} for row in all_checkpoint_errors
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {
        "video_results": video_results,
        "checkpoint_errors": all_checkpoint_errors,
        "final_mae": summary["final_mae"],
        "final_mean_relative_error": summary["final_mean_relative_error"],
        "checkpoint_mae": summary["checkpoint_mae"],
        "checkpoint_mean_relative_error": summary["checkpoint_mean_relative_error"],
        "cache_dir": cache_dir,
        "viz_dir": viz_dir,
        "summary_json": summary_path,
    }


def test_trained_model(
    model_path: str,
    video_dir: str,
    annotations_dir: str,
    volume_csv_path: str,
    device,
    clip_length=12,
    clips_per_sequence=5,
    stride=5,
    batch_size=1,
    input_size=(200, 320),
    viz_dir="./test_sequence_viz",
    specific_test_video=None,
):
    """
    Load a trained model and test it on the test set without training.
    """
    console.print(f"[bold cyan]Loading trained model and testing...[/bold cyan]")

    # ===== Load model =====
    console.print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        console.print(f"[red]Error: Model file not found at {model_path}[/red]")
        return None

    model = VolumeSequenceModel(hidden_dim=256, dropout=0.3)

    # Load trained weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        console.print(f"[green]✅ Model loaded successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return None

    # ===== Create dataloaders again =====
    console.print(f"Creating dataloaders using existing function...")

    try:
        # Create dataloaders for testing, no need for train and val
        _, _, test_loader = create_dataloaders(
            video_dir=video_dir,
            annotation_dir=annotations_dir,
            volume_csv_path=volume_csv_path,
            batch_size=batch_size,
            num_workers=2,
            clip_length=clip_length,
            clips_per_sequence=clips_per_sequence,
            stride=stride,
            train_split=0.7,
            val_split=0.15,
            input_size=input_size,
            max_sequences_per_video=None,
            testing=False,  # Use normal splits
            specific_test_video=specific_test_video,
        )

    except Exception as e:
        console.print(f"[red]Error creating dataloaders: {e}[/red]")
        return None

    # ===== 3. RUN TESTING =====
    console.rule("[bold green]Running Inference", style="green")

    # Call the main test function with the test_loader
    test_results = test_model_NEW(
        model=model,
        test_loader=test_loader,
        device=device,
        viz_dir=viz_dir,
        cache_dir=os.path.join(viz_dir, "cache"),
    )

    console.print(f"[bold green]Testing completed![/bold green]")
    console.print(f"Visualizations saved to: {viz_dir}")

    return test_results


SEED = 41
FPS_2 = True
LR_BB = 3e-5
LR_LSTM = 1e-4


def load_config(config_path="./train_config.json"):
    """Load training configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def main():
    home_dir = "/home/r.rohangirish/mt_ble/video-based-bloodloss-assessment/code/3dcnn"

    # Load config from JSON
    parser = argparse.ArgumentParser(description="Bleeding Quantification Model")
    parser.add_argument(
        "--config", type=str, default="./train_config.json", help="Path to config JSON"
    )
    parser.add_argument("--dev", type=int, help="Device ID (overrides config)")
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (overrides config)"
    )
    parser.add_argument("--batch", type=int, help="Batch size (overrides config)")
    parser.add_argument(
        "--train", action="store_true", help="Train a new model (overrides config)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode (opposite of --train)"
    )
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--model-name", type=str, help="Model name (overrides config)")
    args = parser.parse_args()

    # Load config and apply command-line overrides
    config = load_config(args.config)

    device = args.dev if args.dev is not None else config.get("device", 0)
    epochs = args.epochs if args.epochs is not None else config.get("epochs", 20)
    batch_size = args.batch if args.batch is not None else config.get("batch_size", 2)
    seed = args.seed if args.seed is not None else config.get("seed", 41)
    model_name = (
        args.model_name if args.model_name else config.get("model_name", "model-dec5")
    )

    # Determine train mode: --train flag sets True, --test flag sets False, otherwise use config
    if args.train:
        train_mode = True
    elif args.test:
        train_mode = False
    else:
        train_mode = config.get("train", False)

    # Paths
    volume_csv = config.get(
        "volume_csv",
        "/home/r.rohangirish/mt_ble/data/labels_quantification/BL_data_combined.csv",
    )
    VIDEO_DIR_1FPS = config.get(
        "video_dir_1fps", "/raid/dsl/users/r.rohangirish/data/videos_1_fps"
    )
    VIDEO_DIR_2FPS = config.get(
        "video_dir_2fps", "/raid/dsl/users/r.rohangirish/data/videos_2_fps"
    )
    ANNO_FOLDER = config.get(
        "annotations_dir", "/home/r.rohangirish/mt_ble/data/labels_xml"
    )
    output_dir_models = config.get("output_dir_models", "./MODELS_NOV2025")

    # Training parameters
    clip_length = config.get("clip_length", 12)
    clips_per_sequence = config.get("clips_per_sequence", 6)
    stride = config.get("stride", 12)
    learning_rate = config.get("learning_rate", 1e-5)
    input_size = tuple(config.get("input_size", [224, 224]))
    use_2fps = config.get("use_2fps", False)

    # Test parameters
    test_model_path = config.get(
        "test_model_path", "./BLE_MODELS_NEW/v5-FINAL-f1/v5-FINAL-f1_best.pth"
    )
    specific_test_video = config.get("specific_test_video", "Dvpb")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Update global learning rates and FPS_2 flag
    global LR_BB, LR_LSTM, FPS_2
    LR_BB = config.get("lr_backbone", 3e-5)
    LR_LSTM = config.get("lr_lstm", 1e-4)
    FPS_2 = use_2fps

    output_dir = os.path.join(output_dir_models, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Select video directory based on FPS setting
    VIDEO_DIR = VIDEO_DIR_2FPS if use_2fps else VIDEO_DIR_1FPS

    # Save parameters
    with open(os.path.join(output_dir, "params.txt"), "w") as f:
        f.write(f"Model name: {model_name}\n")
        f.write(f"Bleeding density Weighted Model\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Backbone LR: {LR_BB}, LSTM: {LR_LSTM}, dropout: 0.5\n")
        f.write(
            f"Clip length: {clip_length}, Clips per seq: {clips_per_sequence}, Stride: {stride}\n"
        )
        f.write(f"Input size: {input_size[0]}x{input_size[1]}\n")
        f.write(f"FPS: {'2' if use_2fps else '1'}\n")
        f.write("MULTI TASK, NO VOL WEIGHTING\n")

    if train_mode:
        model, history, test_metrics, test_loader = train_sequence_model(
            video_dir=VIDEO_DIR,
            annotations_dir=ANNO_FOLDER,
            volume_csv_path=volume_csv,
            device=torch.device(f"cuda:{device}"),
            clip_length=clip_length,
            clips_per_sequence=clips_per_sequence,
            stride=stride,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name,
            input_size=input_size,
            output_dir=output_dir,
            viz_dir=os.path.join(output_dir, f"{model_name}_viz"),
        )

        plot_training_history(
            os.path.join(output_dir, f"{model_name}_history.json"),
            save_dir=output_dir,
            model_name=model_name,
        )
    else:
        # TESTING AND VISUALIZATION
        console.print(
            Panel.fit(
                "[bold yellow]LOADING & TESTING MODEL[/bold yellow]",
                style="yellow",
                border_style="blue",
            )
        )

        test_video_dir = VIDEO_DIR_2FPS if use_2fps else VIDEO_DIR_1FPS
        full_test_model_path = (
            os.path.join(home_dir, test_model_path)
            if not os.path.isabs(test_model_path)
            else test_model_path
        )

        _ = test_trained_model(
            model_path=full_test_model_path,
            clip_length=clip_length,
            clips_per_sequence=clips_per_sequence,
            video_dir=test_video_dir,
            annotations_dir=ANNO_FOLDER,
            volume_csv_path=volume_csv,
            device=torch.device(f"cuda:{device}"),
            input_size=input_size,
            viz_dir=os.path.join(output_dir, f"test_viz"),
            specific_test_video=specific_test_video,
        )


if __name__ == "__main__":
    main()
