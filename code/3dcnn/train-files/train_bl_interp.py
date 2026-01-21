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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights,
    r2plus1d_18,
    R2Plus1D_18_Weights,
)
from typing import List, Dict, Tuple, Optional
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
import json
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from collections import defaultdict
from VideBleedingDetectorLSTM import VideoBleedingDetectorLSTM

console = Console()
SEED = 40

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

class VideoBleedingDetector(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()

        try:
            full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(full_model.children())[:-2])
            in_features = 512
            console.print("------------------------")
            console.print(
                "Using R(2+1)D backbone for feature extraction", style="bold green"
            )
        except Exception as e:
            console.print(f"Problem with loading the model: {e}")
            raise

        # Global pooling to handle variable spatial dimensions
        self.global_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head: bleeding yes/no
        self.classifier = nn.Linear(in_features, num_classes)

        self.volume_regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # Initialize final layer to output small positive values
        nn.init.xavier_uniform_(self.volume_regressor[-1].weight)
        nn.init.constant_(self.volume_regressor[-1].bias, 0.1)

    def forward(self, x):  # x: [B, C, T, H, W]
        features = self.backbone(x)  # [B, 512, T', H', W']

        # Global pooling to get fixed-size features
        features = self.global_avgpool(features)  # [B, 512, 1, 1, 1]
        features = torch.flatten(features, 1)  # [B, 512]

        # Apply dropout
        features = self.dropout(features)

        # Classification: bleeding probability
        clip_pred = self.classifier(features)  # [B, 2]

        # Volume regression: delta prediction
        delta_volume_pred = self.volume_regressor(features).squeeze(-1)  # [B]
        # delta_volume_pred = torch.relu(
        #     delta_volume_pred
        # )  # ReLU for non-zero volume deltas FIXME removed to see what is happening without ReLU, might be clamping to 0 everything

        return clip_pred, delta_volume_pred


class SurgicalInterpDataset(Dataset):
    """
    Dataset with true volume interpolation between measurement checkpoints.
    Uses linear interpolation when checkpoints available, falls back to uniform distribution.
    """

    def __init__(
        self,
        video_paths: List[str],
        annotation_paths: List[str],  # XML files
        volume_csv_path: str,
        clip_length: int = 6,
        stride: int = 3,
        transform=None,
        max_clips_per_video: Optional[int] = None,
    ):
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.max_clips_per_video = max_clips_per_video

        # Load volume data from CSV
        self.volume_data = self._load_volume_csv(volume_csv_path)

        # Create flat list of all clips with interpolated volume targets
        self.all_clips = self._prepare_all_clips_with_interpolation(
            video_paths, annotation_paths
        )

        # Balance dataset: keep all bleeding, sample equal number of non-bleeding
        self.all_clips = self._balance_bleeding_clips(self.all_clips)

        self._print_dataset_stats()

    def _load_volume_csv(self, path: str) -> Dict:
        """Load volume measurements from CSV"""
        df = pd.read_csv(path)
        df = df.dropna(subset=["video_name"])

        data = defaultdict(lambda: {"checkpoints": [], "total_volume": 0.0})

        for _, row in df.iterrows():
            video_name = str(row["video_name"])

            # Store total volume (a_e_bl)
            if pd.notna(row.get("a_e_bl")):
                data[video_name]["total_volume"] = float(row["a_e_bl"])

            # Store measurement checkpoints
            if pd.notna(row.get("measurement_frame")) and pd.notna(row.get("bl_loss")):
                measurement_frame = int(row["measurement_frame"])
                cumulative_volume = float(row["bl_loss"])
                data[video_name]["checkpoints"].append(
                    (measurement_frame, cumulative_volume)
                )

        # Sort checkpoints by frame number
        for video_name in data:
            data[video_name]["checkpoints"].sort(key=lambda x: x[0])

        return data

    def _parse_xml_annotations(self, xml_path: str) -> List[Dict]:
        """Parse CVAT XML file to extract bleeding annotations"""
        if not os.path.exists(xml_path):
            return []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotations = []

            # Get video dimensions
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

            # Parse tracks
            for track in root.findall("track"):
                track_label = track.get("label")
                if track_label in ["BL_Low", "BL_Medium", "BL_High"]:
                    for box in track.findall("box"):
                        frame_num = int(box.get("frame"))
                        outside = int(box.get("outside", "0"))

                        if outside == 1:
                            continue

                        annotations.append(
                            {
                                "frame": frame_num,
                                "label": track_label,
                                "original_width": orig_width,
                                "original_height": orig_height,
                            }
                        )

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return []

        return annotations

    def _xml_to_frame_labels(self, xml_path: str, total_frames: int) -> np.ndarray:
        """Convert XML annotations to per-frame bleeding labels"""
        frame_labels = np.zeros(total_frames, dtype=np.int32)
        severity_map = {"BL_Low": 1, "BL_Medium": 2, "BL_High": 3}

        annotations = self._parse_xml_annotations(xml_path)

        for anno in annotations:
            frame_num = anno["frame"]
            label = anno["label"]
            severity = severity_map.get(label, 1)

            if 0 <= frame_num < total_frames:
                frame_labels[frame_num] = max(frame_labels[frame_num], severity)

        return frame_labels

    def _interpolate_volume_between_checkpoints(
        self,
        video_clips: List[Dict],
        checkpoints: List[Tuple[int, float]],
        total_volume: float,
    ) -> List[Dict]:
        """
        True interpolation: assign volume based on linear interpolation between incremental measurement checkpoints.
        Checkpoints contain incremental blood loss between measurements, not cumulative totals.
        """
        if not checkpoints:
            print(f"  No checkpoints available, using uniform distribution")
            return self._assign_volume_uniform(video_clips, total_volume)

        # Sort checkpoints by frame number
        checkpoints = sorted(checkpoints, key=lambda x: x[0])
        print(f"  Using {len(checkpoints)} incremental checkpoints for interpolation")

        # Convert incremental checkpoints to cumulative
        cumulative_checkpoints = [(0, 0.0)]  # Start at frame 0 with 0ml
        cumulative_volume = 0.0

        for frame, incremental_loss in checkpoints:
            cumulative_volume += incremental_loss
            cumulative_checkpoints.append((frame, cumulative_volume))

        print(f"  Converted to cumulative: {cumulative_checkpoints}")

        # Sort clips by temporal order
        video_clips = sorted(video_clips, key=lambda x: x["center_frame"])

        for clip in video_clips:
            center_frame = clip["center_frame"]

            # Find surrounding cumulative checkpoints
            before_checkpoint = None
            after_checkpoint = None

            for i, (frame, cum_volume) in enumerate(cumulative_checkpoints):
                if frame <= center_frame:
                    before_checkpoint = (frame, cum_volume)
                else:
                    after_checkpoint = (frame, cum_volume)
                    break

            # Interpolate cumulative volume at this clip's center frame
            if before_checkpoint and after_checkpoint:
                # Linear interpolation between two checkpoints
                frame1, vol1 = before_checkpoint
                frame2, vol2 = after_checkpoint

                # Interpolation factor
                t = (center_frame - frame1) / (frame2 - frame1)
                interpolated_cumulative = vol1 + t * (vol2 - vol1)

                clip["target_cumulative_volume"] = float(interpolated_cumulative)

            elif before_checkpoint and not after_checkpoint:
                # After last checkpoint - use last known cumulative volume
                clip["target_cumulative_volume"] = float(before_checkpoint[1])

            elif not before_checkpoint and after_checkpoint:
                # Before first checkpoint - interpolate from 0 to first checkpoint
                frame2, vol2 = after_checkpoint
                t = center_frame / frame2 if frame2 > 0 else 0
                interpolated_cumulative = t * vol2
                clip["target_cumulative_volume"] = float(interpolated_cumulative)

            else:
                # No checkpoints (shouldn't happen due to earlier check)
                clip["target_cumulative_volume"] = 0.0

        # Calculate volume deltas from cumulative volumes
        prev_cumulative = 0.0
        for clip in video_clips:
            current_cumulative = clip["target_cumulative_volume"]

            if clip["has_bleeding"]:
                # Volume delta is the increase from previous clip
                clip["volume_delta"] = max(0.0, current_cumulative - prev_cumulative)
            else:
                # Non-bleeding clips contribute no volume increase
                clip["volume_delta"] = 0.0
                # But maintain cumulative for next clip
                clip["target_cumulative_volume"] = prev_cumulative

            prev_cumulative = clip["target_cumulative_volume"]

        return video_clips

    def _assign_volume_uniform(
        self, video_clips: List[Dict], total_volume: float
    ) -> List[Dict]:
        """
        Fallback: uniform distribution when no checkpoints available.
        """
        bleeding_clips = [clip for clip in video_clips if clip["has_bleeding"]]

        if not bleeding_clips:
            for clip in video_clips:
                clip["target_cumulative_volume"] = 0.0
                clip["volume_delta"] = 0.0
            return video_clips

        volume_per_bleeding_clip = total_volume / len(bleeding_clips)
        cumulative_volume = 0.0

        for clip in video_clips:
            if clip["has_bleeding"]:
                cumulative_volume += volume_per_bleeding_clip
                clip["target_cumulative_volume"] = cumulative_volume
                clip["volume_delta"] = volume_per_bleeding_clip
            else:
                clip["target_cumulative_volume"] = cumulative_volume
                clip["volume_delta"] = 0.0

        return video_clips

    def _assign_volume(
        self, video_clips: List[Dict], total_volume: float
    ) -> List[Dict]:
        """
        Assign volume using interpolation between checkpoints when available.
        Falls back to uniform distribution if no checkpoints.
        """
        video_id = video_clips[0]["video_id"] if video_clips else None
        if not video_id:
            return video_clips

        # Get checkpoints for this video
        volume_info = self.volume_data.get(
            video_id, {"checkpoints": [], "total_volume": 0.0}
        )
        checkpoints = volume_info["checkpoints"]

        if checkpoints:
            return self._interpolate_volume_between_checkpoints(
                video_clips, checkpoints, total_volume
            )
        else:
            return self._assign_volume_uniform(video_clips, total_volume)

    def _prepare_all_clips_with_interpolation(
        self, video_paths: List[str], annotation_paths: List[str]
    ) -> List[Dict]:
        """
        Create flat list of all clips with interpolated volume targets.
        """
        all_clips = []

        for video_path, anno_path in zip(video_paths, annotation_paths):
            video_id = os.path.basename(video_path).split(".")[0]

            if video_id not in self.volume_data:
                print(f"Warning: Skipping {video_id} - no volume data")
                continue

            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if frame_count <= self.clip_length:
                continue

            # Parse bleeding annotations
            bleeding_labels = self._xml_to_frame_labels(anno_path, frame_count)

            # Get volume data
            volume_info = self.volume_data.get(
                video_id, {"checkpoints": [], "total_volume": 0.0}
            )
            total_volume = volume_info["total_volume"]

            # Create clips
            video_clips = []
            for start_idx in range(0, frame_count - self.clip_length + 1, self.stride):
                end_idx = start_idx + self.clip_length
                center_frame = (start_idx + end_idx) // 2

                # Bleeding info
                clip_bleeding = bleeding_labels[start_idx:end_idx]
                has_bleeding = np.max(clip_bleeding) > 0
                max_severity = int(np.max(clip_bleeding))

                clip_info = {
                    "video_id": video_id,
                    "video_path": video_path,
                    "start_frame": start_idx,
                    "end_frame": end_idx - 1,
                    "center_frame": center_frame,
                    "has_bleeding": has_bleeding,
                    "max_severity": max_severity,
                    "bleeding_frames": clip_bleeding.tolist(),
                    "video_total_volume": total_volume,
                    "clip_index_in_video": len(video_clips),
                }
                video_clips.append(clip_info)

            # Apply clip limit if specified
            if self.max_clips_per_video and len(video_clips) > self.max_clips_per_video:
                # Keep balanced ratio of bleeding/non-bleeding when limiting
                bleeding_clips = [c for c in video_clips if c["has_bleeding"]]
                non_bleeding_clips = [c for c in video_clips if not c["has_bleeding"]]

                max_bleeding = min(len(bleeding_clips), self.max_clips_per_video // 2)
                max_non_bleeding = min(
                    len(non_bleeding_clips), self.max_clips_per_video - max_bleeding
                )

                # Sample evenly from the video to maintain temporal distribution
                if len(bleeding_clips) > max_bleeding:
                    step = len(bleeding_clips) // max_bleeding
                    bleeding_clips = bleeding_clips[::step][:max_bleeding]

                if len(non_bleeding_clips) > max_non_bleeding:
                    step = len(non_bleeding_clips) // max_non_bleeding
                    non_bleeding_clips = non_bleeding_clips[::step][:max_non_bleeding]

                video_clips = bleeding_clips + non_bleeding_clips
                video_clips.sort(
                    key=lambda x: x["clip_index_in_video"]
                )  # Restore temporal order

            # Assign volumes using interpolation or uniform distribution
            video_clips = self._assign_volume(video_clips, total_volume)

            all_clips.extend(video_clips)
            bleeding_count = sum(1 for c in video_clips if c["has_bleeding"])

        return all_clips

    def _balance_bleeding_clips(self, clips: List[Dict]) -> List[Dict]:
        """
        Balance dataset: keep ALL bleeding clips, sample equal number of non-bleeding clips
        """
        random.seed(SEED)
        bleeding_clips = [c for c in clips if c["has_bleeding"]]
        non_bleeding_clips = [c for c in clips if not c["has_bleeding"]]

        if len(non_bleeding_clips) > len(bleeding_clips):
            # Randomly sample non-bleeding clips to match bleeding count
            random.shuffle(non_bleeding_clips)
            non_bleeding_clips = non_bleeding_clips[: len(bleeding_clips)]

        # Combine and shuffle
        balanced_clips = bleeding_clips + non_bleeding_clips
        random.shuffle(balanced_clips)

        print(
            f"  Balanced: {len(bleeding_clips)} bleeding, {len(non_bleeding_clips)} non-bleeding"
        )
        console.print(
            f"[cyan]Total clips (post balancing): {len(balanced_clips)} [/cyan]"
        )

        return balanced_clips

    def _print_dataset_stats(self):
        """Print dataset statistics with interpolation info"""
        if not self.all_clips:
            return

        videos = set(c["video_id"] for c in self.all_clips)
        bleeding_clips = [c for c in self.all_clips if c["has_bleeding"]]
        non_bleeding_clips = [c for c in self.all_clips if not c["has_bleeding"]]

        # Volume statistics
        bleeding_volumes = [c["target_cumulative_volume"] for c in bleeding_clips]
        bleeding_deltas = [c["volume_delta"] for c in bleeding_clips]

        # Count videos with checkpoints vs uniform distribution
        videos_with_checkpoints = 0
        videos_uniform = 0

        for video_id in videos:
            volume_info = self.volume_data.get(video_id, {"checkpoints": []})
            if volume_info["checkpoints"]:
                videos_with_checkpoints += 1
            else:
                videos_uniform += 1

        # console.print(f"[cyan]Dataset Statistics:[/cyan]")
        # console.print(f"  Total clips: {len(self.all_clips):,}")
        # console.print(f"  Bleeding clips: {len(bleeding_clips):,} ({len(bleeding_clips)/len(self.all_clips)*100:.1f}%)")
        # console.print(f"  Non-bleeding clips: {len(non_bleeding_clips):,}")

        # console.print(f"[yellow]Volume Assignment Method:[/yellow]")
        # console.print(f"  Videos with interpolation: {videos_with_checkpoints}")
        # console.print(f"  Videos with uniform distribution: {videos_uniform}")

        # if bleeding_volumes:
        #     console.print(f"[green]Volume Statistics:[/green]")
        #     console.print(f"  Cumulative range: {min(bleeding_volumes):.1f} - {max(bleeding_volumes):.1f}ml")
        #     console.print(f"  Mean delta: {np.mean(bleeding_deltas):.3f}ml")
        #     console.print(f"  Delta range: {min(bleeding_deltas):.3f} - {max(bleeding_deltas):.3f}ml")
        #     console.print(f"  Delta std: {np.std(bleeding_deltas):.3f}ml")

        console.print("----------------------------------")

    def visualize_interpolation_example(self, video_id: str):
        """
        Visualize how interpolation works for a specific video (for debugging).
        """
        import matplotlib.pyplot as plt

        volume_info = self.volume_data.get(
            video_id, {"checkpoints": [], "total_volume": 0.0}
        )
        checkpoints = volume_info["checkpoints"]

        if not checkpoints:
            print(f"No checkpoints for video {video_id}")
            return

        # Get clips for this video
        video_clips = [c for c in self.all_clips if c["video_id"] == video_id]
        video_clips = sorted(video_clips, key=lambda x: x["center_frame"])

        # Extract data for plotting
        clip_frames = [c["center_frame"] for c in video_clips]
        clip_volumes = [c["target_cumulative_volume"] for c in video_clips]
        clip_bleeding = [c["has_bleeding"] for c in video_clips]

        checkpoint_frames = [c[0] for c in checkpoints]
        checkpoint_volumes = [c[1] for c in checkpoints]

        # Plot
        plt.figure(figsize=(12, 6))

        # Plot interpolated volumes
        bleeding_frames = [f for f, b in zip(clip_frames, clip_bleeding) if b]
        bleeding_volumes = [v for v, b in zip(clip_volumes, clip_bleeding) if b]
        non_bleeding_frames = [f for f, b in zip(clip_frames, clip_bleeding) if not b]
        non_bleeding_volumes = [v for v, b in zip(clip_volumes, clip_bleeding) if not b]

        plt.plot(
            bleeding_frames,
            bleeding_volumes,
            "ro-",
            label="Bleeding clips",
            markersize=4,
        )
        plt.plot(
            non_bleeding_frames,
            non_bleeding_volumes,
            "bo-",
            label="Non-bleeding clips",
            markersize=4,
        )

        # Plot checkpoints
        plt.plot(
            checkpoint_frames,
            checkpoint_volumes,
            "g^",
            label="Measurement checkpoints",
            markersize=8,
        )

        plt.xlabel("Frame Number")
        plt.ylabel("Cumulative Blood Loss (ml)")
        plt.title(f"Volume Interpolation for Video {video_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Checkpoints: {checkpoints}")
        print(f"Total clips: {len(video_clips)}")
        print(f"Bleeding clips: {sum(clip_bleeding)}")

    def __len__(self) -> int:
        return len(self.all_clips)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, torch.Tensor, int, float, float]:
        """
        Get a single clip with all its labels.

        Returns:
            clip_tensor: [C, T, H, W]
            clip_label: 0/1 for bleeding classification
            frame_labels: [T] frame-level bleeding severity
            severity: 0-3 max severity in clip
            target_volume: Cumulative volume target (interpolated)
            volume_delta: Expected volume increase (interpolated)
        """
        clip_info = self.all_clips[idx]

        # Load frames for this clip
        frames = self._load_clip_frames(
            clip_info["video_path"],
            clip_info["start_frame"],
            clip_info["end_frame"] + 1,
        )

        # Handle insufficient frames
        if len(frames) < self.clip_length:
            last_frame = (
                frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            )
            frames.extend([last_frame] * (self.clip_length - len(frames)))

        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]

        # Prepare labels
        clip_label = 1 if clip_info["has_bleeding"] else 0
        frame_labels = torch.tensor(clip_info["bleeding_frames"], dtype=torch.float32)
        severity = clip_info["max_severity"]

        target_volume = torch.tensor(
            clip_info["target_cumulative_volume"], dtype=torch.float32
        )
        volume_delta = torch.tensor(clip_info["volume_delta"], dtype=torch.float32)

        return (
            clip_tensor,
            clip_label,
            frame_labels,
            severity,
            target_volume,
            volume_delta,
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


# class SurgicalUniformDataset(Dataset):
#     """
#     Fast dataset that assigns volume targets only to bleeding clips UNIFORMLY.
#     Non-bleeding clips get zero volume. Dataset is balanced 1:1 bleeding to non-bleeding.
#     """

#     def __init__(
#         self,
#         video_paths: List[str],
#         annotation_paths: List[str],  # XML files
#         volume_csv_path: str,
#         clip_length: int = 6,
#         stride: int = 3,
#         transform=None,
#         max_clips_per_video: Optional[int] = None,
#     ):
#         self.clip_length = clip_length
#         self.stride = stride
#         self.transform = transform
#         self.max_clips_per_video = max_clips_per_video

#         # Load volume data from CSV
#         self.volume_data = self._load_volume_csv(volume_csv_path)

#         # Create flat list of all clips with bleeding-only volume targets
#         self.all_clips = self._prepare_all_clips_bleeding_only(
#             video_paths, annotation_paths
#         )

#         # Balance dataset: keep all bleeding, sample equal number of non-bleeding
#         self.all_clips = self._balance_bleeding_clips(self.all_clips)

#         # print(f"Created balanced dataset with {len(self.all_clips):,} clips")
#         self._print_dataset_stats()

#     def _load_volume_csv(self, path: str) -> Dict:
#         """Load volume measurements from CSV"""
#         df = pd.read_csv(path)
#         df = df.dropna(subset=["video_name"])

#         data = defaultdict(lambda: {"checkpoints": [], "total_volume": 0.0})

#         for _, row in df.iterrows():
#             video_name = str(row["video_name"])

#             # Store total volume (a_e_bl)
#             if pd.notna(row.get("a_e_bl")):
#                 data[video_name]["total_volume"] = float(row["a_e_bl"])

#             # Store measurement checkpoints
#             if pd.notna(row.get("measurement_frame")) and pd.notna(row.get("bl_loss")):
#                 measurement_frame = int(row["measurement_frame"])
#                 cumulative_volume = float(row["bl_loss"])
#                 data[video_name]["checkpoints"].append(
#                     (measurement_frame, cumulative_volume)
#                 )

#         # Sort checkpoints by frame number
#         for video_name in data:
#             data[video_name]["checkpoints"].sort(key=lambda x: x[0])

#         return data

#     def _parse_xml_annotations(self, xml_path: str) -> List[Dict]:
#         """Parse CVAT XML file to extract bleeding annotations"""
#         if not os.path.exists(xml_path):
#             return []

#         try:
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             annotations = []

#             # Get video dimensions
#             meta = root.find("meta")
#             if meta is not None:
#                 original_size = meta.find("original_size")
#                 if original_size is not None:
#                     orig_width = int(original_size.find("width").text)
#                     orig_height = int(original_size.find("height").text)
#                 else:
#                     orig_width = orig_height = 1920
#             else:
#                 orig_width = orig_height = 1920

#             # Parse tracks
#             for track in root.findall("track"):
#                 track_label = track.get("label")
#                 if track_label in ["BL_Low", "BL_Medium", "BL_High"]:
#                     for box in track.findall("box"):
#                         frame_num = int(box.get("frame"))
#                         outside = int(box.get("outside", "0"))

#                         if outside == 1:
#                             continue

#                         annotations.append(
#                             {
#                                 "frame": frame_num,
#                                 "label": track_label,
#                                 "original_width": orig_width,
#                                 "original_height": orig_height,
#                             }
#                         )

#         except Exception as e:
#             print(f"Error parsing {xml_path}: {e}")
#             return []

#         return annotations

#     def _xml_to_frame_labels(self, xml_path: str, total_frames: int) -> np.ndarray:
#         """Convert XML annotations to per-frame bleeding labels"""
#         frame_labels = np.zeros(total_frames, dtype=np.int32)
#         severity_map = {"BL_Low": 1, "BL_Medium": 2, "BL_High": 3}

#         annotations = self._parse_xml_annotations(xml_path)

#         for anno in annotations:
#             frame_num = anno["frame"]
#             label = anno["label"]
#             severity = severity_map.get(label, 1)

#             if 0 <= frame_num < total_frames:
#                 frame_labels[frame_num] = max(frame_labels[frame_num], severity)

#         return frame_labels

#     def _assign_volume(
#         self,
#         video_clips: List[Dict],
#         total_volume: float
#     ) -> List[Dict]:
#         """
#         Assign volume ONLY to bleeding clips, zero to non-bleeding clips.
#         Distribute total video volume evenly among all bleeding clips.
#         """
#         # Find all bleeding clips
#         bleeding_clips = [clip for clip in video_clips if clip["has_bleeding"]]

#         if not bleeding_clips:
#             # No bleeding clips - assign zero to all
#             for clip in video_clips:
#                 clip["target_cumulative_volume"] = 0.0
#                 clip["volume_delta"] = 0.0
#             return video_clips

#         # Distribute total volume evenly among bleeding clips
#         volume_per_bleeding_clip = total_volume / len(bleeding_clips)

#         # print(f"{len(bleeding_clips)} bleeding clips, {volume_per_bleeding_clip:.3f}ml per clip")

#         # Assign volumes in temporal order
#         cumulative_volume = 0.0
#         bleeding_clip_index = 0

#         for clip in video_clips:
#             if clip["has_bleeding"]:
#                 # This is a bleeding clip - assign volume
#                 cumulative_volume += volume_per_bleeding_clip
#                 clip["target_cumulative_volume"] = cumulative_volume
#                 clip["volume_delta"] = volume_per_bleeding_clip
#                 bleeding_clip_index += 1
#             else:
#                 # Non-bleeding clip - zero volume delta, maintain cumulative
#                 clip["target_cumulative_volume"] = cumulative_volume
#                 clip["volume_delta"] = 0.0

#         return video_clips

#     def _prepare_all_clips_bleeding_only(
#         self, video_paths: List[str], annotation_paths: List[str]
#     ) -> List[Dict]:
#         """
#         Create flat list of all clips with bleeding-only volume targets.
#         """
#         all_clips = []

#         for video_path, anno_path in zip(video_paths, annotation_paths):
#             video_id = os.path.basename(video_path).split(".")[0]

#             # print(f"Processing {video_id}...")
#             if video_id not in self.volume_data:
#                 print(f"Warning: Skipping {video_id} - no volume data")
#                 continue
#             # Get video metadata
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 continue

#             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             cap.release()

#             if frame_count <= self.clip_length:
#                 continue

#             # Parse bleeding annotations
#             bleeding_labels = self._xml_to_frame_labels(anno_path, frame_count)

#             # Get volume data
#             volume_info = self.volume_data.get(
#                 video_id, {"checkpoints": [], "total_volume": 0.0}
#             )
#             total_volume = volume_info["total_volume"]

#             # print(f"  {frame_count} frames, total volume: {total_volume:.1f}ml")

#             # Create clips
#             video_clips = []
#             for start_idx in range(0, frame_count - self.clip_length + 1, self.stride):
#                 end_idx = start_idx + self.clip_length
#                 center_frame = (start_idx + end_idx) // 2

#                 # Bleeding info
#                 clip_bleeding = bleeding_labels[start_idx:end_idx]
#                 has_bleeding = np.max(clip_bleeding) > 0
#                 max_severity = int(np.max(clip_bleeding))

#                 clip_info = {
#                     "video_id": video_id,
#                     "video_path": video_path,
#                     "start_frame": start_idx,
#                     "end_frame": end_idx - 1,
#                     "center_frame": center_frame,
#                     "has_bleeding": has_bleeding,
#                     "max_severity": max_severity,
#                     "bleeding_frames": clip_bleeding.tolist(),
#                     "video_total_volume": total_volume,
#                     "clip_index_in_video": len(video_clips),
#                 }
#                 video_clips.append(clip_info)

#             # Apply clip limit if specified
#             if self.max_clips_per_video and len(video_clips) > self.max_clips_per_video:
#                 # Keep balanced ratio of bleeding/non-bleeding when limiting
#                 bleeding_clips = [c for c in video_clips if c["has_bleeding"]]
#                 non_bleeding_clips = [c for c in video_clips if not c["has_bleeding"]]

#                 max_bleeding = min(len(bleeding_clips), self.max_clips_per_video // 2)
#                 max_non_bleeding = min(len(non_bleeding_clips), self.max_clips_per_video - max_bleeding)

#                 # Sample evenly from the video to maintain temporal distribution
#                 if len(bleeding_clips) > max_bleeding:
#                     step = len(bleeding_clips) // max_bleeding
#                     bleeding_clips = bleeding_clips[::step][:max_bleeding]

#                 if len(non_bleeding_clips) > max_non_bleeding:
#                     step = len(non_bleeding_clips) // max_non_bleeding
#                     non_bleeding_clips = non_bleeding_clips[::step][:max_non_bleeding]

#                 video_clips = bleeding_clips + non_bleeding_clips
#                 video_clips.sort(key=lambda x: x["clip_index_in_video"])  # Restore temporal order

#             # Assign volumes (bleeding clips only)
#             video_clips = self._assign_volume(video_clips, total_volume)

#             all_clips.extend(video_clips)
#             bleeding_count = sum(1 for c in video_clips if c["has_bleeding"])
#             # print(f"  ✅ Added {len(video_clips)} clips ({bleeding_count} bleeding, {len(video_clips)-bleeding_count} non-bleeding)")

#         return all_clips

#     def _balance_bleeding_clips(self, clips: List[Dict]) -> List[Dict]:
#         """
#         Balance dataset: keep ALL bleeding clips, sample equal number of non-bleeding clips
#         """
#         bleeding_clips = [c for c in clips if c["has_bleeding"]]
#         non_bleeding_clips = [c for c in clips if not c["has_bleeding"]]

#         # print(f"\nBalancing dataset:")
#         # print(f"  Original: {len(bleeding_clips)} bleeding, {len(non_bleeding_clips)} non-bleeding")

#         if len(non_bleeding_clips) > len(bleeding_clips):
#             # Randomly sample non-bleeding clips to match bleeding count
#             random.shuffle(non_bleeding_clips)
#             non_bleeding_clips = non_bleeding_clips[:len(bleeding_clips)]

#         # Combine and shuffle
#         balanced_clips = bleeding_clips + non_bleeding_clips
#         random.shuffle(balanced_clips)

#         print(f"  Balanced: {len(bleeding_clips)} bleeding, {len(non_bleeding_clips)} non-bleeding")
#         console.print(f"[cyan]Total clips (post balancing): {len(balanced_clips)} [/cyan]")

#         return balanced_clips

#     def _print_dataset_stats(self):
#         """Print dataset statistics"""
#         if not self.all_clips:
#             return
#         videos = set(c["video_id"] for c in self.all_clips)
#         # print(f"Videos: {len(videos)}")

#         bleeding_clips = [c for c in self.all_clips if c["has_bleeding"]]
#         non_bleeding_clips = [c for c in self.all_clips if not c["has_bleeding"]]

#         # Volume statistics for bleeding clips only
#         bleeding_volumes = [c["target_cumulative_volume"] for c in bleeding_clips]
#         bleeding_deltas = [c["volume_delta"] for c in bleeding_clips]

#         # Check non-bleeding volumes (should all be zero deltas)
#         non_bleeding_deltas = [c["volume_delta"] for c in non_bleeding_clips]

#         # console.print(f"\n[green bold]Dataset Statistics: [/green bold]")
#         # console.print(f"Total balanced clips: {len(self.all_clips):,}")
#         # console.print(f"Bleeding clips: {len(bleeding_clips):,} ({len(bleeding_clips)/len(self.all_clips)*100:.1f}%)")
#         # console.print(f"Non-bleeding clips: {len(non_bleeding_clips):,} ({len(non_bleeding_clips)/len(self.all_clips)*100:.1f}%)")

#         if bleeding_volumes:
#             print(f"Bleeding volume range: {min(bleeding_volumes):.1f} - {max(bleeding_volumes):.1f}ml")
#             # print(f"Mean bleeding delta: {np.mean(bleeding_deltas):.3f}ml ± {np.std(bleeding_deltas):.3f}ml")
#             print(f"Mean Bleeding delta: {np.mean(bleeding_deltas):.3f}ml")
#         # if non_bleeding_deltas:
#             # print(f"Non-bleeding delta range: {min(non_bleeding_deltas):.3f} - {max(non_bleeding_deltas):.3f}ml")
#             # print(f"Non-bleeding clips with non-zero delta: {sum(1 for d in non_bleeding_deltas if d > 0)}")

#         # Show volume distribution by video
#         console.print("----------------------------------")


#     def __len__(self) -> int:
#         return len(self.all_clips)

#     def __getitem__(
#         self, idx: int
#     ) -> Tuple[torch.Tensor, int, torch.Tensor, int, float, float]:
#         """
#         Get a single clip with all its labels.

#         Returns:
#             clip_tensor: [C, T, H, W]
#             clip_label: 0/1 for bleeding classification
#             frame_labels: [T] frame-level bleeding severity
#             severity: 0-3 max severity in clip
#             target_volume: Cumulative volume target (for bleeding clips only)
#             volume_delta: Expected volume increase (bleeding clips only, others get 0)
#         """
#         clip_info = self.all_clips[idx]

#         # Load frames for this clip
#         frames = self._load_clip_frames(
#             clip_info["video_path"],
#             clip_info["start_frame"],
#             clip_info["end_frame"] + 1,
#         )

#         # Handle insufficient frames
#         if len(frames) < self.clip_length:
#             last_frame = (
#                 frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
#             )
#             frames.extend([last_frame] * (self.clip_length - len(frames)))

#         # Apply transforms
#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]

#         clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]

#         # Prepare labels
#         clip_label = 1 if clip_info["has_bleeding"] else 0
#         frame_labels = torch.tensor(clip_info["bleeding_frames"], dtype=torch.float32)
#         severity = clip_info["max_severity"]

#         target_volume = torch.tensor(clip_info["target_cumulative_volume"], dtype=torch.float32)
#         volume_delta = torch.tensor(clip_info["volume_delta"], dtype=torch.float32)


#         return (
#             clip_tensor,
#             clip_label,
#             frame_labels,
#             severity,
#             target_volume,
#             volume_delta,
#         )

#     def _load_clip_frames(
#         self, video_path: str, start_frame: int, end_frame: int
#     ) -> List[np.ndarray]:
#         """Load specific frames from video"""
#         frames = []

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return frames

#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#         for _ in range(end_frame - start_frame):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)

#         cap.release()
#         return frames


def create_dataloaders(
    video_dir: str,
    annotation_dir: str,
    volume_csv_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    clip_length: int = 6,
    stride: int = 3,
    train_split: float = 0.7,
    val_split: float = 0.15,
    input_size: Tuple[int, int] = (328, 512),
    max_clips_per_video: Optional[int] = None,
    testing=False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create fast DataLoaders using pooled clip approach.
    """
    import glob
    from torchvision import transforms
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Collect video paths
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    annotation_paths = []

    for vp in video_paths:
        base = os.path.basename(vp).split(".")[0]
        xml_path = os.path.join(annotation_dir, f"{base}.xml")
        if os.path.exists(xml_path):
            annotation_paths.append(xml_path)
        else:
            video_paths.remove(vp)

    console.print(f"Found {len(video_paths)} videos with XML annotations")

    # Video-level split (maintain video integrity)
    unique_ids = sorted(list(set(os.path.basename(p).split(".")[0] for p in video_paths)))
    random.seed(SEED)
    random.shuffle(unique_ids)
    min_num = 1 if testing else 2
    n_test = max(min_num, int((1 - train_split - val_split) * len(unique_ids)))
    n_val = max(min_num, int(val_split * len(unique_ids)))

    test_ids = unique_ids[:n_test]
    val_ids = unique_ids[n_test : n_test + n_val]
    train_ids = unique_ids[n_test + n_val :]

    def split_paths(ids):
        vids, annos = [], []
        for vp, ap in zip(video_paths, annotation_paths):
            if os.path.basename(vp).split(".")[0] in ids:
                vids.append(vp)
                annos.append(ap)
        return vids, annos

    train_vids, train_annos = split_paths(train_ids)
    val_vids, val_annos = split_paths(val_ids)
    test_vids, test_annos = split_paths(test_ids)

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.05, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    console.print("Creating datasets...", style="cyan")

    train_dataset = SurgicalInterpDataset(
        train_vids,
        train_annos,
        volume_csv_path,
        clip_length=clip_length,
        stride=stride,
        transform=train_transform,
        max_clips_per_video=max_clips_per_video,
    )

    val_dataset = SurgicalInterpDataset(
        val_vids,
        val_annos,
        volume_csv_path,
        clip_length=clip_length,
        stride=stride * 2,
        transform=val_transform,
        max_clips_per_video=max_clips_per_video // 2 if max_clips_per_video else None,
    )

    test_dataset = SurgicalInterpDataset(
        test_vids,
        test_annos,
        volume_csv_path,
        clip_length=clip_length,
        stride=stride * 2,
        transform=val_transform,
        max_clips_per_video=max_clips_per_video // 2 if max_clips_per_video else None,
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
    table = Table(title="Dataset Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Videos", justify="right", style="green")
    table.add_column("Clips", justify="right", style="yellow")

    table.add_row("Train", str(len(train_vids)), f"{len(train_dataset):,}")
    table.add_row("Validation", str(len(val_vids)), f"{len(val_dataset):,}")
    table.add_row("Test", str(len(test_vids)), f"{len(test_dataset):,}")

    console.print(table)

    return train_loader, val_loader, test_loader


def volume_loss_with_mask(predictions, targets, clip_labels):
    """
    Focus loss on bleeding clips and use Huber loss for robustness
    """
    # Huber loss is less sensitive to outliers than MSE
    huber_loss = F.smooth_l1_loss(predictions, targets, reduction='none')

    # Create weights: higher for bleeding clips
    weights = torch.where(clip_labels == 1,
                         torch.ones_like(targets) * 2.0,    # 2x weight for bleeding
                         torch.ones_like(targets) * 0.5)     # 0.5x weight for non-bleeding

    weighted_loss = (huber_loss * weights).mean()
    return weighted_loss

def compute_volume_loss_separate(predictions, targets, clip_labels):
    """
    Compute loss separately for bleeding and non-bleeding clips.
    Penalizes low predictions on bleeding clips.
    """
    bleeding_mask = clip_labels == 1
    non_bleeding_mask = ~bleeding_mask

    total_loss = torch.tensor(0.0, device=predictions.device)
    loss_components = {}

    # Loss for bleeding clips (should predict > 0)
    if bleeding_mask.any():
        bleeding_preds = predictions[bleeding_mask]
        bleeding_targets = targets[bleeding_mask]

        # Huber loss for robustness
        bleeding_loss = F.smooth_l1_loss(bleeding_preds, bleeding_targets)

        # Penalty for predicting too low on bleeding clips with actual volume
        actual_bleeding_mask = bleeding_targets > 0.01
        if actual_bleeding_mask.any():
            low_pred_penalty = torch.relu(0.1 - bleeding_preds[actual_bleeding_mask]).mean()
        else:
            low_pred_penalty = torch.tensor(0.0, device=predictions.device)

        total_loss = total_loss + bleeding_loss + 0.5 * low_pred_penalty
        loss_components['bleeding_loss'] = bleeding_loss.item()
        loss_components['low_pred_penalty'] = low_pred_penalty.item()

    # Loss for non-bleeding clips (should predict 0)
    if non_bleeding_mask.any():
        non_bleeding_preds = predictions[non_bleeding_mask]
        non_bleeding_targets = targets[non_bleeding_mask]

        # Lower weight for non-bleeding
        non_bleeding_loss = F.smooth_l1_loss(non_bleeding_preds, non_bleeding_targets)
        total_loss = total_loss + 0.2 * non_bleeding_loss
        loss_components['non_bleeding_loss'] = non_bleeding_loss.item()

    return total_loss, loss_components


def train_model(
    video_dir,
    annotations_dir,
    volume_csv_path,
    device,
    clip_length=6,
    stride=3,
    epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    output_dir="./models_bleeding",
    model_name="interp_bl_model",
    patience=10,
    small_dataset=False,
):
    """
    Train bleeding detection model with separate optimizers for classification and volume regression.
    """
    os.makedirs(output_dir, exist_ok=True)
    # model = VideoBleedingDetector(num_classes=2, dropout_rate=0.3)
    model = VideoBleedingDetectorLSTM(dropout_rate=0.5)
    console.print("[bold]Using R2Plus1D, combined with LSTM Model [/bold]", style="cyan")

    model = model.to(device)

    # Separate optimizers with different learning rates
    clf_params = list(model.backbone.parameters()) + list(model.classifier.parameters())
    vol_params = list(model.volume_regressor.parameters())

    clf_optimizer = torch.optim.Adam(
        clf_params,
        lr=learning_rate,
        weight_decay=1e-4
    )

    vol_optimizer = torch.optim.Adam(
        vol_params,
        lr=learning_rate * 50,  # 50x higher learning rate for volume
        weight_decay=1e-5
    )

    # Separate schedulers
    clf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        clf_optimizer, mode='min', factor=0.7, patience=5, verbose=True
    )

    vol_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        vol_optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Tracking
    history = {
        "train_loss": [], "train_clf_loss": [], "train_vol_loss": [],
        "val_loss": [], "val_clf_loss": [], "val_vol_loss": [],
        "val_precision": [], "val_recall": [], "val_accuracy": [], "val_f1": [],
        "val_volume_mae": [], "val_volume_rmse": []
    }

    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")

    console.print(
        Panel.fit(
            f"Bleeding Detection & Quantification\n"
            f"Device: {device}\n"
            f"Epochs: {epochs} | Batch Size: {batch_size}\n"
            f"CLF LR: {learning_rate} | VOL LR: {learning_rate * 50}\n"
            f"Tasks: Classification + Volume Regression\n",
            title="Training Configuration",
            border_style="white",
        )
    )

    # Create DataLoaders
    console.print(f"Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        video_dir=video_dir,
        annotation_dir=annotations_dir,
        volume_csv_path=volume_csv_path,
        batch_size=batch_size,
        num_workers=4,
        clip_length=clip_length,
        stride=stride,
        train_split=0.7,
        val_split=0.15,
        input_size=(328, 512),
        max_clips_per_video=None,
        testing=small_dataset,
    )

    # Training loop
    for epoch in range(epochs):
        console.print(f"\n[bold blue]Epoch {epoch + 1}/{epochs}[/bold blue]")

        # Calculate volume loss weight with warmup
        vol_weight = min(5.0, 0.5 + (epoch / 3) * 4.5)  # Warmup over 3 epochs
        console.print(f"Volume loss weight: {vol_weight:.2f}")

        # Training
        model.train()
        train_metrics = _train_epoch_dual_opt(
            model, train_loader, clf_optimizer, vol_optimizer, device, vol_weight
        )

        # Validation
        model.eval()
        val_metrics = _validate_epoch(model, val_loader, device)

        # Update history
        for key in train_metrics:
            history[f"train_{key}"].append(train_metrics[key])
        for key in val_metrics:
            history[f"val_{key}"].append(val_metrics[key])

        # Update both schedulers
        clf_scheduler.step(val_metrics["clf_loss"])
        vol_scheduler.step(val_metrics["vol_loss"])

        # Results table
        table = Table(title=f"Epoch {epoch+1}/{epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", justify="right", style="green")
        table.add_column("Validation", justify="right", style="yellow")

        table.add_row("Total Loss", f"{train_metrics['loss']:.4f}", f"{val_metrics['loss']:.4f}")
        table.add_row("Classification Loss", f"{train_metrics['clf_loss']:.4f}", f"{val_metrics['clf_loss']:.4f}")
        table.add_row("Volume Loss", f"{train_metrics['vol_loss']:.4f}", f"{val_metrics['vol_loss']:.4f}")
        table.add_row("Accuracy", "-", f"{val_metrics['accuracy']:.4f}")
        table.add_row("F1 Score", "-", f"{val_metrics['f1']:.4f}")

        console.print(table)

        # Print current learning rates
        console.print(f"CLF LR: {clf_optimizer.param_groups[0]['lr']:.2e}, "
                     f"VOL LR: {vol_optimizer.param_groups[0]['lr']:.2e}")

        # Save best model based on F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'clf_optimizer_state_dict': clf_optimizer.state_dict(),
                'vol_optimizer_state_dict': vol_optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1
            }, best_model_path)
            console.print(f"[bold green]New best model saved: F1 = {best_f1:.4f}[/bold green]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print("[yellow]Early stopping triggered[/yellow]")
                break

    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)

    with open(os.path.join(output_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    console.print(
        Panel.fit(
            f"[bold green]Training completed[/bold green]\n"
            f"Best F1: {best_f1:.4f}\n"
            f"Best model: {best_model_path}\n"
            f"Final model: {final_model_path}",
            title="Training Complete",
            border_style="green",
        )
    )

    # Test evaluation
    test_metrics = test_model(model, test_loader, device)

    return model, history, test_metrics


def _train_epoch_dual_opt(model, train_loader, clf_optimizer, vol_optimizer, device, vol_weight=5.0):
    """Training epoch with dual optimizers and better volume loss handling"""
    total_loss = 0.0
    total_clf_loss = 0.0
    total_vol_loss = 0.0

    # Track loss components
    all_loss_components = []

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

        task = progress.add_task("Training", total=len(train_loader))

        for batch_idx, (clips, clip_labels, _, _, _, volume_deltas) in enumerate(train_loader):
            clips = clips.to(device, dtype=torch.float32)
            volume_deltas = volume_deltas.to(device, dtype=torch.float32)
            clip_labels = clip_labels.to(device)

            # Zero gradients for both optimizers
            clf_optimizer.zero_grad()
            vol_optimizer.zero_grad()

            # Forward pass
            clip_preds, volume_delta_preds = model(clips)

            # Classification loss
            clf_loss = F.cross_entropy(clip_preds, clip_labels)

            # Volume loss using separation method
            vol_loss, loss_components = compute_volume_loss_separate(
                volume_delta_preds, volume_deltas, clip_labels
            )

            # Store components for tracking
            all_loss_components.append(loss_components)

            # Combined loss
            total_loss_batch = clf_loss + vol_weight * vol_loss

            # Backward pass
            total_loss_batch.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update both optimizers
            clf_optimizer.step()
            vol_optimizer.step()

            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_clf_loss += clf_loss.item()
            total_vol_loss += vol_loss.item()

            # Debug logging every 100 batches
            # Debug logging every 100 batches
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    # Need to recreate bleeding_mask here
                    bleeding_mask = clip_labels == 1

                    if bleeding_mask.any():
                        bleeding_preds = volume_delta_preds[bleeding_mask]
                        bleeding_targets = volume_deltas[bleeding_mask]
                        console.print(
                            f"Batch {batch_idx}: Bleeding preds mean: {bleeding_preds.mean():.4f}, "
                            f"std: {bleeding_preds.std():.4f}, targets mean: {bleeding_targets.mean():.4f}",
                            style="dim"
                        )

            progress.update(task, advance=1)

    # Calculate average loss components
    avg_components = {}
    for key in ['bleeding_loss', 'low_pred_penalty', 'non_bleeding_loss']:
        values = [lc[key] for lc in all_loss_components if key in lc]
        if values:
            avg_components[key] = np.mean(values)

    # Print summary
    if avg_components:
        console.print(f"\nTraining loss components:")
        for key, value in avg_components.items():
            console.print(f"  {key}: {value:.4f}")

    return {
        "loss": total_loss / len(train_loader),
        "clf_loss": total_clf_loss / len(train_loader),
        "vol_loss": total_vol_loss / len(train_loader),
    }


def _train_epoch(model, train_loader, optimizer, device):
    total_loss = 0.0
    total_clf_loss = 0.0
    total_vol_loss = 0.0

    for batch_idx, (clips, clip_labels, _, _, _, volume_deltas) in enumerate(train_loader):
        clips = clips.to(device, dtype=torch.float32)
        volume_deltas = volume_deltas.to(device, dtype=torch.float32)
        clip_labels = clip_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        clip_preds, volume_delta_preds = model(clips)

        # Classification loss
        clf_loss = F.cross_entropy(clip_preds, clip_labels)
        vol_loss = volume_loss_with_mask(volume_delta_preds, volume_deltas, clip_labels)

        # Scale volume loss up significantly since it's stuck
        total_loss_batch = clf_loss + 5.0 * vol_loss  # Increased from 0.5 to 5.0, 10x
        total_loss_batch.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += total_loss_batch.item()
        total_clf_loss += clf_loss.item()
        total_vol_loss += vol_loss.item()

        # Debug every 10 batches
        if batch_idx % 10 == 0:
            with torch.no_grad():
                bleeding_mask = clip_labels == 1
                if bleeding_mask.any():
                    print(f"Batch {batch_idx} - Bleeding preds mean: {volume_delta_preds[bleeding_mask].mean():.4f}, "
                          f"targets mean: {volume_deltas[bleeding_mask].mean():.4f}")

    return {
        "loss": total_loss / len(train_loader),
        "clf_loss": total_clf_loss / len(train_loader),
        "vol_loss": total_vol_loss / len(train_loader),
    }


def _validate_epoch(model, val_loader, device):
    """Single validation epoch with consistent loss calculation"""
    total_loss = 0.0
    total_clf_loss = 0.0
    total_vol_loss = 0.0
    all_preds = []
    all_labels = []

    # For tracking volume predictions
    all_volume_preds = []
    all_volume_targets = []
    bleeding_volume_preds = []
    bleeding_volume_targets = []

    with torch.no_grad():
        with Progress(
            TextColumn("Validation"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:

            task = progress.add_task("Validating", total=len(val_loader))

            for batch_idx, (clips, clip_labels, _, _, _, volume_deltas) in enumerate(val_loader):
                clips = clips.to(device, dtype=torch.float32, non_blocking=True)
                volume_deltas = volume_deltas.to(device, dtype=torch.float32, non_blocking=True)
                clip_labels = clip_labels.to(device)

                # Forward pass
                clip_preds, volume_delta_preds = model(clips)

                # Classification loss
                clf_loss = F.cross_entropy(clip_preds, clip_labels)

                # Use separate volume losses for bleeding & non bleeding
                vol_loss, loss_components = compute_volume_loss_separate(
                    volume_delta_preds, volume_deltas, clip_labels
                )

                # Use same loss weight as final training weight (5.0)
                total_loss_batch = clf_loss + 5.0 * vol_loss

                # Accumulate losses
                total_loss += total_loss_batch.item()
                total_clf_loss += clf_loss.item()
                total_vol_loss += vol_loss.item()

                # Collect classification predictions
                _, predicted = torch.max(clip_preds, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(clip_labels.cpu().numpy())

                # Collect volume predictions for analysis
                all_volume_preds.extend(volume_delta_preds.cpu().numpy())
                all_volume_targets.extend(volume_deltas.cpu().numpy())

                # Still need bleeding_mask for collecting bleeding-specific predictions
                bleeding_mask = clip_labels == 1
                if bleeding_mask.any():
                    bleeding_volume_preds.extend(volume_delta_preds[bleeding_mask].cpu().numpy())
                    bleeding_volume_targets.extend(volume_deltas[bleeding_mask].cpu().numpy())

                # Debug logging every 50 batches
                if batch_idx % 50 == 0:
                    # Overall stats
                    console.print(
                        f"Batch {batch_idx} - Vol preds: mean={volume_delta_preds.mean():.4f}, "
                        f"std={volume_delta_preds.std():.4f}, "
                        f"min={volume_delta_preds.min():.4f}, max={volume_delta_preds.max():.4f}",
                        style="dim"
                    )

                    # Bleeding-specific stats
                    if bleeding_mask.any():
                        bleeding_preds = volume_delta_preds[bleeding_mask]
                        bleeding_targets = volume_deltas[bleeding_mask]
                        console.print(
                            f"  Bleeding only - preds: {bleeding_preds.mean():.4f}, "
                            f"targets: {bleeding_targets.mean():.4f}",
                            style="dim"
                        )

                progress.update(task, advance=1)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Volume prediction metrics
    all_volume_preds = np.array(all_volume_preds)
    all_volume_targets = np.array(all_volume_targets)

    # Calculate volume MAE and RMSE
    volume_mae = np.mean(np.abs(all_volume_preds - all_volume_targets))
    volume_rmse = np.sqrt(np.mean((all_volume_preds - all_volume_targets) ** 2))

    # Print summary statistics
    console.print("\n[cyan]Validation Volume Statistics:[/cyan]")
    console.print(f"Overall - MAE: {volume_mae:.4f}, RMSE: {volume_rmse:.4f}")

    if bleeding_volume_preds:
        bleeding_mae = np.mean(np.abs(np.array(bleeding_volume_preds) - np.array(bleeding_volume_targets)))
        console.print(f"Bleeding clips - MAE: {bleeding_mae:.4f}")
        console.print(f"Bleeding predictions - mean: {np.mean(bleeding_volume_preds):.4f}, "
                     f"std: {np.std(bleeding_volume_preds):.4f}")

    return {
        "loss": float(total_loss / len(val_loader)),
        "clf_loss": float(total_clf_loss / len(val_loader)),
        "vol_loss": float(total_vol_loss / len(val_loader)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "volume_mae": float(volume_mae),
        "volume_rmse": float(volume_rmse),
    }


# def _validate_epoch(model, val_loader, device):
#     """Single validation epoch"""
#     total_loss = 0.0
#     total_clf_loss = 0.0
#     total_vol_loss = 0.0
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         with Progress(
#             TextColumn("Validation"),
#             BarColumn(),
#             TextColumn("{task.percentage:>3.0f}%"),
#             console=console,
#             transient=True,
#         ) as progress:

#             task = progress.add_task("Validating", total=len(val_loader))

#             for clips, clip_labels, _, _, _, volume_deltas in val_loader:

#                 clips = clips.to(device, dtype=torch.float32, non_blocking=True)
#                 volume_deltas = volume_deltas.to(
#                     device, dtype=torch.float32, non_blocking=True
#                 )
#                 clip_labels = clip_labels.to(device)

#                 # Forward pass
#                 clip_preds, volume_delta_preds = model(clips)
#                 volume_delta_preds = (
#                     volume_delta_preds.float()
#                 )  # Ensure float type for regression
#                 # Losses
#                 clf_loss = F.cross_entropy(clip_preds, clip_labels)
#                 vol_loss = F.mse_loss(volume_delta_preds.squeeze(), volume_deltas)
#                 total_loss_batch = clf_loss + 0.5 * vol_loss

#                 # Accumulate
#                 total_loss += total_loss_batch.item()
#                 total_clf_loss += clf_loss.item()
#                 total_vol_loss += vol_loss.item()

#                 # Collect predictions
#                 _, predicted = torch.max(clip_preds, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(clip_labels.cpu().numpy())

#                 # Add this in validation loop to debug
#                 print(
#                     f"Volume deltas stats: min={volume_deltas.min():.6f}, max={volume_deltas.max():.6f}, std={volume_deltas.std():.6f}"
#                 )
#                 print(
#                     f"Predictions stats: min={volume_delta_preds.min():.6f}, max={volume_delta_preds.max():.6f}, std={volume_delta_preds.std():.6f}"
#                 )
#                 progress.update(task, advance=1)

#     # Calculate metrics
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, zero_division=0)
#     recall = recall_score(all_labels, all_preds, zero_division=0)
#     f1 = f1_score(all_labels, all_preds, zero_division=0)

#     return {
#         "loss": total_loss / len(val_loader),
#         "clf_loss": total_clf_loss / len(val_loader),
#         "vol_loss": total_vol_loss / len(val_loader),
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }


def test_model(model, test_loader, device):
    """
    Test model and evaluate blood loss accumulation per video.
    """
    model.eval()
    test_dataset = test_loader.dataset

    # Group clips by video from dataset metadata
    video_clips = defaultdict(list)
    for i, clip_info in enumerate(test_dataset.all_clips):
        video_clips[clip_info["video_id"]].append((i, clip_info))

    console.print(f"Testing on {len(video_clips)} videos")

    video_results = {}

    with torch.no_grad():
        with Progress(
            TextColumn("Testing videos"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:

            task = progress.add_task("Processing", total=len(video_clips))

            for video_id, clips_info in video_clips.items():
                # Sort clips by temporal order
                clips_info.sort(key=lambda x: x[1]["start_frame"])

                # Process all clips for this video
                all_volume_deltas = []
                all_clip_preds = []

                # Process clips in batches for memory efficiency
                batch_size = test_loader.batch_size
                for i in range(0, len(clips_info), batch_size):
                    batch_indices = [
                        clip_idx for clip_idx, _ in clips_info[i : i + batch_size]
                    ]

                    # Get batch data directly from dataset
                    batch_clips = []
                    for idx in batch_indices:
                        clip_tensor, _, _, _, _, _ = test_dataset[idx]
                        batch_clips.append(clip_tensor)

                    if not batch_clips:
                        continue

                    batch_clips = torch.stack(batch_clips).to(device)

                    # Forward pass
                    clip_preds, volume_delta_preds = model(batch_clips)

                    # Store predictions
                    all_volume_deltas.extend(volume_delta_preds.cpu().numpy())
                    _, predicted = torch.max(clip_preds, 1)
                    all_clip_preds.extend(predicted.cpu().numpy())

                # Calculate cumulative volume for this video
                predicted_cumulative = np.cumsum(all_volume_deltas)
                final_predicted_volume = (
                    predicted_cumulative[-1] if predicted_cumulative.size > 0 else 0.0
                )

                # Get ground truth volume
                ground_truth_volume = clips_info[0][1]["video_total_volume"]

                # Classification metrics for this video
                true_labels = [clip_info["has_bleeding"] for _, clip_info in clips_info]
                video_accuracy = accuracy_score(true_labels, all_clip_preds)

                video_results[video_id] = {
                    "predicted_volume": float(final_predicted_volume),
                    "ground_truth_volume": float(ground_truth_volume),
                    "volume_error": float(
                        abs(final_predicted_volume - ground_truth_volume)
                    ),
                    "relative_error": float(
                        abs(final_predicted_volume - ground_truth_volume)
                        / max(ground_truth_volume, 1e-6)
                    ),
                    "accuracy": float(video_accuracy),
                    "num_clips": len(clips_info),
                }

                progress.update(task, advance=1)

    # Overall statistics
    predicted_volumes = [r["predicted_volume"] for r in video_results.values()]
    ground_truth_volumes = [r["ground_truth_volume"] for r in video_results.values()]
    volume_errors = [r["volume_error"] for r in video_results.values()]
    relative_errors = [r["relative_error"] for r in video_results.values()]

    volume_mse = np.mean(
        [(p - g) ** 2 for p, g in zip(predicted_volumes, ground_truth_volumes)]
    )
    volume_mae = np.mean(volume_errors)
    mean_relative_error = np.mean(relative_errors)

    # Results table
    table = Table(title="Test Results - Volume Prediction")
    table.add_column("Video ID", style="cyan")
    table.add_column("Predicted (ml)", justify="right", style="green")
    table.add_column("Ground Truth (ml)", justify="right", style="yellow")
    table.add_column("Error (ml)", justify="right", style="red")
    table.add_column("Relative Error (%)", justify="right", style="magenta")
    table.add_column("Accuracy", justify="right", style="blue")

    for video_id, results in sorted(video_results.items()):
        table.add_row(
            video_id,
            f"{results['predicted_volume']:.1f}",
            f"{results['ground_truth_volume']:.1f}",
            f"{results['volume_error']:.1f}",
            f"{results['relative_error']*100:.1f}%",
            f"{results['accuracy']:.3f}",
        )

    console.print(table)

    # Summary statistics
    summary_table = Table(title="Overall Test Performance")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Volume MSE", f"{volume_mse:.2f}")
    summary_table.add_row("Volume MAE", f"{volume_mae:.2f} ml")
    summary_table.add_row("Mean Relative Error", f"{mean_relative_error*100:.1f}%")
    summary_table.add_row("Total Videos", str(len(video_results)))

    console.print(summary_table)

    test_metrics = {
        "video_results": video_results,
        "volume_mse": float(volume_mse),
        "volume_mae": float(volume_mae),
        "mean_relative_error": float(mean_relative_error),
        "predicted_volumes": predicted_volumes,
        "ground_truth_volumes": ground_truth_volumes,
    }

    return test_metrics


def plot_training_history(history_or_path, save_dir=None, model_name="model"):
    """
    Plot training history with clean seaborn styling.
    Creates loss comparison and overview plots.

    Args:
        history_or_path: Either a history dictionary or path to JSON file
        save_dir: Directory to save plots (optional)
        model_name: Name for plot titles
    """
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Handle both dictionary and file path inputs
    if isinstance(history_or_path, str):
        # if file path, load JSON
        with open(history_or_path, "r") as f:
            history = json.load(f)
        print(f"Loaded history from {history_or_path}")
    elif isinstance(history_or_path, dict):
        # already dict
        history = history_or_path
    else:
        raise ValueError(
            "Input must be either a history dictionary or path to JSON file"
        )

    # Set seaborn style
    plt.style.use("default")
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.2)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Loss Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        history["train_loss"],
        "o-",
        linewidth=2,
        markersize=4,
        label="Train Loss",
        color="#1f77b4",
    )
    ax.plot(
        epochs,
        history["val_loss"],
        "o-",
        linewidth=2,
        markersize=4,
        label="Validation Loss",
        color="#ff7f0e",
    )

    ax.set_title("Training vs Validation Loss", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Add min validation loss annotation
    min_val_idx = np.argmin(history["val_loss"])
    min_val_loss = history["val_loss"][min_val_idx]
    ax.annotate(
        f"Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_idx + 1}",
        xy=(min_val_idx + 1, min_val_loss),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2: Overview Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Training Overview for {model_name}", fontsize=16, fontweight="bold", y=0.98
    )

    # Loss plot (top left)
    axes[0, 0].plot(
        epochs,
        history["train_loss"],
        "o-",
        linewidth=2,
        label="Train",
        color="#2ca02c",
        markersize=3,
    )
    axes[0, 0].plot(
        epochs,
        history["val_loss"],
        "o-",
        linewidth=2,
        label="Validation",
        color="#d62728",
        markersize=3,
    )
    axes[0, 0].set_title("Loss", fontweight="bold")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot (top right)
    axes[0, 1].plot(
        epochs,
        history["val_accuracy"],
        "o-",
        linewidth=2,
        label="Accuracy",
        color="#9467bd",
        markersize=3,
    )
    axes[0, 1].set_title("Validation Accuracy", fontweight="bold")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score (bottom left)
    axes[1, 0].plot(
        epochs,
        history["val_f1"],
        "o-",
        linewidth=2,
        label="F1 Score",
        color="#ff7f0e",
        markersize=3,
    )
    axes[1, 0].set_title("F1 Score", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Volume Loss Components (bottom right)
    axes[1, 1].plot(
        epochs,
        history["train_vol_loss"],
        "o-",
        linewidth=2,
        label="Train Vol Loss",
        color="#17becf",
        markersize=3,
    )
    axes[1, 1].plot(
        epochs,
        history["val_vol_loss"],
        "o-",
        linewidth=2,
        label="Val Vol Loss",
        color="#e377c2",
        markersize=3,
    )
    axes[1, 1].set_title("Volume Loss", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Volume Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/training_overview.png", dpi=450, bbox_inches="tight")
    plt.show()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser(description="Bleeding Quantification Model")
    parser.add_argument("--dev", type=int, default=0, help="Device ID)")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--small_dataset",
        type=bool,
        default=False,
        help="Use small dataset for quick testing",
    )
    args = parser.parse_args()

    if args.small_dataset:
        print("Using small dataset for quick testing")
        VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/test_videos"
        ANNO_FOLDER = "/home/r.rohangirish/mt_ble/data/test_labels_xml"
    else:
        VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/videos"
        ANNO_FOLDER = "/home/r.rohangirish/mt_ble/data/labels_xml"

    output_dir = "./models_interpolation"
    model_name="interp_bl_model_v3"

    model, history, _ = train_model(
        video_dir=VIDEO_DIR,
        annotations_dir=ANNO_FOLDER,
        volume_csv_path="/home/r.rohangirish/mt_ble/data/labels_quantification/BL_data_combined.csv",
        device=torch.device(f"cuda:{args.dev}"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        patience=10,
        output_dir=output_dir,
        model_name=model_name,
        small_dataset=args.small_dataset,
    )

    plot_training_history(
        os.path.join(output_dir, f"{model_name}_history.json"),
        save_dir="./models_interpolation",
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
