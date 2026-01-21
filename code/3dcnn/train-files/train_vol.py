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

FPS_2 = True

# torch.use_deterministic_algorithms(True)
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
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
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
        clip_length: int = 12,
        stride: int = 6,
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
        self.check_volume_alignment()

    def _load_volume_csv(self, path: str) -> Dict:
        """Load volume measurements from CSV"""
        df = pd.read_csv(path)
        df = df.dropna(subset=["video_name"])

        data = defaultdict(lambda: {"checkpoints": [], "total_volume": 0.0})
        skip_videos = []
        for _, row in df.iterrows():
            video_name = str(row["video_name"])

            if pd.notna(row.get("a_e_bl")):
                total_volume = float(row["a_e_bl"])
                if total_volume > 0.0:
                    data[video_name]["total_volume"] = total_volume
                else:
                    skip_videos.append(video_name)
                    continue  # Skip entries with zero volume

            if pd.notna(row.get("measurement_frame")) and pd.notna(row.get("bl_loss")):
                measurement_frame = int(row["measurement_frame"])
                if FPS_2:
                    measurement_frame *= 2  # scale to 2fps
                cumulative_volume = float(row["bl_loss"])
                data[video_name]["checkpoints"].append(
                    (measurement_frame, cumulative_volume)
                )

        for video_name in data:
            data[video_name]["checkpoints"].sort(key=lambda x: x[0])
        skip_videos = list(set(skip_videos))
        console.print(
            f"[bold green]Skipped videos: {', '.join(set(skip_videos))} with 0.0 a_e_bl.[/bold green]"
        )
        return data

    def _parse_xml_annotations(self, xml_path: str) -> List[Dict]:
        """Parse CVAT XML file to extract bleeding annotations (upsampled if FPS_2)"""
        if not os.path.exists(xml_path):
            return []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotations = []

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

            for track in root.findall("track"):
                track_label = track.get("label")
                if track_label in ["BL_Low", "BL_Medium", "BL_High"]:
                    last_frame = None
                    for box in track.findall("box"):
                        if int(box.get("outside", "0")) == 1:
                            continue

                        frame_num = int(box.get("frame"))
                        if FPS_2:
                            frame_num *= 2  # double for 2fps

                        annotations.append(
                            {
                                "frame": frame_num,
                                "label": track_label,
                                "original_width": orig_width,
                                "original_height": orig_height,
                            }
                        )

                        if (
                            FPS_2
                            and last_frame is not None
                            and frame_num - last_frame == 2
                        ):
                            mid_frame = last_frame + 1
                            annotations.append(
                                {
                                    "frame": mid_frame,
                                    "label": track_label,
                                    "original_width": orig_width,
                                    "original_height": orig_height,
                                }
                            )

                        last_frame = frame_num

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return []

        return annotations

    def _xml_to_frame_labels(self, xml_path: str, total_frames: int) -> np.ndarray:
        """Convert XML annotations to per-frame bleeding labels"""
        if FPS_2:
            frame_labels = np.zeros(total_frames * 2, dtype=np.int32)
        else:
            frame_labels = np.zeros(total_frames, dtype=np.int32)

        severity_map = {"BL_Low": 1, "BL_Medium": 2, "BL_High": 3}
        annotations = self._parse_xml_annotations(xml_path)

        for anno in annotations:
            frame_num = anno["frame"]
            label = anno["label"]
            severity = severity_map.get(label, 1)

            if 0 <= frame_num < len(frame_labels):
                frame_labels[frame_num] = max(frame_labels[frame_num], severity)

        return frame_labels

    def _interpolate_volume_between_checkpoints(
        self,
        video_clips: List[Dict],
        checkpoints: List[Tuple[int, float]],
        total_volume: float,
    ) -> List[Dict]:
        """
        Assign volume by interpolating within segments defined by checkpoints.
        Each segment's volume is distributed only among bleeding clips in that segment.
        """
        if not checkpoints:
            print(f"  No checkpoints available, using uniform distribution")
            return self._assign_volume_uniform(video_clips, total_volume)

        # Sort everything by frame order
        checkpoints = sorted(checkpoints, key=lambda x: x[0])
        video_clips = sorted(video_clips, key=lambda x: x["center_frame"])

        # Create segments: [(start_frame, end_frame, volume_for_segment)]
        segments = []

        # First segment: 0 to first checkpoint
        segments.append((0, checkpoints[0][0], checkpoints[0][1]))

        # Middle segments: between consecutive checkpoints
        for i in range(1, len(checkpoints)):
            start_frame = checkpoints[i - 1][0]
            end_frame = checkpoints[i][0]
            segment_volume = checkpoints[i][1]  # This checkpoint's incremental volume
            segments.append((start_frame, end_frame, segment_volume))

        print(f"\n[{video_clips[0]['video_id']}] Segment-based interpolation:")
        print(f"  Segments: {segments}")

        # First, assign deltas based on segments
        for clip in video_clips:
            clip["volume_delta"] = 0.0  # Initialize all to 0

        # Process each segment
        for seg_start, seg_end, seg_volume in segments:
            # Find bleeding clips in this segment
            segment_clips = []
            for clip in video_clips:
                if clip["has_bleeding"] and seg_start <= clip["center_frame"] < seg_end:
                    segment_clips.append(clip)

            if not segment_clips:
                print(
                    f"  Segment [{seg_start}-{seg_end}]: No bleeding clips, skipping {seg_volume}ml"
                )
                continue

            # Distribute segment volume among bleeding clips
            volume_per_clip = seg_volume / len(segment_clips)

            print(
                f"  Segment [{seg_start}-{seg_end}]: {len(segment_clips)} bleeding clips, "
                f"{volume_per_clip:.2f}ml each (total: {seg_volume}ml)"
            )

            # Assign delta volumes
            for clip in segment_clips:
                clip["volume_delta"] = volume_per_clip

        # Now calculate cumulative volumes in temporal order
        cumulative = 0.0
        for clip in video_clips:  # Already sorted by center_frame
            cumulative += clip[
                "volume_delta"
            ]  # Add this clip's contribution (0 for non-bleeding)
            clip["target_cumulative_volume"] = cumulative

        # Final summary
        total_assigned = sum(c["volume_delta"] for c in video_clips)
        final_cumulative = (
            video_clips[-1]["target_cumulative_volume"] if video_clips else 0.0
        )

        print(
            f"  Total deltas: {total_assigned:.2f}ml, Final cumulative: {final_cumulative:.2f}ml, Expected: {total_volume:.2f}ml"
        )

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
            console.print(f"\n[{video_id}] Assigned uniform distribution")
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
            cap.release()

            # Get volume info
            volume_info = self.volume_data.get(
                video_id, {"checkpoints": [], "total_volume": 0.0}
            )
            total_volume = volume_info["total_volume"]
            checkpoints = volume_info["checkpoints"].copy()

            # Check if we need to add a final segment
            sum_incremental = sum(v for _, v in checkpoints)
            if abs(total_volume - sum_incremental) > 0.1:
                # Remaining volume goes from last checkpoint (or 0) to video end
                if checkpoints:
                    # There's remaining volume after last checkpoint
                    last_checkpoint_frame = max(f for f, _ in checkpoints)
                    if (
                        frame_count - 1 > last_checkpoint_frame
                    ):  # Only add if there's space
                        remaining_volume = total_volume - sum_incremental
                        checkpoints.append((frame_count - 1, remaining_volume))
                        print(
                            f"[{video_id}] Added final segment: {last_checkpoint_frame} to {frame_count-1}, "
                            f"volume: {remaining_volume:.1f}ml"
                        )
                else:
                    # No checkpoints at all - entire volume goes to end
                    checkpoints.append((frame_count - 1, total_volume))
                    print(
                        f"[{video_id}] No checkpoints, entire {total_volume}ml assigned to video end"
                    )

            # Parse bleeding annotations
            bleeding_labels = self._xml_to_frame_labels(anno_path, frame_count)

            # Create clips
            video_clips = []
            for start_idx in range(0, frame_count - self.clip_length + 1, self.stride):
                end_idx = start_idx + self.clip_length
                center_frame = (start_idx + end_idx) // 2

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

            # Assign volumes using segment-based interpolation
            video_clips = self._assign_volume(video_clips, total_volume)
            all_clips.extend(video_clips)

        return all_clips

    def check_volume_alignment(self):
        """Check if final bleeding clip cumulative volume ≈ expected total volume."""
        for vid in set(c["video_id"] for c in self.all_clips):
            video_clips = [c for c in self.all_clips if c["video_id"] == vid]
            video_clips.sort(key=lambda x: x["center_frame"])

            # Get the LAST BLEEDING clip, not just the last clip
            bleeding_clips = [c for c in video_clips if c["has_bleeding"]]

            if bleeding_clips:
                last_bleeding_clip = bleeding_clips[
                    -1
                ]  # Last bleeding clip by temporal order
                final_predicted = last_bleeding_clip["target_cumulative_volume"]
                expected = self.volume_data.get(vid, {}).get("total_volume", 0.0)

                if abs(final_predicted - expected) > 1.0:  # 1ml tolerance
                    print(f"!! Volume mismatch for {vid}:")
                    print(f"     Predicted final: {final_predicted:.1f}ml")
                    print(f"     Expected total: {expected:.1f}ml")
                    print(f"     Difference: {abs(final_predicted - expected):.1f}ml")
            else:
                # No bleeding clips at all
                expected = self.volume_data.get(vid, {}).get("total_volume", 0.0)
                if expected > 0:
                    print(
                        f"!! Volume mismatch for {vid}: No bleeding clips but expected {expected:.1f}ml"
                    )

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
        """Print dataset statistics with frame count comparison"""
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

        console.print("\n[bold cyan]Frame Count Analysis:[/bold cyan]")

        # Check frame counts for each video
        for video_id in sorted(videos):
            volume_info = self.volume_data.get(video_id, {"checkpoints": []})
            if volume_info["checkpoints"]:
                videos_with_checkpoints += 1
            else:
                videos_uniform += 1

            # Get actual video frame count
            video_clips = [c for c in self.all_clips if c["video_id"] == video_id]
            if video_clips:
                video_path = video_clips[0]["video_path"]

                # Get actual frame count from video
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                else:
                    actual_frames = -1

                # Get max frame from clips
                max_clip_frame = max(c["end_frame"] for c in video_clips)

                # Get max frame from checkpoints
                checkpoints = volume_info.get("checkpoints", [])
                max_checkpoint_frame = max([f for f, _ in checkpoints], default=0)

                # Get max frame from bleeding annotations
                bleeding_frames = []
                for c in video_clips:
                    if c["has_bleeding"]:
                        bleeding_frames.append(c["center_frame"])
                max_bleeding_frame = max(bleeding_frames, default=0)

                # Print comparison
                console.print(f"\n[yellow]{video_id}:[/yellow]")
                console.print(f"  Actual video frames: {actual_frames}")
                # console.print(f"  Max clip end frame: {max_clip_frame}")
                # console.print(f"  Max checkpoint frame: {max_checkpoint_frame}")
                # console.print(f"  Max bleeding annotation frame: {max_bleeding_frame}")

                if max_checkpoint_frame > actual_frames:
                    console.print(
                        f"  [red]⚠️  Checkpoint beyond video! ({max_checkpoint_frame} > {actual_frames})[/red]"
                    )
                if max_bleeding_frame > actual_frames:
                    console.print(
                        f"  [red]⚠️  Bleeding annotation beyond video! ({max_bleeding_frame} > {actual_frames})[/red]"
                    )

        console.print(f"\n[cyan]Dataset Statistics:[/cyan]")
        console.print(f"  Total clips: {len(self.all_clips):,}")
        console.print(
            f"  Bleeding clips: {len(bleeding_clips):,} ({len(bleeding_clips)/len(self.all_clips)*100:.1f}%)"
        )
        # console.print(f"  Non-bleeding clips: {len(non_bleeding_clips):,}")

        console.print(f"\n[yellow]Volume Assignment Method:[/yellow]")
        console.print(f"  Videos with interpolation: {videos_with_checkpoints}")
        console.print(f"  Videos with uniform distribution: {videos_uniform}")

        # if bleeding_volumes:
        #     console.print(f"\n[green]Volume Statistics:[/green]")
        #     console.print(f"  Cumulative range: {min(bleeding_volumes):.1f} - {max(bleeding_volumes):.1f}ml")
        #     console.print(f"  Mean delta: {np.mean(bleeding_deltas):.3f}ml")
        #     console.print(f"  Delta range: {min(bleeding_deltas):.3f} - {max(bleeding_deltas):.3f}ml")
        #     console.print(f"  Delta std: {np.std(bleeding_deltas):.3f}ml")

        console.print("----------------------------------")

    def __len__(self) -> int:
        return len(self.all_clips)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, torch.Tensor, int, float, float]:
        """
        Get a single clip with all its labels.
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
    # Collect all video paths
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

    # Video-level split (maintain video integrity)
    unique_ids = sorted(
        list(set(os.path.basename(p).split(".")[0] for p in filtered_video_paths))
    )
    console.print(f"[bold green]Video IDs:[/bold green] {', '.join(unique_ids)}")

    console.print(
        f"Using {len(annotation_paths)} videos that have corresponding XML annotations"
    )

    random.shuffle(unique_ids)
    min_num = 1 if testing else 2
    # n_test = max(min_num, int((1 - train_split - val_split) * len(unique_ids)))
    n_test = 2
    n_val = max(min_num, int(val_split * len(unique_ids)))

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

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            # transforms.RandomHorizontalFlip(p=0.1),
            # transforms.ColorJitter(brightness=0.05, contrast=0.1),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
    table.add_column("Vids", justify="right", style="yellow")

    def get_dataset_video_ids(dataset):
        return sorted(list(set(c["video_id"] for c in dataset.all_clips)))

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


def compute_volume_loss(predictions, targets, clip_labels):
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
            low_pred_penalty = torch.relu(
                0.1 - bleeding_preds[actual_bleeding_mask]
            ).mean()
        else:
            low_pred_penalty = torch.tensor(0.0, device=predictions.device)

        total_loss = total_loss + bleeding_loss + 0.6 * low_pred_penalty
        loss_components["bleeding_loss"] = bleeding_loss.item()
        loss_components["low_pred_penalty"] = low_pred_penalty.item()

    # Loss for non-bleeding clips (should predict 0)
    if non_bleeding_mask.any():
        non_bleeding_preds = predictions[non_bleeding_mask]
        non_bleeding_targets = targets[non_bleeding_mask]

        # Lower weight for non-bleeding
        non_bleeding_loss = F.smooth_l1_loss(non_bleeding_preds, non_bleeding_targets)
        total_loss = total_loss + 0.2 * non_bleeding_loss
        loss_components["non_bleeding_loss"] = non_bleeding_loss.item()

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
    # model = VideoBleedingDetector(num_classes=2, dropout_rate=0.5)
    model = VideoBleedingDetectorLSTM(dropout_rate=0.5)
    console.print(
        "[bold]Using R2Plus1D, combined with LSTM Model [/bold]", style="cyan"
    )

    model = model.to(device)

    # Separate optimizers with different learning rates
    # clf_params = list(model.backbone.parameters()) + list(model.classifier.parameters())
    vol_params = list(model.volume_regressor.parameters())

    vol_optimizer = torch.optim.Adam(
        vol_params,
        lr=learning_rate * 10,  # 10x higher learning rate for volume, 1e-4 now
        weight_decay=1e-4,
    )

    vol_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        vol_optimizer, mode="min", factor=0.5, patience=2
    )

    # Tracking
    history = {
        "train_loss": [],
        "train_clf_loss": [],
        "train_vol_loss": [],
        "val_loss": [],
        "val_clf_loss": [],
        "val_vol_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_volume_mae": [],
        "val_volume_rmse": [],
    }

    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")

    console.print(
        Panel.fit(
            f"Bleeding Detection & Quantification\n"
            f"Device: {device}\n"
            f"Epochs: {epochs} | Batch Size: {batch_size}\n"
            f"VOL LR: {learning_rate}\n"
            f"Tasks: Volume Regression\n",
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
        train_metrics = _train_epoch_volume_only(
            model, train_loader, vol_optimizer, device
        )

        # Validation
        model.eval()
        val_metrics = _validate_epoch_volume_only(model, val_loader, device)

        # Update history
        for key in train_metrics:
            history[f"train_{key}"].append(train_metrics[key])
        for key in val_metrics:
            history[f"val_{key}"].append(val_metrics[key])

        # Update both schedulers
        # clf_scheduler.step(val_metrics["clf_loss"])
        vol_scheduler.step(val_metrics["vol_loss"])

        # Results table
        table = Table(title=f"Epoch {epoch+1}/{epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", justify="right", style="green")
        table.add_column("Validation", justify="right", style="yellow")

        # table.add_row("Total Loss", f"{train_metrics['loss']:.4f}", f"{val_metrics['loss']:.4f}")
        table.add_row(
            "Volume Loss",
            f"{train_metrics['vol_loss']:.4f}",
            f"{val_metrics['vol_loss']:.4f}",
        )
        table.add_row("Volume MAE", "-", f"{val_metrics['volume_mae']:.4f}")
        table.add_row("Volume RMSE", "-", f"{val_metrics['volume_rmse']:.4f}")

        console.print(table)

    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)

    with open(os.path.join(output_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    console.print(
        Panel.fit(
            f"[bold green]Training completed[/bold green]\n",
            # f"Best F1: {best:.4f}\n"
            # f"Best model: {best_model_path}\n"
            # f"Final model: {final_model_path}",
            title="Training Complete",
            border_style="green",
        )
    )

    # Test evaluation
    test_metrics = test_model(model, test_loader, device)

    return model, history, test_metrics, test_loader


def _train_epoch_volume_only(
    model, train_loader, vol_optimizer, device, vol_weight=1.0
):
    """Training epoch with only volume loss"""
    total_vol_loss = 0.0
    all_loss_components = []

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

        task = progress.add_task("Training (Volume Only)", total=len(train_loader))

        for batch_idx, (clips, clip_labels, _, _, _, volume_deltas) in enumerate(
            train_loader
        ):
            clips = clips.to(device, dtype=torch.float32)
            volume_deltas = volume_deltas.to(device, dtype=torch.float32)

            vol_optimizer.zero_grad()

            # Forward
            _, volume_delta_preds = model(clips)

            # Volume loss only
            vol_loss, loss_components = compute_volume_loss(
                volume_delta_preds,
                volume_deltas,
                clip_labels,  # Keep if masking bleeding only
            )

            (vol_weight * vol_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            vol_optimizer.step()

            total_vol_loss += vol_loss.item()
            all_loss_components.append(loss_components)

            # Debug logging
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    bleeding_mask = clip_labels == 1
                    if bleeding_mask.any():
                        bleeding_preds = volume_delta_preds[bleeding_mask]
                        bleeding_targets = volume_deltas[bleeding_mask]
                        console.print(
                            f"Batch {batch_idx}: Bleeding preds mean: {bleeding_preds.mean():.4f}, "
                            f"std: {bleeding_preds.std():.4f}, targets mean: {bleeding_targets.mean():.4f}",
                            style="dim",
                        )

            progress.update(task, advance=1)

    # Average loss breakdown
    avg_components = {}
    for key in ["bleeding_loss", "low_pred_penalty", "non_bleeding_loss"]:
        values = [lc[key] for lc in all_loss_components if key in lc]
        if values:
            avg_components[key] = np.mean(values)

    console.print(f"\n[bold green]Volume-Only Training Summary:[/bold green]")
    for key, value in avg_components.items():
        console.print(f"  {key}: {value:.4f}")

    return {
        "vol_loss": total_vol_loss / len(train_loader),
    }


def _validate_epoch_volume_only(model, val_loader, device):
    """Validation epoch using only volume loss"""
    total_vol_loss = 0.0

    all_volume_preds = []
    all_volume_targets = []
    bleeding_volume_preds = []
    bleeding_volume_targets = []

    model.eval()

    with torch.no_grad():
        with Progress(
            TextColumn("Validation (Volume Only)"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:

            task = progress.add_task("Validating", total=len(val_loader))

            for batch_idx, (clips, clip_labels, _, _, _, volume_deltas) in enumerate(
                val_loader
            ):

                clips = clips.to(device, dtype=torch.float32, non_blocking=True)
                volume_deltas = volume_deltas.to(
                    device, dtype=torch.float32, non_blocking=True
                )
                clip_labels = clip_labels.to(device)

                # Forward
                _, volume_delta_preds = model(clips)

                # Volume loss only
                vol_loss, _ = compute_volume_loss(
                    volume_delta_preds, volume_deltas, clip_labels
                )
                total_vol_loss += vol_loss.item()

                # Gather metrics
                all_volume_preds.extend(volume_delta_preds.cpu().numpy())
                all_volume_targets.extend(volume_deltas.cpu().numpy())

                bleeding_mask = clip_labels == 1
                if bleeding_mask.any():
                    bleeding_volume_preds.extend(
                        volume_delta_preds[bleeding_mask].cpu().numpy()
                    )
                    bleeding_volume_targets.extend(
                        volume_deltas[bleeding_mask].cpu().numpy()
                    )

                if batch_idx % 50 == 0:
                    console.print(
                        f"Batch {batch_idx} - Vol preds: mean={volume_delta_preds.mean():.4f}, "
                        f"std={volume_delta_preds.std():.4f}, "
                        f"min={volume_delta_preds.min():.4f}, max={volume_delta_preds.max():.4f}",
                        style="dim",
                    )

                    if bleeding_mask.any():
                        console.print(
                            f"  Bleeding only - preds mean: {volume_delta_preds[bleeding_mask].mean():.4f}, "
                            f"targets mean: {volume_deltas[bleeding_mask].mean():.4f}",
                            style="dim",
                        )

                progress.update(task, advance=1)

    # Convert to numpy
    all_volume_preds = np.array(all_volume_preds)
    all_volume_targets = np.array(all_volume_targets)

    volume_mae = np.mean(np.abs(all_volume_preds - all_volume_targets))
    volume_rmse = np.sqrt(np.mean((all_volume_preds - all_volume_targets) ** 2))

    console.print("\n[cyan]Validation Volume Statistics:[/cyan]")
    console.print(f"Overall - MAE: {volume_mae:.4f}, RMSE: {volume_rmse:.4f}")

    if bleeding_volume_preds:
        bleeding_mae = np.mean(
            np.abs(np.array(bleeding_volume_preds) - np.array(bleeding_volume_targets))
        )
        console.print(f"Bleeding clips - MAE: {bleeding_mae:.4f}")
        console.print(
            f"Bleeding predictions - mean: {np.mean(bleeding_volume_preds):.4f}, "
            f"std: {np.std(bleeding_volume_preds):.4f}"
        )

    return {
        "vol_loss": total_vol_loss / len(val_loader),
        "volume_mae": float(volume_mae),
        "volume_rmse": float(volume_rmse),
    }


def test_model(model, test_loader, device, viz_dir="./test_viz"):
    """
    Test model and evaluate blood loss accumulation per video (volume only).
    Also generates per-video cumulative prediction plots with GT overlay.
    """
    import matplotlib.pyplot as plt
    import os

    model.eval()
    test_dataset = test_loader.dataset

    video_clips = defaultdict(list)
    for i, clip_info in enumerate(test_dataset.all_clips):
        video_clips[clip_info["video_id"]].append((i, clip_info))

    os.makedirs(viz_dir, exist_ok=True)
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
                clips_info.sort(key=lambda x: x[1]["start_frame"])
                all_volume_deltas = []
                frame_centers = []

                batch_size = test_loader.batch_size
                for i in range(0, len(clips_info), batch_size):
                    batch_indices = [
                        clip_idx for clip_idx, _ in clips_info[i : i + batch_size]
                    ]
                    batch_clips = []

                    for idx in batch_indices:
                        clip_tensor, *_, volume_delta = test_dataset[idx]
                        batch_clips.append(clip_tensor)
                        frame_centers.append(
                            test_dataset.all_clips[idx]["center_frame"]
                        )

                    if not batch_clips:
                        continue

                    batch_clips = torch.stack(batch_clips).to(device)
                    _, volume_delta_preds = model(batch_clips)
                    all_volume_deltas.extend(volume_delta_preds.cpu().numpy())

                predicted_cumulative = np.cumsum(all_volume_deltas)
                final_predicted_volume = (
                    predicted_cumulative[-1] if predicted_cumulative.size > 0 else 0.0
                )
                ground_truth_volume = clips_info[0][1]["video_total_volume"]

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
                    "num_clips": len(clips_info),
                }

                # === Visualization ===
                volume_data = test_dataset.volume_data.get(video_id, {})
                checkpoints = {int(f): v for f, v in volume_data.get("checkpoints", [])}
                sorted_gt = sorted(checkpoints.items())
                gt_frames = [f for f, _ in sorted_gt]
                gt_volumes = np.cumsum([v for _, v in sorted_gt])

                plt.figure(figsize=(12, 5))
                plt.plot(
                    frame_centers,
                    predicted_cumulative,
                    label="Predicted",
                    color="blue",
                    linewidth=2,
                )

                if gt_frames:
                    plt.scatter(
                        gt_frames,
                        gt_volumes,
                        color="red",
                        label="GT Checkpoints",
                        zorder=5,
                    )
                    for f, v in zip(gt_frames, gt_volumes):
                        plt.text(
                            f,
                            v,
                            f"{v:.1f}",
                            color="red",
                            fontsize=8,
                            ha="left",
                            va="bottom",
                        )

                plt.title(
                    f"Video: {video_id} | Predicted: {final_predicted_volume:.1f} ml | GT: {ground_truth_volume:.1f} ml"
                )
                plt.xlabel("Frame Index")
                plt.ylabel("Cumulative Blood Loss (ml)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                save_path = os.path.join(viz_dir, f"video_pred_{video_id}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()

                progress.update(task, advance=1)

    # === Summary Table ===
    predicted_volumes = [r["predicted_volume"] for r in video_results.values()]
    ground_truth_volumes = [r["ground_truth_volume"] for r in video_results.values()]
    volume_errors = [r["volume_error"] for r in video_results.values()]
    relative_errors = [r["relative_error"] for r in video_results.values()]

    volume_mse = np.mean(
        [(p - g) ** 2 for p, g in zip(predicted_volumes, ground_truth_volumes)]
    )
    volume_mae = np.mean(volume_errors)
    mean_relative_error = np.mean(relative_errors)

    table = Table(title="Test Results - Volume Prediction")
    table.add_column("Video ID", style="cyan")
    table.add_column("Predicted (ml)", justify="right", style="green")
    table.add_column("Ground Truth (ml)", justify="right", style="yellow")
    table.add_column("Error (ml)", justify="right", style="red")
    table.add_column("Relative Error (%)", justify="right", style="magenta")

    for video_id, results in sorted(video_results.items()):
        table.add_row(
            video_id,
            f"{results['predicted_volume']:.1f}",
            f"{results['ground_truth_volume']:.1f}",
            f"{results['volume_error']:.1f}",
            f"{results['relative_error']*100:.1f}%",
        )
    console.print(table)

    summary_table = Table(title="Overall Test Performance")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")
    summary_table.add_row("Volume MSE", f"{volume_mse:.2f}")
    summary_table.add_row("Volume MAE", f"{volume_mae:.2f} ml")
    summary_table.add_row("Mean Relative Error", f"{mean_relative_error*100:.1f}%")
    summary_table.add_row("Total Videos", str(len(video_results)))
    console.print(summary_table)

    return {
        "video_results": video_results,
        "volume_mse": float(volume_mse),
        "volume_mae": float(volume_mae),
        "mean_relative_error": float(mean_relative_error),
        "predicted_volumes": predicted_volumes,
        "ground_truth_volumes": ground_truth_volumes,
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

    epochs = np.arange(1, len(history["train_vol_loss"]) + 1)

    # === Dark Theme ===
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("bright")
    sns.set_context("talk", font_scale=1.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor("#222222")

    for ax in axs:
        ax.set_facecolor("#2b2b2b")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(True, alpha=0.3)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Training Loss
    axs[0].plot(
        epochs,
        history["train_vol_loss"],
        "o-",
        label="Train Loss",
        color="#66c2a5",
        linewidth=2,
    )
    axs[0].set_title(f"{model_name} - Training Loss", fontsize=14)
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Validation Loss
    axs[1].plot(
        epochs,
        history["val_vol_loss"],
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
        VIDEO_DIR = "/raid/dsl/users/r.rohangirish/data/videos_2_fps"
        ANNO_FOLDER = "/home/r.rohangirish/mt_ble/data/labels_xml"

    output_dir = "./models_interpolation"
    model_name = "bl_LSTM_cl_16"

    model, history, _, _ = train_model(
        video_dir=VIDEO_DIR,
        annotations_dir=ANNO_FOLDER,
        volume_csv_path="/home/r.rohangirish/mt_ble/data/labels_quantification/BL_data_combined.csv",
        device=torch.device(f"cuda:{args.dev}"),
        clip_length=16,
        stride=8,
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
