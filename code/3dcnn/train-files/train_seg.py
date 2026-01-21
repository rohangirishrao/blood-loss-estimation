import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Dict, Optional
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import time
import argparse
import shutil
import seaborn as sns
from pathlib import Path
import json

console = Console()

NUM_THREADS = 6
# video_dir = "/home/r.rohangirish/mt_ble/data/test_videos"
# anno_dir = "/home/r.rohangirish/mt_ble/data/test_labels_xml"

video_dir = "/home/r.rohangirish/mt_ble/data/videos"
anno_dir = "/home/r.rohangirish/mt_ble/data/labels_xml"


# ===================================================
# ========== WORKING VERSION (no temp diff)==========


class VideoBleedingDetector(nn.Module):
    def __init__(
        self, num_classes=2, severity_levels=4, input_size=(328, 512), dropout_rate=0.7
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.input_size = input_size

        try:
            full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(full_model.children())[:-2])
            in_features = 512
            console.print("Using R2Plus1D-18 backbone for feature extraction")
        except:
            console.print("Problem with loading the model. Exiting..")
            return

        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze backbone parameters

        # Feature extraction gives us [B, 512, T/8, H/8, W/8]
        self.feature_channels = in_features
        self.backbone.train()
        # Global pooling for classification tasks
        self.global_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.dropout = nn.Dropout(0.5)

        # Classification heads
        self.classifier = nn.Linear(self.feature_channels, num_classes)
        self.severity_classifier = nn.Linear(self.feature_channels, severity_levels)

        # Segmentation head - upsample to original spatial resolution
        # self.segmentation_head = nn.Sequential(
        #     # Reduce channels first
        #     nn.Conv3d(self.feature_channels, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     # First upsample: from H/8, W/8 (28x28) to H/4, W/4 (56x56)
        #     nn.ConvTranspose3d(
        #         256, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        #     ),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     # Second upsample: from H/4, W/4 (56x56) to H/2, W/2 (112x112)
        #     nn.ConvTranspose3d(
        #         128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        #     ),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     # Third upsample: from H/2, W/2 (112x112) to H, W (224x224)
        #     nn.ConvTranspose3d(
        #         64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        #     ),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     # Final layer to get single channel output
        #     nn.Conv3d(32, 1, kernel_size=1),
        #     nn.Sigmoid(),
        # )
        # Much simpler segmentation head to see if performance improves
        self.segmentation_head = nn.Sequential(
            nn.Conv3d(self.feature_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.3),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.Conv3d(64, 1, kernel_size=1),
        )

    def forward(self, x, train=True):
        # x: [batch_size, channels, frames, height, width]
        features = self.backbone(x)  # [B, 512, T/8, H/8, W/8]

        if train:
            self.backbone.train()  # Ensure backbone is in training mode

        # pooling for classification tasks
        pooled_features = self.global_avgpool(features)  # [B, 512, 1, 1, 1]
        pooled_features = torch.flatten(pooled_features, 1)  # [B, 512]
        pooled_features = self.dropout(pooled_features)

        # Classification predictions
        clip_pred = self.classifier(pooled_features)
        severity_pred = self.severity_classifier(pooled_features)

        # Segmentation prediction (clip-level)
        segmentation_pred = self.segmentation_head(features)  # [B, 1, T/8, H, W]

        # Take temporal average for clip-level segmentation
        segmentation_pred = torch.mean(segmentation_pred, dim=2)  # [B, 1, H, W]
        segmentation_pred = segmentation_pred.squeeze(1)  # [B, H, W]

        # Ensure segmentation output matches target size
        if segmentation_pred.shape[-2:] != self.input_size:
            segmentation_pred = F.interpolate(
                segmentation_pred.unsqueeze(1),
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return clip_pred, severity_pred, segmentation_pred


class SurgicalVideoDataset(Dataset):
    def __init__(
        self, video_paths, annotation_paths, clip_length=6, stride=3, transform=None
    ):
        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform

        # Create clips with annotations
        with console.status("[bold green]Preparing video clips..."):
            self.clips = self._prepare_clips()

        # Display dataset statistics
        self._display_dataset_stats()

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
                console.print(f"⚠️  Could not open video {video_path}", style="yellow")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                console.print(
                    f"⚠️  Invalid frame count for {video_path}: {frame_count}",
                    style="yellow",
                )
                continue

            # Parse XML annotations
            try:
                bleeding_frames = self._parse_xml_to_frames(anno_path, frame_count)
            except Exception as e:
                console.print(
                    f"❌ Error parsing annotations for {video_path}: {e}", style="red"
                )
                continue

            # Create overlapping clips
            for start_idx in range(0, frame_count - self.clip_length + 1, self.stride):
                end_idx = start_idx + self.clip_length
                clip_bleeding = bleeding_frames[start_idx:end_idx]

                if len(clip_bleeding) != self.clip_length:
                    continue

                has_bleeding = np.max(clip_bleeding) > 0
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

        # Balance dataset
        severity_counts = {
            severity: len(clips) for severity, clips in bleeding_clips.items()
        }
        total_bleeding = sum(severity_counts.values())
        total_non_bleeding = len(non_bleeding_clips)

        if total_bleeding < total_non_bleeding:
            random.shuffle(non_bleeding_clips)
            non_bleeding_clips = non_bleeding_clips[:total_bleeding]

        # Combine all clips
        all_clips = non_bleeding_clips.copy()
        for severity, clips in bleeding_clips.items():
            all_clips.extend(clips)

        return all_clips

    def _parse_xml_annotations(self, xml_path):
        """Parse CVAT XML file to extract bounding box annotations"""
        if not os.path.exists(xml_path):
            return []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            annotations = []

            # Get original video dimensions
            meta = root.find("meta")
            original_size = meta.find("original_size")
            orig_width = int(original_size.find("width").text)
            orig_height = int(original_size.find("height").text)

            # Parse each track (bleeding region)
            for track in root.findall("track"):
                track_label = track.get("label")  # BL_Low, BL_Medium, BL_High
                track_id = track.get("id")

                # Only process bleeding labels
                if track_label in ["BL_Low", "BL_Medium", "BL_High"]:
                    # Parse each box in the track
                    for box in track.findall("box"):
                        frame_num = int(box.get("frame"))
                        outside = int(box.get("outside", "0"))

                        # Skip boxes marked as outside
                        if outside == 1:
                            continue

                        # Extract bounding box coordinates
                        xtl = float(box.get("xtl"))  # x top left
                        ytl = float(box.get("ytl"))  # y top left
                        xbr = float(box.get("xbr"))  # x bottom right
                        ybr = float(box.get("ybr"))  # y bottom right

                        annotations.append(
                            {
                                "frame": frame_num,
                                "bbox": [xtl, ytl, xbr, ybr],
                                "label": track_label,
                                "track_id": track_id,
                                "original_width": orig_width,
                                "original_height": orig_height,
                            }
                        )

        except ET.ParseError as e:
            console.print(f"❌ XML parsing error for {xml_path}: {e}", style="red")
            return []
        except Exception as e:
            console.print(f"❌ Unexpected error parsing {xml_path}: {e}", style="red")
            return []

        return annotations

    def _parse_xml_to_frames(self, xml_path, total_frames):
        """Convert XML annotations to per-frame bleeding labels and segmentation masks"""
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

    def _create_segmentation_mask_from_xml(
        self, xml_path, start_frame, end_frame, target_size=(328, 512)
    ):
        """Create clip-level segmentation mask from XML bounding boxes"""
        mask = np.zeros(target_size, dtype=np.float32)
        severity_map = {"BL_Low": 0.33, "BL_Medium": 0.66, "BL_High": 1.0}

        annotations = self._parse_xml_annotations(xml_path)

        # Only debug for ApVr.mp4 to avoid spam
        debug_this = "ApVr" in xml_path
        debug_this = False  # Set to True to enable debugging
        if debug_this:
            print(f"DEBUG: Looking for frames {start_frame}-{end_frame}")

            # Show what frame numbers actually exist in XML
            frame_numbers = [anno["frame"] for anno in annotations]
            min_frame = min(frame_numbers) if frame_numbers else None
            max_frame = max(frame_numbers) if frame_numbers else None

            print(f"DEBUG: XML has frames from {min_frame} to {max_frame}")
            print(
                f"DEBUG: Sample XML frames: {sorted(set(frame_numbers))[:10]}..."
            )  # First 10 unique frames

            # Check if ANY frames are close
            close_frames = [f for f in frame_numbers if abs(f - start_frame) < 50]
            print(f"DEBUG: Frames within 50 of {start_frame}: {close_frames[:5]}")

        boxes_used = 0
        # Union of all bounding boxes in the clip frames
        for anno in annotations:
            frame_num = anno["frame"]

            # Check if this annotation falls within our clip
            if start_frame <= frame_num <= end_frame:
                bbox = anno["bbox"]
                label = anno["label"]
                orig_width = anno["original_width"]
                orig_height = anno["original_height"]

                if debug_this:
                    print(
                        f"DEBUG: Using frame {frame_num}, bbox: {bbox}, label: {label}"
                    )
                    video_name = os.path.basename(xml_path).replace(".xml", "")
                    print(
                        f"DEBUG: Video {video_name} - Looking for frames {start_frame}-{end_frame}"
                    )

                # Scale bounding box to target size
                xtl, ytl, xbr, ybr = bbox

                # Scale coordinates
                h_scale = target_size[0] / orig_height
                w_scale = target_size[1] / orig_width

                x1_scaled = int(xtl * w_scale)
                y1_scaled = int(ytl * h_scale)
                x2_scaled = int(xbr * w_scale)
                y2_scaled = int(ybr * h_scale)

                if debug_this:
                    print(f"DEBUG: Original coords: ({xtl}, {ytl}, {xbr}, {ybr})")
                    print(
                        f"DEBUG: Scaled coords: ({x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled})"
                    )

                # Ensure coordinates are within bounds
                x1_scaled = max(0, min(x1_scaled, target_size[1] - 1))
                y1_scaled = max(0, min(y1_scaled, target_size[0] - 1))
                x2_scaled = max(0, min(x2_scaled, target_size[1] - 1))
                y2_scaled = max(0, min(y2_scaled, target_size[0] - 1))

                if debug_this:
                    print(
                        f"DEBUG: Bounded coords: ({x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled})"
                    )

                # Set mask values based on severity
                severity_value = severity_map.get(label, 0.33)
                if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                    if debug_this:
                        print(
                            f"DEBUG: Setting mask region with severity {severity_value}"
                        )
                    current_max = (
                        mask[y1_scaled:y2_scaled, x1_scaled:x2_scaled].max()
                        if mask[y1_scaled:y2_scaled, x1_scaled:x2_scaled].size > 0
                        else 0
                    )
                    mask[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = max(
                        current_max, severity_value
                    )
                    boxes_used += 1
                else:
                    if debug_this:
                        print(f"DEBUG: Invalid box dimensions after scaling!")

        if debug_this:
            print(f"DEBUG: Used {boxes_used} boxes, final mask sum: {mask.sum()}")
            print("=" * 50)  # Separator for clarity

        return mask

    def _display_dataset_stats(self):
        """Display dataset statistics using rich"""
        bleeding_clips = {1: [], 2: [], 3: []}
        non_bleeding_clips = []

        for clip in self.clips:
            if clip["has_bleeding"]:
                bleeding_clips[clip["max_severity"]].append(clip)
            else:
                non_bleeding_clips.append(clip)

        # Create statistics table
        table = Table(title="Dataset Statistics")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right", style="white")

        table.add_row("Total Clips", str(len(self.clips)))
        table.add_row("Non-Bleeding", str(len(non_bleeding_clips)))
        table.add_row("BL_Low (Severity 1)", str(len(bleeding_clips[1])))
        table.add_row("BL_Medium (Severity 2)", str(len(bleeding_clips[2])))
        table.add_row("BL_High (Severity 3)", str(len(bleeding_clips[3])))

        total_bleeding = sum(len(clips) for clips in bleeding_clips.values())
        table.add_row("Total Bleeding", str(total_bleeding))

        console.print(table)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]

        # Load video clip frames
        frames = self._load_clip(
            clip_info["video_path"],
            clip_info["start_frame"],
            clip_info["end_frame"] + 1,
        )

        if len(frames) < self.clip_length:
            last_frame = (
                frames[-1] if frames else np.zeros((328, 512, 3), dtype=np.uint8)
            )
            frames.extend([last_frame] * (self.clip_length - len(frames)))

        # Create segmentation mask for this clip
        xml_path = None
        for video_path, anno_path in zip(self.video_paths, self.annotation_paths):
            if video_path == clip_info["video_path"]:
                xml_path = anno_path
                break

        segmentation_mask = np.zeros((328, 512), dtype=np.float32)
        if xml_path:
            segmentation_mask = self._create_segmentation_mask_from_xml(
                xml_path,
                clip_info["start_frame"],
                clip_info["end_frame"],
                target_size=(328, 512),
            )
            if segmentation_mask.sum() == 0 and clip_info["has_bleeding"]:
                print(f"MISMATCH DEBUG:")
                print(f"  Clip: {clip_info['start_frame']}-{clip_info['end_frame']}")
                print(f"  Labeled as bleeding: {clip_info['has_bleeding']}")
                print(f"  Bleeding frames array: {clip_info['bleeding_frames']}")
                print(f"  Max in array: {np.max(clip_info['bleeding_frames'])}")

            # Check what XML annotations exist for this range
            # if xml_path:
            #     annotations = self._parse_xml_annotations(xml_path)
            #     frames_in_range = [anno['frame'] for anno in annotations
            #                     if clip_info['start_frame'] <= anno['frame'] <= clip_info['end_frame']]
            #     # print(f"  XML frames in clip range: {frames_in_range}")

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into tensor [T, C, H, W] then reshape to [C, T, H, W]
        clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)

        # Get labels
        clip_label = 1 if clip_info["has_bleeding"] else 0
        frame_labels = torch.tensor(clip_info["bleeding_frames"], dtype=torch.float32)
        severity = torch.tensor(clip_info["max_severity"], dtype=torch.long)
        segmentation_tensor = torch.tensor(segmentation_mask, dtype=torch.float32)

        return clip_tensor, clip_label, frame_labels, severity, segmentation_tensor

    def _load_clip(self, video_path, start_frame, end_frame):
        t0 = time.time()
        frames = []

        cap = cv2.VideoCapture(video_path)
        params = [cv2.CAP_PROP_N_THREADS, NUM_THREADS]
        cap.open(video_path, apiPreference=cv2.CAP_FFMPEG, params=params)

        if not cap.isOpened():
            console.print(f"⚠️  Could not open video {video_path}", style="yellow")
            return frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        t1 = time.time()
        # print(f"Loaded {len(frames)} frames from {video_path} in {t1 - t0:.2f}s")
        return frames


def calculate_iou(pred_mask, true_mask, threshold=0.3):
    """Calculate Intersection over Union for segmentation masks"""
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > 0).float()

    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


def calculate_bleeding_area_percentage(segmentation_mask, threshold=0.3):
    """Calculate bleeding area as percentage of total frame area"""
    binary_mask = segmentation_mask > threshold
    total_pixels = segmentation_mask.numel()
    bleeding_pixels = binary_mask.sum().item()
    return (bleeding_pixels / total_pixels) * 100


def visualize_bleeding_masks(
    model,
    test_loader,
    device,
    num_samples=30,
    path="./visualizations",
    cleanup=True,
    video_section="last_third",
):
    """Visualize bleeding clips from specific sections of videos

    Args:
        video_section: "first_third", "middle_third", "last_third", or "all" - which part of videos to sample from
    """

    # Create directory first
    os.makedirs(path, exist_ok=True)

    # Then handle cleanup if requested
    if cleanup and os.path.exists(path):
        # Remove contents but keep directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                console.print(f"Failed to delete {file_path}: {e}")
        console.print(f"🗑️  Cleaned up existing visualizations in {path}")

    # Filter clips based on video section preference
    filtered_clips_info = []

    for batch_idx, (
        clips,
        clf_labels,
        frame_labels,
        sev_labels,
        seg_masks,
    ) in enumerate(test_loader):
        for i in range(clips.size(0)):
            # Skip non-bleeding clips
            if clf_labels[i].item() == 0:
                continue

            # Get true mask area to check if we have annotations
            true_mask = seg_masks[i].cpu().numpy()
            true_area = calculate_bleeding_area_percentage(torch.tensor(true_mask))

            # Skip clips without annotations
            if true_area == 0.0:
                continue

            # Get clip info to determine video section
            clip_idx = batch_idx * test_loader.batch_size + i
            if clip_idx >= len(test_loader.dataset.clips):
                continue

            clip_info = test_loader.dataset.clips[clip_idx]

            # Get video info to determine which third this clip is from
            video_path = clip_info["video_path"]

            # Get total frames in video
            cap = cv2.VideoCapture(video_path)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_video_frames <= 0:
                continue

            clip_start = clip_info["start_frame"]
            clip_position_ratio = clip_start / total_video_frames

            # Determine which section this clip belongs to
            if video_section == "first_third" and clip_position_ratio < 0.33:
                include_clip = True
            elif video_section == "middle_third" and 0.33 <= clip_position_ratio < 0.67:
                include_clip = True
            elif video_section == "last_third" and clip_position_ratio >= 0.67:
                include_clip = True
            elif video_section == "all":
                include_clip = True
            else:
                include_clip = False

            if include_clip:
                filtered_clips_info.append(
                    {
                        "batch_idx": batch_idx,
                        "clip_idx_in_batch": i,
                        "clip_info": clip_info,
                        "position_ratio": clip_position_ratio,
                        "clips": clips,
                        "seg_masks": seg_masks,
                        "true_area": true_area,
                    }
                )

    # Sort by position in video (latest first for last_third)
    if video_section == "last_third":
        filtered_clips_info.sort(key=lambda x: x["position_ratio"], reverse=True)
    else:
        filtered_clips_info.sort(key=lambda x: x["position_ratio"])

    console.print(
        f"📍 Found {len(filtered_clips_info)} bleeding clips in {video_section} of videos"
    )

    if len(filtered_clips_info) == 0:
        console.print(f"❌ No bleeding clips found in {video_section} section")
        return

    model.eval()
    samples_found = 0

    with torch.no_grad():
        for clip_data in filtered_clips_info[
            :num_samples
        ]:  # Take only requested number
            if samples_found >= num_samples:
                break

            # Extract data
            clips = clip_data["clips"].to(device)
            seg_masks = clip_data["seg_masks"].to(device)
            i = clip_data["clip_idx_in_batch"]
            clip_info = clip_data["clip_info"]
            true_area = clip_data["true_area"]
            position_ratio = clip_data["position_ratio"]

            # Get predictions for this specific clip
            single_clip = clips[i : i + 1]  # [1, C, T, H, W]
            single_mask = seg_masks[i : i + 1]  # [1, H, W]

            _, _, seg_pred = model(single_clip)

            # Get middle frame for visualization
            middle_frame = clips[i, :, clips.size(2) // 2, :, :].cpu()
            middle_frame = denormalize_frame(middle_frame)

            # Handle segmentation prediction
            if seg_pred[0].dim() == 3:  # [T, H, W]
                pred_mask = (
                    seg_pred[0][seg_pred[0].shape[0] // 2].cpu().numpy()
                )  # Middle temporal frame
            else:  # [H, W]
                pred_mask = seg_pred[0].cpu().numpy()

            # Calculate metrics
            iou = calculate_iou(seg_pred[0], seg_masks[i])
            pred_area = calculate_bleeding_area_percentage(torch.tensor(pred_mask))

            # Create 3-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Original frame
            axes[0].imshow(middle_frame)
            axes[0].set_title("Original Frame")
            axes[0].axis("off")

            # Ground truth overlay
            true_mask = seg_masks[i].cpu().numpy()
            axes[1].imshow(middle_frame)
            axes[1].imshow(true_mask, cmap="Reds", alpha=0.6)
            axes[1].set_title(f"Ground Truth ({true_area:.1f}%)")
            axes[1].axis("off")

            # Prediction overlay
            axes[2].imshow(middle_frame)
            axes[2].imshow(pred_mask, cmap="Blues", alpha=0.6)
            axes[2].set_title(f"Prediction ({pred_area:.1f}%)")
            axes[2].axis("off")

            # Title with info
            video_name = os.path.basename(clip_info["video_path"])
            start_frame = clip_info["start_frame"]
            end_frame = clip_info["end_frame"]
            middle_frame_num = start_frame + clips.size(2) // 2
            severity = clip_info["max_severity"]

            plt.suptitle(
                f"#{samples_found+1} | {video_name} | Frames {start_frame}-{end_frame} (frame {middle_frame_num}) | "
                f"Severity: {severity} | IoU: {iou:.3f} | Video pos: {position_ratio:.1%} ({video_section})",
                fontsize=9,
            )

            plt.tight_layout()

            # Create safe filename
            safe_video_name = "".join(
                c for c in video_name if c.isalnum() or c in ("-", "_", ".")
            )
            filename = f"{safe_video_name}_clip{samples_found+1}_{video_section}_pos{position_ratio:.2f}_iou{iou:.3f}.png"
            filepath = os.path.join(path, filename)

            try:
                fig.savefig(filepath, bbox_inches="tight", dpi=330)
                console.print(
                    f"[bold green]Sample {samples_found+1}:[/bold green] "
                    f"Video: {video_name} | "
                    f"Frame {middle_frame_num} ({position_ratio:.1%} through video) | "
                    f"IoU: {iou:.3f} | "
                    f"Areas: GT {true_area:.1f}% vs Pred {pred_area:.1f}%"
                )
            except Exception as e:
                console.print(f"[red]Error saving {filepath}: {e}[/red]")
                # Try with simpler filename
                simple_filename = (
                    f"sample_{samples_found+1}_{video_section}_iou{iou:.3f}.png"
                )
                simple_filepath = os.path.join(path, simple_filename)
                fig.savefig(simple_filepath, bbox_inches="tight", dpi=330)
                console.print(f"[yellow]Saved as: {simple_filename}[/yellow]")

            plt.close()
            samples_found += 1

        # Summary
        console.print(f"\n📊 Visualization Summary:")
        console.print(
            f"  ✅ Showed {samples_found} clips from {video_section} of videos"
        )
        console.print(f"  🎯 Video section: {video_section}")
        console.print(
            f"  📹 Clips ranged from {min([c['position_ratio'] for c in filtered_clips_info[:samples_found]]):.1%} to {max([c['position_ratio'] for c in filtered_clips_info[:samples_found]]):.1%} through videos"
        )
        console.print(f"  💾 Saved to: {os.path.abspath(path)}")


def denormalize_frame(frame):
    """Denormalize frame for display"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    frame = frame.permute(1, 2, 0).numpy()
    frame = frame * std + mean
    return np.clip(frame, 0, 1)


def train_bleeding_detector(
    video_dir,
    annotation_dir,
    output_dir="./models",
    epochs=30,
    batch_size=4,
    clip_length=6,
    stride=3,
    learning_rate=0.0001,
    device="cuda:0",
    model_name="bleeding_detector",
    test=False,
    split_loss=True,  # Whether to split loss into classification, severity, and segmentation
):
    """Train bleeding detection model with video-level splitting and comprehensive metrics"""

    # Setup
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    console.print(
        Panel.fit(
            f"🚀 Bleeding Detection & Quantification\n"
            f"Device: {device}\n"
            f"Epochs: {epochs} | Batch Size: {batch_size}\n"
            f"Clip Length: {clip_length} | Learning Rate: {learning_rate}\n"
            f"Tasks: Classification + Severity + Segmentation\n",
            title="Training Configuration",
            border_style="white",
        )
    )

    # Video-level data collection and splitting
    console.print("\n📂 Collecting videos and annotations...")

    video_files = [
        f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi", ".mov"))
    ]

    # Match videos with XML annotations
    matched_pairs = []
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        xml_file = f"{video_name}.xml"
        xml_path = os.path.join(annotation_dir, xml_file)

        if os.path.exists(xml_path):
            video_path = os.path.join(video_dir, video_file)
            matched_pairs.append((video_path, xml_path))
        else:
            console.print(f"⚠️  No annotation found for {video_file}", style="yellow")

    if len(matched_pairs) < 4:
        console.print(
            "❌ Need at least 4 video-annotation pairs for splitting", style="red"
        )
        return None, None, None

    console.print(f"✅ Found {len(matched_pairs)} video-annotation pairs")

    # Video-level splitting (80% train, at least 2 test, 1 val)
    random.shuffle(matched_pairs)
    total_videos = len(matched_pairs)

    # Ensure minimum requirements
    minim = 1 if test else 2
    min_test = max(2, int(0.1 * total_videos))
    min_val = max(3, int(0.1 * total_videos))

    val_size = min_val
    test_size = min_test
    train_size = total_videos - val_size - test_size

    train_pairs = matched_pairs[:train_size]
    val_pairs = matched_pairs[train_size : train_size + val_size]
    test_pairs = matched_pairs[train_size + val_size :]

    # Display split info
    split_table = Table(title="Video-Level Data Split")
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Videos", justify="right", style="green")
    split_table.add_column("Percentage", justify="right", style="cyan")

    split_table.add_row(
        "Train", str(len(train_pairs)), f"{len(train_pairs)/total_videos*100:.1f}%"
    )
    split_table.add_row(
        "Validation", str(len(val_pairs)), f"{len(val_pairs)/total_videos*100:.1f}%"
    )
    split_table.add_row(
        "Test", str(len(test_pairs)), f"{len(test_pairs)/total_videos*100:.1f}%"
    )
    split_table.add_row("Total", str(total_videos), "100.0%")

    console.print(split_table)

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((328, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((328, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    console.print("\n🔄 Creating datasets...")

    train_videos, train_annotations = zip(*train_pairs) if train_pairs else ([], [])
    val_videos, val_annotations = zip(*val_pairs) if val_pairs else ([], [])
    test_videos, test_annotations = zip(*test_pairs) if test_pairs else ([], [])

    train_dataset = SurgicalVideoDataset(
        list(train_videos),
        list(train_annotations),
        clip_length=clip_length,
        stride=stride,
        transform=train_transform,
    )

    val_dataset = SurgicalVideoDataset(
        list(val_videos),
        list(val_annotations),
        clip_length=clip_length,
        stride=stride,
        transform=val_test_transform,
    )

    test_dataset = SurgicalVideoDataset(
        list(test_videos),
        list(test_annotations),
        clip_length=clip_length,
        stride=stride,
        transform=val_test_transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    console.print(
        f"📊 Clips - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    model = VideoBleedingDetector(num_classes=2, input_size=(328, 512)).to(  # TEMP DIFF
        device
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"🧠 Total parameters: {total_params:,}")

    # Training setup
    criterion_clf = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()
    # criterion_seg = nn.BCELoss() # WORKING VERSION

    pos_weight = torch.tensor([3.0]).to(device)  # TODO adjust this later maybe
    criterion_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.1
    )  # changed from 0.01 to 0.1
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=5, verbose=True
    )  # changed from .5 to .8 factor and patience 3->5

    # Metrics tracking
    metrics = {
        "train_loss": [],
        "train_clf_acc": [],
        "train_sev_acc": [],
        "train_seg_iou": [],
        "val_loss": [],
        "val_clf_loss": [],
        "val_clf_acc": [],
        "val_sev_acc": [],
        "val_seg_iou": [],
    }
    torch.cuda.reset_peak_memory_stats()
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    st = time.time()

    # ========== Training Loop ===========
    initial_weights = model.classifier.weight.clone()
    for epoch in range(epochs):
        console.print(f"\n[bold blue]Epoch {epoch+1}/{epochs}[/bold blue]")
        if epoch == 1:
            current_weights = model.classifier.weight
            weight_change = (current_weights - initial_weights).abs().mean()
            console.print(f"Weight change after epoch 1: {weight_change:.8f}")

        # === Training ===
        model.train()
        train_loss_total = 0
        train_clf_correct = 0
        train_sev_correct = 0
        train_seg_iou_total = 0
        train_total = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as progress:

            train_task = progress.add_task("Training", total=len(train_loader))

            for batch_idx, (
                clips,
                clf_labels,
                frame_labels,
                sev_labels,
                seg_masks,
            ) in enumerate(train_loader):
                t9 = time.time()
                console.print(
                    f"Processing batch {batch_idx + 1}/{len(train_loader)}"
                )
                clips = clips.to(device)
                clf_labels = clf_labels.to(device)
                sev_labels = sev_labels.to(device)
                seg_masks = seg_masks.to(device)

                optimizer.zero_grad()

                clf_pred, sev_pred, seg_pred = model(clips)

                # Calculate losses
                loss_clf = criterion_clf(clf_pred, clf_labels)
                loss_sev = criterion_sev(sev_pred, sev_labels)

                # expanded_seg_masks = seg_masks.unsqueeze(1).expand(-1, seg_pred.shape[1], -1, -1)
                # loss_seg = criterion_seg(seg_pred, expanded_seg_masks)

                loss_seg = criterion_seg(seg_pred, seg_masks)

                total_loss = loss_clf + 0.3 * loss_sev  # + 0.5 * loss_seg

                total_loss.backward()
                optimizer.step()

                # Metrics
                train_loss_total += total_loss.item()
                train_clf_correct += clf_pred.argmax(1).eq(clf_labels).sum().item()
                train_sev_correct += sev_pred.argmax(1).eq(sev_labels).sum().item()
                train_total += clf_labels.size(0)

                batch_iou = sum(
                    calculate_iou(seg_pred[i], seg_masks[i])
                    for i in range(clips.size(0))
                )
                train_seg_iou_total += batch_iou

                if batch_idx % 10 == 0:
                    console.print(
                        f"Training: [{batch_idx:3d}/{len(train_loader):3d}] Loss: {total_loss.item():.4f}",
                        end="\r",
                    )
                time_taken = time.time() - t9
                console.print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} processed in {time_taken:.2f}s"
                )
                progress.update(train_task, advance=1)

        # === Validation ===
        model.eval()
        val_loss_total = 0
        val_clf_loss_total = 0
        val_clf_correct = 0
        val_sev_correct = 0
        val_seg_iou_total = 0
        val_total = 0

        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                console=console,
                refresh_per_second=4,
                transient=False,
            ) as progress:

                val_task = progress.add_task("Validation", total=len(val_loader))

                for (
                    clips,
                    clf_labels,
                    frame_labels,
                    sev_labels,
                    seg_masks,
                ) in val_loader:
                    clips = clips.to(device)
                    clf_labels = clf_labels.to(device)
                    sev_labels = sev_labels.to(device)
                    seg_masks = seg_masks.to(device)

                    clf_pred, sev_pred, seg_pred = model(clips)

                    # Losses
                    loss_clf = criterion_clf(clf_pred, clf_labels)
                    loss_sev = criterion_sev(sev_pred, sev_labels)
                    # expanded_seg_masks = seg_masks.unsqueeze(1).expand(-1, seg_pred.shape[1], -1, -1)
                    # loss_seg = criterion_seg(seg_pred, expanded_seg_masks)

                    loss_seg = criterion_seg(seg_pred, seg_masks)  # WORKING VERSION

                    total_loss = loss_clf + 0.3 * loss_sev + 0.5 * loss_seg

                    # Metrics
                    val_loss_total += total_loss.item()
                    if split_loss:
                        val_clf_loss_total += loss_clf.item()

                    val_clf_correct += clf_pred.argmax(1).eq(clf_labels).sum().item()
                    val_sev_correct += sev_pred.argmax(1).eq(sev_labels).sum().item()
                    val_total += clf_labels.size(0)

                    batch_iou = sum(
                        calculate_iou(seg_pred[i], seg_masks[i])
                        for i in range(clips.size(0))
                    )
                    val_seg_iou_total += batch_iou

                    progress.update(val_task, advance=1)

        # ====== Epoch metrics ======
        # ===========================
        train_loss_avg = train_loss_total / len(train_loader)
        train_clf_acc = 100.0 * train_clf_correct / train_total
        train_sev_acc = 100.0 * train_sev_correct / train_total
        train_seg_iou = train_seg_iou_total / train_total

        val_loss_avg = val_loss_total / len(val_loader)
        val_clf_loss_avg = val_clf_loss_total / len(val_loader) if split_loss else 0
        val_clf_acc = 100.0 * val_clf_correct / val_total
        val_sev_acc = 100.0 * val_sev_correct / val_total
        val_seg_iou = val_seg_iou_total / val_total

        # Save metrics
        metrics["train_loss"].append(train_loss_avg)
        metrics["train_clf_acc"].append(train_clf_acc)
        metrics["train_sev_acc"].append(train_sev_acc)
        metrics["train_seg_iou"].append(train_seg_iou)
        metrics["val_loss"].append(val_loss_avg)
        metrics["val_clf_acc"].append(val_clf_acc)
        metrics["val_sev_acc"].append(val_sev_acc)
        metrics["val_seg_iou"].append(val_seg_iou)
        if split_loss:
            metrics["val_clf_loss"].append(val_clf_loss_avg)
        else:
            metrics["val_clf_loss"].append(0)  # Keep consistent length

        # === Print results ===
        results_table = Table(title=f"Epoch {epoch+1} Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Train", justify="right", style="green")
        results_table.add_column("Validation", justify="right", style="yellow")

        results_table.add_row("Loss", f"{train_loss_avg:.4f}", f"{val_loss_avg:.4f}")
        results_table.add_row(
            "Classification Acc", f"{train_clf_acc:.2f}%", f"{val_clf_acc:.2f}%"
        )
        results_table.add_row(
            "Severity Acc", f"{train_sev_acc:.2f}%", f"{val_sev_acc:.2f}%"
        )
        results_table.add_row(
            "Segmentation IoU", f"{train_seg_iou:.4f}", f"{val_seg_iou:.4f}"
        )

        if split_loss:
            results_table.add_row("Classifier Loss", "-", f"{val_clf_loss_avg:.4f}")

        console.print(results_table)

        # === Save best ===
        current_val_acc = val_clf_acc
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            patience_counter = 0
            torch.save(
                model.state_dict(), os.path.join(output_dir, f"{model_name}_best.pth")
            )
            console.print("💾 New best model saved!", style="bold green")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print("⏰ Early stopping triggered", style="yellow")
                break

        scheduler.step(val_loss_avg)

    # === Final save ===
    et = time.time()
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_final.pth"))

    import json

    with open(os.path.join(output_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed_time = et - st
    time_str = f"{int(elapsed_time//3600):02d}h {int((elapsed_time%3600)//60):02d}m {int(elapsed_time%60):02d}s"

    console.print(
        Panel.fit(
            f"✅ Training completed in {time_str}!\n"
            f"Best validation accuracy: {best_val_acc:.2f}%\n"
            f"Final epoch: {epoch+1}/{epochs}\n"
            f"Models and metrics saved to: {output_dir}",
            title="Training Complete",
            border_style="green",
        )
    )
    console.print(f"Training Time: {time_str}")
    # === Test ===

    console.print("\n🧪 Starting test evaluation...")
    test_metrics = evaluate_model(model, test_loader, device)
    console.print("\n🔍 Visualizing bleeding segmentation results...")
    os.makedirs("./visualization", exist_ok=True)
    visualize_bleeding_masks(
        model,
        test_loader,
        device,
        num_samples=30,
        video_section="last_third",
        path="./visualizations",
        cleanup=True,
    )

    return model, metrics, test_metrics


def evaluate_model(model, test_loader, device):
    """Comprehensive test evaluation with classification and segmentation metrics"""
    model.eval()

    # Storage for predictions and targets
    all_clf_preds = []
    all_clf_targets = []
    all_sev_preds = []
    all_sev_targets = []

    test_loss_total = 0
    test_seg_iou_total = 0
    test_seg_dice_total = 0
    test_seg_precision_total = 0
    test_seg_recall_total = 0
    test_total = 0

    criterion_clf = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()

    # FIXED: Use same loss as training
    pos_weight = torch.tensor([3.0]).to(device)
    criterion_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as progress:

            test_task = progress.add_task("Testing", total=len(test_loader))

            for clips, clf_labels, _, sev_labels, seg_masks in test_loader:
                clips = clips.to(device)
                clf_labels = clf_labels.to(device)
                sev_labels = sev_labels.to(device)
                seg_masks = seg_masks.to(device)

                clf_pred, sev_pred, seg_pred = model(clips)

                # Calculate losses
                loss_clf = criterion_clf(clf_pred, clf_labels)
                loss_sev = criterion_sev(sev_pred, sev_labels)

                # FIXED: Handle temporal dimension mismatch like in training
                # if seg_pred.dim() == 4:  # [B, T, H, W]
                #     expanded_seg_masks = seg_masks.unsqueeze(1).expand(-1, seg_pred.shape[1], -1, -1)
                #     loss_seg = criterion_seg(seg_pred, expanded_seg_masks)
                # else:  # [B, H, W]
                #     loss_seg = criterion_seg(seg_pred, seg_masks)

                loss_seg = criterion_seg(seg_pred, seg_masks)
                # Use same loss weights as training
                total_loss = loss_clf + 0.3 * loss_sev # + 0.5 * loss_seg

                test_loss_total += total_loss.item()
                test_total += clf_labels.size(0)

                # Store predictions for detailed metrics
                all_clf_preds.extend(clf_pred.argmax(1).cpu().numpy())
                all_clf_targets.extend(clf_labels.cpu().numpy())
                all_sev_preds.extend(sev_pred.argmax(1).cpu().numpy())
                all_sev_targets.extend(sev_labels.cpu().numpy())

                # Segmentation metrics - FIXED: Apply sigmoid to logits for metrics
                for i in range(clips.size(0)):
                    # Convert logits to probabilities for metric calculation
                    if seg_pred[i].dim() == 3:  # [T, H, W]
                        pred_probs = torch.sigmoid(seg_pred[i])
                        # Use middle temporal frame or average
                        pred_mask = pred_probs[pred_probs.shape[0] // 2]
                    else:  # [H, W]
                        pred_mask = torch.sigmoid(seg_pred[i])

                    true_mask = seg_masks[i]

                    # IoU
                    iou = calculate_iou(pred_mask, true_mask)
                    test_seg_iou_total += iou

                    # Dice coefficient
                    dice = calculate_dice_coefficient(pred_mask, true_mask)
                    test_seg_dice_total += dice

                    # Pixel-wise precision and recall
                    precision, recall = calculate_pixel_metrics(pred_mask, true_mask)
                    test_seg_precision_total += precision
                    test_seg_recall_total += recall

                progress.update(test_task, advance=1)

    # Calculate classification metrics
    clf_acc = accuracy_score(all_clf_targets, all_clf_preds)
    clf_precision, clf_recall, clf_f1, _ = precision_recall_fscore_support(
        all_clf_targets, all_clf_preds, average="weighted", zero_division=0
    )

    sev_acc = accuracy_score(all_sev_targets, all_sev_preds)
    sev_precision, sev_recall, sev_f1, _ = precision_recall_fscore_support(
        all_sev_targets, all_sev_preds, average="weighted", zero_division=0
    )

    # Calculate segmentation metrics
    seg_iou_avg = test_seg_iou_total / test_total
    seg_dice_avg = test_seg_dice_total / test_total
    seg_precision_avg = test_seg_precision_total / test_total
    seg_recall_avg = test_seg_recall_total / test_total
    seg_f1_avg = (
        2 * (seg_precision_avg * seg_recall_avg) / (seg_precision_avg + seg_recall_avg)
        if (seg_precision_avg + seg_recall_avg) > 0
        else 0
    )

    # Display comprehensive results
    console.print("\n[bold green]🎯 Test Results[/bold green]")

    # Classification results
    clf_table = Table(title="Classification Metrics")
    clf_table.add_column("Task", style="cyan")
    clf_table.add_column("Accuracy", justify="right", style="green")
    clf_table.add_column("Precision", justify="right", style="yellow")
    clf_table.add_column("Recall", justify="right", style="yellow")
    clf_table.add_column("F1-Score", justify="right", style="magenta")

    clf_table.add_row(
        "Bleeding Detection",
        f"{clf_acc:.4f}",
        f"{clf_precision:.4f}",
        f"{clf_recall:.4f}",
        f"{clf_f1:.4f}",
    )
    clf_table.add_row(
        "Severity Classification",
        f"{sev_acc:.4f}",
        f"{sev_precision:.4f}",
        f"{sev_recall:.4f}",
        f"{sev_f1:.4f}",
    )

    console.print(clf_table)

    # Segmentation results
    seg_table = Table(title="Segmentation Metrics")
    seg_table.add_column("Metric", style="cyan")
    seg_table.add_column("Value", justify="right", style="green")

    seg_table.add_row("IoU (Intersection over Union)", f"{seg_iou_avg:.4f}")
    seg_table.add_row("Dice Coefficient", f"{seg_dice_avg:.4f}")
    seg_table.add_row("Pixel-wise Precision", f"{seg_precision_avg:.4f}")
    seg_table.add_row("Pixel-wise Recall", f"{seg_recall_avg:.4f}")
    seg_table.add_row("Pixel-wise F1-Score", f"{seg_f1_avg:.4f}")

    console.print(seg_table)

    # Calculate bleeding area statistics
    avg_bleeding_area = calculate_average_bleeding_area(model, test_loader, device)
    console.print(
        f"\n📏 Average bleeding area in test set: {avg_bleeding_area:.2f}% of frame"
    )

    return {
        "test_loss": test_loss_total / len(test_loader),
        "classification": {
            "bleeding_accuracy": clf_acc,
            "bleeding_precision": clf_precision,
            "bleeding_recall": clf_recall,
            "bleeding_f1": clf_f1,
            "severity_accuracy": sev_acc,
            "severity_precision": sev_precision,
            "severity_recall": sev_recall,
            "severity_f1": sev_f1,
        },
        "segmentation": {
            "iou": seg_iou_avg,
            "dice": seg_dice_avg,
            "precision": seg_precision_avg,
            "recall": seg_recall_avg,
            "f1": seg_f1_avg,
        },
        "bleeding_area_percentage": avg_bleeding_area,
    }


def calculate_dice_coefficient(pred_mask, true_mask, threshold=0.5):
    """Calculate Dice coefficient for segmentation masks"""
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > 0).float()

    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def calculate_pixel_metrics(pred_mask, true_mask, threshold=0.5):
    """Calculate pixel-wise precision and recall"""
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > 0).float()

    true_positives = (pred_binary * true_binary).sum()
    false_positives = (pred_binary * (1 - true_binary)).sum()
    false_negatives = ((1 - pred_binary) * true_binary).sum()

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 1.0
    )

    return precision, recall


def calculate_average_bleeding_area(model, test_loader, device):
    """Calculate average bleeding area percentage across test set"""
    model.eval()
    total_area = 0
    count = 0

    with torch.no_grad():
        for clips, clf_labels, frame_labels, sev_labels, seg_masks in test_loader:
            clips = clips.to(device)

            _, _, seg_pred = model(clips)

            for i in range(clips.size(0)):
                if clf_labels[i] == 1:  # Only for bleeding clips
                    area_pct = calculate_bleeding_area_percentage(seg_pred[i])
                    total_area += area_pct
                    count += 1

    return total_area / count if count > 0 else 0


def load_and_evaluate_model(
    video_dir,
    annotation_dir,
    model_path,
    device="cuda:2",
    batch_size=4,
    clip_length=6,
    stride=3,
    train_split=0.7,
    val_split=0.15,
    seed=43,
):
    """Load trained model and evaluate on the same test split as training"""
    console.print(f"DEBUG: Received device parameter: {device}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    console.print(
        Panel.fit(
            f"🔍 Loading Model and Evaluating\n"
            f"Model: {model_path}\n"
            f"Device: {device}",
            title="Model Evaluation",
            border_style="blue",
        )
    )

    # Recreate the exact same data split as training
    console.print("📂 Recreating data splits...")

    # Get video files and match with annotations (same logic as training)
    video_files = [f for f in os.listdir(video_dir) if f.endswith((".mp4"))]

    matched_pairs = []
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        xml_file = f"{video_name}.xml"
        xml_path = os.path.join(annotation_dir, xml_file)

        if os.path.exists(xml_path):
            video_path = os.path.join(video_dir, video_file)
            matched_pairs.append((video_path, xml_path))

    # Use SAME random seed to ensure same split as training
    random.seed(seed)
    random.shuffle(matched_pairs)

    total_videos = len(matched_pairs)
    min_test = max(2, int(0.1 * total_videos))
    min_val = max(2, int(0.1 * total_videos))

    val_size = min_val
    test_size = min_test
    train_size = total_videos - val_size - test_size

    test_pairs = matched_pairs[train_size + val_size :]  # Same test split

    console.print(f"✅ Test set: {len(test_pairs)} videos")

    # Create test dataset with same transforms as training
    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((328, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_videos, test_annotations = zip(*test_pairs) if test_pairs else ([], [])
    test_dataset = SurgicalVideoDataset(
        list(test_videos),
        list(test_annotations),
        clip_length=clip_length,
        stride=stride,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Load the trained model
    console.print(f"🧠 Loading model from {model_path}...")
    # model = VideoBleedingDetector(
    #     num_classes=2, severity_levels=4, input_size=(328, 512)
    # )
    model = VideoBleedingDetector(num_classes=2, input_size=(328, 512))

    # Always load to CPU first, then move to target device
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        console.print("✅ Model loaded successfully to CPU")

        # Now move to target device
        model = model.to(device)
        console.print(f"✅ Model moved to {device}")
    except Exception as e:
        console.print(f"❌ Error loading model: {e}", style="red")
        return None, None, None

    # Evaluate
    console.print("🧪 Starting evaluation...")
    test_metrics = evaluate_model(model, test_loader, device)

    # Optional: Show segmentation samples
    console.print("\n🔍 Showing segmentation samples...")
    # visualize_bleeding_masks(model, test_loader, device, num_samples=10)

    return model, test_metrics, test_loader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Surgical Video Bleeding Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training/evaluation mode
    parser.add_argument(
        "--train",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to train model or evaluate existing one",
    )

    # Device settings
    parser.add_argument(
        "--dev", "--device", type=int, default=0, help="CUDA device number"
    )

    # Data paths
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/r.rohangirish/mt_ble/data/videos",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--anno_dir",
        "--annotation_dir",
        type=str,
        default="/home/r.rohangirish/mt_ble/data/labels_xml",
        help="Directory containing XML annotation files",
    )

    # Model settings
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/bleeding_detector_v1_best.pth",
        help="Path to saved model (for evaluation mode)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bleeding_detector_v1",
        help="Name for saving new model (for training mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save models and outputs",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training/evaluation"
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--clip_length", type=int, default=6, help="Number of frames per video clip"
    )
    parser.add_argument(
        "--stride", type=int, default=3, help="Stride for clip sampling"
    )

    # Evaluation settings
    parser.add_argument(
        "--num_viz",
        type=int,
        default=10,
        help="Number of segmentation samples to visualize",
    )
    parser.add_argument(
        "--seed", type=int, default=43, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cleanup",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to clean up temporary files after visualization",
    )

    return parser.parse_args()


def load_and_plot_metrics(json_file_path, save_dir="./models/plots", save_plots=True):
    """Load metrics and create beautiful seaborn plots"""

    # Load data
    with open(json_file_path, "r") as f:
        metrics = json.load(f)

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(metrics["train_loss"]) + 1))

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.2)

    # Plot 1: Loss Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(
        epochs,
        metrics["train_loss"],
        "o-",
        linewidth=3,
        markersize=6,
        label="Train Loss",
        color="#1f77b4",
    )
    ax1.set_title("Training Loss", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation Loss + Validation Classification Loss
    ax2.plot(
        epochs,
        metrics["val_loss"],
        "o-",
        linewidth=3,
        markersize=6,
        label="Val Loss",
        color="#ff7f0e",
    )
    # ax2.plot(
    #     epochs,
    #     metrics["val_clf_loss"],
    #     "s--",
    #     linewidth=3,
    #     markersize=6,
    #     label="Val Clf Loss",
    #     color="#2ca02c",
    # )
    ax2.set_title("Validation Losses", fontsize=16, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/loss_comparison.pdf", bbox_inches="tight")
    plt.close()

    # Plot 2: All Accuracies
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        epochs,
        metrics["train_clf_acc"],
        "o-",
        linewidth=3,
        label="Train Classification",
        color="#2ca02c",
    )
    ax.plot(
        epochs,
        metrics["val_clf_acc"],
        "o--",
        linewidth=3,
        label="Val Classification",
        color="#2ca02c",
        alpha=0.7,
    )
    ax.plot(
        epochs,
        metrics["train_sev_acc"],
        "s-",
        linewidth=3,
        label="Train Severity",
        color="#d62728",
    )
    ax.plot(
        epochs,
        metrics["val_sev_acc"],
        "s--",
        linewidth=3,
        label="Val Severity",
        color="#d62728",
        alpha=0.7,
    )

    ax.set_title("Classification & Severity Accuracy", fontsize=18, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([50, 100])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/accuracy_comparison.pdf", bbox_inches="tight")
    plt.close()

    # Plot 3: Segmentation IoU
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        metrics["train_seg_iou"],
        "D-",
        linewidth=4,
        markersize=7,
        label="Train IoU",
        color="#9467bd",
    )
    ax.plot(
        epochs,
        metrics["val_seg_iou"],
        "D-",
        linewidth=4,
        markersize=7,
        label="Val IoU",
        color="#ff7f0e",
    )
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="IoU = 0.5 (Good)")

    ax.set_title("Segmentation Performance (IoU)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("IoU Score", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.8])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/segmentation_iou.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/segmentation_iou.pdf", bbox_inches="tight")
    plt.close()

    # Plot 4: Overview Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss trends
    ax1.plot(epochs, metrics["train_loss"], "o-", label="Train", color="blue")
    ax1.plot(epochs, metrics["val_loss"], "o-", label="Val", color="orange")
    ax1.set_title("Loss Trends", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Classification accuracy
    ax2.plot(epochs, metrics["train_clf_acc"], "o-", label="Train", color="green")
    ax2.plot(epochs, metrics["val_clf_acc"], "o-", label="Val", color="lightgreen")
    ax2.set_title("Classification Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Severity accuracy
    ax3.plot(epochs, metrics["train_sev_acc"], "o-", label="Train", color="red")
    ax3.plot(epochs, metrics["val_sev_acc"], "o-", label="Val", color="lightcoral")
    ax3.set_title("Severity Accuracy", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Segmentation IoU
    ax4.plot(epochs, metrics["train_seg_iou"], "o-", label="Train", color="purple")
    ax4.plot(epochs, metrics["val_seg_iou"], "o-", label="Val", color="plum")
    ax4.set_title("Segmentation IoU", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Training Overview Dashboard", fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_overview.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/training_overview.pdf", bbox_inches="tight")
    plt.close()

    # Calculate metrics
    final_train_loss = metrics["train_loss"][-1]
    final_val_loss = metrics["val_loss"][-1]
    final_train_clf = metrics["train_clf_acc"][-1]
    final_val_clf = metrics["val_clf_acc"][-1]
    final_train_sev = metrics["train_sev_acc"][-1]
    final_val_sev = metrics["val_sev_acc"][-1]
    final_train_iou = metrics["train_seg_iou"][-1]
    final_val_iou = metrics["val_seg_iou"][-1]

    best_val_clf = max(metrics["val_clf_acc"])
    best_val_iou = max(metrics["val_seg_iou"])
    overfitting_gap = final_train_clf - final_val_clf

    # Create Rich Table
    table = Table(
        title="🎯 Training Results Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Train", style="green")
    table.add_column("Validation", style="yellow")
    table.add_column("Best Val", style="bright_green")

    table.add_row(
        "Loss",
        f"{final_train_loss:.3f}",
        f"{final_val_loss:.3f}",
        f"{min(metrics['val_loss']):.3f}",
    )
    table.add_row(
        "Classification",
        f"{final_train_clf:.1f}%",
        f"{final_val_clf:.1f}%",
        f"{best_val_clf:.1f}%",
    )
    table.add_row(
        "Severity",
        f"{final_train_sev:.1f}%",
        f"{final_val_sev:.1f}%",
        f"{max(metrics['val_sev_acc']):.1f}%",
    )
    table.add_row(
        "Segmentation IoU",
        f"{final_train_iou:.3f}",
        f"{final_val_iou:.3f}",
        f"{best_val_iou:.3f}",
    )

    # Status panel
    if overfitting_gap > 15:
        status_color = "red"
        status_emoji = "🚨"
        status_msg = "High Overfitting"
    elif overfitting_gap > 8:
        status_color = "yellow"
        status_emoji = "⚠️"
        status_msg = "Moderate Overfitting"
    else:
        status_color = "green"
        status_emoji = "✅"
        status_msg = "Good Generalization"

    # Create panels
    console.print("\n")
    console.print(table)
    console.print("\n")

    # Key metrics panel
    from rich.text import Text

    metrics_text = Text()
    metrics_text.append(f"Overfitting Gap: ", style="bold")
    metrics_text.append(f"{overfitting_gap:.1f}%", style=status_color)
    metrics_text.append(f" {status_emoji} {status_msg}\n", style=status_color)
    metrics_text.append(f"Total Epochs: {len(epochs)}\n")
    metrics_text.append(f"Plots Saved: {save_dir}", style="dim")

    console.print(
        Panel(
            metrics_text, title="📊 Key Insights", border_style="blue", padding=(1, 2)
        )
    )

    return metrics


def main():
    """Main function to run the complete pipeline"""

    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Convert string boolean
    train_mode = args.train.lower() == "true"
    device_str = f"cuda:{args.dev}"

    # Display configuration
    config_table = Table(title="Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Mode", "Training" if train_mode else "Evaluation")
    config_table.add_row("Device", device_str)
    config_table.add_row("Video Directory", args.video_dir)
    config_table.add_row("Annotation Directory", args.anno_dir)
    config_table.add_row("Random Seed", str(args.seed))

    if train_mode:
        config_table.add_row("Epochs", str(args.epochs))
        config_table.add_row("Batch Size", str(args.batch_size))
        config_table.add_row("Learning Rate", str(args.lr))
        config_table.add_row("Model Name", args.model_name)
    else:
        config_table.add_row("Model Path", args.model_path)
        config_table.add_row("Visualizations", str(args.num_viz))

    console.print(config_table)

    # Check paths exist
    if not os.path.exists(args.video_dir):
        console.print(
            f"❌ Video directory does not exist: {args.video_dir}", style="red"
        )
        return

    if not os.path.exists(args.anno_dir):
        console.print(
            f"❌ Annotation directory does not exist: {args.anno_dir}", style="red"
        )
        return

    if train_mode:
        # Training mode
        console.print("🎯 Starting training mode...", style="bold green")

        model, metrics, test_metrics = train_bleeding_detector(
            video_dir=video_dir,
            annotation_dir=anno_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip_length=args.clip_length,
            stride=args.stride,
            learning_rate=args.lr,
            device=device_str,
            model_name=args.model_name,
        )

        # Plot training metrics
        console.print("📈 Generating training plots...")
        load_and_plot_metrics(
            json_file_path=os.path.join(
                args.output_dir, f"{args.model_name}_metrics.json"
            ),
            save_plots=True,
        )
        console.print("✅ Training completed successfully!", style="bold green")

    else:
        # Evaluation mode
        console.print("🔍 Starting evaluation mode...", style="bold blue")

        # if not os.path.exists(args.model_path):
        #     console.print(
        #         f"❌ Model file does not exist: {args.model_path}", style="red"
        #     )
        #     console.print("💡 Available models in ./models/:", style="yellow")
        #     if os.path.exists("./models"):
        #         model_files = [f for f in os.listdir("./models") if f.endswith(".pth")]
        #         for model_file in model_files:
        #             console.print(f"   - {model_file}")
        #     return
        model_path = "./models/seg_v2_cache_best.pth"
        model, test_metrics, test_loader = load_and_evaluate_model(
            video_dir=args.video_dir,
            annotation_dir=args.anno_dir,
            model_path=model_path,
            device=device_str,
            batch_size=args.batch_size,
            clip_length=args.clip_length,
            stride=args.stride,
            seed=args.seed,
        )

        # Show additional visualizations
        if args.num_viz > 0:
            console.print(f"🎨 Showing {args.num_viz} segmentation samples...")
            visualize_bleeding_masks(
                model,
                test_loader,
                device_str,
                video_section="last_third",
                num_samples=args.num_viz,
                cleanup=args.cleanup,
            )

        load_and_plot_metrics(
            json_file_path=os.path.join(
                args.output_dir, f"{args.model_name}_metrics.json"
            ),
            save_plots=True,
        )
    console.print("✅ Pipeline completed successfully!", style="bold green")


if __name__ == "__main__":
    main()
