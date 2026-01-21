import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torchvision.ops
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
import train_seg

console = Console()

NUM_THREADS = 12
TESTING = False

if TESTING:
    VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/test_videos"
    ANNO_DIR = "/home/r.rohangirish/mt_ble/data/test_labels_xml"
else:
    VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/videos"
    ANNO_DIR = "/home/r.rohangirish/mt_ble/data/labels_xml"


def custom_collate_fn(batch):
    clips, clf_labels, frame_labels, sev_labels, gt_bboxes = zip(*batch)

    return (
        torch.stack(clips),
        torch.tensor(clf_labels),
        torch.stack(frame_labels),
        torch.tensor(sev_labels),
        list(gt_bboxes)  # Just keep as list!
    )

""" def custom_collate_fn(batch):
    clips, clf_labels, frame_labels, sev_labels, gt_bboxes = zip(*batch)

    # Stack fixed-size tensors
    clips = torch.stack(clips)  # [B, C, T, H, W]
    clf_labels = torch.tensor(clf_labels)  # [B]
    frame_labels = torch.stack(frame_labels)  # [B, T]
    sev_labels = torch.tensor(sev_labels)  # [B]

    # Pad bounding boxes
    max_boxes = max(len(b) for b in gt_bboxes)
    B = len(gt_bboxes)

    padded_boxes = torch.zeros((B, max_boxes, 5), dtype=torch.float32)
    # bbox_masks = torch.zeros((B, max_boxes), dtype=torch.bool)

    for i, boxes in enumerate(gt_bboxes):
        for j, b in enumerate(boxes):
            if j >= max_boxes:
                break
            padded_boxes[i, j, :4] = torch.tensor(b["bbox"])
            padded_boxes[i, j, 4] = b["class"]  # last column = class
           # bbox_masks[i, j] = 1  # mark as valid
    # print(f"custom_collate_fn took {t1 - t0:.2f}s for {len(batch)} samples")

    return clips, clf_labels, frame_labels, sev_labels, padded_boxes """


class BleedingDetector(nn.Module):
    def __init__(
        self,
        num_classes=2,
        severity_levels=4,
        input_size=(328, 512),
        dropout_rate=0.7,
        max_detections=10,
        num_anchors=1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.input_size = input_size
        self.max_detections = max_detections
        self.num_anchors = num_anchors
        self._initialize_weights()
        try:
            full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(full_model.children())[:-2])
            in_features = 512
            console.print("Using R2Plus1D-18 backbone for feature extraction")
        except:
            console.print("Problem with loading the model. Exiting..")
            return

        # for param in self.backbone.parameters():
        #     param.requires_grad = False  # Freeze backbone parameters

        # Feature extraction gives us [B, 512, T/8, H/8, W/8]
        self.feature_channels = in_features
        self.backbone.train()

        # Global pooling for classification tasks
        self.global_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classification heads
        self.classifier = nn.Linear(self.feature_channels, num_classes)
        self.severity_classifier = nn.Linear(self.feature_channels, severity_levels)

        # Detection head - outputs bounding boxes instead of segmentation
        # Each spatial location can predict multiple anchors
        # Output: [x_center, y_center, width, height, objectness, class1, class2, class3]
        self.detection_head = nn.Sequential(
            nn.Conv3d(self.feature_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.3),
            # Final detection layer: 8 outputs per anchor
            # [x, y, w, h, objectness, class_BL_Low, class_BL_Medium, class_BL_High]
            nn.Conv3d(128, num_anchors * 8, kernel_size=1),
        )
        self.anchors = torch.tensor([[0.20, 0.29]])
        # Anchor boxes (different sizes for different bleeding severities)
        # self.register_buffer(
        #     "anchors",
        #     torch.tensor(
        #         [
        #             [0.15, 0.20],  # Small anchor for BL_Low
        #             # [0.15, 0.20],  # Medium anchor for BL_Medium
        #            # [0.25, 0.35],  # Large anchor for BL_High
        #         ]
        #     ),
        # )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, train=True):
        # x: [batch_size, channels, frames, height, width]
        features = self.backbone(x)  # [B, 512, T/8, H/8, W/8]

        if train:
            self.backbone.train()  # Ensure backbone is in training mode

        # Classification from global pooled features
        pooled_features = self.global_avgpool(features)  # [B, 512, 1, 1, 1]
        pooled_features = torch.flatten(pooled_features, 1)  # [B, 512]
        pooled_features = self.dropout(pooled_features)

        # Classification predictions
        clip_pred = self.classifier(pooled_features)
        severity_pred = self.severity_classifier(pooled_features)

        # Detection prediction
        detection_raw = self.detection_head(
            features
        )  # [B, num_anchors*8, T/8, H/8, W/8]

        # Average over temporal dimension to get clip-level detections
        detection_raw = torch.mean(detection_raw, dim=2)  # [B, num_anchors*8, H/8, W/8]

        # Reshape for easier processing
        B, _, H, W = detection_raw.shape
        detection_raw = detection_raw.view(
            B, self.num_anchors, 8, H, W
        )  # [B, 3, 8, H/8, W/8]
        detection_raw = detection_raw.permute(0, 1, 3, 4, 2)  # [B, 3, H/8, W/8, 8]

        # Parse detections: [x, y, w, h, objectness, class1, class2, class3]
        detection_pred = self.parse_detections(detection_raw)

        return clip_pred, severity_pred, detection_pred

    def parse_detections(self, raw_detections, conf_threshold=0.2):
        """
        Fast, fully vectorized parsing of raw detections.
        Input: raw_detections: [B, num_anchors, H, W, 8]
        Output: list of detections per batch (list of dicts)
        """
        B, A, H, W, _ = raw_detections.shape
        device = raw_detections.device

        # Activation
        x_offset = torch.sigmoid(raw_detections[..., 0])
        y_offset = torch.sigmoid(raw_detections[..., 1])
        w_scale = torch.exp(raw_detections[..., 2])
        h_scale = torch.exp(raw_detections[..., 3])
        objectness = torch.sigmoid(raw_detections[..., 4])
        class_scores = torch.softmax(raw_detections[..., 5:], dim=-1)

        # Grids
        grid_x = torch.arange(W, device=device).view(1, 1, 1, W).float() / W
        grid_y = torch.arange(H, device=device).view(1, 1, H, 1).float() / H
        grid_x = grid_x.expand(B, A, H, W)
        grid_y = grid_y.expand(B, A, H, W)

        anchor_w = self.anchors[:, 0].view(1, A, 1, 1).to(device)
        anchor_h = self.anchors[:, 1].view(1, A, 1, 1).to(device)

        x_center = (grid_x + x_offset) * self.input_size[1]
        y_center = (grid_y + y_offset) * self.input_size[0]
        box_w = anchor_w * w_scale * self.input_size[1]
        box_h = anchor_h * h_scale * self.input_size[0]

        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        # Confidence = objectness × best class score
        best_class_score, best_class_idx = class_scores.max(dim=-1)
        confidence = objectness * best_class_score

        # Flatten for easy masking
        x1 = x1.reshape(B, -1)
        y1 = y1.reshape(B, -1)
        x2 = x2.reshape(B, -1)
        y2 = y2.reshape(B, -1)
        confidence = confidence.reshape(B, -1)
        best_class_idx = best_class_idx.reshape(B, -1)

        detections = []
        for b in range(B):
            mask = confidence[b] > conf_threshold
            if mask.sum() == 0:
                detections.append([])
                continue

            boxes = torch.stack(
                [x1[b][mask], y1[b][mask], x2[b][mask], y2[b][mask]], dim=1
            )
            scores = confidence[b][mask]
            classes = best_class_idx[b][mask]

            # Optional: torchvision NMS here
            keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.3)
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            dets = [
                {
                    "bbox": box.tolist(),
                    "confidence": score.item(),
                    "class": cls.item(),
                    "class_name": ["BL_Low", "BL_Medium", "BL_High"][cls.item()],
                }
                for box, score, cls in zip(boxes, scores, classes)
            ]
            detections.append(dets)

        return detections

    def nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression to remove overlapping boxes"""
        if len(detections) == 0:
            return detections

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        keep = []
        while len(detections) > 0:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)

            # Remove all detections with high IoU overlap
            remaining = []
            for det in detections:
                if self.bbox_iou(current["bbox"], det["bbox"]) < iou_threshold:
                    remaining.append(det)
            detections = remaining

        return keep[: self.max_detections]  # Limit number of detections

    def bbox_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class SurgicalVideoDataset(Dataset):
    def __init__(
        self, video_paths, annotation_paths, clip_length=6, stride=3, transform=None
    ):
        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.annotation_cache = {
            path: self._parse_xml_annotations(path) for path in self.annotation_paths
        }

        # Create clips with annotations
        with console.status("[bold green]Preparing video clips..."):
            self.clips = self._prepare_clips()

        # Display dataset statistics
        self._display_dataset_stats()

    def _prepare_clips(self):
        """Prepare clips with consistent bleeding labels derived from bounding boxes"""
        bleeding_clips = {1: [], 2: [], 3: []}
        non_bleeding_clips = []

        # Statistics tracking
        stats = {
            "total_clips_processed": 0,
            "clips_with_boxes": 0,
            "clips_without_boxes": 0,
            "error_clips": 0,
            "videos_processed": 0
        }

        console.print("[bold green]Preparing video clips...[/bold green]")

        for video_idx, (video_path, anno_path) in enumerate(zip(self.video_paths, self.annotation_paths)):
            video_id = os.path.basename(video_path).split(".")[0]
            stats["videos_processed"] += 1

            # Open video to get frame count
            cap = cv2.VideoCapture(video_path)
            params = [cv2.CAP_PROP_N_THREADS, NUM_THREADS]
            cap.open(video_path, apiPreference=cv2.CAP_FFMPEG, params=params)

            if not cap.isOpened():
                console.print(f"⚠️  Could not open video {video_path}", style="yellow")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                console.print(f"⚠️  Invalid frame count for {video_path}: {frame_count}", style="yellow")
                continue

            # Process clips from this video
            video_clips_with_boxes = 0
            video_clips_without_boxes = 0

            for start_idx in range(0, frame_count - self.clip_length + 1, self.stride):
                end_idx = start_idx + self.clip_length
                stats["total_clips_processed"] += 1

                # Get ground truth boxes for this clip range
                gt_bboxes = self._create_gt_bboxes_from_xml(
                    anno_path, start_idx, end_idx - 1, target_size=(328, 512)
                )

                # Derive everything from the boxes
                has_bleeding = len(gt_bboxes) > 0

                if has_bleeding:
                    # Calculate severity from actual boxes
                    severity_classes = [box['class'] for box in gt_bboxes]
                    max_severity_class = max(severity_classes)
                    # Convert class (0,1,2) to severity (1,2,3)
                    max_severity = max_severity_class + 1

                    # Create frame-level labels based on boxes
                    frame_labels = np.zeros(self.clip_length, dtype=np.int32)
                    for box in gt_bboxes:
                        if 'frame_in_clip' in box:
                            frame_idx = box['frame_in_clip']
                            if 0 <= frame_idx < self.clip_length:
                                frame_labels[frame_idx] = max(frame_labels[frame_idx], max_severity)
                        else:
                            # If no frame info, mark all frames
                            frame_labels[:] = max_severity

                    clip_info = {
                        "video_path": video_path,
                        "annotation_path": anno_path,
                        "video_id": video_id,
                        "start_frame": start_idx,
                        "end_frame": end_idx - 1,
                        "has_bleeding": True,
                        "bleeding_frames": frame_labels,
                        "max_severity": max_severity,
                        "gt_bboxes": gt_bboxes,
                        "num_boxes": len(gt_bboxes)
                    }

                    bleeding_clips[max_severity].append(clip_info)
                    stats["clips_with_boxes"] += 1
                    video_clips_with_boxes += 1

                else:
                    # Non-bleeding clip
                    clip_info = {
                        "video_path": video_path,
                        "annotation_path": anno_path,
                        "video_id": video_id,
                        "start_frame": start_idx,
                        "end_frame": end_idx - 1,
                        "has_bleeding": False,
                        "bleeding_frames": np.zeros(self.clip_length, dtype=np.int32),
                        "max_severity": 0,
                        "gt_bboxes": [],
                        "num_boxes": 0
                    }

                    non_bleeding_clips.append(clip_info)
                    stats["clips_without_boxes"] += 1
                    video_clips_without_boxes += 1

            # Log video statistics
            if video_idx < 5 or video_idx % 10 == 0:  # Log first 5 and every 10th
                console.print(
                    f"  Video {video_id}: {video_clips_with_boxes} bleeding clips, "
                    f"{video_clips_without_boxes} non-bleeding clips"
                )

        # Calculate totals before balancing
        severity_counts = {s: len(clips) for s, clips in bleeding_clips.items()}
        total_bleeding = sum(severity_counts.values())
        total_non_bleeding = len(non_bleeding_clips)

        # console.print("\n[bold]Pre-balancing Statistics:[/bold]")
        # console.print(f"  Total bleeding clips: {total_bleeding}")
        # console.print(f"    - Low severity: {severity_counts[1]}")
        # console.print(f"    - Medium severity: {severity_counts[2]}")
        # console.print(f"    - High severity: {severity_counts[3]}")
        # console.print(f"  Total non-bleeding clips: {total_non_bleeding}")

        # Balance dataset
        if total_non_bleeding > total_bleeding:
            # Reduce non-bleeding to match bleeding
            random.shuffle(non_bleeding_clips)
            non_bleeding_clips = non_bleeding_clips[:total_bleeding]
            console.print(f"\n[cyan]Balanced: Reduced non-bleeding from {total_non_bleeding} to {total_bleeding}[/cyan]")
        elif total_bleeding > total_non_bleeding * 1.5:
            # If too few non-bleeding, warn
            console.print(
                f"\n[yellow]Warning: Bleeding clips ({total_bleeding}) significantly "
                f"outnumber non-bleeding ({total_non_bleeding})[/yellow]"
            )

        # Combine all clips
        all_clips = []
        all_clips.extend(non_bleeding_clips)
        for severity in [1, 2, 3]:
            all_clips.extend(bleeding_clips[severity])

        # Shuffle to mix bleeding and non-bleeding
        random.shuffle(all_clips)

        # Final statistics
        # console.print("\n[bold green]Dataset Preparation Complete:[/bold green]")
        # console.print(f"  Total clips in dataset: {len(all_clips)}")
        # console.print(f"  Bleeding clips: {sum(1 for c in all_clips if c['has_bleeding'])}")
        # console.print(f"  Non-bleeding clips: {sum(1 for c in all_clips if not c['has_bleeding'])}")

        # Sanity check
        bleeding_without_boxes = sum(1 for c in all_clips if c['has_bleeding'] and len(c['gt_bboxes']) == 0)
        if bleeding_without_boxes > 0:
            console.print(f"\n[red]ERROR: Found {bleeding_without_boxes} bleeding clips without boxes![/red]")
            console.print("[red]This should not happen - check annotation parsing![/red]")

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
                        xtl = float(box.get("xtl"))
                        ytl = float(box.get("ytl"))
                        xbr = float(box.get("xbr"))
                        ybr = float(box.get("ybr"))

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


    def _create_gt_bboxes_from_xml(self, xml_path, start_frame, end_frame, target_size=(328, 512)):
        """Create clip-level ground truth bounding boxes by aggregating frame boxes"""
        frame_boxes = {}  # Group by track_id
        severity_map = {"BL_Low": 0, "BL_Medium": 1, "BL_High": 2}

        if not xml_path or not os.path.exists(xml_path):
            return []

        annotations = self.annotation_cache.get(xml_path, [])

        # Collect all boxes in this clip range, grouped by track
        for anno in annotations:
            frame_num = anno["frame"]

            # Check if this annotation falls within our clip
            if start_frame <= frame_num <= end_frame:
                track_id = anno.get("track_id", f"track_{frame_num}")  # Fallback if no track_id

                if track_id not in frame_boxes:
                    frame_boxes[track_id] = []

                # Scale bounding box to target size
                xtl, ytl, xbr, ybr = anno["bbox"]
                orig_width = anno["original_width"]
                orig_height = anno["original_height"]

                # Scale coordinates
                h_scale = target_size[0] / orig_height
                w_scale = target_size[1] / orig_width

                x1_scaled = xtl * w_scale
                y1_scaled = ytl * h_scale
                x2_scaled = xbr * w_scale
                y2_scaled = ybr * h_scale

                # Clamp to image bounds
                x1_scaled = max(0, min(x1_scaled, target_size[1] - 1))
                y1_scaled = max(0, min(y1_scaled, target_size[0] - 1))
                x2_scaled = max(0, min(x2_scaled, target_size[1] - 1))
                y2_scaled = max(0, min(y2_scaled, target_size[0] - 1))

                # Only keep valid bounding boxes
                if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                    frame_boxes[track_id].append({
                        "bbox": [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                        "label": anno["label"],
                        "frame": frame_num,
                        "frame_in_clip": frame_num - start_frame
                    })

        # Aggregate boxes by track to create clip-level boxes
        gt_boxes = []

        for track_id, boxes in frame_boxes.items():
            if not boxes:
                continue

            # Option 1: Average position across frames (for stable objects)
            avg_x1 = np.mean([b["bbox"][0] for b in boxes])
            avg_y1 = np.mean([b["bbox"][1] for b in boxes])
            avg_x2 = np.mean([b["bbox"][2] for b in boxes])
            avg_y2 = np.mean([b["bbox"][3] for b in boxes])

            # Option 2: Union of all boxes (for moving objects)
            # union_x1 = min([b["bbox"][0] for b in boxes])
            # union_y1 = min([b["bbox"][1] for b in boxes])
            # union_x2 = max([b["bbox"][2] for b in boxes])
            # union_y2 = max([b["bbox"][3] for b in boxes])

            # Get most common label for this track
            labels = [b["label"] for b in boxes]
            most_common_label = max(set(labels), key=labels.count)

            # Calculate presence ratio (how many frames this appears in)
            presence_ratio = len(boxes) / self.clip_length

            gt_boxes.append({
                "bbox": [avg_x1, avg_y1, avg_x2, avg_y2],
                "class": severity_map.get(most_common_label, 0),
                "class_name": most_common_label,
                "confidence": 1.0,
                "track_id": track_id,
                "num_frames": len(boxes),
                "presence_ratio": presence_ratio,
                "frame_in_clip": boxes[len(boxes)//2]["frame_in_clip"]  # Middle frame
            })

        return gt_boxes

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
        t0 = time.time()
        # print(f"🌀 __getitem__[{idx}] START")

        clip_info = self.clips[idx]

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

        # Use cached bounding boxes
        gt_bboxes = clip_info["gt_bboxes"]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        clip_tensor = torch.stack(frames).permute(1, 0, 2, 3)

        clip_label = 1 if clip_info["has_bleeding"] else 0
        frame_labels = torch.tensor(clip_info["bleeding_frames"], dtype=torch.float32)
        severity = torch.tensor(clip_info["max_severity"], dtype=torch.long)

        t1 = time.time()
        # print(f"__getitem__[{idx}] took {t1 - t0:.2f}s")

        return clip_tensor, clip_label, frame_labels, severity, gt_bboxes

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
        # print(f"🕒 Loaded {len(frames)} frames from {video_path} in {t1 - t0:.2f}s")
        return frames


def calculate_detection_loss_complex(detections, gt_bboxes, device):
    """Fixed detection loss with better error handling"""
    total_loss = 0.0
    batch_size = len(detections)

    if batch_size == 0:
        return torch.tensor(0.0, device=device)

    for b in range(batch_size):
        pred_boxes = detections[b]
        gt_boxes = gt_bboxes[b]

        # Convert predictions to tensors
        if pred_boxes:
            pred_data = [
                (box["bbox"], box["confidence"], box["class"]) for box in pred_boxes
            ]
            if pred_data:
                bboxes, confs, classes = zip(*pred_data)
                pred_coords = torch.tensor(bboxes, dtype=torch.float32, device=device)
                pred_confs = torch.tensor(confs, dtype=torch.float32, device=device)
                pred_classes = torch.tensor(classes, dtype=torch.float32, device=device)
            else:
                pred_coords = torch.zeros((0, 4), device=device)
                pred_confs = torch.zeros((0,), device=device)
                pred_classes = torch.zeros((0,), device=device)
        else:
            pred_coords = torch.zeros((0, 4), device=device)
            pred_confs = torch.zeros((0,), device=device)
            pred_classes = torch.zeros((0,), device=device)

        # Process ground truth
        valid_gt = [box for box in gt_boxes if isinstance(box, dict) and "bbox" in box]

        if valid_gt:
            gt_coords = torch.tensor(
                [box["bbox"] for box in valid_gt], dtype=torch.float32, device=device
            )
            gt_classes = torch.tensor(
                [box["class"] for box in valid_gt], dtype=torch.float32, device=device
            )
        else:
            gt_coords = torch.zeros((0, 4), device=device)
            gt_classes = torch.zeros((0,), device=device)

        # Calculate losses based on different scenarios
        if len(pred_coords) > 0 and len(gt_coords) > 0:
            # Match predictions to ground truth using IoU
            ious = torch.zeros((len(pred_coords), len(gt_coords)), device=device)
            for i, pred in enumerate(pred_coords):
                for j, gt in enumerate(gt_coords):
                    ious[i, j] = calculate_iou_tensor(pred, gt)

            # Find best matches
            matched_indices = ious.max(dim=1)[1]
            max_ious = ious.max(dim=1)[0]

            # Coordinate loss for matched boxes
            coord_loss = F.l1_loss(pred_coords, gt_coords[matched_indices])

            # Confidence loss (high for good matches, low for poor matches)
            conf_targets = (max_ious > 0.5).float()
            conf_loss = F.mse_loss(pred_confs, conf_targets)

            # Class loss for matched boxes
            class_loss = F.mse_loss(pred_classes, gt_classes[matched_indices])

            total_loss += coord_loss + conf_loss + class_loss

        elif len(pred_coords) > 0 and len(gt_coords) == 0:
            # False positives - penalize confidence
            conf_loss = F.mse_loss(pred_confs, torch.zeros_like(pred_confs))
            total_loss += conf_loss

        elif len(pred_coords) == 0 and len(gt_coords) > 0:
            # Missed detections - add penalty
            missed_penalty = torch.tensor(1.0, device=device) * len(gt_coords)
            total_loss += missed_penalty

    return total_loss / max(batch_size, 1)


def calculate_detection_loss(detections, gt_bboxes, device):
    """Fixed detection loss with proper device handling"""
    total_loss = torch.tensor(0.0, device=device)
    num_samples = 0

    for b in range(len(detections)):
        pred_boxes = detections[b]
        gt_boxes_batch = gt_bboxes[b]

        # Process GT boxes
        if isinstance(gt_boxes_batch, torch.Tensor):
            valid_mask = gt_boxes_batch[:, 4] >= 0
            gt_boxes = gt_boxes_batch[valid_mask]
            # Ensure GT boxes are on correct device
            if gt_boxes.device != device:
                gt_boxes = gt_boxes.to(device)
        else:
            # Convert list of dicts to tensor on correct device
            valid_boxes = [box for box in gt_boxes_batch if isinstance(box, dict) and 'bbox' in box]
            if valid_boxes:
                gt_boxes = torch.tensor(
                    [[box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3], box['class']]
                     for box in valid_boxes],
                    device=device, dtype=torch.float32
                )
            else:
                gt_boxes = torch.empty((0, 5), device=device, dtype=torch.float32)

        if len(gt_boxes) == 0:
            # No GT boxes - penalize any predictions
            if len(pred_boxes) > 0:
                total_loss += len(pred_boxes) * 0.5
            continue

        if len(pred_boxes) == 0:
            # Missed detections - strong penalty
            total_loss += len(gt_boxes) * 2.0
            num_samples += 1
            continue

        # For each GT box, find closest prediction
        for gt_box in gt_boxes:
            gt_bbox = gt_box[:4]
            gt_class = int(gt_box[4].item())

            min_dist = float('inf')
            best_pred = None

            for pred in pred_boxes:
                # Create pred_bbox tensor on the SAME device as gt_bbox
                pred_bbox = torch.tensor(pred['bbox'], device=device, dtype=torch.float32)

                # Simple L1 distance
                dist = torch.abs(pred_bbox - gt_bbox).sum()
                if dist < min_dist:
                    min_dist = dist
                    best_pred = pred

            if best_pred is not None:
                # Bbox regression loss - ensure same device
                pred_bbox = torch.tensor(best_pred['bbox'], device=device, dtype=torch.float32)
                bbox_loss = F.smooth_l1_loss(pred_bbox, gt_bbox)

                # Class loss
                class_loss = 0.5 if best_pred['class'] != gt_class else 0.0

                # Confidence loss - encourage high confidence for matches
                conf_loss = (1.0 - best_pred['confidence']) ** 2

                total_loss += bbox_loss + class_loss + conf_loss
            else:
                # No prediction matched this GT
                total_loss += 2.0

            num_samples += 1

    return total_loss / max(num_samples, 1)



def calculate_iou_tensor(box1, box2):
    """Calculate IoU between two boxes (tensors)"""
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def calculate_detection_metrics(detections, gt_bboxes):
    """Fixed detection metrics calculation"""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, gt_list in zip(detections, gt_bboxes):
        # Process ground truth
        # Convert tensor-based GTs to dict format
        if isinstance(gt_list, torch.Tensor):
            gt_list = [
                {"bbox": box[:4].tolist(), "class": int(box[4].item())}
                for box in gt_list
                if box[4] != -1  # filter out padding if applicable
            ]
        elif isinstance(gt_list, list):
            gt_list = [box for box in gt_list if isinstance(box, dict) and "bbox" in box]
        else:
            gt_list = []

        valid_gt = [box for box in gt_list if isinstance(box, dict) and "bbox" in box]

        if not preds and not valid_gt:
            continue

        if not preds and valid_gt:
            total_fn += len(valid_gt)
            continue

        if preds and not valid_gt:
            total_fp += len(preds)
            continue

        # Match predictions to ground truth
        matched_gt = set()
        for pred in preds:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(valid_gt):
                iou = calculate_bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > 0.5:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

        # Count unmatched ground truth as false negatives
        total_fn += len(valid_gt) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    return precision, recall


def calculate_bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def debug_detection_batch(detections, gt_bboxes, batch_idx, phase="train"):
    """Debug detection outputs vs ground truth"""
    console.print(f"\n[{phase.upper()} DEBUG] --- Batch {batch_idx} ---")

    for i in range(len(detections)):
        pred_boxes = detections[i]
        gt_boxes = gt_bboxes[i]

        # Count valid GT boxes
        if isinstance(gt_boxes, torch.Tensor):
            valid_gt = (gt_boxes[:, 4] >= 0).sum().item()
            gt_list = [
                {"bbox": gt_boxes[j, :4].tolist(), "class": int(gt_boxes[j, 4].item())}
                for j in range(gt_boxes.shape[0]) if gt_boxes[j, 4] >= 0
            ]
        else:
            valid_gt = len([b for b in gt_boxes if isinstance(b, dict)])
            gt_list = gt_boxes

        console.print(f"  Sample {i}: {len(pred_boxes)} predicted, {valid_gt} GT boxes")

        # Show first few predictions
        if len(pred_boxes) > 0:
            console.print(f"    First pred: conf={pred_boxes[0]['confidence']:.3f}, "
                         f"class={pred_boxes[0]['class_name']}, "
                         f"bbox={[f'{x:.1f}' for x in pred_boxes[0]['bbox']]}")

        # Show first GT box
        if valid_gt > 0 and len(gt_list) > 0:
            console.print(f"    First GT: class={gt_list[0]['class']}, "
                         f"bbox={[f'{x:.1f}' for x in gt_list[0]['bbox']]}")


def validate_ground_truth(train_loader, num_batches=5):
    """Check if GT boxes are valid"""
    # console.print("\n[bold]Validating Ground Truth Data:[/bold]")

    total_clips = 0
    clips_with_boxes = 0
    total_boxes = 0
    box_sizes = []

    for i, (_, clf_labels, _, sev_labels, gt_bboxes) in enumerate(train_loader):
        if i >= num_batches:
            break

        for j in range(len(clf_labels)):
            total_clips += 1

            # Check if this is supposed to have bleeding
            if clf_labels[j] == 1:  # Bleeding clip
                # Count valid boxes
                if isinstance(gt_bboxes[j], torch.Tensor):
                    valid_boxes = (gt_bboxes[j][:, 4] >= 0).sum().item()
                else:
                    valid_boxes = len([b for b in gt_bboxes[j] if isinstance(b, dict)])

                if valid_boxes > 0:
                    clips_with_boxes += 1
                    total_boxes += valid_boxes

                    # Check box sizes
                    if isinstance(gt_bboxes[j], torch.Tensor):
                        for k in range(gt_bboxes[j].shape[0]):
                            if gt_bboxes[j][k, 4] >= 0:
                                w = (gt_bboxes[j][k, 2] - gt_bboxes[j][k, 0]).item()
                                h = (gt_bboxes[j][k, 3] - gt_bboxes[j][k, 1]).item()
                                box_sizes.append((w, h))
                else:
                    console.print(f"[red]WARNING: Bleeding clip without GT boxes![/red]")

    console.print(f"Total clips checked: {total_clips}")
    console.print(f"Clips with boxes: {clips_with_boxes}")
    console.print(f"Total boxes: {total_boxes}")
    console.print(f"Avg boxes per bleeding clip: {total_boxes/clips_with_boxes if clips_with_boxes > 0 else 0:.2f}")

    if box_sizes:
        avg_w = np.mean([s[0] for s in box_sizes])
        avg_h = np.mean([s[1] for s in box_sizes])
        console.print(f"Average box size: {avg_w:.1f} x {avg_h:.1f}")


def train_detector(
    video_dir,
    annotation_dir,
    output_dir="./models_detector",
    epochs=30,
    batch_size=4,
    clip_length=6,
    stride=3,
    learning_rate=0.0001,
    device="cuda:0",
    model_name="bleeding_detector",
    test=False,
    split_loss=True,  # Whether to split loss into classification, severity, and detection
):
    """Train bleeding detection model with video-level splitting and comprehensive metrics"""

    # Setup

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # torch.cuda.empty_cache()
    os.makedirs(output_dir, exist_ok=True)

    console.print(
        Panel.fit(
            f"Bleeding Detection & Quantification\n"
            f"Device: {device}\n"
            f"Epochs: {epochs} | Batch Size: {batch_size}\n"
            f"Clip Length: {clip_length} | Learning Rate: {learning_rate}\n"
            f"Tasks: Classification + Severity + Detection",
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
    if TESTING:
        min_test = 1
        min_val = 1

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
    console.print("\nCreating datasets...")

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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    console.print(
        f"📊 Clips - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    console.print(
        f"Validating Ground Truth Data in Train Loader..."
    )
    validate_ground_truth(train_loader, num_batches=10)

    model = BleedingDetector(num_classes=2, input_size=(328, 512)).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"🧠 Total parameters: {total_params:,}")

    # Training setup
    criterion_clf = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=5, verbose=True
    )

    # Metrics tracking
    metrics = {
        "train_loss": [],
        "train_clf_acc": [],
        "train_sev_acc": [],
        "train_det_precision": [],
        "train_det_recall": [],
        "val_loss": [],
        "val_clf_loss": [],
        "val_clf_acc": [],
        "val_sev_acc": [],
        "val_det_precision": [],
        "val_det_recall": [],
    }

    # torch.cuda.reset_peak_memory_stats()
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    st = time.time()

    # Training loop
    initial_weights = model.classifier.weight.clone()
    for epoch in range(epochs):
        console.print(f"\n[bold blue]========= Epoch {epoch+1}/{epochs} ========= [/bold blue]")
        if epoch == 1:
            current_weights = model.classifier.weight
            weight_change = (current_weights - initial_weights).abs().mean()
            console.print(f"Weight change after epoch 1: {weight_change:.8f}")

        # === Training ===
        model.train()
        train_loss_total = 0
        train_clf_correct = 0
        train_sev_correct = 0
        train_det_precision_total = 0
        train_det_recall_total = 0
        train_total = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[green]{task.percentage:>3.0f}%[/green]"),
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
                gt_bboxes,
            ) in enumerate(train_loader):
                t0 = time.time()
                # console.print(
                #     f"==== Processing batch {batch_idx + 1}/{len(train_loader)} =====",
                # )
                # print(f"Processing batch {batch_idx + 1}/{len(train_loader)}", end="\r")
                clips = clips.to(device)
                clf_labels = clf_labels.to(device)
                sev_labels = sev_labels.to(device)

                optimizer.zero_grad()

                t0_model = time.time()
                clf_pred, sev_pred, detections = model(clips)
                t1_model = time.time()
                # console.print(
                #     f"Model forward pass took {t1_model - t0_model:.2f}s for batch {batch_idx + 1}"
                # )

                # Calculate losses
                loss_clf = criterion_clf(clf_pred, clf_labels)
                loss_sev = criterion_sev(sev_pred, sev_labels)
                loss_det = calculate_detection_loss(detections, gt_bboxes, device)

                total_loss = loss_clf + 0.3 * loss_sev + 0.5 * loss_det

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Metrics
                train_loss_total += total_loss.item()
                train_clf_correct += clf_pred.argmax(1).eq(clf_labels).sum().item()
                train_sev_correct += sev_pred.argmax(1).eq(sev_labels).sum().item()
                train_total += clf_labels.size(0)

                batch_precision, batch_recall = calculate_detection_metrics(
                    detections, gt_bboxes
                )
                train_det_precision_total += batch_precision
                train_det_recall_total += batch_recall

                if batch_idx % 10 == 0:
                    console.print(
                        f"Training: [{batch_idx:3d}/{len(train_loader):3d}] Loss: {total_loss.item():.4f}",
                        end="\r",
                    )

                time_taken = time.time() - t0

                if batch_idx % 250 == 0:
                    # console.print(
                    #     f"Batch {batch_idx}/{len(train_loader)} processed in {time_taken:.2f}s"
                    # )
                    # console.print(
                    #     f"Detection output shape: {detections[0].__len__()} boxes in sample"
                    # )
                    console.print(f"\n[bold]Debugging Detection Output for batch {batch_idx+1}:[/bold]")
                    debug_detection_batch(detections, gt_bboxes, batch_idx, phase="train")

                    # console.print(f"\nBatch {batch_idx} losses:")
                    # console.print(f"  Classification: {loss_clf.item():.4f}")
                    # console.print(f"  Severity: {loss_sev.item():.4f}")
                    # console.print(f"  Detection: {loss_det.item():.4f}")
                    # console.print(f"  Total: {total_loss.item():.4f}")

                    # Check detection stats
                    num_preds = sum(len(d) for d in detections)
                    num_gt = sum(len(g) for g in gt_bboxes)
                    console.print(f"  Predictions: {num_preds}, GT boxes: {num_gt}")

                progress.update(train_task, advance=1)

        # === Validation ===
        model.eval()
        val_loss_total = 0
        val_clf_loss_total = 0
        val_clf_correct = 0
        val_sev_correct = 0
        val_det_precision_total = 0
        val_det_recall_total = 0
        val_total = 0
        gt_boxes_total = 0
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
                    gt_bboxes,
                ) in val_loader:
                    # print(f"\n[VAL DEBUG] --- Batch {batch_idx} ---")
                    # for i, (preds, bboxes) in enumerate(zip(detections, gt_bboxes)):
                    #     gt_boxes_total += len(bboxes)
                    #     print(f"  Sample {i}: {len(preds)} predicted boxes")
                    #     if preds:
                    #         print(f"    First pred box: {preds[0]}")
                    #     else:
                    #         print("    No predicted boxes")

                    #     if isinstance(bboxes, torch.Tensor):
                    #         bboxes = [
                    #             {"bbox": box[:4].tolist(), "class": int(box[4])}
                    #             for box in bboxes
                    #             if box.shape[0] >= 5 and box[4] != -1
                    #         ]
                    #     print(f"    GT boxes: {len(bboxes)}")
                    #     if bboxes:
                    #         print(f"    First GT box: {bboxes[0]}")

                    clips = clips.to(device)
                    clf_labels = clf_labels.to(device)
                    sev_labels = sev_labels.to(device)

                    clf_pred, sev_pred, detections = model(clips)

                    # Losses
                    loss_clf = criterion_clf(clf_pred, clf_labels)
                    loss_sev = criterion_sev(sev_pred, sev_labels)
                    loss_det = calculate_detection_loss(detections, gt_bboxes, device)

                    total_loss = loss_clf + 0.1 * loss_sev + 1.5 * loss_det

                    # Metrics
                    val_loss_total += total_loss.item()
                    if split_loss:
                        val_clf_loss_total += loss_clf.item()

                    val_clf_correct += clf_pred.argmax(1).eq(clf_labels).sum().item()
                    val_sev_correct += sev_pred.argmax(1).eq(sev_labels).sum().item()
                    val_total += clf_labels.size(0)

                    # gt_bboxes_fixed = []
                    # for b in gt_bboxes:
                    #     b = [
                    #         box for box in b
                    #         if box["bbox"] != [0.0, 0.0, 0.0, 0.0]
                    #     ]
                    #     gt_bboxes_fixed.append(b)

                    batch_precision, batch_recall = calculate_detection_metrics(
                        detections, gt_bboxes
                    )

                    val_det_precision_total += batch_precision
                    val_det_recall_total += batch_recall
                    # console.print(f"GT Boxes Total: {gt_boxes_total}")
                    progress.update(val_task, advance=1)

        # ====== Epoch metrics ======

        train_loss_avg = train_loss_total / len(train_loader)
        train_clf_acc = 100.0 * train_clf_correct / train_total
        train_sev_acc = 100.0 * train_sev_correct / train_total
        train_det_precision = train_det_precision_total / len(train_loader)
        train_det_recall = train_det_recall_total / len(train_loader)

        val_loss_avg = val_loss_total / len(val_loader)
        val_clf_loss_avg = val_clf_loss_total / len(val_loader) if split_loss else 0
        val_clf_acc = 100.0 * val_clf_correct / val_total
        val_sev_acc = 100.0 * val_sev_correct / val_total
        val_det_precision = val_det_precision_total / len(val_loader)
        val_det_recall = val_det_recall_total / len(val_loader)

        metrics["train_loss"].append(train_loss_avg)
        metrics["train_clf_acc"].append(train_clf_acc)
        metrics["train_sev_acc"].append(train_sev_acc)
        metrics["train_det_precision"].append(train_det_precision)
        metrics["train_det_recall"].append(train_det_recall)
        metrics["val_loss"].append(val_loss_avg)
        metrics["val_clf_acc"].append(val_clf_acc)
        metrics["val_sev_acc"].append(val_sev_acc)
        metrics["val_det_precision"].append(val_det_precision)
        metrics["val_det_recall"].append(val_det_recall)
        if split_loss:
            metrics["val_clf_loss"].append(val_clf_loss_avg)
        else:
            metrics["val_clf_loss"].append(0)

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
            "Detection Precision",
            f"{train_det_precision:.4f}",
            f"{val_det_precision:.4f}",
        )
        results_table.add_row(
            "Detection Recall", f"{train_det_recall:.4f}", f"{val_det_recall:.4f}"
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

    # ========= Final Save ==========

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
    console.print(f"Plotting metrics...")
    # =========== Testing ============

    console.print("\n🧪 Starting test evaluation...")
    test_metrics = evaluate_model(model, test_loader, device)

    console.print("\n🔍 Visualizing bleeding detection results...")
    os.makedirs("./visualization", exist_ok=True)
    visualize_bleeding_detections(
        model,
        test_loader,
        device,
        num_samples=30,
        path="./detection_viz",
        cleanup=True,
    )
    console.print("✅ Visualization complete!")
    return model, metrics, test_metrics


def evaluate_model(model, test_loader, device):
    """Comprehensive test evaluation with classification and detection metrics"""
    model.eval()

    # Storage for predictions and targets
    all_clf_preds = []
    all_clf_targets = []
    all_sev_preds = []
    all_sev_targets = []

    test_loss_total = 0
    test_det_precision_total = 0
    test_det_recall_total = 0
    test_total = 0

    criterion_clf = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()

    with torch.no_grad():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[green]{task.percentage:>3.0f}%[/green]"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:

            test_task = progress.add_task("Testing", total=len(test_loader))

            for clips, clf_labels, frame_labels, sev_labels, gt_bboxes in test_loader:
                clips = clips.to(device)
                clf_labels = clf_labels.to(device)
                sev_labels = sev_labels.to(device)

                # Forward pass
                clf_pred, sev_pred, detections = model(clips, train=False)

                # Calculate losses
                loss_clf = criterion_clf(clf_pred, clf_labels)
                loss_sev = criterion_sev(sev_pred, sev_labels)
                loss_det = calculate_detection_loss(detections, gt_bboxes, device)
                total_loss = loss_clf + 0.3 * loss_sev + 0.5 * loss_det

                test_loss_total += total_loss.item()
                test_total += clf_labels.size(0)

                # Collect predictions for metrics
                all_clf_preds.extend(clf_pred.argmax(1).cpu().numpy())
                all_clf_targets.extend(clf_labels.cpu().numpy())
                all_sev_preds.extend(sev_pred.argmax(1).cpu().numpy())
                all_sev_targets.extend(sev_labels.cpu().numpy())

                # Convert gt_bboxes if they are tensors
                gt_bboxes_fixed = []
                for b in gt_bboxes:
                    if isinstance(b, torch.Tensor):
                        boxes = [
                            {"bbox": box[:4].tolist(), "class": int(box[4])}
                            for box in b
                            if box[4] != -1  # skip padding if needed
                        ]
                    else:
                        boxes = b
                    gt_bboxes_fixed.append(boxes)

                # Debug print: are predictions and GTs non-empty?
                for i, (preds, gts) in enumerate(zip(detections, gt_bboxes_fixed)):
                    print(f"[DEBUG] Sample {i}: {len(preds)} preds, {len(gts)} GTs")
                    if preds:
                        print(f"  First pred box: {preds[0]}")
                    if gts:
                        print(f"  First GT box: {gts[0]}")

                # Detection metrics
                batch_precision, batch_recall = calculate_detection_metrics(
                    detections, gt_bboxes
                )
                test_det_precision_total += batch_precision
                test_det_recall_total += batch_recall

                progress.update(test_task, advance=1)

    # Calculate classification metrics
    clf_acc = (
        accuracy_score(all_clf_targets, all_clf_preds)
        if len(all_clf_targets) > 0
        else 0
    )
    clf_precision, clf_recall, clf_f1, _ = precision_recall_fscore_support(
        all_clf_targets, all_clf_preds, average="weighted", zero_division=0
    )

    sev_acc = (
        accuracy_score(all_sev_targets, all_sev_preds)
        if len(all_sev_targets) > 0
        else 0
    )
    sev_precision, sev_recall, sev_f1, _ = precision_recall_fscore_support(
        all_sev_targets, all_sev_preds, average="weighted", zero_division=0
    )

    # Calculate detection metrics averages
    det_precision_avg = (
        test_det_precision_total / len(test_loader) if len(test_loader) > 0 else 0
    )
    det_recall_avg = (
        test_det_recall_total / len(test_loader) if len(test_loader) > 0 else 0
    )
    det_f1_avg = (
        2
        * (det_precision_avg * det_recall_avg)
        / (det_precision_avg + det_recall_avg + 1e-6)
    )

    # Display comprehensive results
    console.print("\n[bold green]🎯 Test Results[/bold green]")

    # Classification results table
    clf_table = Table(title="Classification Metrics")
    clf_table.add_column("Task", style="cyan")
    clf_table.add_column("Accuracy", justify="right", style="green")
    clf_table.add_column("Precision", justify="right", style="yellow")
    clf_table.add_column("Recall", justify="right", style="yellow")
    clf_table.add_column("F1-Score", justify="right", style="red")

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

    # Detection results table
    det_table = Table(title="Detection Metrics")
    det_table.add_column("Metric", style="cyan")
    det_table.add_column("Value", justify="right", style="green")

    det_table.add_row("Precision", f"{det_precision_avg:.4f}")
    det_table.add_row("Recall", f"{det_recall_avg:.4f}")
    det_table.add_row("F1-Score", f"{det_f1_avg:.4f}")

    console.print(det_table)

    # Calculate confusion matrix for bleeding classification
    if len(all_clf_targets) > 0:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(all_clf_targets, all_clf_preds)

        cm_table = Table(title="Confusion Matrix (Bleeding Classification)")
        cm_table.add_column("True\\Pred", style="cyan")
        cm_table.add_column("No Bleeding", justify="right")
        cm_table.add_column("Bleeding", justify="right")

        cm_table.add_row("No Bleeding", str(cm[0, 0]), str(cm[0, 1]))
        cm_table.add_row("Bleeding", str(cm[1, 0]), str(cm[1, 1]))

        console.print(cm_table)

    # Calculate average detection count
    avg_detections = calculate_average_detection_count(model, test_loader, device)
    console.print(f"\nAverage detections per bleeding clip: {avg_detections:.2f}")

    # Summary statistics
    summary_table = Table(title="Summary Statistics")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Test Loss", f"{test_loss_total / len(test_loader):.4f}")
    summary_table.add_row("Total Test Samples", str(test_total))
    summary_table.add_row(
        "Bleeding Samples", str(sum(1 for x in all_clf_targets if x == 1))
    )
    summary_table.add_row(
        "Non-Bleeding Samples", str(sum(1 for x in all_clf_targets if x == 0))
    )

    console.print(summary_table)

    # Return comprehensive metrics dictionary
    return {
        "test_loss": test_loss_total / len(test_loader) if len(test_loader) > 0 else 0,
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
        "detection": {
            "precision": det_precision_avg,
            "recall": det_recall_avg,
            "f1": det_f1_avg,
        },
        "avg_detections_per_clip": avg_detections,
        "confusion_matrix": cm.tolist() if len(all_clf_targets) > 0 else None,
        "total_samples": test_total,
        "bleeding_samples": sum(1 for x in all_clf_targets if x == 1),
        "non_bleeding_samples": sum(1 for x in all_clf_targets if x == 0),
    }


def calculate_average_detection_count(model, test_loader, device):
    """Calculate average number of detections per bleeding clip"""
    model.eval()
    total_detections = 0
    bleeding_clips = 0

    with torch.no_grad():
        for clips, clf_labels, frame_labels, sev_labels, gt_bboxes in test_loader:
            clips = clips.to(device)

            _, _, detections = model(clips)

            for i in range(clips.size(0)):
                if clf_labels[i] == 1:  # Only for bleeding clips
                    total_detections += len(detections[i])
                    bleeding_clips += 1

    return total_detections / bleeding_clips if bleeding_clips > 0 else 0


def visualize_bleeding_detections(
    model,
    test_loader,
    device,
    num_samples=30,
    path="./detection_visualizations",
    cleanup=True,
):
    """Visualize bleeding detection results with bounding boxes"""
    os.makedirs(path, exist_ok=True)

    if cleanup and os.path.exists(path):
        for f in os.listdir(path):
            p = os.path.join(path, f)
            if os.path.isfile(p) or os.path.islink(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        console.print(f"🧹 Cleaned existing visualizations in [bold]{path}[/bold]")

    model.eval()
    samples_found = 0
    skipped = 0

    def denormalize_frame(frame_tensor):
        """Denormalize a frame tensor for visualization - handles GPU tensors"""
        # Ensure tensor is on CPU
        frame = frame_tensor.cpu()

        # Standard ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Denormalize
        frame = frame * std + mean
        frame = torch.clamp(frame, 0, 1)

        # Convert to numpy
        frame = frame.permute(1, 2, 0).numpy()
        return (frame * 255).astype(np.uint8)

    def draw_bbox(ax, bbox, label, color, conf=None):
        """Draw bounding box on axes"""
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Add label
        text = f"{label}" + (f" ({conf:.2f})" if conf else "")
        ax.text(
            x1,
            y1 - 5,
            text,
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
        )

    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-6)

    with torch.no_grad():
        for batch_idx, (
            clips,
            clf_labels,
            frame_labels,
            sev_labels,
            gt_bboxes,
        ) in enumerate(test_loader):
            if samples_found >= num_samples:
                break

            clips = clips.to(device)
            clf_pred, sev_pred, detections = model(clips, train=False)

            # Process each sample in the batch
            for i in range(clips.size(0)):
                if samples_found >= num_samples:
                    break

                # Skip non-bleeding clips
                if clf_labels[i].item() == 0:
                    continue

                # Process ground truth boxes
                gt_boxes = []
                if isinstance(gt_bboxes, torch.Tensor):
                    # Handle tensor format from custom_collate_fn
                    for j in range(gt_bboxes.shape[1]):
                        if gt_bboxes[i, j, 4] >= 0:  # Valid box (class >= 0)
                            gt_boxes.append(
                                {
                                    "bbox": gt_bboxes[i, j, :4].tolist(),
                                    "class": int(gt_bboxes[i, j, 4].item()),
                                    "class_name": ["BL_Low", "BL_Medium", "BL_High"][
                                        int(gt_bboxes[i, j, 4].item())
                                    ],
                                }
                            )
                else:
                    # Handle list format
                    gt_boxes = gt_bboxes[i] if i < len(gt_bboxes) else []

                if len(gt_boxes) == 0:
                    skipped += 1
                    continue

                # Get middle frame
                middle_frame_idx = clips.size(2) // 2
                middle_frame = denormalize_frame(
                    clips[i, :, middle_frame_idx]
                )

                # Get predictions for this sample
                pred_boxes = detections[i] if i < len(detections) else []

                # Calculate metrics
                tp = 0
                for pred in pred_boxes:
                    for gt in gt_boxes:
                        if calculate_iou(pred["bbox"], gt["bbox"]) > 0.5:
                            tp += 1
                            break

                precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
                recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                # Create visualization
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # Original frame
                axs[0].imshow(middle_frame)
                axs[0].set_title("Original Frame")
                axs[0].axis("off")

                # Ground truth
                axs[1].imshow(middle_frame)
                for gt in gt_boxes:
                    draw_bbox(axs[1], gt["bbox"], gt["class_name"], "red")
                axs[1].set_title(f"Ground Truth ({len(gt_boxes)} boxes)")
                axs[1].axis("off")

                # Predictions
                axs[2].imshow(middle_frame)
                for pred in pred_boxes:
                    draw_bbox(
                        axs[2],
                        pred["bbox"],
                        pred["class_name"],
                        "blue",
                        pred["confidence"],
                    )
                axs[2].set_title(f"Predictions ({len(pred_boxes)} boxes)")
                axs[2].axis("off")

                # Add title with metrics
                severity_names = ["None", "Low", "Medium", "High"]
                fig.suptitle(
                    f"Batch {batch_idx}, Sample {i} | "
                    f"Severity: {severity_names[sev_labels[i].item()]} | "
                    f"P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}",
                    fontsize=12,
                )

                # Save figure
                filename = (
                    f"{samples_found:03d}_batch{batch_idx}_sample{i}_f1_{f1:.2f}.png"
                )
                filepath = os.path.join(path, filename)
                fig.savefig(filepath, dpi=150, bbox_inches="tight")
                plt.close(fig)

                console.print(
                    f"[green]✅ Sample {samples_found + 1}[/green] - "
                    f"F1: {f1:.2f}, P: {precision:.2f}, R: {recall:.2f} → {filename}"
                )
                samples_found += 1

    # Summary
    console.print(f"\n📊 [bold]Visualization Summary[/bold]")
    console.print(f"  ✅ Visualized: {samples_found}")
    console.print(f"  ⚠️  Skipped (no GT boxes): {skipped}")
    console.print(f"  💾 Saved to: {os.path.abspath(path)}")

    if samples_found == 0:
        console.print(
            "[yellow]⚠️  No bleeding samples with ground truth boxes found![/yellow]"
        )


def load_and_plot_metrics(json_file_path, save_dir="./models/plots"):
    """Load and plot training metrics"""
    os.makedirs(save_dir, exist_ok=True)

    with open(json_file_path, "r") as f:
        metrics = json.load(f)

    epochs = list(range(1, len(metrics["train_loss"]) + 1))

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Loss
    ax1.plot(epochs, metrics["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, metrics["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_title("Training and Validation Loss", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Classification Accuracy
    ax2.plot(epochs, metrics["train_clf_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, metrics["val_clf_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_title("Classification Accuracy", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Severity Accuracy
    ax3.plot(epochs, metrics["train_sev_acc"], "b-", label="Train Sev", linewidth=2)
    ax3.plot(epochs, metrics["val_sev_acc"], "r-", label="Val Sev", linewidth=2)
    ax3.set_title("Severity Classification Accuracy", fontsize=14)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Detection Metrics
    ax4.plot(
        epochs,
        metrics["train_det_precision"],
        "g-",
        label="Train Precision",
        linewidth=2,
    )
    ax4.plot(
        epochs, metrics["train_det_recall"], "g--", label="Train Recall", linewidth=2
    )
    ax4.plot(
        epochs, metrics["val_det_precision"], "m-", label="Val Precision", linewidth=2
    )
    ax4.plot(epochs, metrics["val_det_recall"], "m--", label="Val Recall", linewidth=2)
    ax4.set_title("Detection Metrics", fontsize=14)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Training Metrics Overview", fontsize=16)
    plt.tight_layout()

    # Save plots
    plt.savefig(
        os.path.join(save_dir, "training_metrics.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(os.path.join(save_dir, "training_metrics.pdf"), bbox_inches="tight")
    plt.close()

    # Print summary
    console.print("\n[bold green]Training Summary:[/bold green]")
    console.print(f"Final Train Loss: {metrics['train_loss'][-1]:.4f}")
    console.print(f"Final Val Loss: {metrics['val_loss'][-1]:.4f}")
    console.print(f"Best Val Accuracy: {max(metrics['val_clf_acc']):.2f}%")
    console.print(
        f"Final Val Detection F1: {2 * metrics['val_det_precision'][-1] * metrics['val_det_recall'][-1] / (metrics['val_det_precision'][-1] + metrics['val_det_recall'][-1] + 1e-6):.4f}"
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bleeding Detection Training')

    # Training arguments
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize detection results')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='bleeding_detector', help='Model name for saving')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for testing')

    # Data arguments
    parser.add_argument('--video_dir', type=str, default=VIDEO_DIR, help='Directory containing videos')
    parser.add_argument('--anno_dir', type=str, default=ANNO_DIR, help='Directory containing annotations')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--clip_length', type=int, default=6, help='Number of frames per clip')
    parser.add_argument('--stride', type=int, default=3, help='Stride for clip sampling')

    # System arguments
    parser.add_argument('--dev', type=int, default=0, help='CUDA device number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    return parser.parse_args()


def main():
    """Main function to run the training/testing pipeline"""

    # Parse arguments
    args = parse_args()
    device_str = f"cuda:{args.dev}" if torch.cuda.is_available() else "cpu"

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = None

    # Training mode
    if args.train:
        console.print("\n[bold green]Starting Training...[/bold green]")

        model, metrics, test_metrics = train_detector(
            video_dir=VIDEO_DIR,
            annotation_dir=ANNO_DIR,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip_length=args.clip_length,
            stride=args.stride,
            learning_rate=args.lr,
            device=device_str,
            model_name=args.model_name,
            test=args.test
        )

        if model is not None:
            # Plot metrics
            metrics_path = os.path.join(args.output_dir, f"{args.model_name}_metrics.json")
            if os.path.exists(metrics_path):
                console.print("\n[bold blue]Plotting training metrics...[/bold blue]")
                load_and_plot_metrics(metrics_path, save_dir=args.output_dir)

            # Save test metrics
            if test_metrics:
                test_metrics_path = os.path.join(args.output_dir, f"{args.model_name}_test_metrics.json")
                with open(test_metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                console.print(f"[green]Test metrics saved to: {test_metrics_path}[/green]")

    # Testing/Visualization mode
    elif args.test or args.visualize:
        if not args.checkpoint:
            # Try to find the best model
            best_model_path = os.path.join(args.output_dir, f"{args.model_name}_best.pth")
            if os.path.exists(best_model_path):
                args.checkpoint = best_model_path
            else:
                console.print("[red]Error: No checkpoint specified and no best model found![/red]")
                return

        console.print(f"\n[bold green]Loading model from: {args.checkpoint}[/bold green]")

        # Initialize model
        model = BleedingDetector(num_classes=2, input_size=(328, 512)).to(device_str)

        # Load weights
        checkpoint = torch.load(args.checkpoint, map_location=device_str)
        model.load_state_dict(checkpoint)
        model.eval()

        # Create test dataset
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((328, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Get all videos and annotations
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        matched_pairs = []

        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            xml_file = f"{video_name}.xml"
            xml_path = os.path.join(args.anno_dir, xml_file)

            if os.path.exists(xml_path):
                video_path = os.path.join(args.video_dir, video_file)
                matched_pairs.append((video_path, xml_path))

        # Use last 20% for testing
        test_size = max(1, int(0.2 * len(matched_pairs)))
        test_pairs = matched_pairs[-test_size:]

        test_videos, test_annotations = zip(*test_pairs) if test_pairs else ([], [])

        test_dataset = SurgicalVideoDataset(
            list(test_videos),
            list(test_annotations),
            clip_length=args.clip_length,
            stride=args.stride,
            transform=transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn
        )

        if args.test:
            console.print("\n[bold blue]Running evaluation...[/bold blue]")
            test_metrics = evaluate_model(model, test_loader, device_str)

            # Save test results
            test_results_path = os.path.join(args.output_dir, f"{args.model_name}_test_results.json")
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            console.print(f"[green]Test results saved to: {test_results_path}[/green]")

        if args.visualize:
            console.print("\n[bold blue]Creating visualizations...[/bold blue]")
            viz_path = os.path.join(args.output_dir, "detection_visualizations")
            visualize_bleeding_detections(
                model, test_loader, device_str,
                num_samples=30,
                path=viz_path,
                cleanup=True
            )

    else:
        console.print("[yellow]No action specified! Use --train, --test, or --visualize[/yellow]")
        return

    console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")


if __name__ == "__main__":
    main()

