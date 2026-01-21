import argparse
import os

tmp_dir = '/home/r.rohangirish/mt_ble/tmp'
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

def predict_video(model, video_path, output_path=None, clip_length=6, stride=3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    model = model.to(device)

    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Read video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize arrays for predictions
    clip_predictions = []

    print("Predicting Video...")

    # Process in clips
    with torch.no_grad():
        for start_idx in range(0, frame_count - clip_length + 1, stride):
            end_idx = start_idx + clip_length

            # Extract frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            frames = []
            for _ in range(clip_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            if len(frames) < clip_length:
                continue

            processed_frames = [transform(frame) for frame in frames]
            clip_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0)
            clip_tensor = clip_tensor.to(device)

            # Get predictions
            clip_pred, severity_pred = model(clip_tensor)

            # Convert to probabilities
            clip_prob = F.softmax(clip_pred, dim=1).cpu().numpy()[0, 1]  # Probability for bleeding
            severity_probs = F.softmax(severity_pred, dim=1).cpu().numpy()[0]  # Severity probabilities

            # Store clip predictions with frame ranges
            clip_predictions.append({
                'start_frame': start_idx,
                'end_frame': end_idx - 1,
                'bleeding_prob': clip_prob,
                'severity_probs': severity_probs,
                'predicted_severity': np.argmax(severity_probs)
            })

    print("Prediction complete!")

    # Create frame-level results by assigning clip predictions to frames
    # the idea here is to average the predictions of all clips that contain a given frame, this gives us some information
    # about the bleeding status of that frame. i think..

    frame_results = []
    for frame_idx in range(frame_count):
        # Find all clips that contain this frame
        containing_clips = [clip for clip in clip_predictions
                          if clip['start_frame'] <= frame_idx <= clip['end_frame']]

        if containing_clips:
            # Average predictions from all clips containing this frame
            avg_bleeding_prob = np.mean([clip['bleeding_prob'] for clip in containing_clips])

            # For severity, take the weighted average of severity probabilities
            all_severity_probs = np.array([clip['severity_probs'] for clip in containing_clips])
            avg_severity_probs = np.mean(all_severity_probs, axis=0)
            predicted_severity = np.argmax(avg_severity_probs)

            frame_results.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'bleeding_prob': avg_bleeding_prob,
                'has_bleeding': avg_bleeding_prob > 0.5,
                'predicted_severity': predicted_severity,
                'severity_0_prob': avg_severity_probs[0],
                'severity_1_prob': avg_severity_probs[1],
                'severity_2_prob': avg_severity_probs[2],
                'severity_3_prob': avg_severity_probs[3]
            })
        else:
            # No predictions for this frame
            frame_results.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'bleeding_prob': 0.0,
                'has_bleeding': False,
                'predicted_severity': 0,
                'severity_0_prob': 1.0,
                'severity_1_prob': 0.0,
                'severity_2_prob': 0.0,
                'severity_3_prob': 0.0
            })

    # Convert to DataFrame
    results = pd.DataFrame(frame_results)

    if output_path:
        create_visualization(video_path, results, clip_predictions, output_path)

    cap.release()
    return results, clip_predictions


def create_visualization(video_path, predictions, clip_predictions, output_path, clip_length=6, stride=3):
    """
    Create a visualization showing:
    1. Entire clips (6 frames) with bleeding colored distinctly
    2. Frame-level averaged predictions as text
    3. Severity information for each clip
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define severity colors
    severity_colors = {
        0: (0, 255, 0),     # None - green
        1: (0, 255, 255),   # Low - yellow
        2: (0, 165, 255),   # Medium - orange
        3: (0, 0, 255)      # High - red
    }

    severity_labels = {
        0: "None",
        1: "Low",
        2: "Medium",
        3: "High"
    }

    print("Creating Visualization..")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame-level prediction (from averaging)
        if frame_idx < len(predictions):
            # Use 'frame' column instead of 'frame_number'
            row = predictions[predictions['frame'] == frame_idx].iloc[0]
            frame_bleeding_prob = row['bleeding_prob']
            frame_has_bleeding = row['has_bleeding']
            frame_severity = row['predicted_severity']

            # Find clips containing this frame
            containing_clips = []
            for clip_idx, clip in enumerate(clip_predictions):
                if clip['start_frame'] <= frame_idx <= clip['end_frame']:
                    containing_clips.append((clip_idx, clip))

            # STEP 1: Color the entire frame based on clips that have bleeding
            # Create a copy of the frame for overlays
            overlay = frame.copy()

            # Flag to track if any clip has bleeding
            any_clip_has_bleeding = False
            max_clip_severity = 0

            # Process all clips containing this frame
            for clip_idx, clip in enumerate(clip_predictions):
                if clip['start_frame'] <= frame_idx <= clip['end_frame']:
                    # Check if this clip has bleeding
                    clip_has_bleeding = clip['bleeding_prob'] > 0.5

                    if clip_has_bleeding:
                        any_clip_has_bleeding = True
                        clip_severity = clip['predicted_severity']
                        max_clip_severity = max(max_clip_severity, clip_severity)

            # Add a colored overlay if any clip has bleeding
            if any_clip_has_bleeding:
                # Create full-frame overlay with the appropriate color
                cv2.rectangle(overlay, (0, 0), (width, height),
                             severity_colors[max_clip_severity], -1)

                # Blend the overlay with the original frame
                alpha = 0.3  # Fixed transparency
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Add a colored border
                cv2.rectangle(frame, (0, 0), (width, height),
                             severity_colors[max_clip_severity], 5)

            # STEP 2: Add text information for frame-level predictions
            # Frame info
            frame_text = f"Frame {frame_idx}: Bleeding Prob = {frame_bleeding_prob:.2f}"
            cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            # Frame severity
            if frame_has_bleeding:
                severity_text = f"Frame Severity: {severity_labels[frame_severity]}"
                cv2.putText(frame, severity_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, severity_colors[frame_severity], 2)

            # STEP 3: Add information about individual clips
            y_offset = 100
            if containing_clips:
                cv2.putText(frame, "Clips containing this frame:", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30

                for clip_idx, clip in containing_clips:
                    clip_bleeding = clip['bleeding_prob'] > 0.5
                    clip_severity = clip['predicted_severity']

                    # Create clip status text
                    status = "BLEEDING" if clip_bleeding else "No bleeding"
                    color = severity_colors[clip_severity] if clip_bleeding else (200, 200, 200)

                    clip_text = f"Clip {clip_idx} ({clip['start_frame']}-{clip['end_frame']}): "
                    clip_text += f"{status}, Prob={clip['bleeding_prob']:.2f}"

                    if clip_bleeding:
                        clip_text += f", Severity={severity_labels[clip_severity]}"

                    cv2.putText(frame, clip_text, (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25

        # Write frame to output video
        out.write(frame)
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Visualization saved to {output_path}")

def evaluate_clip_predictions(clip_predictions, ground_truth_csv, clip_length=6):
    """
    Evaluates the accuracy of clip-level predictions.

    Args:
        clip_predictions: List of dictionaries with clip predictions
        ground_truth_csv: Path to CSV with ground truth annotations
        clip_length: Length of each clip in frames

    Returns:
        Dictionary with evaluation metrics
    """
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_csv)

    # Create frame-by-frame ground truth array
    # First get the maximum frame from clip predictions
    max_frame = max([clip['end_frame'] for clip in clip_predictions])
    gt_bleeding = np.zeros(max_frame + 1, dtype=bool)
    gt_severity = np.zeros(max_frame + 1, dtype=int)

    # Fill in the bleeding frames from ground truth
    for _, row in gt_df.iterrows():
        start = int(row['start_frame'])
        end = int(row['end_frame'])

        # Get severity from label
        severity = 0
        if 'BL_Low' in row['label']:
            severity = 1
        elif 'BL_Medium' in row['label']:
            severity = 2
        elif 'BL_High' in row['label']:
            severity = 3

        gt_bleeding[start:end+1] = True
        gt_severity[start:end+1] = severity

    # Evaluate each clip
    clip_true_labels = []
    clip_pred_labels = []
    bleeding_true_labels = []
    bleeding_pred_labels = []

    for clip in clip_predictions:
        start = clip['start_frame']
        end = clip['end_frame']

        # Check if at least half of the frames in the clip have bleeding
        clip_frames = np.arange(start, end+1)
        clip_gt_bleeding = gt_bleeding[clip_frames]
        clip_has_bleeding = np.sum(clip_gt_bleeding) >= clip_length / 2

        # Predicted label
        clip_pred_bleeding = clip['bleeding_prob'] > 0.5

        # Store true and predicted labels
        clip_true_labels.append(clip_has_bleeding)
        clip_pred_labels.append(clip_pred_bleeding)

        # For bleeding-only accuracy, only include clips that have bleeding in GT
        if clip_has_bleeding:
            bleeding_true_labels.append(clip_has_bleeding)
            bleeding_pred_labels.append(clip_pred_bleeding)

    # Calculate metrics
    clip_accuracy = accuracy_score(clip_true_labels, clip_pred_labels)

    # Calculate bleeding-only accuracy if there are any bleeding clips
    if len(bleeding_true_labels) > 0:
        bleeding_accuracy = accuracy_score(bleeding_true_labels, bleeding_pred_labels)
    else:
        bleeding_accuracy = None

    # Count true positives, false positives, etc.
    tp = sum(1 for t, p in zip(clip_true_labels, clip_pred_labels) if t and p)
    fp = sum(1 for t, p in zip(clip_true_labels, clip_pred_labels) if not t and p)
    fn = sum(1 for t, p in zip(clip_true_labels, clip_pred_labels) if t and not p)
    tn = sum(1 for t, p in zip(clip_true_labels, clip_pred_labels) if not t and not p)

    # Print results in a simple format
    print("\n===== CLIP-LEVEL EVALUATION =====")
    print(f"Total clips evaluated: {len(clip_true_labels)}")
    print(f"Ground truth positive clips: {sum(clip_true_labels)}")
    print(f"Model predicted positive clips: {sum(clip_pred_labels)}")
    print(f"Clip-level accuracy: {clip_accuracy:.4f}")

    if bleeding_accuracy is not None:
        print(f"Bleeding-only accuracy: {bleeding_accuracy:.4f}")
    else:
        print("Bleeding-only accuracy: N/A (no bleeding clips in ground truth)")

    print("\nConfusion Matrix:")
    print(f"True Positive: {tp}, False Positive: {fp}")
    print(f"False Negative: {fn}, True Negative: {tn}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.4f}")
    else:
        print("Precision: N/A")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.4f}")
    else:
        print("Recall: N/A")

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"F1 Score: {f1:.4f}")
    else:
        print("F1 Score: N/A")

    return {
        'clip_accuracy': clip_accuracy,
        'bleeding_accuracy': bleeding_accuracy,
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'true_negative': tn
    }


def visualize_bleeding_detection(model, video_path, output_path, clip_length=6, stride=3,
                                show_only_bleeding=True, bleeding_threshold=0.5, dev="cuda:5"):
    """
    Visualize model attention using GradCAM.
    """

    device = torch.device(dev)
    model = model.to(device)
    print(f"Using device: {device}")
    print("Model device:", next(model.parameters()).device)
    model.eval()

    # Model wrapper for GradCAM
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, device):
            super().__init__()
            self.model = model.to(device)  # Force model to this device
            self.device = device

        def forward(self, x):
            # Force input to correct device
            x = x.to(self.device)
            out = self.model(x)
            if isinstance(out, tuple):
                return out[0]  # Return classification output
            return out

    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            target_layers.append(module)

    target_layers = [target_layers[-1]]  # Use last Conv3D layer

    wrapped_model = ModelWrapper(model, device)
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    print(cam.device)
    # Setup for video processing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Read and setup video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

    print("Creating GradCAM visualization...")

    # Store information about which frames have been visualized
    visualized_frames = set()
    bleeding_count = 0
    total_count = 0

    # Process clips
    for start_idx in range(0, frame_count - clip_length + 1, stride):
        # Extract frames for this clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        original_frames = []

        for _ in range(clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            original_frames.append(frame.copy())
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if len(frames) < clip_length:
            continue

        # Process frames
        processed_frames = [transform(frame) for frame in frames]
        clip_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0)
        clip_tensor = clip_tensor.to(device)

        # Get model prediction first to filter non-bleeding clips
        with torch.no_grad():
            clip_pred, severity_pred = model(clip_tensor)
            bleeding_prob = torch.softmax(clip_pred, dim=1)[0, 1].item()
            severity = torch.argmax(severity_pred, dim=1).item()

        total_count += 1

        # show only bleeding clips
        if show_only_bleeding and bleeding_prob < bleeding_threshold:
            continue

        bleeding_count += 1

        # Only get bleeding
        targets = [ClassifierOutputTarget(1)]

        # This is where gradcam is computed
        grayscale_cam = cam(input_tensor=clip_tensor, targets=targets)

        # Visualize each frame
        for i, (frame, cam_map) in enumerate(zip(original_frames, grayscale_cam[0])):
            frame_idx = start_idx + i
            vis_frame = frame.copy()

            # Resize CAM to match original frame size
            cam_map_resized = cv2.resize(cam_map, (frame.shape[1], frame.shape[0]))
            frame_norm = vis_frame / 255.0

            # Create heatmap overlay
            heatmap_overlay = show_cam_on_image(frame_norm, cam_map_resized, use_rgb=True)
            heatmap_overlay = (heatmap_overlay * 255).astype(np.uint8)
            thickness = 4 if i == 0 or i == clip_length-1 else 1

            cv2.rectangle(vis_frame, (0, 0), (width, height), (0, 0, 255), thickness)
            cv2.rectangle(heatmap_overlay, (0, 0), (width, height), (0, 0, 255), thickness)

            info_text = f"Clip: {start_idx}-{start_idx+clip_length-1} | Frame: {frame_idx}"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            pred_text = f"Bleeding: {bleeding_prob:.2f} | Severity: {severity}"
            cv2.putText(vis_frame, pred_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Note that this is a clip prediction applied to all frames
            cv2.putText(vis_frame, "CLIP-LEVEL PREDICTION", (10, height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            comparison = np.hstack((vis_frame, heatmap_overlay))
            out.write(comparison)
            visualized_frames.add(frame_idx)

    # Release resources
    cap.release()
    out.release()

    print(f"GradCAM visualization saved to {output_path}")
    print(f"Displayed {bleeding_count}/{total_count} clips with bleeding probability ≥ {bleeding_threshold}")
    print(f"Visualized {len(visualized_frames)} frames out of {frame_count} total frames")


def apply_layercam(model, video_path, ground_truth_csv, output_path,clip_length=6, stride=3, samples_per_class=400,
                                          dev="cuda:5"):
    """
    Apply LayerCAM to selected clips from each class (bleeding and non-bleeding)
    based on ground truth annotations

    Args:
        model: Trained 3D CNN model
        video_path: Path to input video
        ground_truth_csv: Path to CSV with ground truth annotations
        output_path: Path to save output visualization
        clip_length: Length of clip in frames
        stride: Stride between consecutive clips
        samples_per_class: Number of frames to sample from each class
    """
    device = torch.device(dev)
    model = model.to(device)
    model.eval()

    # Load ground truth annotations
    print("Loading ground truth annotations...")
    gt_df = pd.read_csv(ground_truth_csv)

    # Create frame-by-frame ground truth array
    # Read video to get frame count
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create ground truth array (1 for bleeding, 0 for non-bleeding)
    gt_bleeding = np.zeros(frame_count, dtype=bool)

    # Fill in the bleeding frames from ground truth
    for _, row in gt_df.iterrows():
        start = int(row['start_frame'])
        end = int(row['end_frame'])
        gt_bleeding[start:end + 1] = True

    # Find bleeding and non-bleeding clips
    bleeding_clips = []
    non_bleeding_clips = []

    for start_idx in range(0, frame_count - clip_length + 1, stride):
        end_idx = start_idx + clip_length - 1
        clip_frames = np.arange(start_idx, end_idx + 1)

        # Check if majority of frames in this clip have bleeding
        if np.mean(gt_bleeding[clip_frames]) >= 0.5:
            bleeding_clips.append((start_idx, end_idx))
        else:
            non_bleeding_clips.append((start_idx, end_idx))

    print(f"Found {len(bleeding_clips)} bleeding clips and {len(non_bleeding_clips)} non-bleeding clips")

    # Sample clips from each class
    import random
    random.seed(42)  # For reproducibility

    # Calculate how many clips to sample
    clips_per_class = max(1, samples_per_class // clip_length)

    if len(bleeding_clips) > clips_per_class:
        sampled_bleeding_clips = random.sample(bleeding_clips, clips_per_class)
    else:
        sampled_bleeding_clips = bleeding_clips

    if len(non_bleeding_clips) > clips_per_class:
        sampled_non_bleeding_clips = random.sample(non_bleeding_clips, clips_per_class)
    else:
        sampled_non_bleeding_clips = non_bleeding_clips

    print(f"Sampled {len(sampled_bleeding_clips)} bleeding clips and {len(sampled_non_bleeding_clips)} non-bleeding clips")

    # Combine sampled clips
    sampled_clips = [(start, end, True) for start, end in sampled_bleeding_clips]
    sampled_clips.extend([(start, end, False) for start, end in sampled_non_bleeding_clips])

    # Sort by start frame
    sampled_clips.sort(key=lambda x: x[0])

    # Define a wrapper model for the classifier output
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                return outputs[0]  # Return only classification output
            return outputs

    # Wrap the model
    wrapped_model = ModelWrapper(model)

    # Find target layers at different depths
    # This is model-specific - adjust for your architecture
    target_layers = []
    layer_names = []

    # Find layers based on model architecture
    if hasattr(model, 'backbone'):
        # Early layer (after first few convolutions)
        if hasattr(model.backbone, 'layer1') and len(model.backbone.layer1) > 0:
            target_layers.append(model.backbone.layer1[-1])
            layer_names.append("Early Layer")

        # Middle layer
        if hasattr(model.backbone, 'layer3') and len(model.backbone.layer3) > 0:
            target_layers.append(model.backbone.layer3[-1])
            layer_names.append("Middle Layer")

        # Deep layer (final conv layer)
        if hasattr(model.backbone, 'layer4') and len(model.backbone.layer4) > 0:
            target_layers.append(model.backbone.layer4[-1])
            layer_names.append("Deep Layer")
    else:
        # Fallback for non-ResNet architectures
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                conv_layers.append((name, module))

        if len(conv_layers) >= 3:
            early_idx = len(conv_layers) // 5  # ~20% through the network
            mid_idx = len(conv_layers) // 2    # Middle of the network
            deep_idx = -1                      # Last layer

            target_layers.append(conv_layers[early_idx][1])
            layer_names.append(f"Early Layer ({conv_layers[early_idx][0]})")

            target_layers.append(conv_layers[mid_idx][1])
            layer_names.append(f"Middle Layer ({conv_layers[mid_idx][0]})")

            target_layers.append(conv_layers[deep_idx][1])
            layer_names.append(f"Deep Layer ({conv_layers[deep_idx][0]})")

    if not target_layers:
        raise ValueError("Could not find suitable target layers!")

    # Set up video processing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set up output videos
    # Separate videos for bleeding and non-bleeding
    bleeding_output = output_path.replace('.mp4', '_bleeding.mp4')
    non_bleeding_output = output_path.replace('.mp4', '_non_bleeding.mp4')

    bleeding_writer = cv2.VideoWriter(
        bleeding_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width * len(target_layers), height)
    )

    non_bleeding_writer = cv2.VideoWriter(
        non_bleeding_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width * len(target_layers), height)
    )

    print(f"Applying LayerCAM to {len(sampled_clips)} sampled clips...")

    # Process sampled clips
    cap = cv2.VideoCapture(video_path)
    total_frames_processed = 0

    for clip_idx, (start_idx, end_idx, is_bleeding) in enumerate(sampled_clips):
        print(f"Processing clip {clip_idx+1}/{len(sampled_clips)}: frames {start_idx}-{end_idx} "
              f"({'Bleeding' if is_bleeding else 'Non-bleeding'})")

        # Extract clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        original_frames = []

        for _ in range(clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            original_frames.append(frame.copy())
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if len(frames) < clip_length:
            continue

        # Update total frames counter
        total_frames_processed += len(frames)

        # Process frames - create processed tensors
        processed_tensors = [transform(frame) for frame in frames]
        clip_tensor = torch.stack(processed_tensors).permute(1, 0, 2, 3).unsqueeze(0)
        clip_tensor = clip_tensor.to(device)

        # Get model prediction
        with torch.enable_grad():
            clip_tensor.requires_grad = True
            outputs = wrapped_model(clip_tensor)
            bleeding_prob = torch.softmax(outputs, dim=1)[0, 1].item()

        # Target the bleeding class (always check for bleeding features)
        targets = [ClassifierOutputTarget(1)]

        # Apply LayerCAM to each target layer
        layer_cams = []
        for i, target_layer in enumerate(target_layers):
            # Initialize LayerCAM for this layer
            cam = LayerCAM(
                model=wrapped_model,
                target_layers=[target_layer]
            )

            # Generate LayerCAM
            grayscale_cam = cam(input_tensor=clip_tensor, targets=targets)
            layer_cams.append(grayscale_cam)

        # Visualize each frame with LayerCAM from different layers
        for frame_idx, original_frame in enumerate(original_frames):
            frame_rgb = original_frame.copy()
            combined_frame = np.zeros((height, width * len(target_layers), 3), dtype=np.uint8)

            for i, layer_cam in enumerate(layer_cams):
                # Get CAM for this frame
                cam_frame = layer_cam[0, frame_idx]

                # Resize to match original frame size
                cam_resized = cv2.resize(cam_frame, (width, height))

                # Create heatmap overlay
                frame_norm = frame_rgb / 255.0
                heatmap = show_cam_on_image(frame_norm, cam_resized, use_rgb=True)
                heatmap = np.uint8(heatmap * 255)

                # Add text
                gt_text = "GT: Bleeding" if is_bleeding else "GT: No Bleeding"
                cv2.putText(heatmap, gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)

                # Add layer info
                cv2.putText(heatmap, layer_names[i], (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)

                # Add prediction info
                pred_text = f"Pred: Bleeding ({bleeding_prob:.2f})" if bleeding_prob > 0.5 else f"Pred: No Bleeding ({bleeding_prob:.2f})"
                cv2.putText(heatmap, pred_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)

                # Add frame info
                frame_text = f"Frame: {start_idx + frame_idx}"
                cv2.putText(heatmap, frame_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)

                # Add to combined frame
                combined_frame[:, i*width:(i+1)*width] = heatmap

            # Write to appropriate output video
            if is_bleeding:
                bleeding_writer.write(combined_frame)
            else:
                non_bleeding_writer.write(combined_frame)

    # Release resources
    cap.release()
    bleeding_writer.release()
    non_bleeding_writer.release()

    print(f"Processed a total of {total_frames_processed} frames")
    print(f"LayerCAM visualization complete.")
    print(f"Bleeding clips saved to: {bleeding_output}")
    print(f"Non-bleeding clips saved to: {non_bleeding_output}")

    return bleeding_output, non_bleeding_output