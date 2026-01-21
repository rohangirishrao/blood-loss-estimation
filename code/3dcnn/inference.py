#!/usr/bin/env python3
"""
Usage:
    python3 inference.py --model path/to/model.pth --videos VIDEO_FOLDER --output path/to/results.json
"""

import torch
import cv2
import numpy as np
import pandas as pd
import os
import glob
import json
import argparse
from torchvision import transforms
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Import models from train_BLE
from models import VolumeSequenceModel, MultiTaskVolumeSequenceModel

console = Console()


def infer_bloodloss(
    model_path,
    video_folder,
    output_json="./inference_results.json",
    volume_csv=None,
    clip_length=24,
    clips_per_sequence=6,
    input_size=(224, 224),
    device=None,
):
    """
    Infer blood loss for all videos in a folder WITHOUT needing XML annotations.

    Args:
        model_path: Path to trained .pth model
        video_folder: Folder containing .mp4 videos
        output_json: Where to save results
        volume_csv: Optional CSV with ground truth
        clip_length: Frames per clip (must match training)
        clips_per_sequence: Clips per sequence (must match training)
        input_size: (H, W) for frames (must match training)
        device: torch device
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    console.print(
        Panel.fit(
            f"[bold cyan]Blood Loss Inference[/bold cyan]\n"
            f"Model: {model_path}\n"
            f"Videos: {video_folder}\n"
            f"Device: {device}",
            title="Configuration",
        )
    )

    # Load model
    if not os.path.exists(model_path):
        console.print(f"[red]Error: Model not found at {model_path}[/red]")
        return None

    try:
        checkpoint = torch.load(model_path, map_location=device)
        # is_multitask = any('clip_classifier' in key for key in checkpoint.keys())
        is_multitask = True
        if is_multitask:
            model = MultiTaskVolumeSequenceModel(
                num_classes=2, hidden_dim=256, dropout=0.5
            )
            console.print("[green]Multi-Task Model loaded[/green]")
        else:
            model = VolumeSequenceModel(hidden_dim=256, dropout=0.3)
            console.print("[green]Single-Task Model loaded[/green]")

        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return None

    # Load ground truth if available
    volume_data = {}
    if volume_csv and os.path.exists(volume_csv):
        df = pd.read_csv(volume_csv)
        df = df.dropna(subset=["video_name"])
        for video_name in df["video_name"].unique():
            video_rows = df[df["video_name"] == video_name]
            total_volume_row = video_rows[video_rows["a_e_bl"].notna()]
            if not total_volume_row.empty:
                volume_data[str(video_name)] = float(total_volume_row.iloc[0]["a_e_bl"])
        console.print(
            f"[yellow]Loaded ground truth for {len(volume_data)} videos[/yellow]"
        )

    # Find videos
    video_paths = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    console.print(f"[cyan]Found {len(video_paths)} videos[/cyan]\n")

    if not video_paths:
        console.print("[red]No .mp4 videos found![/red]")
        return None

    # Transform
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )

    results = {
        "model_path": model_path,
        "config": {
            "clip_length": clip_length,
            "clips_per_sequence": clips_per_sequence,
            "input_size": input_size,
        },
        "videos": {},
    }

    # Process videos
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing videos", total=len(video_paths))

        for video_path in video_paths:
            video_id = os.path.basename(video_path).split(".")[0]
            console.print(f"[bold cyan]{video_id}[/bold cyan]")

            try:
                # Get video info
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    console.print(f"  [red]Cannot open video[/red]")
                    progress.update(task, advance=1)
                    continue

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # Calculate sequences
                frames_per_sequence = clip_length * clips_per_sequence
                num_sequences = max(1, frame_count // frames_per_sequence)

                console.print(
                    f"  Frames: {frame_count}, Sequences: {num_sequences} (frames/seq: {frames_per_sequence})"
                )

                sequence_predictions = []

                # Process each sequence
                for seq_idx in range(num_sequences):
                    start_frame = seq_idx * frames_per_sequence

                    # Load sequence
                    clip_tensors = []
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    for clip_idx in range(clips_per_sequence):
                        frames = []
                        for _ in range(clip_length):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)

                        # Pad if necessary
                        if len(frames) > 0 and len(frames) < clip_length:
                            frames.extend([frames[-1]] * (clip_length - len(frames)))

                        if len(frames) == clip_length:
                            frame_tensors = [transform(f) for f in frames]
                            clip_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3)
                            clip_tensors.append(clip_tensor)

                    cap.release()

                    # Pad clips if needed
                    if len(clip_tensors) > 0 and len(clip_tensors) < clips_per_sequence:
                        while len(clip_tensors) < clips_per_sequence:
                            clip_tensors.append(clip_tensors[-1])

                    if len(clip_tensors) == clips_per_sequence:
                        # Run inference
                        sequence_tensor = (
                            torch.stack(clip_tensors).unsqueeze(0).to(device)
                        )

                        with torch.no_grad():
                            model_output = model(sequence_tensor)

                            # Extract volume
                            if isinstance(model_output, tuple):
                                _, volume_pred, _ = model_output
                            else:
                                volume_pred = model_output

                            sequence_predictions.append(float(volume_pred.item()))

                # Calculate total - match training code exactly
                if sequence_predictions:
                    predicted_cumulative = np.cumsum(sequence_predictions)
                    total_predicted = float(predicted_cumulative[-1])
                else:
                    total_predicted = 0.0

                ground_truth = volume_data.get(video_id, None)

                error = None
                rel_error = None
                if ground_truth is not None:
                    error = abs(total_predicted - ground_truth)
                    rel_error = error / max(ground_truth, 1e-6)

                # Store results
                results["videos"][video_id] = {
                    "predicted_ml": total_predicted,
                    "ground_truth_ml": ground_truth,
                    "error_ml": error,
                    "error_percent": rel_error * 100 if rel_error else None,
                    "num_sequences": len(sequence_predictions),
                    # "sequence_predictions": sequence_predictions,  # Add this for debugging
                }

                # Print
                console.print(f"  Sequences: {len(sequence_predictions)}")
                # console.print(f"  Per-seq: {[f'{p:.1f}' for p in sequence_predictions]}")
                console.print(f"  Predicted: [green]{total_predicted:.2f} ml[/green]")
                if ground_truth:
                    console.print(f"  Truth: [yellow]{ground_truth:.2f} ml[/yellow]")
                    if rel_error is not None:
                        console.print(
                            f"  Error: [red]{error:.2f} ml ({rel_error * 100:.1f}%)[/red]"
                        )

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

            progress.update(task, advance=1)

    # Calculate stats
    videos_with_gt = {
        k: v for k, v in results["videos"].items() if v["ground_truth_ml"] is not None
    }

    if videos_with_gt:
        errors = [v["error_ml"] for v in videos_with_gt.values()]
        results["summary"] = {
            "total_videos": len(results["videos"]),
            "videos_with_ground_truth": len(videos_with_gt),
            "mean_error_ml": float(np.mean(errors)),
            "std_error_ml": float(np.std(errors)),
        }

    # Save JSON
    os.makedirs(
        os.path.dirname(output_json) if os.path.dirname(output_json) else ".",
        exist_ok=True,
    )
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    console.print(
        f"\n[bold green]✅ Done! Results saved to: {output_json}[/bold green]"
    )

    # Print summary table
    if results["videos"]:
        table = Table(title="Results Summary")
        table.add_column("Video", style="cyan")
        table.add_column("Predicted (ml)", justify="right", style="green")
        table.add_column("Ground Truth (ml)", justify="right", style="yellow")
        table.add_column("Error (ml)", justify="right", style="red")

        for video_id, data in sorted(results["videos"].items()):
            gt = f"{data['ground_truth_ml']:.1f}" if data["ground_truth_ml"] else "N/A"
            err = f"{data['error_ml']:.1f}" if data["error_ml"] else "N/A"
            table.add_row(video_id, f"{data['predicted_ml']:.1f}", gt, err)

        console.print(table)

    if "summary" in results:
        console.print(
            f"\n[bold]Mean Error: {results['summary']['mean_error_ml']:.2f} ml[/bold]"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Blood Loss Inference (No XML needed)")
    parser.add_argument(
        "--config",
        type=str,
        default="./inference_config.json",
        help="Path to config JSON file (default: ./inference_config.json)",
    )
    parser.add_argument(
        "--model", type=str, help="Path to model .pth file (overrides config)"
    )
    parser.add_argument(
        "--videos", type=str, help="Folder with .mp4 videos (overrides config)"
    )
    parser.add_argument(
        "--output", type=str, help="Output JSON path (overrides config)"
    )
    parser.add_argument("--csv", type=str, help="Ground truth CSV (overrides config)")
    parser.add_argument(
        "--clip-length", type=int, help="Frames per clip (overrides config)"
    )
    parser.add_argument(
        "--clips-per-seq", type=int, help="Clips per sequence (overrides config)"
    )
    parser.add_argument(
        "--input-size", type=int, nargs=2, help="H W (overrides config)"
    )
    parser.add_argument("--device", type=int, help="CUDA device ID (overrides config)")

    args = parser.parse_args()

    # Load config file
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        console.print(f"[cyan]Loaded config from: {args.config}[/cyan]")
    else:
        console.print(
            f"[yellow]Config file not found: {args.config}, using command line args[/yellow]"
        )

    # Command line args override config
    model_path = args.model if args.model else config.get("model_path")
    video_folder = args.videos if args.videos else config.get("video_folder")
    output_json = (
        args.output
        if args.output
        else config.get("output_json", "./inference_results.json")
    )
    volume_csv = args.csv if args.csv else config.get("volume_csv")
    clip_length = (
        args.clip_length if args.clip_length else config.get("clip_length", 24)
    )
    clips_per_seq = (
        args.clips_per_seq
        if args.clips_per_seq
        else config.get("clips_per_sequence", 6)
    )
    input_size = (
        tuple(args.input_size)
        if args.input_size
        else tuple(config.get("input_size", [224, 224]))
    )
    device_id = args.device if args.device is not None else config.get("device", 0)

    # Validate required params
    if not model_path:
        console.print("[red]Error: --model required (or set in config)[/red]")
        return
    if not video_folder:
        console.print("[red]Error: --videos required (or set in config)[/red]")
        return

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    results = infer_bloodloss(
        model_path=model_path,
        video_folder=video_folder,
        output_json=output_json,
        volume_csv=volume_csv,
        clip_length=clip_length,
        clips_per_sequence=clips_per_seq,
        input_size=input_size,
        device=device,
    )

    if results:
        console.print(
            f"\n-->[bold green]Successfully processed {len(results['videos'])} videos [/bold green]"
        )
    else:
        console.print("\n[red]!! Inference failed !![/red]")


if __name__ == "__main__":
    main()
