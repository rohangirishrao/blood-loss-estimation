import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
from torch.utils.data import DataLoader, Subset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import os
import glob
import json
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
# Import from your existing model file
from torchvision import transforms
from clf_model import SurgicalVideoDataset
console = Console()

# --------------------------------------------------------------
# This script contains the code for 3-fold cross-validation at the video level, 
# for the first task of evaluating classification accuracy of bleeding vs non-bleeding videos.
# --------------------------------------------------------------

class VideoBleedingDetector(nn.Module):
    def __init__(self, num_classes=2, severity_levels=4, dropout_rate=0.5):
        super().__init__()

        full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(full_model.children())[:-1])
        self.backbone.add_module('flatten', nn.Flatten())
        in_features = 512
        print("Using torchvision r2plus1d_18 model")
        # Classification head
        self.dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Linear(in_features, num_classes)
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


def run_3fold_cv(
    video_dir,
    annotation_dir,
    epochs=15,
    batch_size=16,
    clip_length=6,
    stride=3,
    learning_rate=1e-4,
    device="cuda:0",
    input_size=(200, 320),
    seed=42,
):
    """Run 3-fold cross-validation on video-level splits"""

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]3-Fold Cross-Validation[/bold cyan]")
    console.print(f"Device: {device}")

    # Collect all video-annotation pairs
    all_video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    video_paths, annotation_paths = [], []

    for vid_path in all_video_paths:
        vid_name = os.path.basename(vid_path).split(".")[0]
        anno_path = os.path.join(annotation_dir, f"{vid_name}.xml")
        if os.path.exists(anno_path):
            video_paths.append(vid_path)
            annotation_paths.append(anno_path)

    console.print(f"Found {len(video_paths)} videos with annotations")

    # Get unique video IDs for video-level splitting
    unique_video_ids = list(set(os.path.basename(p).split(".")[0] for p in video_paths))
    random.shuffle(unique_video_ids)

    # 3-fold split on videos
    kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    fold_results = []
    exclude_per_fold = {
        0: ["Dvpb", "MBHK"],      # Fold 1
        1: ["BMtu", "QmhB"],      # Fold 2
        2: ["DOOn", "DrOJ"]       # Fold 3
    }
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(unique_video_ids)):
        console.print(f"\n[bold blue]Fold {fold + 1}/3[/bold blue]")

        # Your existing code
        test_video_ids = [unique_video_ids[i] for i in test_idx]
        train_val_video_ids = [unique_video_ids[i] for i in train_val_idx]

        # ADD THIS: Remove excluded videos from training pool for this specific fold
        exclude_from_training_this_fold = exclude_per_fold.get(fold, [])  # Get exclusions for this fold
        excluded_this_fold = [vid for vid in train_val_video_ids if vid in exclude_from_training_this_fold]
        train_val_video_ids = [vid for vid in train_val_video_ids if vid not in exclude_from_training_this_fold]

        # Your existing split logic
        split_point = int(0.8 * len(train_val_video_ids))
        train_video_ids = train_val_video_ids[:split_point]
        val_video_ids = train_val_video_ids[split_point:]

        # ADD excluded videos to validation
        val_video_ids.extend(excluded_this_fold)

        console.print(f"Train: {len(train_video_ids)} videos")
        console.print(f"Val: {len(val_video_ids)} videos")
        console.print(f"Test: {len(test_video_ids)} videos")
        if excluded_this_fold:
            console.print(f"[yellow]Excluded from training this fold: {excluded_this_fold}[/yellow]")

        # Your existing training call
        fold_results_single = train_single_fold(
            video_paths,
            annotation_paths,
            train_video_ids,
            val_video_ids,
            test_video_ids,
            epochs,
            batch_size,
            clip_length,
            stride,
            learning_rate,
            device,
            input_size,
            fold,
        )

        fold_results.append(fold_results_single)

    # Aggregate results across folds
    print_cv_summary(fold_results)
    return fold_results

def filter_paths_by_ids(video_paths, annotation_paths, video_ids):
    filtered_videos, filtered_annos = [], []
    for vp, ap in zip(video_paths, annotation_paths):
        if os.path.basename(vp).split(".")[0] in video_ids:
            filtered_videos.append(vp)
            filtered_annos.append(ap)
    return filtered_videos, filtered_annos

def train_single_fold(
    video_paths,
    annotation_paths,
    train_video_ids,
    val_video_ids,
    test_video_ids,
    epochs,
    batch_size,
    clip_length,
    stride,
    learning_rate,
    device,
    input_size,
    fold_num,
    save_dir="./fold_results",
):
    """Train and evaluate a single fold with comprehensive severity tracking"""
    import json
    import os

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Filter paths by video IDs
    train_videos, train_annos = filter_paths_by_ids(
        video_paths, annotation_paths, train_video_ids
    )
    val_videos, val_annos = filter_paths_by_ids(
        video_paths, annotation_paths, val_video_ids
    )
    test_videos, test_annos = filter_paths_by_ids(
        video_paths, annotation_paths, test_video_ids
    )


    # Create transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
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
        transform=val_transform,
        fps_2=False,
    )
    test_dataset = SurgicalVideoDataset(
        test_videos,
        test_annos,
        clip_length=clip_length,
        stride=stride,
        transform=val_transform,
        fps_2=False,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    console.print(
        f"Clips - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Initialize model
    model = VideoBleedingDetector(num_classes=2, severity_levels=4, dropout_rate=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion_clf = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()

    # Initialize comprehensive metrics tracking
    fold_metrics = {
        "fold": fold_num + 1,
        "epochs": [],

        # Combined metrics
        "train_total_loss": [],
        "val_total_loss": [],

        # Classification metrics
        "train_clf_loss": [],
        "val_clf_loss": [],
        "val_clf_accuracy": [],
        "val_clf_precision": [],
        "val_clf_recall": [],
        "val_clf_f1": [],

        # Severity classification metrics
        "train_sev_loss": [],
        "val_sev_loss": [],
        "val_sev_accuracy": [],
        "val_sev_precision": [],
        "val_sev_recall": [],
        "val_sev_f1": [],

        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "clip_length": clip_length,
            "stride": stride,
            "learning_rate": learning_rate,
            "input_size": input_size,
            "train_videos": len(train_video_ids),
            "val_videos": len(val_video_ids),
            "test_videos": len(test_video_ids),
        }
    }

    # Training loop
    best_val_clf_f1 = 0.0
    best_val_sev_f1 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training with progress bar
        model.train()
        train_total_loss = 0.0
        train_clf_loss = 0.0
        train_sev_loss = 0.0

        with Progress(
            TextColumn(f"Epoch {epoch+1}/{epochs}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            task = progress.add_task("Training", total=len(train_loader))

            for clips, clip_labels, _, severity in train_loader:
                clips = clips.to(device)
                clip_labels = clip_labels.to(device)
                severity = severity.to(device)

                optimizer.zero_grad()
                clip_preds, severity_preds = model(clips)

                loss_clf = criterion_clf(clip_preds, clip_labels)
                loss_sev = criterion_sev(severity_preds, severity.clamp(0, 3))
                total_loss = loss_clf + 0.3 * loss_sev

                total_loss.backward()
                optimizer.step()

                train_total_loss += total_loss.item()
                train_clf_loss += loss_clf.item()
                train_sev_loss += loss_sev.item()

                progress.update(task, advance=1)

        # Validation with detailed metrics
        val_metrics = evaluate_model(
            model, val_loader, criterion_clf, criterion_sev, device
        )

        # Update best model based on classification F1
        if val_metrics["clf_f1"] > best_val_clf_f1:
            best_val_clf_f1 = val_metrics["clf_f1"]
            best_val_sev_f1 = val_metrics["sev_f1"]
            best_model_state = model.state_dict().copy()

        # Save epoch metrics
        avg_train_total_loss = train_total_loss / len(train_loader)
        avg_train_clf_loss = train_clf_loss / len(train_loader)
        avg_train_sev_loss = train_sev_loss / len(train_loader)

        fold_metrics["epochs"].append(epoch + 1)

        # Combined losses
        fold_metrics["train_total_loss"].append(avg_train_total_loss)
        fold_metrics["val_total_loss"].append(val_metrics["total_loss"])

        # Classification metrics
        fold_metrics["train_clf_loss"].append(avg_train_clf_loss)
        fold_metrics["val_clf_loss"].append(val_metrics["clf_loss"])
        fold_metrics["val_clf_accuracy"].append(val_metrics["clf_accuracy"])
        fold_metrics["val_clf_precision"].append(val_metrics["clf_precision"])
        fold_metrics["val_clf_recall"].append(val_metrics["clf_recall"])
        fold_metrics["val_clf_f1"].append(val_metrics["clf_f1"])

        # Severity metrics
        fold_metrics["train_sev_loss"].append(avg_train_sev_loss)
        fold_metrics["val_sev_loss"].append(val_metrics["sev_loss"])
        fold_metrics["val_sev_accuracy"].append(val_metrics["sev_accuracy"])
        fold_metrics["val_sev_precision"].append(val_metrics["sev_precision"])
        fold_metrics["val_sev_recall"].append(val_metrics["sev_recall"])
        fold_metrics["val_sev_f1"].append(val_metrics["sev_f1"])

        if epoch % 5 == 0:
            console.print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Total Loss: {avg_train_total_loss:.4f} - "
                f"Val Clf F1: {val_metrics['clf_f1']:.4f} - "
                f"Val Sev F1: {val_metrics['sev_f1']:.4f}"
            )

    best_model_path = os.path.join(save_dir, f"fold_{fold_num+1}_best_model.pth")
    torch.save(best_model_state, best_model_path)
    # Test with best model
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_model(
        model, test_loader, criterion_clf, criterion_sev, device
    )

    # Add final test results to metrics
    fold_metrics["test_results"] = {
        "total_loss": test_metrics["total_loss"],

        # Classification test results
        "clf_accuracy": test_metrics["clf_accuracy"],
        "clf_precision": test_metrics["clf_precision"],
        "clf_recall": test_metrics["clf_recall"],
        "clf_f1": test_metrics["clf_f1"],
        "clf_loss": test_metrics["clf_loss"],

        # Severity test results
        "sev_accuracy": test_metrics["sev_accuracy"],
        "sev_precision": test_metrics["sev_precision"],
        "sev_recall": test_metrics["sev_recall"],
        "sev_f1": test_metrics["sev_f1"],
        "sev_loss": test_metrics["sev_loss"],

        # Detailed severity breakdown
        "sev_class_report": test_metrics.get("sev_class_report", {}),
        "sev_confusion_matrix": test_metrics.get("sev_confusion_matrix", []),
    }

    # Best validation scores
    fold_metrics["best_val_clf_f1"] = best_val_clf_f1
    fold_metrics["best_val_sev_f1"] = best_val_sev_f1

    # Save comprehensive metrics to single file per fold
    comprehensive_metrics_file = os.path.join(save_dir, f"fold_{fold_num+1}_comprehensive_metrics.json")
    with open(comprehensive_metrics_file, 'w') as f:
        json.dump(fold_metrics, f, indent=2)

    console.print(
        f"[green]Fold {fold_num+1} Results:[/green]\n"
        f"  Classification - Acc: {test_metrics['clf_accuracy']:.3f}, F1: {test_metrics['clf_f1']:.3f}\n"
        f"  Severity - Acc: {test_metrics['sev_accuracy']:.3f}, F1: {test_metrics['sev_f1']:.3f}"
    )
    console.print(f"[blue]All metrics saved to: {comprehensive_metrics_file}[/blue]")

    return {
        "fold": fold_num + 1,

        # Classification results
        "test_clf_accuracy": test_metrics["clf_accuracy"],
        "test_clf_precision": test_metrics["clf_precision"],
        "test_clf_recall": test_metrics["clf_recall"],
        "test_clf_f1": test_metrics["clf_f1"],

        # Severity results
        "test_sev_accuracy": test_metrics["sev_accuracy"],
        "test_sev_precision": test_metrics["sev_precision"],
        "test_sev_recall": test_metrics["sev_recall"],
        "test_sev_f1": test_metrics["sev_f1"],

        # Best validation scores
        "best_val_clf_f1": best_val_clf_f1,
        "best_val_sev_f1": best_val_sev_f1,

        # Dataset info
        "train_videos": len(train_video_ids),
        "val_videos": len(val_video_ids),
        "test_videos": len(test_video_ids),
        "metrics_file": comprehensive_metrics_file,
    }


def evaluate_model(model, data_loader, criterion_clf, criterion_sev, device):
    """Enhanced evaluation with separate classification and severity metrics"""

    model.eval()
    total_loss = 0.0
    clf_loss_total = 0.0
    sev_loss_total = 0.0

    all_clf_preds = []
    all_clf_labels = []
    all_sev_preds = []
    all_sev_labels = []

    with torch.no_grad():
        for clips, clip_labels, _, severity in data_loader:
            clips = clips.to(device)
            clip_labels = clip_labels.to(device)
            severity = severity.to(device)

            clip_preds, severity_preds = model(clips)

            # Calculate losses
            clf_loss = criterion_clf(clip_preds, clip_labels)
            sev_loss = criterion_sev(severity_preds, severity.clamp(0, 3))
            batch_total_loss = clf_loss + 0.3 * sev_loss

            total_loss += batch_total_loss.item()
            clf_loss_total += clf_loss.item()
            sev_loss_total += sev_loss.item()

            # Get predictions
            _, clf_predicted = torch.max(clip_preds, 1)
            _, sev_predicted = torch.max(severity_preds, 1)

            all_clf_preds.extend(clf_predicted.cpu().numpy())
            all_clf_labels.extend(clip_labels.cpu().numpy())
            all_sev_preds.extend(sev_predicted.cpu().numpy())
            all_sev_labels.extend(severity.clamp(0, 3).cpu().numpy())

    # Calculate classification metrics
    clf_accuracy = accuracy_score(all_clf_labels, all_clf_preds)
    clf_precision = precision_score(all_clf_labels, all_clf_preds, zero_division=0)
    clf_recall = recall_score(all_clf_labels, all_clf_preds, zero_division=0)
    clf_f1 = f1_score(all_clf_labels, all_clf_preds, zero_division=0)

    # Calculate severity metrics
    sev_accuracy = accuracy_score(all_sev_labels, all_sev_preds)
    sev_precision = precision_score(all_sev_labels, all_sev_preds, average='weighted', zero_division=0)
    sev_recall = recall_score(all_sev_labels, all_sev_preds, average='weighted', zero_division=0)
    sev_f1 = f1_score(all_sev_labels, all_sev_preds, average='weighted', zero_division=0)

    # Detailed severity analysis
    try:
        sev_class_report = classification_report(all_sev_labels, all_sev_preds, output_dict=True, zero_division=0)
        sev_confusion_matrix = confusion_matrix(all_sev_labels, all_sev_preds).tolist()
    except:
        sev_class_report = {}
        sev_confusion_matrix = []

    return {
        "total_loss": total_loss / len(data_loader),

        # Classification metrics
        "clf_loss": clf_loss_total / len(data_loader),
        "clf_accuracy": clf_accuracy,
        "clf_precision": clf_precision,
        "clf_recall": clf_recall,
        "clf_f1": clf_f1,

        # Severity metrics
        "sev_loss": sev_loss_total / len(data_loader),
        "sev_accuracy": sev_accuracy,
        "sev_precision": sev_precision,
        "sev_recall": sev_recall,
        "sev_f1": sev_f1,

        # Detailed severity analysis
        "sev_class_report": sev_class_report,
        "sev_confusion_matrix": sev_confusion_matrix,
    }


def print_cv_summary(fold_results):
    """Print cross-validation summary"""

    # Results table
    table = Table(title="3-Fold Cross-Validation Results")
    table.add_column("Fold", style="cyan")
    table.add_column("Test Acc", justify="right", style="green")
    table.add_column("Test Prec", justify="right", style="yellow")
    table.add_column("Test Rec", justify="right", style="blue")
    table.add_column("Test F1", justify="right", style="magenta")
    table.add_column("Best Val F1", justify="right", style="white")

    for result in fold_results:
        table.add_row(
            str(result["fold"]),
            f"{result['test_accuracy']:.3f}",
            f"{result['test_precision']:.3f}",
            f"{result['test_recall']:.3f}",
            f"{result['test_f1']:.3f}",
            f"{result['best_val_f1']:.3f}",
        )

    # Calculate means and stds
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    means = {metric: np.mean([r[metric] for r in fold_results]) for metric in metrics}
    stds = {metric: np.std([r[metric] for r in fold_results]) for metric in metrics}

    # Add mean row
    table.add_row(
        "[bold]Mean ± Std[/bold]",
        f"[bold]{means['test_accuracy']:.3f} ± {stds['test_accuracy']:.3f}[/bold]",
        f"[bold]{means['test_precision']:.3f} ± {stds['test_precision']:.3f}[/bold]",
        f"[bold]{means['test_recall']:.3f} ± {stds['test_recall']:.3f}[/bold]",
        f"[bold]{means['test_f1']:.3f} ± {stds['test_f1']:.3f}[/bold]",
        "-",
    )

    console.print(table)

    # Summary panel
    console.print(
        Panel.fit(
            f"[bold green]Cross-Validation Summary[/bold green]\n"
            f"Mean Test F1: {means['test_f1']:.3f} ± {stds['test_f1']:.3f}\n"
            f"Mean Test Accuracy: {means['test_accuracy']:.3f} ± {stds['test_accuracy']:.3f}\n"
            f"Total Videos: {sum(r['train_videos'] + r['val_videos'] + r['test_videos'] for r in fold_results) // 3}",
            title="Final Results",
        )
    )

def get_cv_dataset_statistics_only(
    video_dir,
    annotation_dir,
    clip_length=6,
    stride=3,
    input_size=(200, 320),
    seed=42,
):
    """Get dataset statistics for 3-fold CV without training"""

    import glob
    import random
    from sklearn.model_selection import KFold
    import os
    from rich.console import Console

    console = Console()
    console.print(f"[bold cyan]3-Fold Dataset Statistics Only[/bold cyan]")

    # Collect all video-annotation pairs
    all_video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    video_paths, annotation_paths = [], []

    for vid_path in all_video_paths:
        vid_name = os.path.basename(vid_path).split(".")[0]
        anno_path = os.path.join(annotation_dir, f"{vid_name}.xml")
        if os.path.exists(anno_path):
            video_paths.append(vid_path)
            annotation_paths.append(anno_path)

    console.print(f"Found {len(video_paths)} videos with annotations")

    # Get unique video IDs for video-level splitting
    unique_video_ids = list(set(os.path.basename(p).split(".")[0] for p in video_paths))
    random.seed(seed)
    random.shuffle(unique_video_ids)

    # 3-fold split on videos
    kfold = KFold(n_splits=3, shuffle=True, random_state=seed)

    def filter_paths_by_ids(video_paths, annotation_paths, video_ids):
        filtered_videos, filtered_annos = [], []
        for vp, ap in zip(video_paths, annotation_paths):
            if os.path.basename(vp).split(".")[0] in video_ids:
                filtered_videos.append(vp)
                filtered_annos.append(ap)
        return filtered_videos, filtered_annos

    # Create minimal transform for statistics
    basic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_fold_stats = {}

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(unique_video_ids)):
        console.print(f"\n[bold blue]Processing Fold {fold + 1}/3 Statistics[/bold blue]")

        # Split video IDs
        test_video_ids = [unique_video_ids[i] for i in test_idx]
        train_val_video_ids = [unique_video_ids[i] for i in train_val_idx]

        # Further split train_val into train and val (80-20)
        split_point = int(0.8 * len(train_val_video_ids))
        train_video_ids = train_val_video_ids[:split_point]
        val_video_ids = train_val_video_ids[split_point:]

        console.print(f"Video split - Train: {len(train_video_ids)}, Val: {len(val_video_ids)}, Test: {len(test_video_ids)}")

        # Filter paths by video IDs
        train_videos, train_annos = filter_paths_by_ids(video_paths, annotation_paths, train_video_ids)
        val_videos, val_annos = filter_paths_by_ids(video_paths, annotation_paths, val_video_ids)
        test_videos, test_annos = filter_paths_by_ids(video_paths, annotation_paths, test_video_ids)

        # Create datasets to get clip statistics
        train_dataset = SurgicalVideoDataset(
            train_videos, train_annos,
            clip_length=clip_length, stride=stride,
            transform=basic_transform, fps_2=False
        )
        val_dataset = SurgicalVideoDataset(
            val_videos, val_annos,
            clip_length=clip_length, stride=stride,
            transform=basic_transform, fps_2=False
        )
        test_dataset = SurgicalVideoDataset(
            test_videos, test_annos,
            clip_length=clip_length, stride=stride,
            transform=basic_transform, fps_2=False
        )

        # Get detailed statistics
        fold_stats = SurgicalVideoDataset.compare_fold_statistics(
            train_dataset, val_dataset, test_dataset, fold + 1
        )

        all_fold_stats[f"fold_{fold + 1}"] = fold_stats

    # Save all statistics
    os.makedirs("./dataset_statistics", exist_ok=True)
    with open("./dataset_statistics/cv_dataset_stats.json", 'w') as f:
        json.dump(all_fold_stats, f, indent=2)

    console.print(f"\n[bold green]All fold statistics saved to: ./dataset_statistics/cv_dataset_stats.json[/bold green]")

    return all_fold_stats


if __name__ == "__main__":
    VIDEO_DIR = "/home/r.rohangirish/mt_ble/data/videos"
    ANNO_DIR = "/home/r.rohangirish/mt_ble/data/labels_xml"
    seed = 40
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    stats = get_cv_dataset_statistics_only(
        video_dir=VIDEO_DIR,
        annotation_dir=ANNO_DIR,
        clip_length=6,
        stride=3,
        input_size=(200, 320),
        seed=seed
    )

    results = run_3fold_cv(
        video_dir=VIDEO_DIR,
        annotation_dir=ANNO_DIR,
        epochs=20,
        batch_size=24,
        clip_length=6,
        stride=3,
        learning_rate=1e-4,
        device="cuda:5",
        input_size=(224, 224),
        seed=seed,
    )
