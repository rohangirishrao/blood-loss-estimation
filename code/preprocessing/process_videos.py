import os
import glob
import tqdm
import boto3
import cv2
import re
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel

LOCAL_DIR = "/raid/dsl/users/r.rohangirish/data/lfs_temp"
SPLIT_DIR = "/raid/dsl/users/r.rohangirish/data/videos_1_fps"

# =========================
# ADD YOUR DIRECTORIES HERE
# LOCAL_DIR = "/home/dsl/MT_BLE/split_videos_temp"
# SPLIT_DIR = "/home/dsl/MT_BLE/split_videos_1_fps"

# =========================
# PARAMETERS HERE
FPS = 1
VIDEOS_TO_PROCESS = 16
# This can be a list of prefixes or a single comma-separated string
# to specify which videos you wish to process. Uncomment and fill in the prefixes.
PREFIX_FILTERS = [
    "kidv, yIDg, fMcS, FRoK, panG, kdju, oSOz, ohtb, nMDR, roiG, teFF, FHzn, ezAz, wPbc, kEiq"
]
# PREFIX_FILTERS = None


console = Console()
# =========================
# S3 CONFIGURATION
BUCKET_NAME = "quantitative-blood-loss-video"
PREFIX = "main/"
ENDPOINT_URL = "https://dbe-lakefs.dbe.unibas.ch:8000"


def get_comb_folders(bucket_name=BUCKET_NAME, prefix=PREFIX, endpoint_url=ENDPOINT_URL):
    """Return list of S3 folders starting with 'comb'."""
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")
    return [
        p["Prefix"].rstrip("/").split("/")[-1]
        for p in response.get("CommonPrefixes", [])
        if p["Prefix"].startswith(prefix + "comb")
    ]


def download_video_from_s3(
    folder, local_dir, bucket_name=BUCKET_NAME, endpoint_url=ENDPOINT_URL
):
    """Download the first .mp4 file from a specific folder in an S3 bucket."""
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    os.makedirs(local_dir, exist_ok=True)

    folder_prefix = f"{PREFIX}{folder}/"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    mp4_files = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".mp4")
        and obj["Key"].count("/") == folder_prefix.count("/")
    ]

    if not mp4_files:
        console.log(f"[yellow]⚠️ No .mp4 files found in {folder_prefix}")
        return None

    file_to_download = next(
        (f for f in mp4_files if os.path.basename(folder) in f),
        mp4_files[0],  # fallback
    )
    file_name = os.path.basename(file_to_download)
    local_path = os.path.join(local_dir, file_name)

    console.log(f"[cyan]!! Downloading [bold]{file_name}[/bold] from {folder_prefix}")
    s3.download_file(bucket_name, file_to_download, local_path)
    return local_path


def split_video(input_video_path, output_folder, fps=1, show_progress=True):
    """Split a video to the specified FPS and save result to output folder."""
    os.makedirs(output_folder, exist_ok=True)

    video_capture = cv2.VideoCapture(input_video_path)
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_fps == 0 or total_frames == 0:
        console.log(f"[red]⚠️ Invalid video: no frames or FPS in {input_video_path}")
        return None

    output_path = os.path.join(output_folder, os.path.basename(input_video_path))
    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    console.log(
        f"[green]Splitting video at {fps} FPS: [italic]{input_video_path}[/italic]"
    )

    frame_counter = 0
    
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Processing frames...", total=total_frames)

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                if frame_counter % int(original_fps / fps) == 0:
                    video_writer.write(frame)
                frame_counter += 1
                progress.update(task, advance=1)
    else:
        # No progress bar - just process frames
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_counter % int(original_fps / fps) == 0:
                video_writer.write(frame)
            frame_counter += 1

    video_capture.release()
    video_writer.release()
    console.log(f"[green]---> Saved split video: {output_path}")
    return output_path


def upload_to_s3(file_path, bucket_name=BUCKET_NAME, endpoint_url=ENDPOINT_URL):
    """Upload a file to the main/split/ directory in S3."""
    if not os.path.exists(file_path):
        return None

    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    s3_key = f"main/split/{os.path.basename(file_path)}"

    console.log(f"[blue]☁️ Uploading to S3 as [italic]{s3_key}[/italic]")
    s3.upload_file(file_path, bucket_name, s3_key)
    console.log("[green]-----> Upload complete.")
    return s3_key


def download_and_process_s3_videos(
    s3_keys,
    local_dir=LOCAL_DIR,
    output_folder=SPLIT_DIR,
    fps=1,
    upload=True,
    cleanup=True,
    bucket_name=BUCKET_NAME,
    endpoint_url=ENDPOINT_URL,
):
    """
    Download specific videos from S3 by their keys and process through the full pipeline.

    Pipeline: Download from S3 -> Split to target FPS -> Upload split video -> Remove original

    Args:
        s3_keys: List of S3 object keys (e.g., ["main/69_merged.mp4", "main/video2.mp4"])
        local_dir: Directory to download original videos to (default: LOCAL_DIR)
        output_folder: Directory to save split videos (default: SPLIT_DIR)
        fps: Target FPS for split videos (default: 1)
        upload: Whether to upload split videos back to S3 (default: True)
        cleanup: Whether to remove original downloaded videos after processing (default: True)
        bucket_name: S3 bucket name
        endpoint_url: S3 endpoint URL

    Returns:
        List of paths to successfully processed split videos
    """
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    console.print(f"\n[cyan]Processing {len(s3_keys)} video(s) from S3[/cyan]")
    console.print(
        f"[cyan]Pipeline: Download → Split ({fps} FPS) → Upload → Cleanup[/cyan]\n"
    )

    processed_videos = []
    failed_videos = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing videos", total=len(s3_keys))

        for idx, s3_key in enumerate(s3_keys, start=1):
            video_name = os.path.basename(s3_key)
            console.rule(f"[bold yellow][{idx}/{len(s3_keys)}]: {video_name}")

            # Step 1: Download from S3
            try:
                local_path = os.path.join(local_dir, video_name)
                console.log(f"[cyan]--> Downloading from S3: {s3_key}[/cyan]")
                s3.download_file(bucket_name, s3_key, local_path)
                console.log(f"[green]✓ Downloaded to: {local_path}[/green]")
            except Exception as e:
                console.log(f"[red]✗ Download failed: {e}[/red]")
                failed_videos.append(s3_key)
                progress.update(task, advance=1)
                continue

            # Step 2: Split video to target FPS
            try:
                split_path = split_video(local_path, output_folder, fps, show_progress=False)

                if not split_path:
                    console.log(f"[red]✗ Failed to split video[/red]")
                    failed_videos.append(s3_key)
                    if cleanup and os.path.exists(local_path):
                        os.remove(local_path)
                    progress.update(task, advance=1)
                    continue

                processed_videos.append(split_path)
                console.log(f"[green]✓ Split video saved[/green]")

            except Exception as e:
                console.log(f"[red]✗ Split failed: {e}[/red]")
                failed_videos.append(s3_key)
                if cleanup and os.path.exists(local_path):
                    os.remove(local_path)
                progress.update(task, advance=1)
                continue

            # Step 3: Upload split video to S3
            if upload:
                try:
                    upload_result = upload_to_s3(split_path, bucket_name, endpoint_url)
                    if upload_result:
                        console.log(f"[green]✓ Uploaded split video to S3[/green]")
                    else:
                        console.log(f"[yellow]⚠ Upload failed[/yellow]")
                except Exception as e:
                    console.log(f"[yellow]⚠ Upload error: {e}[/yellow]")

            # Step 4: Cleanup original downloaded video
            if cleanup and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    console.log(f"[green]✓ Removed original: {video_name}[/green]")
                except Exception as e:
                    console.log(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")

            progress.update(task, advance=1)

    # Summary
    console.print(
        f"\n[bold green]✓ Successfully processed: {len(processed_videos)}/{len(s3_keys)}[/bold green]"
    )
    if failed_videos:
        console.print(f"[bold red]✗ Failed: {len(failed_videos)}[/bold red]")
        for failed in failed_videos:
            console.print(f"  - {failed}")

    return processed_videos


def process_video_paths(video_paths, output_folder, fps=1, upload=False):
    """
    Process a list of LOCAL video file paths through the pipeline: split to target FPS and optionally upload.

    Note: This function expects videos to already be downloaded locally.
    For downloading from S3 first, use download_and_process_s3_videos() instead.

    Args:
        video_paths: List of absolute paths to .mp4 files (already on local disk)
        output_folder: Directory to save the split videos
        fps: Target FPS for the split videos (default: 1)
        upload: Whether to upload split videos to S3 (default: False)

    Returns:
        List of paths to successfully processed split videos
    """
    os.makedirs(output_folder, exist_ok=True)

    console.print(f"\n[cyan]Processing {len(video_paths)} video(s) at {fps} FPS[/cyan]")
    console.print(f"[cyan]Output folder: {output_folder}[/cyan]\n")

    processed_videos = []
    failed_videos = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing videos", total=len(video_paths))

        for idx, video_path in enumerate(video_paths, start=1):
            video_name = os.path.basename(video_path)
            console.rule(f"[bold yellow][{idx}/{len(video_paths)}]: {video_name}")

            # Check if file exists
            if not os.path.exists(video_path):
                console.log(f"[red]File not found: {video_path}[/red]")
                failed_videos.append(video_path)
                progress.update(task, advance=1)
                continue

            # Check if it's an mp4 file
            if not video_path.lower().endswith(".mp4"):
                console.log(f"[yellow]Skipping non-mp4 file: {video_path}[/yellow]")
                failed_videos.append(video_path)
                progress.update(task, advance=1)
                continue

            # Split the video
            try:
                split_path = split_video(video_path, output_folder, fps)

                if split_path:
                    processed_videos.append(split_path)
                    console.log(f"[green] Successfully processed: {video_name}[/green]")

                    # Upload if requested
                    if upload:
                        upload_result = upload_to_s3(split_path)
                        if upload_result:
                            console.log(f"[green]Uploaded to S3[/green]")
                        else:
                            console.log(f"[yellow]Upload failed[/yellow]")
                else:
                    console.log(f"[red]Failed to process: {video_name}[/red]")
                    failed_videos.append(video_path)

            except Exception as e:
                console.log(f"[red]Error processing {video_name}: {e}[/red]")
                failed_videos.append(video_path)

            progress.update(task, advance=1)

    # Summary
    console.print(
        f"\n[bold green]Successfully processed: {len(processed_videos)}/{len(video_paths)}[/bold green]"
    )
    if failed_videos:
        console.print(f"[bold red]Failed: {len(failed_videos)}[/bold red]")
        for failed in failed_videos:
            console.print(f"  - {os.path.basename(failed)}")

    return processed_videos


def process_videos(prefix_filters=None, max_to_process=10):
    """Main video processing pipeline: download, split, and optionally upload."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    processed_txt = os.path.join(LOCAL_DIR, "processed.txt")

    # Extract processed tokens
    def extract_tokens(name):
        """
        Extract anonymized 4-char patient code.

        Rules:
        1. Strip .mp4
        2. Remove '_joined' if present
        3. Find the substring after the last '_' or '-'
        4. Take FIRST 4 alphanumeric chars of that substring
        """
        base = os.path.basename(name)
        base = base.replace(".mp4", "")

        # Remove trailing "_joined"
        if base.endswith("_joined"):
            base = base[:-7]

        # Find last block after "_" or "-"
        if "_" in base or "-" in base:
            # split on both delimiters, take the last valid block
            import re

            parts = re.split(r"[_-]", base)
            anon_block = parts[-1]
        else:
            # no delimiters, whole name is the block
            anon_block = base

        # Keep only alphanumeric characters
        anon_block = "".join(ch for ch in anon_block if ch.isalnum())

        if len(anon_block) < 4:
            return []

        return [anon_block[:4]]

    processed_tokens = set()
    if os.path.exists(processed_txt):
        with open(processed_txt, "r") as f:
            for line in f:
                processed_tokens.update(extract_tokens(line.strip()))
    console.print(processed_tokens)

    def folder_contains_processed_token(folder):
        return any(tok in folder for tok in processed_tokens)

    # Load S3 folders
    comb_folders = get_comb_folders()
    console.print(f"--> Found [cyan]{len(comb_folders)}[/cyan] total folders in S3.")

    # Get and parse the given prefix filters
    parsed_prefixes = []
    if prefix_filters:
        if isinstance(prefix_filters, str):
            parsed_prefixes = [
                p.strip() for p in prefix_filters.split(",") if p.strip()
            ]
        elif isinstance(prefix_filters, (list, tuple, set)):
            flat = []
            for p in prefix_filters:
                if isinstance(p, str) and "," in p:
                    flat.extend([x.strip() for x in p.split(",") if x.strip()])
                elif isinstance(p, str):
                    flat.append(p.strip())
            parsed_prefixes = [p for p in flat if p]
        else:
            parsed_prefixes = [str(prefix_filters).strip()]

    if parsed_prefixes:
        non_4 = [p for p in parsed_prefixes if len(p) != 4]
        if non_4:
            console.log(f"[yellow]Note: some filters are not 4-chars long: {non_4}")

    # Apply prefix + processed-token filtering
    if parsed_prefixes:
        console.log(f"[cyan]Using prefix filters:[/cyan] {parsed_prefixes}")
        comb_folders = [
            folder
            for folder in comb_folders
            if any(p in folder for p in parsed_prefixes)
            and not folder_contains_processed_token(folder)
        ]
    else:
        comb_folders = [
            folder
            for folder in comb_folders
            if not folder_contains_processed_token(folder)
        ]

    comb_folders = comb_folders[:max_to_process]

    console.print(
        f"[bold]{len(comb_folders)}[/bold] folders selected for processing.\n"
    )

    # Process each folder
    for idx, folder in enumerate(comb_folders, start=1):
        console.rule(f"[bold yellow]Processing [{idx}/{len(comb_folders)}]: {folder}")
        video_path = download_video_from_s3(folder, LOCAL_DIR)
        if not video_path:
            continue

        split_path = split_video(video_path, SPLIT_DIR, FPS)
        if not split_path:
            continue

        # upload_to_s3(split_path)

        # Mark as processed
        os.remove(video_path)
        with open(processed_txt, "a") as f:
            f.write(os.path.basename(video_path) + "\n")

        console.log(f"Removed original file: {video_path}")


if __name__ == "__main__":
    console.print(
        Panel.fit("[bold green]Surgical Video Processor[/bold green]", title="Startup")
    )
    # If you have folders to process, uncomment below
    
    # NOTE: this will not upload, as over this is not working over DGX right now. 
    # If you wish to upload, use the DSL VM at dsl@dsl-dbe.dbe.unibas.ch
    # process_videos(max_to_process=VIDEOS_TO_PROCESS, prefix_filters=PREFIX_FILTERS)

    # If you have specific video paths to process, use below
    download_and_process_s3_videos(
        s3_keys=[
            "main/69_merged.mp4",
            "main/71_merged.mp4",
            "main/80_merged.mp4",
            "main/82_merged.mp4",
            "main/83_merged.mp4",
            "main/84_merged.mp4",
            "main/95_merged.mp4",
        ],  # Note: no leading slash!
        fps=1,
        upload=True,
        cleanup=True,
    )
