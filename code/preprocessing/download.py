import os
import glob
import tqdm
import boto3
import cv2
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

BUCKET_NAME = "quantitative-blood-loss-video"
PREFIX = "main/"
ENDPOINT_URL = "https://dbe-lakefs.dbe.unibas.ch:8000"

console = Console()
def download_videos(
    s3_path,
    video_list,
    local_dir,
    bucket_name=BUCKET_NAME,
    endpoint_url=ENDPOINT_URL
):
    """
    Download videos from S3 that match any of the given video identifiers.
    
    Args:
        s3_path: S3 path prefix (e.g., "main/split/")
        video_list: List of 4-character strings to match against video filenames
        local_dir: Local directory to save downloaded videos
        bucket_name: S3 bucket name
        endpoint_url: S3 endpoint URL
    
    Returns:
        List of local paths to downloaded videos
    """
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    os.makedirs(local_dir, exist_ok=True)
    
    # Normalize s3_path to ensure it ends with /
    if not s3_path.endswith("/"):
        s3_path += "/"
    
    console.print(f"[cyan]Searching for videos in: {bucket_name}/{s3_path}[/cyan]")
    console.print(f"[cyan]Matching identifiers: {video_list}[/cyan]\n")
    
    # List all objects in the S3 path
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_path)
    except Exception as e:
        console.print(f"[red]Error listing S3 objects: {e}[/red]")
        return []
    
    # Filter for .mp4 files that contain any of the video identifiers
    mp4_files = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".mp4"):
            filename = os.path.basename(key)
            # Check if any of the video identifiers is in the filename
            if any(vid_id in filename for vid_id in video_list):
                mp4_files.append(key)
    
    if not mp4_files:
        console.print(f"[yellow]No matching videos found for identifiers: {video_list}[/yellow]")
        return []
    
    console.print(f"[green]Found {len(mp4_files)} matching video(s)[/green]\n")
    
    # Download each matched video
    downloaded_paths = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Downloading videos", total=len(mp4_files))
        
        for s3_key in mp4_files:
            filename = os.path.basename(s3_key)
            local_path = os.path.join(local_dir, filename)
            
            try:
                console.log(f"[cyan]⬇️  Downloading: [bold]{filename}[/bold]")
                s3.download_file(bucket_name, s3_key, local_path)
                downloaded_paths.append(local_path)
                console.log(f"[green]✅ Saved to: {local_path}[/green]")
            except Exception as e:
                console.log(f"[red]❌ Error downloading {filename}: {e}[/red]")
            
            progress.update(task, advance=1)
    
    console.print(f"\n[bold green]Successfully downloaded {len(downloaded_paths)} video(s)[/bold green]")

    return downloaded_paths

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
        # Example: comb_Pat_anon_<BLOCK> → take <BLOCK>
        if "_" in base or "-" in base:
            # split on both delimiters, take the last valid block
            import re
            parts = re.split(r"[_-]", base)
            anon_block = parts[-1]
        else:
            # no delimiters → whole name is the block
            anon_block = base

        # Keep only alphanumeric characters
        anon_block = "".join(ch for ch in anon_block if ch.isalnum())

        if len(anon_block) < 4:
            return []

        return [anon_block[:4]]

def rename_files(dir):
    """Rename all .mp4 files in the given directory to their 4-char anonymized codes."""
    mp4_files = glob.glob(os.path.join(dir, "*.mp4"))
    console.print(f"[cyan]Renaming {len(mp4_files)} files in {dir}[/cyan]")
    
    for file_path in mp4_files:
        tokens = extract_tokens(file_path)
        if not tokens:
            console.print(f"[yellow]Skipping file (no valid token): {file_path}[/yellow]")
            continue
        
        new_name = tokens[0] + ".mp4"
        new_path = os.path.join(dir, new_name)
        
        try:
            os.rename(file_path, new_path)
            console.print(f"[green]Renamed:[/green] {file_path} → {new_path}")
        except Exception as e:
            console.print(f"[red]Error renaming {file_path}: {e}[/red]")


if __name__ == "__main__":
    download_videos(
        s3_path="main/split/",
        video_list=["QmhB", "MBHK", "OwRK"],
        local_dir="/raid/dsl/users/r.rohangirish/data/videos_1_fps"
    )
    # rename_files("/raid/dsl/users/r.rohangirish/data/infer_videos_1_fps")