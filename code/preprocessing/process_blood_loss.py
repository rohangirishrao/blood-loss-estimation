import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text
import re


def extract_prefix(folder_name):
    if not isinstance(folder_name, str):
        return None
    m = re.search(r'(?:anon(?:ymized)?[-_])([A-Za-z]{4})', folder_name, re.IGNORECASE)
    return m.group(1) if m else None

def process_all_blood_loss_data(folder_path, output_csv="BL_data_combined.csv", fps=1):
    # Load all data
    bl_data = pd.read_csv(os.path.join(folder_path, "BL_data.csv"))
    bl_ts   = pd.read_csv(os.path.join(folder_path, "BL_data_time_series.csv"))
    matches = pd.read_csv(os.path.join(folder_path, "cleaned_matches.csv"))
    idmap   = pd.read_csv(os.path.join(folder_path, "case_id_record_id.csv"))

    # Get matches from already matched data by Simon
    matches[["patient_id", "case_id"]] = matches["patient"].str.split(",", expand=True)
    matches["case_id"] = matches["case_id"].astype(str).str.strip()
    matches["video_name"] = matches["folder"].apply(extract_prefix)

    matches["start_y"] = pd.to_datetime(matches["start_y"], errors="coerce")
    matches["end_y"]   = pd.to_datetime(matches["end_y"], errors="coerce")
    matches["frame_count"] = (matches["end_y"] - matches["start_y"]).dt.total_seconds().mul(fps).round().fillna(0).astype(int)

    # === Normalize case_id mapping ===
    idmap["case_id"] = idmap["case_id"].astype(str).str.strip()
    idmap = idmap.assign(case_id=idmap["case_id"].str.split(",")).explode("case_id")
    idmap["case_id"] = idmap["case_id"].str.strip()

    # Join matches with idmap to get record_id + timing info
    video_meta = matches.merge(idmap[["record_id", "case_id"]], on="case_id", how="left")
    video_meta = video_meta[["record_id", "video_name", "start_y", "end_y", "frame_count"]].dropna(subset=["record_id"])
    video_meta = video_meta.drop_duplicates(subset=["record_id"])

    bl_data["video_name"] = bl_data["video_repo"].apply(extract_prefix)
    aebl_map = bl_data[["record_id", "a_e_bl", "video_name"]].dropna(subset=["video_name"])

    # Get timestamps of measurements
    bl_ts["measure_time_bl_loss"] = pd.to_datetime(bl_ts["measure_time_bl_loss"], errors="coerce")

    combined = (
        bl_ts
        .merge(video_meta, on="record_id", how="left")
        .merge(aebl_map[["record_id", "a_e_bl", "video_name"]], on=["record_id", "video_name"], how="left")
    )

    # Get frame number of measurement time
    combined["measurement_frame"] = (
        (combined["measure_time_bl_loss"] - combined["start_y"])
        .dt.total_seconds()
        .round()
        .astype("Int64")
    )

    # Get rid of columns with no videos
    combined = combined[combined["video_name"].notna()]

    # Combined output columns
    combined = combined[[
        "record_id",
        "bl_loss",
        "measure_time_bl_loss",
        "start_y",
        "end_y",
        "measurement_frame",
        "frame_count",
        "video_name",
        "a_e_bl"
    ]]

    combined.to_csv(output_csv, index=False)
    # print(f"\n Final CSV saved to: {output_csv}")
    return combined



def open_output(csv):
    result = pd.read_csv(csv)
    print(result.head(5))


if __name__ == "__main__":
    console = Console()
    folder_path = "/home/r.rohangirish/mt_ble/data/labels_quantification"
    fps = 2
    output_csv = os.path.join(folder_path, f"BL_data_combined_{fps}_fps.csv", fps=fps)

    if not os.path.exists(folder_path):
        rprint(f"[bold red]Error:[/bold red] Folder {folder_path} does not exist.")
    else:
        console.print(Panel(Text("Processing blood loss data...", style="bold green")))
        combined_data = process_all_blood_loss_data(folder_path, output_csv)
        console.print(
            Panel(Text(f"Data processed and saved to {output_csv}", style="bold green"))
        )

    open_output(output_csv)
