import os
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import csv

ROOT_DIR = Path("/home/r.rohangirish/mt_ble")

def process_annotation_zips(zip_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for zip_filename in os.listdir(zip_dir):
        if not zip_filename.endswith(".zip"):
            continue

        zip_path = os.path.join(zip_dir, zip_filename)
        xml_name = os.path.splitext(zip_filename)[0] + ".xml"
        xml_output_path = os.path.join(output_dir, xml_name)

        if os.path.exists(xml_output_path):
            print(f"✅ Skipping already processed: {xml_name}")
            continue

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("temp_extracted")

        extracted_xml = os.path.join("temp_extracted", "annotations.xml")
        if os.path.exists(extracted_xml):
            os.rename(extracted_xml, xml_output_path)
            print(f"📁 Saved: {xml_output_path}")

        # Clean up
        for root, dirs, files in os.walk("temp_extracted", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

def parse_xml_to_csv(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the video name from the XML file (using filename without extension)
    video_name = os.path.splitext(os.path.basename(xml_file))[0]

    # Create the output CSV file path
    output_csv = os.path.join(output_dir, f"{video_name}_annotations.csv")

    # Check if CSV already exists
    if os.path.exists(output_csv):
        print(f"CSV for {video_name} already exists. Skipping...")
        return

    # Open the CSV file to write results
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_frame", "end_frame", "label"])  # header

        # Process each track (e.g., BL_Low)
        for track in root.findall('track'):
            label = track.get('label')
            frames = []

            # Extract the frames for each bounding box in the track
            for box in track.findall('box'):
                frame = int(box.get('frame'))  # frame number
                frames.append(frame)

            # Group consecutive frames and create segments
            start_frame = frames[0]
            for i in range(1, len(frames)):
                # Check if the next frame is consecutive
                if frames[i] != frames[i-1] + 1:
                    # End the previous segment and start a new one
                    writer.writerow([start_frame, frames[i-1], label])
                    start_frame = frames[i]

            # Add the last segment (since the loop ends before writing it)
            writer.writerow([start_frame, frames[-1], label])

    print(f"CSV for {video_name} has been created.")

def process_all_xml_in_folder(xml_folder, output_dir):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            parse_xml_to_csv(xml_path, output_dir)

def main():
    # This is the function that takes zips that CVAT produces, makes XML files, then makes those into CSV files!
    xml_folder = ROOT_DIR / "data" / "labels_xml"
    output_csv = ROOT_DIR / "data" / "labels_csv"
    zips = ROOT_DIR / "data" / "labels_zip"
    output = ROOT_DIR / "pytvideo" / "data" / "labels"

    process_annotation_zips(zips, xml_folder)
    process_all_xml_in_folder(xml_folder, output_csv)

if __name__ == "__main__":
    main()