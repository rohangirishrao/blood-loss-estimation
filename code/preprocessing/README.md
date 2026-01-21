# Preprocessing

This folder contains scripts for preprocessing of videos and labels, as well as merging all blood loss values from different CSV's.

1. `process_blood_loss.py` is the file to process all CSV's that have values about patient ID's, videos and their blood loss values.
2. `process_labels.py` processes a folder of zip's into a folder of xml's that are necessary.
3. `process_videos.py` is the main script to process video from S3 storage
    - it downloads specified videos locally (path will have to be given), splits them to 1 or 2 FPS (also specified at the top of the file), then optionally uploads them to S3 to a `split/` folder (Note: not working on DGX, upload will have to be done by the DBE VM). Then, the split video is saved to a specified path, upon which training and inference can be done.
    - Also contains a method to process specific paths to videos.
4. `download.py` is a simple script to download specified videos from S3, with no pre-processing steps. Not very necessary.