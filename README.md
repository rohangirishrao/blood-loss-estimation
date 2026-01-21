# Video-based Blood Loss Estimation
This repository houses the code for the Master Thesis titled "Development of a Video based deep learning algorithm to quantify blood loss in minimally invasive surgery".

It was developed at the **Digital Surgery Lab** of the Department of Biomedical Engineering at the *University of Basel*.

## Description
This project aims to develop a deep-learning model to identify bleeding events minimally invasive surgery, and further quantify the amount of the blood lost.
The ideal workflow for the entire project looks like this: 
1. CVAT (annotation) is set up on a VM on a DBE server. For this, we need downsampled videos, at 1 FPS. This is done by `process_videos.py`, which downloads, split and then uploads back to S3 under `/split` folder. The split videos are also saved locally, and can be moved to wherever they need to be.
2. The annotation procedure of importing videos and exporting annotations is given in the following **CVAT** section.
2. To train the model, `/3dcnn/train_ble.py` is the file that is necessary, more steps are given in the following **DGX** section. 

### Entire Pipeline Now:
Workflow files necessary:

### Processing videos (downlod, split, upload):`process_videos.py`

- In **`/code/preprocessing`** running `python3 process_videos.py` does this:
    - The folders and the FPS needs to be set at the top of the file. Open the file in VSCode and set a temp directory (`LOCAL_DIR` variable), where the full FPS videos are downloaded to. These will be deleted after the processing. 
    - Set the `FPS` variable. For annotation, 1 FPS is the best option. For later model training, 2 was also used, but not to much more success.
    - Set the `SPLIT_DIR` variable, where the processed videos will be saved. `VIDEOS_TO_PROCESS` is the number of videos to process, and the `PREFIX_FILTERS` allows you to choose a string which is contained in the video name, which is then processed, default **None**. Examples are given at the top of the file.

### CVAT:

- New Task in Task tab:
    - choose Project MT_Blood_Loss_Estimation
    - Give video name: VideoX_[First_4_Letters_Of_Video_Name], X: vid number on CVAT, add 1 to the last processed one
    - Advanced Config: 35% Image Quality, Chunk Size: 1000
    - Submit & Continue
    - Annotate Videos
    - Export CVAT 1.1 - results in zip file **locally, save with the first 4 letters, case sensitive**.

## DGX Workstation:

- put zips into **`data/labels_zip`**
- `video-based-bloodloss-assessment/code/process_labels.py`
    - takes zips that CVAT produces, makes XML files, then makes those into CSV files, in labels_xml & labels_csv resp.
    - Alternatively, you can also just extract the zip's and put the xml's into a folder on the server - might be easier.
- split videos from lakefs go into **`data/videos`**
- For further instructions, move into `code` folder, for training and inference. 

```bash
conda env create -f environments.yml
```

Then, open a `tmux` or `byobu` session (so that you don't have to be connected via SSH through the session). Then, `conda activate bleeding-detection`, which should set you up to be ready for all commands in `code` folder and the README there.
