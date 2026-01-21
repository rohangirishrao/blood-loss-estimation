# 3DCNN + LSTM Training and Inference

This folder contains all the main code for the project, specifically the implementation of the final 3DCNN + LSTM model for blood loss prediction. The folder train-files has all older files and tests conducted, while `train_BLE.py` is the main training file for the model. `inference.py` is the meant for inferring on a folder of untrained, but 1/2FPS processed videos. 

The following sections (Train & Inference) contain instructions on how to run 2 parts of the pipeline.

# Conda

Make sure to go back to the original path of this repo, and type the following. This might take a few minutes.
```bash
conda env create -f environments.yml
```

Then, open a `tmux` or `byobu` session (so that you don't have to be connected via SSH through the session). Then, `conda activate bleeding-detection`, which should set you up to be ready for all following commands.

# Training BLE

## Before Usage
Check device availability with `nvitop`, then remember the device ID that is free. If 0 is free in nvitop, you can type in 0 in the json file. This step is very important.

## Usage

**Train using config file (RECOMMENDED):**
```bash
python train_BLE.py --config ./train_config.json
```

**Override config with command-line arguments:**
```bash
python train_BLE.py --train --epochs 50 --batch 4 --dev 1
```

## Configuration Params

Edit `train_config.json` to set default parameters, here with 12sx6 = 72s sequences:

```json
{
  "device": 0,
  "epochs": 20,
  "batch_size": 2,
  "train": true,
  "seed": 41,
  "model_name": "model-dec5",
  "video_dir_1fps": "/path/to/video_dir_1fps",
  "video_dir_2fps": "/path/to/video_dir_2fps",
  "annotations_dir": "/path/to/dir_labels_xml",
  "volume_csv": "../documents/BL_data_combined.csv",
  "output_dir_models": "./MODELS_DEC2025",
  "clip_length": 12, 
  "clips_per_sequence": 6,
  "stride": 12,
  "learning_rate": 1e-5,
  "lr_backbone": 3e-5,
  "lr_lstm": 1e-4,
  "input_size": [224, 224],
  "use_2fps": false,
  "test_model_path": "./path/to/best_model.pth",
  "specific_test_video": "Dvpb"
}
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to config JSON | `./train_config.json` |
| `--train` | Enable training mode | From config |
| `--test` | Enable test mode | From config |
| `--dev` | GPU device ID | From config |
| `--epochs` | Number of epochs | From config |
| `--batch` | Batch size | From config |
| `--seed` | Random seed | From config |
| `--model-name` | Model name | From config |

**Note:** Command-line arguments override config file values.

## Examples

**Train new model with custom settings:**
```bash
python train_BLE.py --train --model-name my-model --epochs 30 --batch 3
```

**Use different config file:**
```bash
python train_BLE.py --config my_config.json --train true
```

# INFERENCE BLE

## Usage

1. Use `inference_config.json`, like the following example. The device ID is important, use `nvitop` to find out which device is free on the cluster - inference needs around 10-15 GB of free VRAM, but always use a free GPU as to not interrupt other processes.

```json
{
  "model_path": "./MODELS_NOV2025/model-dec1/model-dec1_best.pth",
  "video_folder": "/raid/dsl/users/r.rohangirish/data/videos_high_bl",
  "output_json": "./inference_results.json",
  "volume_csv": "/home/r.rohangirish/mt_ble/data/labels_quantification/BL_data_combined.csv",
  "clip_length": 12,
  "clips_per_sequence": 6,
  "input_size": [224, 224],
  "device": 0
}
```

2. Run:

```bash
python inference.py
```
if using the json. Otherwise you can also use it over the CL, with the following line. All possible config params are listed right after. 

```bash
python inference.py --model_path /path/to/trained_model.pth --video_folder /folder/of/mp4s
```
## Config Parameters

- `model_path`: Path to trained model (.pth file)
- `video_folder`: Folder containing .mp4 videos
- `output_json`: Where to save results
- `volume_csv`: (Optional) CSV with ground truth for comparison
- `clip_length`: Frames per clip (default: 24)
- `clips_per_sequence`: Clips per sequence (default: 6)
- `input_size`: Frame size [H, W] (default: [224, 224])
- `device`: GPU device ID (default: 0)
