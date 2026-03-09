# Car Model Identification (Offline, No CLI Args)

This project is controlled by editing the `CONFIG` dict in `main.py`.  
Run the pipeline with a single command:

```powershell
python main.py
```

## Quick Start

1. Edit run variables at the top of `main.py`:
   - `RUN_NAME` controls the output folder.
   - `INFER_IMAGE_PATH` and `INFER_IMAGE_NAME` build the image path for inference.

2. Edit `CONFIG` in `main.py`:
   - Set `mode` to `"train"`, `"eval"`, or `"infer"`.
   - Choose a `model` and `pretrained`:
     - `"cnn"` ignores `pretrained` (always random init)
     - `"resnet50"` / `"efficientnet_v2_s"` use ImageNet weights when `pretrained=True`
   - Toggle label smoothing:
     - `label_smoothing`: `True` or `False`
     - `label_smoothing_value`: float (e.g., `0.1`)
   - Update paths (`data_dir_train`, `data_dir_test`).

3. Run:
```powershell
python main.py
```

## How To Use

### Train
1. Set `mode = "train"`.
2. Set `data_dir_train` to your training folder.
3. Pick your `model`, `pretrained`, and training hyperparameters.
4. Run:
```powershell
python main.py
```

Outputs will be created under `outputs/runs/<RUN_NAME>/`.

### Eval
1. Set `mode = "eval"`.
2. Set `data_dir_test` to your test folder.
3. Set `checkpoint_path` and `classes_path` to the run you want to evaluate.
   - By default they point to `outputs/runs/<RUN_NAME>/...`.
4. Run:
```powershell
python main.py
```

### Infer
1. Set `mode = "infer"`.
2. Set `INFER_IMAGE_PATH` and `INFER_IMAGE_NAME` to a single image file.
3. Set `checkpoint_path` and `classes_path` to the run you want to use.
4. Optional: set `infer_mode = "random_test"` and `infer_random_n` to run on random test images.
5. Optional: set `infer_save_images = True` to copy inferred images into the run folder. If `False`, the image is displayed with true/pred labels.
6. Optional: set `infer_save_annotated = True` to save images labeled with true/pred (on separate lines, auto-fit font).
7. Optional: set `infer_print_console = True` to log true/pred in console.
8. Optional: set `gradcam = True` to save Grad-CAM overlays. When `infer_save_images=False`, a side-by-side display shows original + overlay.
9. Run:
```powershell
python main.py
```

## Examples

### Train (ResNet50, pretrained)
Set in `main.py`:
```python
RUN_NAME = "r50_pretrained_run1"
INFER_IMAGE_PATH = "dataset/infer"
INFER_IMAGE_NAME = "sample.jpg"
CONFIG["mode"] = "train"
CONFIG["model"] = "resnet50"
CONFIG["pretrained"] = True
CONFIG["lr_scheduler"] = "step"
```

### Eval (use a previous run)
```python
RUN_NAME = "r50_pretrained_run1"
CONFIG["mode"] = "eval"
CONFIG["data_dir_test"] = "dataset/test"
```

### Infer (single image)
```python
RUN_NAME = "r50_pretrained_run1"
INFER_IMAGE_PATH = "dataset/infer"
INFER_IMAGE_NAME = "my_car.jpg"
CONFIG["mode"] = "infer"
CONFIG["infer_mode"] = "single"
CONFIG["topk"] = 5
```

### Infer (random test images)
```python
RUN_NAME = "r50_pretrained_run1"
CONFIG["mode"] = "infer"
CONFIG["infer_mode"] = "random_test"
CONFIG["infer_random_n"] = 8
CONFIG["infer_save_images"] = True
```

## CONFIG Summary

Key fields in `main.py`:

- `mode`: `"train" | "eval" | "infer"`
- `data_dir_train`: training root (ImageFolder)
- `data_dir_test`: test root (ImageFolder)
- `out_dir`: output run directory
- `model`: `"cnn" | "resnet50" | "efficientnet_v2_s"`
- `pretrained`: `True/False` (applies to resnet/efficientnet only)
- `label_smoothing`: `True/False`
- `label_smoothing_value`: e.g. `0.1`
- `lr_scheduler`: `"none" | "step" | "plateau"`
- `lr_step_size`: step size for StepLR
- `lr_gamma`: LR decay factor (StepLR and Plateau)
- `lr_plateau_patience`: patience for ReduceLROnPlateau
- `two_stage_finetune`: `True/False`
- `stage1_epochs`: epochs for head-only training
- `stage2_epochs`: epochs for full fine-tuning
- `stage2_lr`: learning rate for stage 2
- `split_ratio`: validation ratio (default `0.1`)
- `epochs`, `batch_size`, `img_size`, `lr`, `seed`, `num_workers`
- `checkpoint_path`, `classes_path`, `image_path`
- `confusion_matrix`: `True/False`
- `infer_mode`: `"single" | "random_test"`
- `infer_random_n`: integer (only for `random_test`)
- `infer_save_images`: `True/False`
- `infer_save_annotated`: `True/False`
- `infer_print_console`: `True/False`
- `topk`: integer
- `use_amp`: `True/False` (CUDA only)
- `gradcam`: `True/False` (infer only)
- `gradcam_alpha`: overlay strength (0.0 to 1.0)

## Outputs

All outputs are written under:

```
outputs/runs/<run_name>/
  checkpoints/
    best.pt
    last.pt
  plots/
    loss_curve.png
    accuracy_curve.png
  splits/
    train.txt
    val.txt
  logs/
    metrics.csv
  classes.json
  config.csv
  plots/confusion_matrix.png (if enabled)
  gradcam/
    *_overlay.png
    *_heatmap.png
  infer/
    *.jpg
    *_annotated.png
```

# `restructure_stanford_cars.py`

Reorganizes Stanford Cars images into class-based folders using `.mat` annotations.

## What it does

- Reads class names from `cars_meta.mat`
- Reads image/class mappings from annotation files
- Creates output like:

```
archive_by_class/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

## Requirements 
- Python 3.8+
- `scipy`

Install dependency:
```
pip install scipy
```

---

## Expected default dataset layout

```
archive/
  cars_train/cars_train/*.jpg
  cars_test/cars_test/*.jpg
  car_devkit/devkit/
    cars_meta.mat
    cars_train_annos.mat
    cars_test_annos.mat
```

---

## Usage

Basic:

```
python3 restructure_stanford_cars.py --split both --mode copy
```
### Arguments
- archive-dir (default: archive)
Root of Stanford Cars archive.

- split (choices: train, test, both; default: both)
Which split to process.

- mode (choices: copy, move, symlink; default: copy)
File placement strategy.

- out-dir (default: archive_by_class)
Output root directory.

- test-annos (optional)
Explicit path to test annotation file (prefer labels-included test annotations).


Done by:
- Kim Bogyeong
- Chia Yuan Jun
- Lee Jinseo (Jason)
- Lee Haeeun (Chloe)
- Sun Zi Xiao (Cindy)
- Su Yi Maung (Monica)
