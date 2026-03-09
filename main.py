from __future__ import annotations

import csv
import json
import os
import random
import shutil
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import get_model
from utils import (
    ImageListDataset,
    build_class_mapping,
    build_items_from_split,
    build_transforms,
    compute_accuracy,
    compute_topk_accuracy,
    ensure_dir,
    generate_gradcam,
    get_gradcam_target_layer,
    load_classes,
    load_split_file,
    save_classes,
    save_gradcam_overlay,
    save_infer_annotated,
    show_infer_image,
    show_infer_with_overlay,
    show_infer_grid,
    build_gradcam_overlay_image,
    save_metrics_csv,
    save_plots,
    seed_everything,
    stratified_split,
    write_split_file,
    save_confusion_matrix,
    get_device,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

RUN_NAME = "e_net50_pretrained_tuned_1"
RUN_DIR = f"outputs/runs/{RUN_NAME}"
INFER_IMAGE_PATH = "dataset/infer"
INFER_IMAGE_NAME = "infer_3.jpg"

CONFIG = {
    # --- Mode ---
    "mode": "infer",  # "train" | "eval" | "infer"

    # --- Data paths ---
    "data_dir_train": "dataset/train",
    "data_dir_test": "dataset/test",
    "out_dir": RUN_DIR,
    "checkpoint_path": f"{RUN_DIR}/checkpoints/best.pt",
    "classes_path": f"{RUN_DIR}/classes.json",
    "image_path": f"{INFER_IMAGE_PATH}/{INFER_IMAGE_NAME}",

    # --- Model ---
    "model": "efficientnet_v2_s",  # "cnn" | "resnet50" | "efficientnet_v2_s"
    "pretrained": True,  # applies to resnet/efficientnet only

    # --- Training setup ---
    "epochs": 20,
    "batch_size": 64,
    "img_size": 224,
    "lr": 1e-4,
    "seed": 42,
    "num_workers": 4,
    "split_ratio": 0.1,  # 90/10 by default
    "use_amp": True,

    # --- Regularization ---
    "label_smoothing": True,
    "label_smoothing_value": 0.1,

    # --- LR Scheduler ---
    "lr_scheduler": "step",  # "none" | "step" | "plateau"
    "lr_step_size": 10,
    "lr_gamma": 0.5,
    "lr_plateau_patience": 2,

    # --- Two-stage fine-tuning ---
    "two_stage_finetune": True,
    "stage1_epochs": 5,
    "stage2_epochs": 25,
    "stage2_lr": 1e-4,

    # --- Eval ---
    "confusion_matrix": False,

    # --- Infer ---
    "infer_mode": "random_test",  # "single" | "random_test"
    "infer_random_n": 5,
    "infer_save_images": False,
    "infer_print_console": True,
    "infer_save_annotated": False,
    "gradcam": True,
    "gradcam_alpha": 0.4,

    "topk": 5
}


def _load_or_create_classes(out_dir: Path, train_dir: Path) -> dict:
    classes_path = out_dir / "classes.json"
    if classes_path.exists():
        print(f"Loading classes from {classes_path}")
        return load_classes(classes_path)
    print(f"Building class mapping from {train_dir}")
    class_names, class_to_idx = build_class_mapping(train_dir)
    print(f"Found {len(class_names)} classes")
    save_classes(classes_path, class_names, class_to_idx)
    print(f"Saved classes to {classes_path}")
    return {"index_to_class": class_names, "class_to_index": class_to_idx}


def _load_or_create_splits(out_dir: Path, items: list[tuple[str, int]], split_ratio: float, seed: int):
    splits_dir = out_dir / "splits"
    ensure_dir(splits_dir)
    train_split_path = splits_dir / "train.txt"
    val_split_path = splits_dir / "val.txt"
    if train_split_path.exists() and val_split_path.exists():
        print(f"Loading existing splits from {splits_dir}")
        train_list = load_split_file(train_split_path)
        val_list = load_split_file(val_split_path)
        return train_list, val_list

    print(f"Creating stratified split with val ratio {split_ratio} and seed {seed}")
    train_list, val_list = stratified_split(items, split_ratio, seed)
    write_split_file(train_split_path, train_list)
    write_split_file(val_split_path, val_list)
    print(f"Saved splits to {splits_dir}")
    return train_list, val_list


def run_train(cfg: dict) -> None:
    print("Mode: train")
    print(f"Model: {cfg['model']} | Pretrained: {cfg['pretrained']}")
    seed_everything(cfg["seed"])
    out_dir = Path(cfg["out_dir"])
    ensure_dir(out_dir)
    ensure_dir(out_dir / "checkpoints")
    ensure_dir(out_dir / "plots")
    ensure_dir(out_dir / "logs")
    config_path = out_dir / "config.csv"
    with config_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key in sorted(cfg.keys()):
            writer.writerow([key, cfg[key]])
    print(f"Saved config to {config_path}")

    train_dir = Path(cfg["data_dir_train"])
    print(f"Training data dir: {train_dir}")
    classes = _load_or_create_classes(out_dir, train_dir)
    class_to_idx = classes["class_to_index"]

    items = build_items_from_split(train_dir, class_to_idx)
    print(f"Indexed {len(items)} images in training dataset")
    train_list, val_list = _load_or_create_splits(out_dir, items, cfg["split_ratio"], cfg["seed"])
    print(f"Train split size: {len(train_list)} | Val split size: {len(val_list)}")

    train_tfms, val_tfms = build_transforms(cfg["img_size"])
    train_ds = ImageListDataset(train_list, train_dir, train_tfms)
    val_ds = ImageListDataset(val_list, train_dir, val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = get_device()
    print(f"Using device: {device}")
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    if use_amp:
        print("AMP: enabled (CUDA)")
    else:
        print("AMP: disabled")
    num_classes = len(classes["index_to_class"])
    print(f"Model: {cfg['model']} | Num classes: {num_classes}")
    model = get_model(cfg["model"], num_classes, pretrained=cfg["pretrained"]).to(device)
    smoothing = cfg["label_smoothing_value"] if cfg["label_smoothing"] else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    def set_trainable(module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def get_head_module(model_name: str):
        if model_name == "resnet50":
            return model.fc
        if model_name == "efficientnet_v2_s":
            return model.classifier
        return None

    def build_optimizer_and_scheduler(params, lr):
        opt = torch.optim.Adam(params, lr=lr)
        sch = None
        if cfg["lr_scheduler"] == "step":
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"])
        elif cfg["lr_scheduler"] == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", patience=cfg["lr_plateau_patience"], factor=cfg["lr_gamma"]
            )
        return opt, sch

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    metrics_csv = out_dir / "logs" / "metrics.csv"
    start_time = time.perf_counter()

    def run_epoch_loop(start_epoch: int, num_epochs: int, optimizer, scheduler, stage_name: str):
        nonlocal best_acc, history, metrics_csv
        print("=" * 60)
        for i in range(num_epochs):
            epoch = start_epoch + i
            print(f"Epoch {epoch}/{total_epochs} ({stage_name})")
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            train_iter = train_loader
            if tqdm is not None:
                train_iter = tqdm(train_loader, desc="Train", leave=False)
            for images, labels in train_iter:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
                correct += compute_accuracy(outputs, labels) * images.size(0)
                total += images.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                val_iter = val_loader
                if tqdm is not None:
                    val_iter = tqdm(val_loader, desc="Val", leave=False)
                for images, labels in val_iter:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    val_correct += compute_accuracy(outputs, labels) * images.size(0)
                    val_total += images.size(0)

            val_loss = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            save_metrics_csv(metrics_csv, epoch, train_loss, val_loss, train_acc, val_acc)

            checkpoint_last = out_dir / "checkpoints" / "last.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, checkpoint_last)
            print(f"Saved last checkpoint to {checkpoint_last}")

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_best = out_dir / "checkpoints" / "best.pt"
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, checkpoint_best)
                print(f"New best checkpoint saved to {checkpoint_best}")

            if scheduler is not None:
                if cfg["lr_scheduler"] == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            epoch_time = time.perf_counter() - epoch_start
            elapsed = time.perf_counter() - start_time
            avg_epoch = elapsed / epoch
            eta = avg_epoch * (total_epochs - epoch)
            print(
                f"Epoch {epoch:03d}/{total_epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"time={epoch_time:.1f}s eta={eta/60:.1f}m"
            )

    use_two_stage = cfg["two_stage_finetune"] and cfg["model"] in {"resnet50", "efficientnet_v2_s"}
    if use_two_stage:
        total_epochs = cfg["stage1_epochs"] + cfg["stage2_epochs"]
        print(f"Two-stage fine-tuning enabled: stage1={cfg['stage1_epochs']} stage2={cfg['stage2_epochs']}")
        head = get_head_module(cfg["model"])
        if head is None:
            raise ValueError("Two-stage fine-tuning requested, but no classifier head found.")

        set_trainable(model, False)
        set_trainable(head, True)
        optimizer, scheduler = build_optimizer_and_scheduler(head.parameters(), cfg["lr"])
        run_epoch_loop(1, cfg["stage1_epochs"], optimizer, None, "stage1-head")

        set_trainable(model, True)
        optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), cfg["stage2_lr"])
        run_epoch_loop(cfg["stage1_epochs"] + 1, cfg["stage2_epochs"], optimizer, scheduler, "stage2-full")
    else:
        total_epochs = cfg["epochs"]
        optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), cfg["lr"])
        run_epoch_loop(1, cfg["epochs"], optimizer, scheduler, "single")

    save_plots(history, out_dir / "plots")
    print(f"Saved plots to {out_dir / 'plots'}")
    print("Training complete. Best val acc:", f"{best_acc:.4f}")


def run_eval(cfg: dict) -> None:
    print("Mode: eval")
    print(f"Model: {cfg['model']} | Pretrained: {cfg['pretrained']}")
    device = get_device()
    print(f"Using device: {device}")
    classes = load_classes(Path(cfg["classes_path"]))
    index_to_class = classes["index_to_class"]
    num_classes = len(index_to_class)

    print(f"Loading checkpoint from {cfg['checkpoint_path']}")
    model = get_model(cfg["model"], num_classes, pretrained=cfg["pretrained"]).to(device)
    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_dir = Path(cfg["data_dir_test"])
    print(f"Test data dir: {test_dir}")
    _, test_tfms = build_transforms(cfg["img_size"])
    test_items = build_items_from_split(test_dir, classes["class_to_index"])
    print(f"Indexed {len(test_items)} test images")
    test_ds = ImageListDataset(test_items, test_dir, test_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    top1_total = 0.0
    top5_total = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        test_iter = test_loader
        if tqdm is not None:
            test_iter = tqdm(test_loader, desc="Test", leave=False)
        for images, labels in test_iter:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            top1_total += compute_topk_accuracy(outputs, labels, k=1) * images.size(0)
            top5_total += compute_topk_accuracy(outputs, labels, k=5) * images.size(0)
            total += images.size(0)
            if cfg["confusion_matrix"]:
                preds = outputs.argmax(dim=1).detach().cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().tolist())

    top1 = top1_total / max(total, 1)
    top5 = top5_total / max(total, 1)
    print(f"Test Top-1 Accuracy: {top1:.4f}")
    print(f"Test Top-5 Accuracy: {top5:.4f}")

    if cfg["confusion_matrix"]:
        out_dir = Path(cfg["out_dir"])
        ensure_dir(out_dir / "plots")
        save_confusion_matrix(all_labels, all_preds, num_classes, out_dir / "plots" / "confusion_matrix.png")
        print(f"Saved confusion matrix to {out_dir / 'plots' / 'confusion_matrix.png'}")


def run_infer(cfg: dict) -> None:
    print("Mode: infer")
    print(f"Model: {cfg['model']} | Pretrained: {cfg['pretrained']}")
    device = get_device()
    print(f"Using device: {device}")
    classes = load_classes(Path(cfg["classes_path"]))
    index_to_class = classes["index_to_class"]
    num_classes = len(index_to_class)

    print(f"Loading checkpoint from {cfg['checkpoint_path']}")
    model = get_model(cfg["model"], num_classes, pretrained=cfg["pretrained"]).to(device)
    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    _, val_tfms = build_transforms(cfg["img_size"])
    out_dir = Path(cfg["out_dir"])
    infer_dir = out_dir / "infer"
    gradcam_dir = out_dir / "gradcam"

    def infer_one(image_path: Path, true_label: int | None) -> tuple[np.ndarray | None, str]:
        print(f"Inferring image: {image_path}")
        dataset = ImageListDataset([(str(image_path), 0)], image_path.parent, val_tfms)
        image, _ = dataset[0]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            topk = min(cfg["topk"], num_classes)
            values, indices = torch.topk(probs, k=topk, dim=1)

        values_list = values.cpu().numpy().tolist()[0]
        indices_list = indices.cpu().numpy().tolist()[0]

        pred_idx = int(indices_list[0])
        pred_name = index_to_class[pred_idx]
        true_name = index_to_class[true_label] if true_label is not None else "unknown"
        title = f"true: {true_name}\npred: {pred_name}"

        if cfg.get("infer_print_console", True):
            print("Top predictions:")
            for rank, (idx, score) in enumerate(zip(indices_list, values_list), start=1):
                print(f"{rank}. {index_to_class[idx]} ({score:.4f})")
            print(f"True: {true_name} | Pred: {pred_name}")

        save_images = bool(cfg.get("infer_save_images", False))
        show_images = not save_images
        if save_images:
            ensure_dir(infer_dir)
            dest = infer_dir / image_path.name
            shutil.copy2(image_path, dest)
            print(f"Saved infer image to {dest}")

        if cfg.get("infer_save_annotated", False):
            ensure_dir(infer_dir)
            annotated_path = infer_dir / f"{image_path.stem}_annotated.png"
            save_infer_annotated(image_path, annotated_path, title)
            print(f"Saved annotated infer image to {annotated_path}")

        if cfg.get("gradcam", False):
            target_layer = get_gradcam_target_layer(model, cfg["model"])
            heatmap, _ = generate_gradcam(model, target_layer, image)
            ensure_dir(gradcam_dir)
            stem = image_path.stem
            overlay_path = gradcam_dir / f"{stem}_overlay.png"
            heatmap_path = gradcam_dir / f"{stem}_heatmap.png"
            save_gradcam_overlay(
                image_path=image_path,
                heatmap=heatmap,
                out_overlay=overlay_path,
                out_heatmap=heatmap_path,
                size=(cfg["img_size"], cfg["img_size"]),
                alpha=float(cfg.get("gradcam_alpha", 0.4)),
            )
            print(f"Saved Grad-CAM overlay to {overlay_path}")
            print(f"Saved Grad-CAM heatmap to {heatmap_path}")
            if show_images:
                overlay_img = build_gradcam_overlay_image(
                    image_path=image_path,
                    heatmap=heatmap,
                    size=(cfg["img_size"], cfg["img_size"]),
                    alpha=float(cfg.get("gradcam_alpha", 0.4)),
                )
                return overlay_img, title
        if show_images:
            show_infer_image(image_path, title)
        return None, ""

    infer_mode = cfg.get("infer_mode", "single")
    if infer_mode == "single":
        image_path = Path(cfg["image_path"]).resolve()
        overlay_img, title = infer_one(image_path, None)
        if overlay_img is not None:
            show_infer_with_overlay(image_path, overlay_img, title)
    elif infer_mode == "random_test":
        test_dir = Path(cfg["data_dir_test"])
        items = build_items_from_split(test_dir, classes["class_to_index"])
        if not items:
            raise ValueError(f"No test images found in {test_dir}")
        n = int(cfg.get("infer_random_n", 1))
        n = max(1, min(n, len(items)))
        chosen = random.sample(items, n)
        grid_images: list[Path] = []
        grid_overlays: list[np.ndarray] = []
        grid_titles: list[str] = []
        for rel_path, label in chosen:
            img_path = (test_dir / rel_path).resolve()
            overlay_img, title = infer_one(img_path, label)
            if overlay_img is not None:
                grid_images.append(img_path)
                grid_overlays.append(overlay_img)
                grid_titles.append(title)
        if grid_images:
            show_infer_grid(grid_images, grid_overlays, grid_titles)
    else:
        raise ValueError("Unknown infer_mode. Use 'single' or 'random_test'.")


def main() -> None:
    mode = CONFIG["mode"].lower().strip()
    if mode == "train":
        run_train(CONFIG)
    elif mode == "eval":
        run_eval(CONFIG)
    elif mode == "infer":
        run_infer(CONFIG)
    else:
        raise ValueError("Unknown mode in CONFIG. Use 'train', 'eval', or 'infer'.")


if __name__ == "__main__":
    main()
