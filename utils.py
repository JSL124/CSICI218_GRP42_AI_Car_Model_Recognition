from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
import textwrap


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_class_mapping(train_dir: Path) -> tuple[list[str], dict[str, int]]:
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_idx


def save_classes(path: Path, index_to_class: list[str], class_to_index: dict[str, int]) -> None:
    ensure_dir(path.parent)
    payload = {"index_to_class": index_to_class, "class_to_index": class_to_index}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_classes(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_items_from_split(root_dir: Path, class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    items = []
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    for class_name, class_idx in class_to_idx.items():
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        for file in class_dir.iterdir():
            if file.is_file() and file.suffix.lower() in allowed:
                items.append((str(file.relative_to(root_dir)), class_idx))
    return items


def stratified_split(
    items: list[tuple[str, int]], split_ratio: float, seed: int
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)
    buckets: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for path, label in items:
        buckets[label].append((path, label))

    train_list = []
    val_list = []

    for label, label_items in buckets.items():
        rng.shuffle(label_items)
        n = len(label_items)
        if n == 1:
            train_list.extend(label_items)
            continue
        val_count = max(1, int(round(n * split_ratio)))
        if n - val_count < 1:
            val_count = n - 1
        val_list.extend(label_items[:val_count])
        train_list.extend(label_items[val_count:])

    rng.shuffle(train_list)
    rng.shuffle(val_list)
    return train_list, val_list


def write_split_file(path: Path, items: Iterable[tuple[str, int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rel_path, label in items:
            f.write(f"{rel_path}\t{label}\n")


def load_split_file(path: Path) -> list[tuple[str, int]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rel_path, label = line.split("\t")
        items.append((rel_path, int(label)))
    return items


class ImageListDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], root_dir: Path, transform=None) -> None:
        self.items = items
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel_path, label = self.items[idx]
        path = Path(rel_path)
        if not path.is_absolute():
            path = self.root_dir / rel_path
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, val_tfms


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().sum().item()
    return correct / max(labels.size(0), 1)


def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    k = min(k, logits.size(1))
    _, pred = logits.topk(k, 1, True, True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().sum().item() / max(labels.size(0), 1)


def save_metrics_csv(path: Path, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])


def save_plots(history: dict, plot_dir: Path) -> None:
    ensure_dir(plot_dir)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy_curve.png")
    plt.close()
    print(f"Saved plots to {plot_dir}")


def save_confusion_matrix(labels: list[int], preds: list[int], num_classes: int, out_path: Path) -> None:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        cm[y, p] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path)
    plt.close()


def get_gradcam_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    name = model_name.lower().strip()
    if name == "resnet50":
        return model.layer4[-1]
    if name == "efficientnet_v2_s":
        return model.features[-1]
    if name == "cnn":
        for module in reversed(list(model.features)):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("Grad-CAM: no Conv2d layer found in cnn features.")
    raise ValueError(f"Grad-CAM not supported for model: {model_name}")


def generate_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inp, out):
        activations["value"] = out

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        outputs = model(input_tensor)
        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1).item())
        score = outputs[:, class_idx].sum()
        score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grads = gradients["value"]
    acts = activations["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = cam * 0.0
    return cam, class_idx


def save_gradcam_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    out_overlay: Path,
    out_heatmap: Path,
    size: tuple[int, int],
    alpha: float = 0.4,
) -> None:
    image = Image.open(image_path).convert("RGB").resize(size)
    img = np.asarray(image).astype(np.float32) / 255.0
    heat_color = cm.jet(heatmap)[..., :3]
    overlay = (1.0 - alpha) * img + alpha * heat_color
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    heat_out = (heat_color * 255.0).clip(0, 255).astype(np.uint8)
    ensure_dir(out_overlay.parent)
    Image.fromarray(overlay).save(out_overlay)
    Image.fromarray(heat_out).save(out_heatmap)


def save_infer_annotated(image_path: Path, out_path: Path, title: str) -> None:
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    title_text = _fit_title(title)
    plt.title(title_text, fontsize=_title_fontsize(title_text))
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path)
    plt.close()


def show_infer_image(image_path: Path, title: str) -> None:
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    title_text = _fit_title(title)
    plt.title(title_text, fontsize=_title_fontsize(title_text))
    plt.tight_layout()
    plt.show()
    plt.close()


def build_gradcam_overlay_image(
    image_path: Path,
    heatmap: np.ndarray,
    size: tuple[int, int],
    alpha: float = 0.4,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(size)
    img = np.asarray(image).astype(np.float32) / 255.0
    heat_color = cm.jet(heatmap)[..., :3]
    overlay = (1.0 - alpha) * img + alpha * heat_color
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return overlay


def show_infer_with_overlay(image_path: Path, overlay: np.ndarray, title: str) -> None:
    image = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    axes[0].imshow(image)
    axes[0].axis("off")
    title_text = _fit_title(title)
    axes[0].set_title(title_text, fontsize=_title_fontsize(title_text))
    axes[1].imshow(overlay)
    axes[1].axis("off")
    axes[1].set_title("Grad-CAM overlay")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def show_infer_grid(image_paths: list[Path], overlays: list[np.ndarray], titles: list[str]) -> None:
    n = len(image_paths)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, (image_path, overlay, title) in enumerate(zip(image_paths, overlays, titles)):
        image = Image.open(image_path).convert("RGB")
        axes[0, i].imshow(image)
        axes[0, i].axis("off")
        title_text = _fit_title(title)
        axes[0, i].set_title(title_text, fontsize=_title_fontsize(title_text))
        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")
        axes[1, i].set_title("Grad-CAM overlay")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def _fit_title(title: str, width: int = 28) -> str:
    return textwrap.fill(title, width=width)


def _title_fontsize(title: str, base: int = 9, min_size: int = 6) -> int:
    lines = title.splitlines() if title else [""]
    max_len = max(len(line) for line in lines)
    if max_len <= 22:
        return base
    if max_len <= 32:
        return base - 1
    if max_len <= 42:
        return max(base - 2, min_size)
    return min_size
