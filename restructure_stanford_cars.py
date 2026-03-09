"""Reorganize Stanford Cars images into class folders using .mat annotations.

Expected default layout:
  archive/
    cars_train/cars_train/*.jpg
    cars_test/cars_test/*.jpg
    car_devkit/devkit/
      cars_meta.mat
      cars_train_annos.mat
      cars_test_annos.mat

Example:
  python3 restructure_stanford_cars.py --split both --mode copy
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

sio = None


def _require_scipy():
    global sio
    if sio is None:
        try:
            import scipy.io as _sio
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "scipy is required. Install first: pip install scipy"
            ) from exc
        sio = _sio


def _mat_str(x) -> str:
    if isinstance(x, str):
        return x
    try:
        return str(x[0])
    except Exception:
        return str(x)


def _to_int(v) -> int:
    return int(v[0][0])


def _safe_folder_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', " ", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _load_class_names(meta_path: Path) -> list[str]:
    _require_scipy()
    meta = sio.loadmat(str(meta_path))
    cn = meta["class_names"]  # shape (1, 196)
    return [_mat_str(cn[0, i]) for i in range(cn.shape[1])]


def _load_annos(annos_path: Path) -> list[tuple[str, int]]:
    _require_scipy()
    data = sio.loadmat(str(annos_path))
    annos = data["annotations"]  # shape (1, N)
    first = annos[0, 0]
    fields = set(first.dtype.names or ())
    if "class" not in fields:
        raise ValueError(
            f"'class' field is missing in {annos_path}. "
            "Use a labels-included annotation file (cars_test_annos_withlabels.mat)."
        )
    rows: list[tuple[str, int]] = []
    for i in range(annos.shape[1]):
        a = annos[0, i]
        fname = _mat_str(a["fname"])
        class_id = _to_int(a["class"])  # 1-based
        rows.append((fname, class_id))
    return rows


def _transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _process_split(
    split: str,
    image_root: Path,
    annos_path: Path,
    class_names: list[str],
    out_root: Path,
    mode: str,
) -> tuple[int, int]:
    records = _load_annos(annos_path)
    ok = 0
    missing = 0

    for fname, class_id in records:
        src = image_root / fname
        class_name = _safe_folder_name(class_names[class_id - 1])
        dst = out_root / split / class_name / fname

        if not src.exists():
            missing += 1
            continue
        _transfer(src, dst, mode)
        ok += 1
    return ok, missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reorganize Stanford Cars train/test images into class folders."
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("archive"),
        help="Root archive directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to process.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move", "symlink"],
        default="copy",
        help="How to place files in destination.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("archive_by_class"),
        help="Output directory.",
    )
    parser.add_argument(
        "--test-annos",
        type=Path,
        default=None,
        help="Optional explicit test annotation .mat path (with labels).",
    )
    return parser


def _resolve_test_annos(args) -> Path:
    if args.test_annos is not None:
        return args.test_annos

    candidates = [
        args.archive_dir / "car_devkit" / "devkit" / "cars_test_annos_withlabels.mat",
        Path("Reference/StanfordCars-Dataset-main/stanford_cars/cars_test_annos_withlabels.mat"),
        args.archive_dir / "car_devkit" / "devkit" / "cars_test_annos.mat",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


def main() -> None:
    args = build_parser().parse_args()

    devkit = args.archive_dir / "car_devkit" / "devkit"
    meta_path = devkit / "cars_meta.mat"
    train_annos = devkit / "cars_train_annos.mat"
    test_annos = _resolve_test_annos(args)

    train_images = args.archive_dir / "cars_train" / "cars_train"
    test_images = args.archive_dir / "cars_test" / "cars_test"

    required = [meta_path]
    if args.split in {"train", "both"}:
        required.extend([train_annos, train_images])
    if args.split in {"test", "both"}:
        required.extend([test_annos, test_images])

    missing_paths = [p for p in required if not p.exists()]
    if missing_paths:
        joined = "\n".join(f"- {p}" for p in missing_paths)
        raise FileNotFoundError(f"Missing required paths:\n{joined}")

    class_names = _load_class_names(meta_path)
    print(f"[INFO] Loaded classes: {len(class_names)}")
    print(f"[INFO] Output root: {args.out_dir.resolve()}")
    print(f"[INFO] Mode: {args.mode}")
    if args.split in {"train", "both"}:
        print(f"[INFO] Train annos: {train_annos}")
    if args.split in {"test", "both"}:
        print(f"[INFO] Test annos: {test_annos}")

    if args.split in {"train", "both"}:
        ok, missing = _process_split(
            split="train",
            image_root=train_images,
            annos_path=train_annos,
            class_names=class_names,
            out_root=args.out_dir,
            mode=args.mode,
        )
        print(f"[TRAIN] processed={ok}, missing={missing}")

    if args.split in {"test", "both"}:
        try:
            ok, missing = _process_split(
                split="test",
                image_root=test_images,
                annos_path=test_annos,
                class_names=class_names,
                out_root=args.out_dir,
                mode=args.mode,
            )
            print(f"[TEST] processed={ok}, missing={missing}")
        except ValueError as exc:
            raise SystemExit(
                f"{exc}\n\n"
                "Try one of the following:\n"
                "1) Provide --test-annos <.../cars_test_annos_withlabels.mat>\n"
                "2) Run train only: --split train"
            )


if __name__ == "__main__":
    main()
