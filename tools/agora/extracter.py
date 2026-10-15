#!/usr/bin/env python3
"""
Extract the RTMDet-Fusion training data from the AGORA dataset.

Random crops are cut from every source image (sizes from config.CROP_SIZES,
optionally rotated) and annotated from the SMPL-X fits:

- person bbox and amodal segmentation (COCO RLE) from the AGORA masks
- 11 head keypoints and 31 body keypoints, as 2D (x, y, v) and in unified
  coordinates (x, y, z, v): image X, pseudo depth, flipped image Y
- head bbox and body reference bbox (from the keypoints, may extend out of frame)
- part mask teachers (head / torso / limbs / hands) and a depth teacher from a
  per-person SMPL-X render, and an occlusion mask teacher from the AGORA masks
- subject_id (scan identity for Re-ID), kid flag and occlusion percentage
- matched_ins_id: the RTMDet-Ins detection matched to each person
  (crops in which RTMDet-Ins matches nobody are retried)

Each crop is saved twice: original and horizontally mirrored.
--crops-per-image sets the number of crops per source image.

Inputs (--agora-dir; default $AGORA_DIR or <repo>/datasets/agora), per split
(train, validation):
    {split}/SMPLX/{split}_*_withjv.pkl
        AGORA SMPL-X camera DataFrames including the gt_joints_2d, gt_joints_3d
        and gt_verts columns (added by the project_joints step of the AGORA
        evaluation toolkit, https://github.com/pixelite1201/agora_evaluation).
        Columns used: imgPath, isValid, gt_joints_2d, gt_joints_3d, gt_verts,
        and gt_path_smplx / kid / occlusion when present.
    {split}/images4K/ (--resolution 4K) or {split}/images/ (720p, *_1280x720.png)
        source images; train images may sit in numbered subfolders 0-9
    {split}/masks4K/ (4K) or {split}/masks/ (720p)
        per-scene folders with the per-person masks and the integrated mask (_00000)

Also required, not distributed with this repository (obtain them under their
own licenses):
    weights/rtmdet-ins_s.pth                     RTMDet-Ins-S COCO weights (MMDetection)
    weights/smplx/SMPLX_NEUTRAL.npz              SMPL-X model (https://smpl-x.is.tue.mpg.de)
    weights/smplx/smplx_vert_segmentation.json   SMPL-X vertex part segmentation (Meshcapade)
Without the SMPL-X assets the part mask and depth teachers are left empty.

Outputs (under data/, which is cleared first):
    {split}/images/[N/]<crop>.jpg, <crop>_m.jpg   crops (train keeps the source subfolder N)
    {split}/depth/[N/]<crop>_<k>.png              depth teachers (uint16 millimetres)
    head_keypoints_{split}.json                   COCO-style annotations

Usage:
    uv run python tools/agora/extracter.py                          # 4K, both splits
    uv run python tools/agora/extracter.py --agora-dir /path/to/agora
    uv run python tools/agora/extracter.py --resolution 720p
    uv run python tools/agora/extracter.py --split train --max-files 3

Press 'q' during extraction to stop safely (processed data is saved).
"""

import argparse
import copy
import json
import random
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from multiprocessing import Manager, Pool, cpu_count, Event as MPEvent
from pathlib import Path
from typing import Optional

# Keyboard input (Windows)
if sys.platform == "win32":
    import msvcrt
else:
    import select

# Add the script directory to the import path
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    AGORA_DIR,
    CROPPED_DIR,
    OUTPUT_DIR,
    CROPS_PER_IMAGE,
    MAX_CROP_RETRIES,
    RANDOM_SEED,
    KEYPOINT_NAMES,
    SKELETON,
    BODY_KEYPOINT_NAMES,
    resolution_config,
    NO_INS_MATCH_SKIP_PROB,
)
from configs.default import RTMDET_INS_S_WEIGHTS
from utils import (
    save_image_jpeg,
    get_jpeg_encoder_name,
    get_focal_length,
    get_image_path,
    find_image_subdir,
    try_single_crop,
    select_crop_params,
    run_ins_inference_and_match,
    check_no_ins_matched_persons,
    mirror_image_array,
    mirror_annotations,
    GPUImageProcessor,
    get_image_processor_name,
)
from utils.rtmdet_ins import RTMDetInsInference


def setup_random_seed():
    """Set the torch random seeds (RTMDet-Ins inference)"""
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Performance over determinism (deterministic mode disabled)
    torch.backends.cudnn.benchmark = True


def log_worker_error(
    worker_id: int,
    error: Exception,
    context: dict = None,
) -> Path:
    """Write a worker error to a log file

    Args:
        worker_id: worker ID
        error: the exception
        context: extra context (pkl_path, row_idx, ...)

    Returns:
        path of the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = CROPPED_DIR / "error_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{worker_id}_{timestamp}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Worker {worker_id} Error Log\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        if context:
            f.write("Context:\n")
            for key, value in context.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write(f"Exception Type: {type(error).__name__}\n")
        f.write(f"Exception Message: {str(error)}\n\n")

        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        f.write("\n")

    return log_path


def keyboard_listener(stop_event: MPEvent, display_event: threading.Event) -> None:
    """Watch the keyboard and send the stop signal on 'q'

    Args:
        stop_event: multiprocessing Event that stops the workers
        display_event: thread Event that ends this listener
    """
    print("\n[INFO] Press 'q' to stop safely (processed data is saved)")

    while not display_event.is_set():
        try:
            if sys.platform == "win32":
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                    if key == "q":
                        print("\n[STOP] 'q' pressed. Stopping workers...")
                        stop_event.set()
                        return
                time.sleep(0.1)
            else:
                # Unix/Linux/Mac
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == "q":
                        print("\n[STOP] 'q' pressed. Stopping workers...")
                        stop_event.set()
                        return
        except Exception:
            pass  # ignore keyboard input errors


# Per-worker RTMDet-Ins instance
_worker_rtmdet_ins: RTMDetInsInference = None


def get_worker_rtmdet_ins() -> RTMDetInsInference:
    """Return this worker's RTMDet-Ins instance"""
    global _worker_rtmdet_ins
    return _worker_rtmdet_ins


def init_worker_rtmdet_ins(weights_path: str = None):
    """Initialize RTMDet-Ins in this worker"""
    global _worker_rtmdet_ins
    if _worker_rtmdet_ins is not None:
        return

    setup_random_seed()

    if weights_path is None and RTMDET_INS_S_WEIGHTS.exists():
        weights_path = str(RTMDET_INS_S_WEIGHTS)

    if weights_path is None:
        print("Warning: RTMDet-Ins weights not found, skipping Ins inference")
        return

    try:
        _worker_rtmdet_ins = RTMDetInsInference(weights_path)
    except Exception as e:
        print(f"Warning: Failed to load RTMDet-Ins: {e}")


def process_pkl_file(
    pkl_path: Path,
    split: str,
    start_image_id: int,
    start_anno_id: int,
    crops_per_image: int = CROPS_PER_IMAGE,
    silent: bool = False,
    tqdm_position: int = 0,
    progress_callback: Optional[callable] = None,
    stop_check: Optional[callable] = None,
) -> tuple[list, list, int, int, int, bool]:
    """
    Process one pkl file.

    For every row (source image), random crops are cut, saved (original and
    mirrored) and annotated.

    Returns:
        images: image info list
        annotations: annotation list
        next_image_id: next image ID
        next_anno_id: next annotation ID
        ignore_count: persons marked ignore (head bbox too small / head out of frame)
        was_stopped: whether processing was interrupted by the stop signal
    """
    with open(pkl_path, "rb") as f:
        data = pd.compat.pickle_compat.load(f)

    images = []
    annotations = []
    image_id = start_image_id
    anno_id = start_anno_id
    ignore_count = 0

    iterator = range(len(data))
    if not silent:
        iterator = tqdm(
            iterator,
            desc=f"  {pkl_path.name}",
            leave=False,
            position=tqdm_position,
        )

    # Row-level error context
    row_context = {
        "pkl_path": str(pkl_path),
        "row_idx": None,
        "img_name": None,
        "crops_done": 0,
    }

    was_stopped = False

    for row_idx in iterator:
        # Stop signal (checked per row)
        if stop_check is not None and stop_check():
            was_stopped = True
            break

        row_context["row_idx"] = row_idx
        row_context["crops_done"] = 0

        try:
            row = data.iloc[row_idx]

            # Source image path
            img_name = row["imgPath"]
            row_context["img_name"] = img_name
            subdir = find_image_subdir(img_name, split) if split == "train" else None
            img_path = get_image_path(img_name, split, subdir)

            if not img_path.exists():
                if progress_callback is not None:
                    progress_callback()
                continue

            # Focal length of the scene
            focal_length = get_focal_length(img_name)

            # Base file name (without the resolution suffix)
            stem = img_path.stem
            if resolution_config.image_suffix and stem.endswith(resolution_config.image_suffix):
                stem = stem[:-len(resolution_config.image_suffix)]

            crops_done = 0
            crop_retries = 0  # reset on success

            # Read the source image once (several crops per image)
            source_img = cv2.imread(str(img_path))
            if source_img is None:
                if progress_callback is not None:
                    progress_callback()
                continue

            # GPU image processor (fast repeated crops of the same image)
            gpu_processor = GPUImageProcessor(source_img)

            # Row cache (masks, SMPL-X renders, ... shared between crop attempts)
            row_cache = {}

            while crops_done < crops_per_image:
                row_context["crops_done"] = crops_done

                crop_width, crop_height, angle_deg = select_crop_params()

                result = try_single_crop(
                    source_img=source_img,
                    row=row,
                    img_name=img_name,
                    split=split,
                    focal_length=focal_length,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    angle_deg=angle_deg,
                    image_id=image_id,
                    anno_id=anno_id,
                    force_valid_person=True,
                    gpu_processor=gpu_processor,
                    row_cache=row_cache,
                )

                if not result.success:
                    crop_retries += 1
                    # Too many retries: give up on this crop
                    if crop_retries >= MAX_CROP_RETRIES:
                        crops_done += 1
                        crop_retries = 0
                        continue
                    # No valid crop position (720p: the crop size cannot fit)
                    if result.skip_reason == "position_not_found":
                        if resolution_config.resolution == "720p":
                            crops_done += 1
                            crop_retries = 0
                        continue
                    # Other failures (incl. no_valid_person): retry
                    continue

                # RTMDet-Ins inference and matching
                result = run_ins_inference_and_match(result, get_worker_rtmdet_ins())

                # No person matched by RTMDet-Ins: retry with probability NO_INS_MATCH_SKIP_PROB
                if check_no_ins_matched_persons(result) and random.random() < NO_INS_MATCH_SKIP_PROB:
                    crop_retries += 1
                    if crop_retries >= MAX_CROP_RETRIES:
                        crops_done = crops_per_image
                        crop_retries = 0
                    continue

                # File name: source stem, '@', crop size, centre and rotation
                cx = result.image_info["cx"]
                cy = result.image_info["cy"]
                angle_suffix = f"_r{int(angle_deg)}" if angle_deg != 0 else ""
                base_filename = f"{stem}@{crop_width}x{crop_height}_{int(cx)}_{int(cy)}{angle_suffix}"

                # Save the depth teachers as PNG and reference them from the annotations.
                # The mirrored annotations share the same file (the dataset flips
                # it using the image's mirrored flag).
                depth_rel_dir = f"depth/{subdir}" if (split == "train" and subdir is not None) else "depth"
                for k, ann in enumerate(result.annotations):
                    depth_arr = ann.pop("_depth_teacher", None)
                    if depth_arr is not None:
                        depth_filename = f"{base_filename}_{k}.png"
                        depth_path = CROPPED_DIR / split / depth_rel_dir / depth_filename
                        depth_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(depth_path), depth_arr)
                        ann["depth_file"] = f"{depth_rel_dir}/{depth_filename}"
                    else:
                        ann["depth_file"] = None

                # Save the original and the mirrored version
                for is_mirrored in [False, True]:
                    if is_mirrored:
                        cropped_filename = f"{base_filename}_m.jpg"
                        cropped_img = mirror_image_array(result.cropped_img)
                        crop_annotations = copy.deepcopy(result.annotations)
                        mirror_annotations(crop_annotations, crop_width)
                    else:
                        cropped_filename = f"{base_filename}.jpg"
                        cropped_img = result.cropped_img
                        crop_annotations = result.annotations

                    if split == "train" and subdir is not None:
                        cropped_rel_path = f"{subdir}/{cropped_filename}"
                        cropped_path = CROPPED_DIR / split / "images" / str(subdir) / cropped_filename
                    else:
                        cropped_rel_path = cropped_filename
                        cropped_path = CROPPED_DIR / split / "images" / cropped_filename

                    image_info = copy.deepcopy(result.image_info)
                    image_info["file_name"] = cropped_rel_path
                    image_info["id"] = image_id
                    image_info["mirrored"] = is_mirrored

                    images.append(image_info)

                    for ann in crop_annotations:
                        ann["id"] = anno_id
                        ann["image_id"] = image_id
                        anno_id += 1

                    # Save the crop (JPEG)
                    save_image_jpeg(cropped_img, cropped_path)

                    annotations.extend(crop_annotations)

                    image_id += 1

                # Statistics (original only; the mirror shows the same persons)
                ignore_count += result.ignore_count

                crops_done += 1
                crop_retries = 0

            # Release GPU memory
            gpu_processor.release()

            # One row (source image) done
            if progress_callback is not None:
                progress_callback()

        except Exception as e:
            # Log the row-level error (worker ID unknown here: -1)
            error_context = {
                **row_context,
                "split": split,
                "image_id": image_id,
                "anno_id": anno_id,
            }
            log_path = log_worker_error(-1, e, error_context)
            print(f"\n[process_pkl_file] ERROR at row {row_idx}: {type(e).__name__}: {e}")
            print(f"[process_pkl_file] Error log saved to: {log_path}")
            raise

    return images, annotations, image_id, anno_id, ignore_count, was_stopped


def save_partial_annotations(
    images: list,
    annotations: list,
    split: str,
    worker_id: int,
    pkl_idx: int,
    ignore_count: int,
) -> Path:
    """
    Save partial annotations to a file.

    Args:
        images: image info list
        annotations: annotation list
        split: train or validation
        worker_id: worker ID
        pkl_idx: pkl file index
        ignore_count: number of ignored persons

    Returns:
        path of the saved file
    """
    partial_dir = OUTPUT_DIR / "partial" / split
    partial_dir.mkdir(parents=True, exist_ok=True)

    partial_path = partial_dir / f"w{worker_id}_p{pkl_idx}.json"

    partial_data = {
        "images": images,
        "annotations": annotations,
        "ignore_count": ignore_count,
    }

    with open(partial_path, "w") as f:
        json.dump(partial_data, f)

    return partial_path


def process_worker(args: tuple) -> tuple[list[Path], int, bool]:
    """
    Worker function (multiprocessing).

    Args:
        args: (pkl_files, split, worker_id, resolution, agora_dir,
               progress_queue, stop_event, crops_per_image)
              progress_queue: progress queue (may be None)
              stop_event: stop signal Event (may be None)

    Returns:
        (partial_file_paths, total_ignore_count, was_stopped)
    """
    (pkl_files, split, worker_id, resolution, agora_dir,
     progress_queue, stop_event, crops_per_image) = args

    # Error context
    current_context = {
        "worker_id": worker_id,
        "split": split,
        "resolution": resolution,
        "pkl_files_count": len(pkl_files),
        "current_pkl": None,
        "current_pkl_idx": None,
        "processed_rows": 0,
        "total_rows": 0,
    }

    was_stopped = False

    try:
        # Per-process source settings (not inherited by spawned processes)
        resolution_config.init(resolution, agora_dir)

        init_worker_rtmdet_ins()

        # Different seed per worker (reproducibility)
        random.seed(RANDOM_SEED + worker_id)

        partial_files = []
        image_id = 1
        anno_id = 1
        total_ignore_count = 0

        # Count all rows up front (progress display)
        total_rows = 0
        if progress_queue is not None:
            for pkl_path in pkl_files:
                with open(pkl_path, "rb") as f:
                    data = pd.compat.pickle_compat.load(f)
                    total_rows += len(data)
            current_context["total_rows"] = total_rows
            try:
                progress_queue.put((worker_id, 0, total_rows))
            except Exception:
                pass  # ignore queue errors

        # Per-row progress callback
        processed_rows = [0]  # list for mutation from the closure

        def on_row_complete():
            processed_rows[0] += 1
            current_context["processed_rows"] = processed_rows[0]
            if progress_queue is not None:
                try:
                    progress_queue.put((worker_id, processed_rows[0], total_rows))
                except Exception:
                    pass

        def should_stop() -> bool:
            if stop_event is not None:
                return stop_event.is_set()
            return False

        for pkl_idx, pkl_path in enumerate(pkl_files):
            if should_stop():
                was_stopped = True
                break

            current_context["current_pkl"] = str(pkl_path)
            current_context["current_pkl_idx"] = pkl_idx

            images, annotations, image_id, anno_id, ic, stopped = process_pkl_file(
                pkl_path, split, image_id, anno_id,
                crops_per_image=crops_per_image,
                silent=True,
                progress_callback=on_row_complete if progress_queue else None,
                stop_check=should_stop
            )

            # Save as a partial file to free memory
            if images:
                partial_path = save_partial_annotations(
                    images, annotations, split, worker_id, pkl_idx, ic
                )
                partial_files.append(partial_path)

            total_ignore_count += ic

            del images
            del annotations

            if stopped:
                was_stopped = True
                break

        return partial_files, total_ignore_count, was_stopped

    except Exception as e:
        log_path = log_worker_error(worker_id, e, current_context)
        print(f"\n[Worker {worker_id}] ERROR: {type(e).__name__}: {e}")
        print(f"[Worker {worker_id}] Error log saved to: {log_path}")
        raise


def progress_monitor(progress_queue, n_workers: int, stop_event: threading.Event) -> None:
    """
    Progress monitor thread: receives worker progress and displays it.

    Args:
        progress_queue: progress queue
        n_workers: number of workers
        stop_event: stop signal
    """
    worker_progress = {}  # worker_id -> (processed, total)

    while not stop_event.is_set():
        try:
            worker_id, processed, total = progress_queue.get(timeout=0.5)
            worker_progress[worker_id] = (processed, total)
            display_progress(worker_progress, n_workers)
        except Exception:
            # Timeout or empty queue
            pass

    display_progress(worker_progress, n_workers, final=True)


def display_progress(worker_progress: dict, n_workers: int, final: bool = False) -> None:
    """
    Display per-worker progress in percent.

    Args:
        worker_progress: worker_id -> (processed, total)
        n_workers: number of workers
        final: final display
    """
    parts = []
    total_processed = 0
    total_images = 0

    for worker_id in range(n_workers):
        if worker_id in worker_progress:
            processed, total = worker_progress[worker_id]
            if total > 0:
                pct = processed / total * 100
            else:
                pct = 0
            parts.append(f"W{worker_id}:{pct:5.1f}%")
            total_processed += processed
            total_images += total
        else:
            parts.append(f"W{worker_id}:  ---")

    if total_images > 0:
        overall_pct = total_processed / total_images * 100
    else:
        overall_pct = 0

    progress_str = " | ".join(parts) + f" || Total:{overall_pct:5.1f}%"

    # Overwrite the current line
    sys.stdout.write(f"\r  {progress_str}")
    sys.stdout.flush()

    if final:
        sys.stdout.write("\n")
        sys.stdout.flush()


def merge_partial_files(
    partial_files: list[Path],
    output_path: Path,
    split: str,
) -> tuple[int, int, int]:
    """
    Merge the partial files into the final JSON.

    Args:
        partial_files: partial file paths
        output_path: final output path
        split: train or validation

    Returns:
        (n_images, n_annotations, total_ignore)
    """
    merged_images = []
    merged_annotations = []
    total_ignore = 0

    global_image_id = 1
    global_anno_id = 1

    # Sorted by file name for a stable order
    for partial_path in sorted(partial_files):
        with open(partial_path, "r") as f:
            data = json.load(f)

        images = data["images"]
        annotations = data["annotations"]

        # Local ID -> global ID
        image_id_map = {}

        for img in images:
            old_id = img["id"]
            img["id"] = global_image_id
            image_id_map[old_id] = global_image_id
            global_image_id += 1

        for ann in annotations:
            ann["id"] = global_anno_id
            ann["image_id"] = image_id_map[ann["image_id"]]
            global_anno_id += 1

        merged_images.extend(images)
        merged_annotations.extend(annotations)
        total_ignore += data["ignore_count"]

        partial_path.unlink()

    # COCO-style output
    coco_data = {
        "info": {
            "description": f"RTMDet-Fusion training data extracted from AGORA ({split})",
            "version": "1.0",
            "year": 2021,
            "contributor": "AGORA + extraction script",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {"id": 1, "name": "AGORA License", "url": "https://agora.is.tue.mpg.de"}
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": KEYPOINT_NAMES,
                "body_keypoints": BODY_KEYPOINT_NAMES,
                "skeleton": [list(pair) for pair in SKELETON],
            }
        ],
        "images": merged_images,
        "annotations": merged_annotations,
    }

    with open(output_path, "w") as f:
        json.dump(coco_data, f)

    return len(merged_images), len(merged_annotations), total_ignore


def main():
    parser = argparse.ArgumentParser(
        description="Extract RTMDet-Fusion training data (random crops) from AGORA"
    )
    parser.add_argument(
        "--agora-dir",
        type=Path,
        default=AGORA_DIR,
        help=f"AGORA dataset root (default: {AGORA_DIR})",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "both"],
        default="both",
        help="split(s) to process",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="maximum number of pkl files per split (debugging)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--resolution",
        choices=["4K", "720p"],
        default="4K",
        help="source resolution (default: 4K)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the per-worker progress display",
    )
    parser.add_argument(
        "--crops-per-image",
        type=int,
        default=CROPS_PER_IMAGE,
        help=f"crops per source image (default: {CROPS_PER_IMAGE})",
    )
    args = parser.parse_args()

    # Pre-flight checks (before the output directory is cleared)
    agora_dir = args.agora_dir.expanduser().resolve()
    if not agora_dir.is_dir():
        sys.exit(
            f"Error: AGORA directory not found: {agora_dir}\n"
            "Pass --agora-dir or set the AGORA_DIR environment variable."
        )
    if not RTMDET_INS_S_WEIGHTS.exists():
        sys.exit(
            f"Error: RTMDet-Ins weights not found: {RTMDET_INS_S_WEIGHTS}\n"
            "Every crop needs an RTMDet-Ins match, so extraction would produce no data."
        )

    n_workers = args.workers if args.workers else cpu_count()

    resolution_config.init(args.resolution, agora_dir)

    print("=" * 60)
    print("AGORA extraction (random crops)")
    print("=" * 60)

    print(f"AGORA directory: {agora_dir}")
    print(f"Resolution: {resolution_config.resolution} ({resolution_config.source_width}x{resolution_config.source_height})")
    print(f"  images: {resolution_config.images_dir}")
    print(f"  masks: {resolution_config.masks_dir}")
    print(f"Keypoints: {len(KEYPOINT_NAMES)} head + {len(BODY_KEYPOINT_NAMES)} body")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Workers: {n_workers}")
    print(f"Crops per image: {args.crops_per_image} (x2: original + mirror)")
    print(f"Image processing: {get_image_processor_name()}")
    print(f"JPEG encoder: {get_jpeg_encoder_name()}, quality 60")
    valid_sizes = resolution_config.get_valid_crop_sizes()
    print(f"Valid crop sizes: {len(valid_sizes)}")
    for cs in valid_sizes:
        print(f"  - {cs['width']}x{cs['height']}")

    # Reset the output directory
    print("\nResetting the output directory...")
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  created: {OUTPUT_DIR}")
    for s in ["train", "validation"]:
        split_dir = CROPPED_DIR / s
        if split_dir.exists():
            shutil.rmtree(split_dir)
            print(f"  removed: {split_dir}")
    for s in ["train", "validation"]:
        json_file = OUTPUT_DIR / f"head_keypoints_{s}.json"
        if json_file.exists():
            json_file.unlink()
            print(f"  removed: {json_file}")
    partial_dir = OUTPUT_DIR / "partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
        print(f"  removed: {partial_dir}")

    splits = ["train", "validation"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\nProcessing: {split}")

        smplx_dir = agora_dir / split / "SMPLX"
        pkl_files = sorted(smplx_dir.glob(f"{split}_*_withjv.pkl"))

        if args.max_files:
            pkl_files = pkl_files[:args.max_files]

        print(f"  pkl files: {len(pkl_files)}")

        if len(pkl_files) == 0:
            print(f"  skipped (no {split}_*_withjv.pkl in {smplx_dir})")
            continue

        # No more workers than files
        actual_workers = min(n_workers, len(pkl_files))
        print(f"  workers used: {actual_workers}")

        # Distribute the pkl files round-robin
        chunks = [[] for _ in range(actual_workers)]
        for i, pkl_path in enumerate(pkl_files):
            chunks[i % actual_workers].append(pkl_path)

        manager = Manager()
        stop_event = manager.Event()  # stops the workers (multiprocessing)
        display_stop_event = threading.Event()  # stops the display threads

        any_worker_stopped = False

        if args.no_progress:
            worker_args = [
                (chunk, split, worker_id, args.resolution, agora_dir,
                 None, stop_event, args.crops_per_image)
                for worker_id, chunk in enumerate(chunks)
            ]

            print("  running workers... (no progress display)")

            keyboard_thread = threading.Thread(
                target=keyboard_listener,
                args=(stop_event, display_stop_event),
                daemon=True,
            )
            keyboard_thread.start()

            with Pool(actual_workers) as pool:
                results = pool.map(process_worker, worker_args)

            display_stop_event.set()
            keyboard_thread.join(timeout=1.0)

            print("  workers finished")
        else:
            progress_queue = manager.Queue()

            worker_args = [
                (chunk, split, worker_id, args.resolution, agora_dir,
                 progress_queue, stop_event, args.crops_per_image)
                for worker_id, chunk in enumerate(chunks)
            ]

            print("  running workers...")

            monitor_thread = threading.Thread(
                target=progress_monitor,
                args=(progress_queue, actual_workers, display_stop_event),
                daemon=True,
            )
            monitor_thread.start()

            keyboard_thread = threading.Thread(
                target=keyboard_listener,
                args=(stop_event, display_stop_event),
                daemon=True,
            )
            keyboard_thread.start()

            with Pool(actual_workers) as pool:
                results = pool.map(process_worker, worker_args)

            # Stop the threads
            time.sleep(0.5)  # grace period for the last updates
            display_stop_event.set()
            monitor_thread.join(timeout=2.0)
            keyboard_thread.join(timeout=1.0)

            print("  workers finished")

        # Collect the partial files
        all_partial_files = []
        total_ignore = 0

        for partial_files, ic, was_stopped in results:
            all_partial_files.extend(partial_files)
            total_ignore += ic
            if was_stopped:
                any_worker_stopped = True

        if any_worker_stopped:
            print("  [STOP] Workers stopped by signal. Saving the processed data.")

        # Merge into the final JSON
        print(f"  merging partial files... ({len(all_partial_files)} files)")
        output_path = OUTPUT_DIR / f"head_keypoints_{split}.json"

        n_images, n_annotations, _ = merge_partial_files(
            all_partial_files, output_path, split
        )

        partial_dir = OUTPUT_DIR / "partial" / split
        if partial_dir.exists():
            shutil.rmtree(partial_dir)

        print(f"  saved: {output_path}")
        print(f"  crops: {CROPPED_DIR / split / 'images'}")
        print(f"  images: {n_images}, annotations: {n_annotations}")
        if total_ignore > 0:
            print(f"  ignore=True (head bbox too small or head out of frame): {total_ignore}")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
