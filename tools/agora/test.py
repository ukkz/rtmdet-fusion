#!/usr/bin/env python3
"""
AGORA extraction preview: extract from the pkl files on the fly and display.

Picks a random pkl row, runs the same extraction code as extracter.py
(config / utils) and shows the result, so the extraction logic can be checked
visually without generating a dataset.

Data sources (--agora-dir; default $AGORA_DIR or <repo>/datasets/agora):
    - pkl files: {split}/SMPLX/{split}_*_withjv.pkl
    - images:    {split}/images4K/ (or images/ for 720p)
    - masks:     {split}/masks4K/ (or masks/ for 720p)
See extracter.py for the full input layout and the required model assets.

Keys:
    Enter/Space: extract and show the next random image
    R: extract with a forced rotation
    X: toggle ignore of the selected person
    V: cycle the view mode (head / body / parts / depth / occlusion)
    Click: select a person, or toggle the RTMDet-Ins mask layer
    Q/ESC: quit

Usage:
    uv run python tools/agora/test.py                     # 4K, train + validation
    uv run python tools/agora/test.py --split train       # train only
    uv run python tools/agora/test.py --resolution 720p   # 720p sources
    uv run python tools/agora/test.py --agora-dir /path/to/agora
"""

import argparse
import random
import sys
from pathlib import Path

# Add the script directory to the import path
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Import torch before PyQt5 (avoids a DLL conflict on Windows)
import torch  # noqa: F401

import cv2
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QScrollArea, QShortcut, QStatusBar
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from config import (
    AGORA_DIR,
    KEYPOINT_NAMES,
    MAX_CROP_RETRIES,
    RANDOM_SEED,
    SKELETON,
    resolution_config,
    NO_INS_MATCH_SKIP_PROB,
)
from utils import (
    get_focal_length,
    get_image_path,
    find_image_subdir,
    try_single_crop,
    select_crop_params,
    run_ins_inference_and_match,
    check_no_ins_matched_persons,
    ImageWidget,
    update_3d_views,
    build_info_lines,
)
from utils.rtmdet_ins import get_rtmdet_ins_inference
from utils.matching import get_unmatched_gt_indices


class MainWindow(QMainWindow):
    """Main window"""

    def __init__(self, split: str = None, resolution: str = "4K", agora_dir: Path = AGORA_DIR):
        super().__init__()
        self.split = split
        self.resolution = resolution

        resolution_config.init(resolution, agora_dir)

        random.seed(RANDOM_SEED)

        # RTMDet-Ins (loaded lazily)
        self.rtmdet_ins = None
        self._rtmdet_init_attempted = False

        self._load_pkl_files()

        self._init_ui()

        # First image
        self._extract_and_display_random()

    def _load_pkl_files(self):
        """Collect the pkl file paths"""
        if self.split is not None:
            splits_to_load = [self.split]
        else:
            splits_to_load = ["train", "validation"]

        self.pkl_files = []  # [(split, pkl_path), ...]

        for split in splits_to_load:
            smplx_dir = resolution_config.agora_dir / split / "SMPLX"
            if not smplx_dir.exists():
                print(f"Warning: {smplx_dir} not found, skipping")
                continue

            pkl_files = sorted(smplx_dir.glob(f"{split}_*_withjv.pkl"))
            for pkl_path in pkl_files:
                self.pkl_files.append((split, pkl_path))
            print(f"{split}: {len(pkl_files)} pkl files")

        if not self.pkl_files:
            raise FileNotFoundError(
                f"no {{split}}/SMPLX/{{split}}_*_withjv.pkl under {resolution_config.agora_dir}"
            )

        print(f"Loaded: {len(self.pkl_files)} pkl files in total")

    def _init_ui(self):
        """Build the UI"""
        split_text = self.split if self.split else "train+validation"
        self.setWindowTitle(f"AGORA Extraction Test - {split_text} ({self.resolution})")

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.image_widget = ImageWidget()
        layout.addWidget(self.image_widget, 1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(5)

        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, 1)

        self.person_3d_widgets = []

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Enter/Space: next image | R: force rotation | X: toggle ignore | "
            "V: view mode | Click: layer | Q: quit"
        )

        self.current_annotations = []
        self.current_img = None
        self.selected_person_idx = None
        self.current_crop_info = {}
        self.current_match_results = []  # matching results
        self.current_ins_masks = []  # predicted masks
        self.current_gt_count = 0  # number of GT masks

        self.image_widget.clicked.connect(self._on_image_clicked)
        self.image_widget.inference_layer_toggled.connect(self._on_inference_layer_toggled)
        self.image_widget.view_mode_changed.connect(self._on_view_mode_changed)

        QShortcut(QKeySequence(Qt.Key_Space), self, self._extract_and_display_random)
        QShortcut(QKeySequence(Qt.Key_Return), self, self._extract_and_display_random)
        QShortcut(QKeySequence(Qt.Key_R), self, self._extract_and_display_with_rotation)
        QShortcut(QKeySequence(Qt.Key_X), self, self._on_toggle_ignored)
        QShortcut(QKeySequence(Qt.Key_V), self, self.image_widget.cycle_view_mode)
        QShortcut(QKeySequence(Qt.Key_Q), self, self.close)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)

    def _on_view_mode_changed(self, mode: str):
        """View mode changed: update the status bar and the 3D views"""
        mode_desc = {
            "head": "head keypoints",
            "body": "31 body keypoints + reference bbox + subject_id",
            "parts": "naked-body part masks (SMPL-X teacher, amodal)",
            "depth": "depth teacher (TURBO, red = near)",
            "occlusion": "occlusion teacher (red = hidden region, over the amodal mask)",
        }
        self.status_bar.showMessage(f"[V] view mode: {mode.upper()} - {mode_desc.get(mode, '')}")
        self._update_3d_views(self.current_annotations)

    def _extract_and_display_random(self):
        """Extract and show a random image"""
        self._extract_and_display(force_rotation=False)

    def _extract_and_display_with_rotation(self):
        """Extract and show with a forced rotation"""
        self._extract_and_display(force_rotation=True)

    def _extract_and_display(self, force_rotation: bool = False):
        """Extract a random pkl row and show it (same logic as extracter.py)"""
        # Retry until success (like the extracter loop)
        for attempt in range(MAX_CROP_RETRIES):
            split, pkl_path = random.choice(self.pkl_files)

            with open(pkl_path, "rb") as f:
                data = pd.compat.pickle_compat.load(f)

            if len(data) == 0:
                continue

            row_idx = random.randint(0, len(data) - 1)
            row = data.iloc[row_idx]

            # Source image path
            img_name = row["imgPath"]
            subdir = find_image_subdir(img_name, split) if split == "train" else None
            img_path = get_image_path(img_name, split, subdir)

            if not img_path.exists():
                continue

            focal_length = get_focal_length(img_name)

            source_img = cv2.imread(str(img_path))
            if source_img is None:
                continue

            crop_width, crop_height, angle_deg = select_crop_params(force_rotation)

            # Crop attempt (original only; mirroring is done by extracter when saving)
            result = try_single_crop(
                source_img=source_img,
                row=row,
                img_name=img_name,
                split=split,
                focal_length=focal_length,
                crop_width=crop_width,
                crop_height=crop_height,
                angle_deg=angle_deg,
                image_id=0,
                anno_id=0,
                force_valid_person=True,   # retry without a valid person, like extracter
            )

            if not result.success:
                continue

            # RTMDet-Ins inference and matching (loaded on first use)
            if not self._rtmdet_init_attempted:
                self._rtmdet_init_attempted = True
                self.rtmdet_ins = get_rtmdet_ins_inference()
                if self.rtmdet_ins:
                    print("RTMDet-Ins ready for inference")
                else:
                    print("RTMDet-Ins not available (click layer toggle disabled)")

            result = run_ins_inference_and_match(result, self.rtmdet_ins)

            # No person matched by RTMDet-Ins: retry with probability NO_INS_MATCH_SKIP_PROB
            if check_no_ins_matched_persons(result) and random.random() < NO_INS_MATCH_SKIP_PROB:
                continue

            # Success: keep for display
            cx = result.image_info["cx"]
            cy = result.image_info["cy"]
            mirrored = result.image_info["mirrored"]

            self.current_annotations = result.annotations
            self.current_img = result.cropped_img.copy()
            self.current_match_results = result.match_results
            self.current_ins_masks = result.ins_masks
            self.selected_person_idx = None
            self.current_crop_info = {
                "split": split,
                "pkl": pkl_path.name,
                "row_idx": row_idx,
                "img_name": img_name,
                "crop_size": f"{crop_width}x{crop_height}",
                "angle": angle_deg,
                "cx": cx,
                "cy": cy,
                "mirrored": mirrored,
                "focal_scaled": result.image_info.get("focal_length"),
            }

            self._display_current_image()

            valid_count = sum(1 for ann in result.annotations if not ann["ignore"])
            mirror_text = " [MIRRORED]" if mirrored else ""
            self.status_bar.showMessage(
                f"[{split}] {pkl_path.name} row={row_idx} | "
                f"Size: {crop_width}x{crop_height} | Angle: {angle_deg:.1f}°{mirror_text} | "
                f"Persons: {len(result.annotations)} (valid={valid_count}, "
                f"ignore={result.ignore_count})"
            )
            return

        # All attempts failed
        self.status_bar.showMessage("Extraction failed (maximum number of attempts reached)")

    def _display_current_image(self, selected_idx: int = None):
        """Show the current image and annotations"""
        if self.current_img is None:
            return

        self.selected_person_idx = selected_idx

        self.image_widget.set_image(self.current_img)

        # Pass the precomputed RTMDet-Ins masks to the widget
        self._run_rtmdet_inference()

        # Info text
        info = self.current_crop_info
        info_lines = build_info_lines(
            split=info.get("split"),
            pkl_name=info.get("pkl"),
            row_idx=info.get("row_idx"),
            crop_size=info.get("crop_size"),
            angle=info.get("angle"),
            mirrored=info.get("mirrored"),
            cx=info.get("cx"),
            cy=info.get("cy"),
            annotations=self.current_annotations,
            shortcuts="[Enter/Space: Next] [R: Rotate] [X: Toggle Ignore] [V: View Mode] [Click: Layer] [Q: Quit]",
        )

        # Teacher statistics line
        anns = self.current_annotations
        n_body = sum(1 for a in anns if a.get("keypoints_body") is not None)
        n_parts = sum(1 for a in anns if a.get("part_masks"))
        n_depth = sum(1 for a in anns if a.get("_depth_teacher") is not None)
        n_occ_valid = sum(1 for a in anns if a.get("occlusion_valid"))
        info_lines.append((
            f"teachers: body_kpts={n_body} parts={n_parts} depth={n_depth} "
            f"occ_teacher={n_occ_valid}/{len(anns)}",
            0.5, (150, 220, 255),
        ))

        # Depth teachers (still arrays in the annotations right after try_single_crop)
        # converted to pelvis-relative pseudo depth (same scale as the body keypoint depth)
        from utils.render import pseudo_depth_from_teacher
        focal_scaled = float(self.current_crop_info.get("focal_scaled", 0) or 0)
        depth_maps = []
        for a in anns:
            raw = a.get("_depth_teacher")
            if raw is not None and focal_scaled > 0:
                depth_maps.append(
                    pseudo_depth_from_teacher(raw, a.get("pelvis_depth"), focal_scaled)
                )
            else:
                depth_maps.append(None)

        self.image_widget.set_annotations(
            self.current_annotations, selected_idx, info_lines, depth_maps=depth_maps
        )

        self._update_3d_views(self.current_annotations)

    def _run_rtmdet_inference(self):
        """Pass the precomputed predicted masks and matching results to the ImageWidget"""
        if self.current_img is None:
            return

        # Number of GT masks (display)
        gt_count = sum(
            1 for anno in self.current_annotations
            if "segmentation" in anno and anno["segmentation"] is not None
        )
        self.current_gt_count = gt_count

        # Log unmatched GT annotations
        if self.current_match_results:
            unmatched_gt = get_unmatched_gt_indices(self.current_match_results, gt_count)
            if unmatched_gt:
                print(f"Unmatched GT annotations: {unmatched_gt}")

        # Inference already ran in _extract_and_display
        self.image_widget.set_inference_masks(self.current_ins_masks, self.current_match_results)

    def _on_inference_layer_toggled(self, visible: bool):
        """Inference layer toggled"""
        if visible:
            # Matching summary
            n_ins = len(self.current_match_results)
            n_matched = sum(1 for r in self.current_match_results if r.matched)
            n_unmatched_ins = n_ins - n_matched
            n_unmatched_gt = self.current_gt_count - n_matched

            avg_iou = 0.0
            avg_cov = 0.0
            if n_matched > 0:
                matched_results = [r for r in self.current_match_results if r.matched]
                avg_iou = sum(r.iou for r in matched_results) / n_matched
                avg_cov = sum(r.coverage for r in matched_results) / n_matched

            msg = (
                f"[INS] Matched: {n_matched}/{n_ins} | "
                f"Unmatched INS: {n_unmatched_ins} | Unmatched GT: {n_unmatched_gt} | "
                f"Avg IoU: {avg_iou:.2f} | Avg Cov: {avg_cov:.2f} | "
                "(click to go back)"
            )
            self.status_bar.showMessage(msg)
        else:
            self.status_bar.showMessage("Showing annotations (click to show the predicted masks)")

    def _update_3d_views(self, annotations: list):
        """Rebuild the 3D views (31 body points in body mode)"""
        use_body = self.image_widget.current_view_mode() == "body"
        self.person_3d_widgets = update_3d_views(
            self.scroll_area,
            self.scroll_layout,
            self.person_3d_widgets,
            annotations,
            KEYPOINT_NAMES,
            SKELETON,
            use_body=use_body,
        )

    def _on_image_clicked(self, img_x: float, img_y: float):
        """Select the person nearest to the click"""
        if not self.current_annotations:
            return

        min_dist = float('inf')
        nearest_idx = None

        for idx, anno in enumerate(self.current_annotations):
            if "bbox" not in anno:
                continue

            bx, by, bw, bh = anno["bbox"]
            cx = bx + bw / 2
            cy = by + bh / 2
            dist = ((img_x - cx) ** 2 + (img_y - cy) ** 2) ** 0.5

            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

        if nearest_idx is not None:
            anno = self.current_annotations[nearest_idx]
            bbox = anno.get("bbox", [])
            bbox_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]" if bbox else "N/A"

            status_msg = f"Selected: #{nearest_idx} | BBox: {bbox_str} | ignore={anno.get('ignore', False)}"
            self.status_bar.showMessage(status_msg)

            self._display_current_image(selected_idx=nearest_idx)

    def _on_toggle_ignored(self):
        """Toggle the ignore flag of the selected person"""
        if self.selected_person_idx is None:
            self.status_bar.showMessage("No person selected. Click a person first.")
            return

        person_idx = self.selected_person_idx
        if person_idx >= len(self.current_annotations):
            return

        anno = self.current_annotations[person_idx]
        anno["ignore"] = not anno.get("ignore", False)

        if anno["ignore"]:
            self.status_bar.showMessage(f"#{person_idx} -> ignore=True")
        else:
            self.status_bar.showMessage(f"#{person_idx} -> ignore=False")

        self._display_current_image(selected_idx=self.selected_person_idx)


def main():
    parser = argparse.ArgumentParser(description="AGORA extraction preview")
    parser.add_argument(
        "--agora-dir",
        type=Path,
        default=AGORA_DIR,
        help=f"AGORA dataset root (default: {AGORA_DIR})",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default=None,
        help="dataset split (default: random from both)",
    )
    parser.add_argument(
        "--resolution",
        choices=["4K", "720p"],
        default="4K",
        help="source resolution (default: 4K)",
    )
    args = parser.parse_args()

    agora_dir = args.agora_dir.expanduser().resolve()
    if not agora_dir.is_dir():
        sys.exit(
            f"Error: AGORA directory not found: {agora_dir}\n"
            "Pass --agora-dir or set the AGORA_DIR environment variable."
        )

    app = QApplication(sys.argv)

    try:
        window = MainWindow(split=args.split, resolution=args.resolution, agora_dir=agora_dir)
        window.show()
        sys.exit(app.exec_())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
