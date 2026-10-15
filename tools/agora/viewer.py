#!/usr/bin/env python3
"""
Viewer for the extracted AGORA training data (two-column view).

Left: crop image with the annotation overlay
Right: per-person 3D views of the unified keypoints (scrollable, rotatable with the mouse)

Data sources:
    - images: data/{split}/images/
    - depth teachers: data/{split}/depth/
    - annotations: data/head_keypoints_{split}.json

Keys:
    Enter: next random image
    Space: random image with overlapping person bboxes
    X: toggle ignore of the selected person (saved to the JSON on exit)
    V: cycle the view mode (head / body / parts / depth / occlusion)
    Click: select a person, or toggle the RTMDet-Ins mask layer
    Q/ESC: quit

Keypoint display:
    - visible (v=2): filled circle with white outline
    - occluded (v=1): hollow circle
    - out of frame (v=0): cross
    - skeleton: solid if both ends are in frame, dashed otherwise
    - a 100px black padding shows keypoints outside the image

Usage:
    uv run python tools/agora/viewer.py                     # train + validation
    uv run python tools/agora/viewer.py --split train       # train only
    uv run python tools/agora/viewer.py --split validation  # validation only
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Add the script directory to the import path
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Import torch before PyQt5 (avoids a DLL conflict on Windows)
import torch  # noqa: F401

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QScrollArea, QShortcut, QStatusBar
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from config import (
    CROPPED_DIR,
    OUTPUT_DIR,
    MATCHING_IOU_THRESHOLD,
    MATCHING_SCORE_THRESHOLD,
    MATCHING_MIN_COVERAGE,
)
from utils import (
    ImageWidget,
    update_3d_views,
    build_info_lines,
)
from utils.rtmdet_ins import get_rtmdet_ins_inference
from utils.matching import decode_rle_mask, match_masks, get_unmatched_gt_indices


def bboxes_overlap(bbox1: list, bbox2: list) -> bool:
    """Whether two COCO bboxes [x, y, w, h] overlap"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def has_overlapping_bboxes(annotations: list) -> bool:
    """Whether any pair of annotation bboxes overlaps"""
    if len(annotations) < 2:
        return False
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            if bboxes_overlap(annotations[i]["bbox"], annotations[j]["bbox"]):
                return True
    return False


class MainWindow(QMainWindow):
    """Main window"""

    def __init__(self, split: str = None):
        super().__init__()
        self.split = split  # None = both splits

        self._load_data()

        self._init_ui()

        # First image
        self._show_random_image()

    def _load_data(self):
        """Load the COCO annotation files"""
        if self.split is not None:
            splits_to_load = [self.split]
        else:
            splits_to_load = ["train", "validation"]

        self.keypoint_names = None
        self.skeleton = None
        self.img_id_to_info = {}  # (split, img_id) -> image_info
        self.img_to_annos = {}    # (split, img_id) -> annotations
        self.valid_entries = []   # [(split, img_id), ...]
        self.coco_data = {}       # split -> full coco_data (for saving)
        self.anno_id_to_anno = {} # (split, anno_id) -> annotation (quick lookup)
        self.modified_splits = set()  # splits changed (saved on exit)

        total_images = 0
        total_annos = 0

        for split in splits_to_load:
            json_path = OUTPUT_DIR / f"head_keypoints_{split}.json"
            if not json_path.exists():
                print(f"Warning: {json_path} not found, skipping")
                continue

            with open(json_path) as f:
                coco_data = json.load(f)

            # Keep the whole COCO data (for saving)
            self.coco_data[split] = coco_data

            # Keypoint names and skeleton from the first split
            if self.keypoint_names is None:
                self.keypoint_names = coco_data["categories"][0]["keypoints"]
                self.skeleton = coco_data["categories"][0]["skeleton"]

            # image_id -> image_info (keyed with the split)
            for img in coco_data["images"]:
                key = (split, img["id"])
                self.img_id_to_info[key] = img

            # image_id -> annotations, anno_id -> annotation
            for anno in coco_data["annotations"]:
                key = (split, anno["image_id"])
                if key not in self.img_to_annos:
                    self.img_to_annos[key] = []
                self.img_to_annos[key].append(anno)
                self.anno_id_to_anno[(split, anno["id"])] = anno

            # Images with annotations
            for key in self.img_to_annos.keys():
                if key[0] == split:
                    self.valid_entries.append(key)

            total_images += len(coco_data["images"])
            total_annos += len(coco_data["annotations"])
            print(f"{split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

        if not self.valid_entries:
            raise FileNotFoundError("no annotated images found")

        print(f"Loaded: {total_images} images, {total_annos} annotations in total")

    def _init_ui(self):
        """Build the UI"""
        split_text = self.split if self.split else "train+validation"
        self.setWindowTitle(f"AGORA Training Data Viewer ({split_text})")

        # Full-screen sized window
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Left / right layout
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Left: image
        self.image_widget = ImageWidget()
        layout.addWidget(self.image_widget, 1)

        # Right: scrollable 3D views
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
        self.status_bar.showMessage("Click to select a person")

        # Current data
        self.current_entry = None
        self.current_annotations = []
        self.current_img = None
        self.selected_person_idx = None

        # RTMDet-Ins (loaded lazily)
        self.rtmdet_ins = None
        self._rtmdet_init_attempted = False

        # Matching results
        self.current_match_results = []
        self.current_ins_masks = []
        self.current_gt_count = 0

        self.image_widget.clicked.connect(self._on_image_clicked)
        self.image_widget.inference_layer_toggled.connect(self._on_inference_layer_toggled)
        self.image_widget.view_mode_changed.connect(self._on_view_mode_changed)

        QShortcut(QKeySequence(Qt.Key_Space), self, self._show_overlapping_image)
        QShortcut(QKeySequence(Qt.Key_Return), self, self._show_random_image)
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
        # The 3D views switch to the body in body mode
        self._update_3d_views(self.current_annotations)

    def _load_depth_maps(self, split: str, annotations: list, img_info: dict) -> list:
        """Load each annotation's depth teacher PNG as pelvis-relative pseudo depth

        - mirrored images flip the PNG (depth_file is shared with the original)
        - returns float32 maps on the same scale as the body keypoint depth (background = NaN)
        """
        from utils.render import pseudo_depth_from_teacher

        mirrored = bool(img_info.get("mirrored"))
        focal_scaled = float(img_info.get("focal_length", 0) or 0)
        depth_maps = []
        for anno in annotations:
            depth_file = anno.get("depth_file")
            arr = None
            if depth_file and focal_scaled > 0:
                depth_path = CROPPED_DIR / split / depth_file
                if depth_path.exists():
                    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if raw is not None:
                        if mirrored:
                            raw = np.ascontiguousarray(raw[:, ::-1])
                        arr = pseudo_depth_from_teacher(
                            raw, anno.get("pelvis_depth"), focal_scaled
                        )
            depth_maps.append(arr)
        return depth_maps

    def _show_random_image(self):
        """Show a random image"""
        entry = random.choice(self.valid_entries)
        self._display_image(entry)

    def _show_overlapping_image(self):
        """Show a random image with overlapping person bboxes"""
        max_attempts = 1000
        for attempt in range(max_attempts):
            entry = random.choice(self.valid_entries)
            annotations = self.img_to_annos[entry]
            if has_overlapping_bboxes(annotations):
                self._display_image(entry)
                self.status_bar.showMessage(
                    f"Overlap found (attempt {attempt + 1}) - {len(annotations)} persons"
                )
                return
        self.status_bar.showMessage(f"No image with overlaps found ({max_attempts} attempts)")

    def _extract_scene_name(self, filename: str) -> str:
        """Scene name from a file name"""
        parts = filename.split("_")
        try:
            bfh_idx = parts.index("bfh")
            return parts[bfh_idx + 1]
        except (ValueError, IndexError):
            return "unknown"

    def _display_image(self, entry: tuple, selected_idx: int = None):
        """Show an image"""
        split, img_id = entry
        img_info = self.img_id_to_info[entry]
        annotations = self.img_to_annos[entry]

        # Crop image path
        img_filename = img_info["file_name"]
        img_path = CROPPED_DIR / split / "images" / img_filename

        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            print(f"Failed to read image: {img_path}")
            return

        # Current data
        self.current_entry = entry
        self.current_annotations = annotations
        self.current_img = img_orig.copy()
        self.selected_person_idx = selected_idx

        scene_name = self._extract_scene_name(img_filename)

        self.image_widget.set_image(img_orig)

        # Info text (everything available from the JSON)
        info_lines = build_info_lines(
            split=split,
            image_id=img_id,
            scene_name=scene_name,
            file_name=img_filename,
            width=img_info.get("width"),
            height=img_info.get("height"),
            angle=img_info.get("rotated"),
            mirrored=img_info.get("mirrored"),
            cx=img_info.get("cx"),
            cy=img_info.get("cy"),
            annotations=annotations,
            shortcuts="[Enter/Space: Next] [X: Toggle Ignore] [V: View Mode] [Click: Layer] [Q: Quit]",
        )

        # Teacher statistics line
        n_body = sum(1 for a in annotations if a.get("keypoints_body") is not None)
        n_parts = sum(1 for a in annotations if a.get("part_masks"))
        n_depth = sum(1 for a in annotations if a.get("depth_file"))
        n_occ_valid = sum(1 for a in annotations if a.get("occlusion_valid"))
        info_lines.append((
            f"teachers: body_kpts={n_body} parts={n_parts} depth={n_depth} "
            f"occ_teacher={n_occ_valid}/{len(annotations)}",
            0.5, (150, 220, 255),
        ))

        # Depth teachers as pseudo depth (depth view mode)
        depth_maps = self._load_depth_maps(split, annotations, img_info)

        self.image_widget.set_annotations(
            annotations, selected_idx, info_lines, depth_maps=depth_maps
        )

        # 3D views on the right
        self._update_3d_views(annotations)

        # RTMDet-Ins inference
        self._run_rtmdet_inference()

    def _run_rtmdet_inference(self):
        """Run RTMDet-Ins and match its masks to the GT"""
        if self.current_img is None:
            return

        # Lazy initialization (attempted once)
        if self.rtmdet_ins is None and not self._rtmdet_init_attempted:
            self._rtmdet_init_attempted = True
            self.rtmdet_ins = get_rtmdet_ins_inference()
            if self.rtmdet_ins:
                print("RTMDet-Ins ready for inference")
            else:
                print("RTMDet-Ins not available (click layer toggle disabled)")

        if self.rtmdet_ins is None:
            self.image_widget.set_inference_masks([])
            return

        try:
            detections = self.rtmdet_ins.infer(self.current_img)
            ins_masks = [det['mask'] for det in detections if det.get('mask') is not None]

            # GT masks (decode the RLEs)
            gt_masks = []
            anno_idx_to_gt_idx = {}
            gt_idx_to_anno_idx = {}

            for anno_idx, anno in enumerate(self.current_annotations):
                if "segmentation" in anno and anno["segmentation"] is not None:
                    try:
                        gt_mask = decode_rle_mask(anno["segmentation"])
                        gt_idx = len(gt_masks)
                        gt_masks.append(gt_mask)
                        anno_idx_to_gt_idx[anno_idx] = gt_idx
                        gt_idx_to_anno_idx[gt_idx] = anno_idx
                    except Exception:
                        pass

            self.current_gt_count = len(gt_masks)

            if ins_masks and gt_masks:
                match_results = match_masks(
                    gt_masks=gt_masks,
                    ins_masks=ins_masks,
                    iou_threshold=MATCHING_IOU_THRESHOLD,
                    score_threshold=MATCHING_SCORE_THRESHOLD,
                    min_coverage=MATCHING_MIN_COVERAGE,
                )
                # gt_idx -> annotation index (display)
                for result in match_results:
                    if result.matched and result.gt_idx in gt_idx_to_anno_idx:
                        result.gt_idx = gt_idx_to_anno_idx[result.gt_idx]

                # Log unmatched GT annotations
                unmatched_gt = get_unmatched_gt_indices(match_results, len(gt_masks))
                if unmatched_gt:
                    unmatched_anno = [gt_idx_to_anno_idx.get(g, g) for g in unmatched_gt]
                    print(f"Unmatched GT annotations: {unmatched_anno}")
            else:
                match_results = []

            self.current_match_results = match_results
            self.current_ins_masks = ins_masks

            self.image_widget.set_inference_masks(ins_masks, match_results)

        except Exception as e:
            print(f"RTMDet-Ins inference failed: {e}")
            import traceback
            traceback.print_exc()
            self.image_widget.set_inference_masks([])

    def _on_inference_layer_toggled(self, visible: bool):
        """Inference layer toggled (show matching statistics)"""
        if visible:
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
            self.keypoint_names,  # from the JSON
            self.skeleton,        # from the JSON
            use_body=use_body,
        )

    def _on_image_clicked(self, img_x: float, img_y: float):
        """Select the person nearest to the click"""
        if not self.current_annotations or self.current_entry is None:
            return

        # Distance to each person's bbox centre
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
            split, img_id = self.current_entry
            img_info = self.img_id_to_info[self.current_entry]

            anno_id = anno.get("id", "N/A")
            bbox = anno.get("bbox", [])
            bbox_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]" if bbox else "N/A"

            status_msg = (
                f"Selected: #{nearest_idx} | "
                f"Split: {split} | "
                f"Image ID: {img_id} | "
                f"Annotation ID: {anno_id} | "
                f"File: {img_info['file_name']} | "
                f"BBox: {bbox_str}"
            )
            self.status_bar.showMessage(status_msg)

            # Redraw with the selection highlighted
            self._display_image(self.current_entry, selected_idx=nearest_idx)

    def _on_toggle_ignored(self):
        """Toggle the ignore flag of the selected person (in memory; saved on exit)"""
        if self.selected_person_idx is None or self.current_entry is None:
            self.status_bar.showMessage("No person selected. Click a person first.")
            return

        split, img_id = self.current_entry
        person_idx = self.selected_person_idx

        annotations = self.img_to_annos.get((split, img_id), [])
        if person_idx >= len(annotations):
            self.status_bar.showMessage("Annotation not found")
            return

        anno = annotations[person_idx]

        current_ignore = anno.get("ignore", False)
        anno["ignore"] = not current_ignore

        self.modified_splits.add(split)

        if anno["ignore"]:
            self.status_bar.showMessage(f"#{person_idx} -> ignore=True (saved on exit)")
        else:
            self.status_bar.showMessage(f"#{person_idx} -> ignore=False (saved on exit)")

        self._display_image(self.current_entry, selected_idx=self.selected_person_idx)

    def _save_modified_coco_files(self):
        """Save the modified COCO files"""
        for split in self.modified_splits:
            json_path = OUTPUT_DIR / f"head_keypoints_{split}.json"
            with open(json_path, "w") as f:
                json.dump(self.coco_data[split], f)
            print(f"Saved: {json_path}")

    def closeEvent(self, event):
        """Save changes on exit"""
        if self.modified_splits:
            self._save_modified_coco_files()
            print(f"Saved changes: {', '.join(self.modified_splits)}")
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Viewer for the extracted AGORA training data")
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default=None,
        help="dataset split (default: random from both)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    try:
        window = MainWindow(split=args.split)
        window.show()
        sys.exit(app.exec_())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run extracter.py first to generate the crops and annotations")
        sys.exit(1)


if __name__ == "__main__":
    main()
