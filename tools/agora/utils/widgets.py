#!/usr/bin/env python3
"""
Shared GUI widgets.

PyQt5 image view and 3D view widgets used by test.py and viewer.py.

ImageWidget draws with QPainter, so overlays have a constant on-screen size
regardless of the source image resolution.
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import (
    KEYPOINT_COLORS_RGB,
    KEYPOINT_COLORS_BGR,
    SKELETON,
    PERSON_COLORS_BGR,
    DISPLAY_PADDING,
    BODY_SKELETON,
    BODY_KEYPOINT_COLORS_BGR,
    BODY_KEYPOINT_COLORS_RGB,
    PART_COLORS_BGR,
    PART_GROUP_NAMES,
)
from utils.matching import MatchResult


def _decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE (counts as str or bytes)"""
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return mask_utils.decode(rle)


class ImageWidget(QWidget):
    """Image view (QPainter based, resolution independent overlays)

    Overlays (bboxes, keypoints, text, ...) are drawn with a constant pixel
    size on the widget regardless of the source image resolution.
    """

    # Drawing sizes (widget pixels)
    FONT_SIZE_LABEL = 24
    FONT_SIZE_KPT = 16
    FONT_SIZE_INFO = 20
    CIRCLE_RADIUS = 6
    CIRCLE_OUTLINE = 8
    LINE_WIDTH_BBOX = 3
    LINE_WIDTH_HEAD_BBOX = 2
    LINE_WIDTH_SKELETON = 2
    DASH_PATTERN = [6, 4]

    # Virtual padding (in image pixels; converted to the actual display scale)
    VIRTUAL_PADDING = DISPLAY_PADDING

    # View modes (cycled with the V key)
    VIEW_MODES = ["head", "body", "parts", "depth", "occlusion"]

    clicked = pyqtSignal(float, float)
    inference_layer_toggled = pyqtSignal(bool)  # inference layer shown / hidden
    view_mode_changed = pyqtSignal(str)  # view mode changed

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)
        self._pixmap: Optional[QPixmap] = None
        self._image_size = (0, 0)  # source image size (w, h)
        self._annotations: List[dict] = []
        self._info_lines: List[tuple] = []  # [(text, font_scale, color), ...]
        self._selected_idx: Optional[int] = None
        # Inference mask layer
        self._inference_masks: List[np.ndarray] = []  # binary masks
        self._show_inference_layer: bool = False
        self._match_results: List[MatchResult] = []  # matching results (Ins -> GT)
        self._gt_to_ins_map: dict = {}  # GT -> Ins map {gt_idx: ins_idx}
        # View mode and depth maps
        self._view_mode: int = 0
        self._depth_maps: List[Optional[np.ndarray]] = []  # depth arrays in annotation order

    def cycle_view_mode(self) -> str:
        """Switch to the next view mode and return its name"""
        self._view_mode = (self._view_mode + 1) % len(self.VIEW_MODES)
        mode = self.VIEW_MODES[self._view_mode]
        self.view_mode_changed.emit(mode)
        self.update()
        return mode

    def current_view_mode(self) -> str:
        return self.VIEW_MODES[self._view_mode]

    def set_image(self, image: np.ndarray):
        """Set an OpenCV (BGR) image"""
        h, w = image.shape[:2]
        self._image_size = (w, h)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def set_annotations(
        self,
        annotations: List[dict],
        selected_idx: Optional[int] = None,
        info_lines: Optional[List[tuple]] = None,
        depth_maps: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """Set annotations and info text

        Args:
            annotations: annotation list (bbox, head_bbox, keypoints, ignore, ...)
            selected_idx: selected person index
            info_lines: info text [(text, font_scale, color_bgr), ...]
            depth_maps: pseudo-depth maps in annotation order, used by the depth
                        view mode (passed separately because they are arrays,
                        not JSON fields)
        """
        self._annotations = annotations
        self._selected_idx = selected_idx
        self._info_lines = info_lines or []
        self._depth_maps = depth_maps or []
        self.update()

    def set_inference_masks(self, masks: List[np.ndarray], match_results: Optional[List[MatchResult]] = None):
        """Set the predicted masks

        Args:
            masks: binary masks (each (h, w))
            match_results: matching results (same order as masks)
        """
        self._inference_masks = masks
        self._match_results = match_results or []
        # GT -> Ins lookup
        self._gt_to_ins_map = {}
        for result in self._match_results:
            if result.matched:
                self._gt_to_ins_map[result.gt_idx] = result.ins_idx
        self.update()

    def toggle_inference_layer(self):
        """Toggle the inference layer"""
        self._show_inference_layer = not self._show_inference_layer
        self.inference_layer_toggled.emit(self._show_inference_layer)
        self.update()

    def is_inference_layer_visible(self) -> bool:
        """Whether the inference layer is shown"""
        return self._show_inference_layer

    def _get_padded_image_rect(self) -> tuple:
        """Virtual image area including padding, and its scale

        Returns:
            (rect, scale): QRectF of the padded area and the image -> widget scale
        """
        if not self._pixmap:
            return QRectF(), 1.0

        widget_w, widget_h = self.width(), self.height()
        img_w, img_h = self._image_size

        if img_w == 0 or img_h == 0:
            return QRectF(), 1.0

        # Virtual size including padding
        padded_w = img_w + self.VIRTUAL_PADDING * 2
        padded_h = img_h + self.VIRTUAL_PADDING * 2

        scale = min(widget_w / padded_w, widget_h / padded_h)
        scaled_w = padded_w * scale
        scaled_h = padded_h * scale

        x = (widget_w - scaled_w) / 2
        y = (widget_h - scaled_h) / 2

        return QRectF(x, y, scaled_w, scaled_h), scale

    def _get_image_rect(self) -> QRectF:
        """Display area of the actual image (inside the padding)"""
        padded_rect, scale = self._get_padded_image_rect()
        if padded_rect.isEmpty():
            return QRectF()

        pad_widget = self.VIRTUAL_PADDING * scale
        return QRectF(
            padded_rect.x() + pad_widget,
            padded_rect.y() + pad_widget,
            padded_rect.width() - pad_widget * 2,
            padded_rect.height() - pad_widget * 2
        )

    def _img_to_widget(self, img_x: float, img_y: float) -> QPointF:
        """Image -> widget coordinates (also outside the padded area)"""
        padded_rect, scale = self._get_padded_image_rect()
        if padded_rect.isEmpty():
            return QPointF(0, 0)

        # Image (0, 0) is at (VIRTUAL_PADDING, VIRTUAL_PADDING) of the padded area
        widget_x = padded_rect.x() + (img_x + self.VIRTUAL_PADDING) * scale
        widget_y = padded_rect.y() + (img_y + self.VIRTUAL_PADDING) * scale

        return QPointF(widget_x, widget_y)

    def _widget_to_img(self, widget_x: float, widget_y: float) -> tuple:
        """Widget -> image coordinates (also outside the padded area)"""
        padded_rect, scale = self._get_padded_image_rect()
        if padded_rect.isEmpty() or scale == 0:
            return (0, 0)

        img_x = (widget_x - padded_rect.x()) / scale - self.VIRTUAL_PADDING
        img_y = (widget_y - padded_rect.y()) / scale - self.VIRTUAL_PADDING

        return (img_x, img_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if not self._pixmap:
            return

        # Image
        rect = self._get_image_rect()
        painter.drawPixmap(rect.toRect(), self._pixmap)

        # Image border
        pen = QPen(QColor(80, 80, 80), 1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        # Inference mask layer (replaces the annotations while shown)
        if self._show_inference_layer and self._inference_masks:
            self._draw_inference_masks(painter)
        else:
            # Annotations (depending on the view mode)
            mode = self.VIEW_MODES[self._view_mode]
            if mode == "head":
                self._draw_annotations(painter)
            elif mode == "body":
                self._draw_body_mode(painter)
            elif mode == "parts":
                self._draw_parts_mode(painter)
            elif mode == "depth":
                self._draw_depth_mode(painter)
            elif mode == "occlusion":
                self._draw_occlusion_mode(painter)
            if mode != "head":
                self._draw_mode_indicator(painter, mode)

        # Info text
        self._draw_info_text(painter)

    def _draw_mode_indicator(self, painter: QPainter, mode: str):
        """Draw the view mode indicator (top right)"""
        font = QFont()
        font.setPixelSize(24)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 0)))
        painter.drawText(QPointF(self.width() - 220, 30), f"[MODE: {mode.upper()}]")

    def _draw_annotations(self, painter: QPainter):
        """head mode: masks, bboxes and head keypoints"""
        if not self._annotations:
            return

        # Masks first (below everything else)
        for person_idx, anno in enumerate(self._annotations):
            if "segmentation" in anno and anno["segmentation"] is not None:
                is_ignored = anno.get("ignore", False)
                if is_ignored:
                    color_bgr = (0, 0, 255)  # red
                else:
                    color_bgr = PERSON_COLORS_BGR[person_idx % len(PERSON_COLORS_BGR)]
                self._draw_mask(painter, anno["segmentation"], color_bgr)

        for person_idx, anno in enumerate(self._annotations):
            is_selected = (self._selected_idx is not None and person_idx == self._selected_idx)
            is_ignored = anno.get("ignore", False)

            # Person colour
            if is_ignored:
                color_bgr = (0, 0, 255)  # red
            else:
                color_bgr = PERSON_COLORS_BGR[person_idx % len(PERSON_COLORS_BGR)]
            color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])

            line_width = self.LINE_WIDTH_BBOX + 2 if is_selected else self.LINE_WIDTH_BBOX

            # Body bbox
            if "bbox" in anno:
                self._draw_body_bbox(painter, anno["bbox"], color, line_width, is_ignored)
                self._draw_person_label(painter, anno["bbox"], person_idx, color, is_ignored, is_selected)

            # Head bbox (dashed)
            if "head_bbox" in anno:
                self._draw_head_bbox(painter, anno["head_bbox"], color, line_width)

            # Keypoints
            if "keypoints" in anno:
                pts, vis = self._draw_keypoints(painter, anno["keypoints"])
                self._draw_skeleton(painter, pts, vis)

    def _draw_body_mode(self, painter: QPainter):
        """body mode: 31 body keypoints + body_ref_bbox + subject_id"""
        if not self._annotations:
            return

        for person_idx, anno in enumerate(self._annotations):
            is_selected = (self._selected_idx is not None and person_idx == self._selected_idx)
            is_ignored = anno.get("body_ignore", False)

            if is_ignored:
                color_bgr = (0, 0, 255)
            else:
                color_bgr = PERSON_COLORS_BGR[person_idx % len(PERSON_COLORS_BGR)]
            color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])
            line_width = self.LINE_WIDTH_BBOX + 2 if is_selected else self.LINE_WIDTH_BBOX

            # Body reference bbox (solid) + label with subject_id
            if anno.get("body_ref_bbox") is not None:
                self._draw_body_bbox(painter, anno["body_ref_bbox"], color, line_width, is_ignored)
                subject = anno.get("subject_id") or ""
                kid = " [kid]" if anno.get("kid") else ""
                label = f"GT#{person_idx} {subject}{kid}" + (" (body_ignore)" if is_ignored else "")
                self._draw_label_at(painter, anno["body_ref_bbox"], label, color, is_selected)

            # Head bbox (dashed, for reference)
            if "head_bbox" in anno:
                self._draw_head_bbox(painter, anno["head_bbox"], color, line_width)

            # Body keypoints + skeleton
            if anno.get("keypoints_body") is not None:
                pts, vis = self._draw_keypoints(
                    painter, anno["keypoints_body"], colors=BODY_KEYPOINT_COLORS_BGR
                )
                self._draw_skeleton(painter, pts, vis, skeleton=BODY_SKELETON)

    def _draw_label_at(self, painter: QPainter, bbox: list, label: str, color: QColor, is_selected: bool):
        """Draw a text label at the top-left of a bbox"""
        p1 = self._img_to_widget(bbox[0], bbox[1])
        font = QFont()
        font.setPixelSize(self.FONT_SIZE_LABEL + (4 if is_selected else 0))
        font.setBold(is_selected)
        painter.setFont(font)
        fm = painter.fontMetrics()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawRect(QRectF(
            p1.x() - 2, p1.y() - 4 - fm.height(),
            fm.horizontalAdvance(label) + 4, fm.height() + 4
        ))
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(QPointF(p1.x(), p1.y() - 4), label)

    def _draw_parts_mode(self, painter: QPainter):
        """parts mode: naked-body part masks (teacher resolution stretched to the image)"""
        if not self._annotations:
            return

        drawn = 0
        for person_idx, anno in enumerate(self._annotations):
            part_masks = anno.get("part_masks")
            if not part_masks:
                continue
            for gid, gname in enumerate(PART_GROUP_NAMES, start=1):
                rle = part_masks.get(gname)
                if rle is None:
                    continue
                mask = _decode_rle(rle)
                self._draw_mask_array(painter, mask, PART_COLORS_BGR[gid], alpha=140)
                drawn += 1
            # Person label
            ref = anno.get("body_ref_bbox") or anno.get("bbox")
            if ref is not None:
                color_bgr = PERSON_COLORS_BGR[person_idx % len(PERSON_COLORS_BGR)]
                color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])
                self._draw_label_at(painter, ref, f"GT#{person_idx}", color, False)

        if drawn == 0:
            self._draw_center_notice(painter, "No part masks (part_masks is empty)")

        # Legend
        font = QFont()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        y = 100
        for gid, gname in enumerate(PART_GROUP_NAMES, start=1):
            c = PART_COLORS_BGR[gid]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(c[2], c[1], c[0])))
            painter.drawRect(QRectF(10, y - 12, 14, 14))
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(QPointF(30, y), gname)
            y += 22

    def _draw_depth_mode(self, painter: QPainter):
        """depth mode: per-person pelvis-relative pseudo-depth heat map

        depth_maps must hold pseudo depth (float32, pelvis = 0, background = NaN,
        same scale as the body keypoint depth). Each person is normalized with
        a range symmetric around the pelvis (0), so every person shows a
        gradient regardless of the number of persons in the scene.
        Red = near (in front of the pelvis), green = around the pelvis, blue = far.
        """
        drawn = 0
        for idx, anno in enumerate(self._annotations or []):
            depth = self._depth_maps[idx] if idx < len(self._depth_maps) else None
            if depth is None:
                continue
            valid = np.isfinite(depth)
            if not valid.any():
                continue

            vals = depth[valid]
            # Symmetric range keeps the pelvis (0) at the centre of the colour map
            radius = max(abs(float(vals.min())), abs(float(vals.max())), 1e-3)
            norm = np.zeros(depth.shape, dtype=np.float32)
            norm[valid] = (depth[valid] + radius) / (2.0 * radius)  # 0..1, 0.5 = pelvis

            colored = cv2.applyColorMap(
                (255 - np.clip(norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )
            h, w = depth.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = colored[:, :, 2]
            rgba[:, :, 1] = colored[:, :, 1]
            rgba[:, :, 2] = colored[:, :, 0]
            rgba[:, :, 3] = valid.astype(np.uint8) * 210
            self._draw_rgba(painter, rgba)
            drawn += 1

            # Person label: pseudo-depth range (pixel-equivalent scale)
            ref = anno.get("body_ref_bbox") or anno.get("bbox")
            if ref is not None:
                color_bgr = PERSON_COLORS_BGR[idx % len(PERSON_COLORS_BGR)]
                color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])
                self._draw_label_at(
                    painter, ref,
                    f"GT#{idx} y:[{vals.min():+.0f}, {vals.max():+.0f}]px", color, False
                )

        if drawn == 0:
            self._draw_center_notice(painter, "No depth maps (depth teacher is empty)")
            return

        # Legend
        font = QFont()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(QPointF(10, 100),
                         "pelvis-relative pseudo-depth: red=near / green=pelvis / blue=far")

    def _draw_occlusion_mode(self, painter: QPainter):
        """occlusion mode: clothed amodal masks (person colour, light) + occluded regions (red)"""
        if not self._annotations:
            return

        for person_idx, anno in enumerate(self._annotations):
            if anno.get("segmentation") is not None:
                color_bgr = PERSON_COLORS_BGR[person_idx % len(PERSON_COLORS_BGR)]
                self._draw_mask(painter, anno["segmentation"], color_bgr)

        drawn_occ = 0
        for person_idx, anno in enumerate(self._annotations):
            occ = anno.get("occlusion_mask")
            if occ is not None:
                self._draw_mask_array(painter, _decode_rle(occ), (0, 0, 255), alpha=180)
                drawn_occ += 1
            # Label: occlusion info
            ref = anno.get("bbox")
            if ref is not None:
                valid = anno.get("occlusion_valid", False)
                pct = anno.get("occlusion_pct")
                pct_str = f"{pct:.0f}%" if pct is not None else "-"
                tag = "OK" if valid else "NG"
                color_bgr = (80, 255, 80) if valid else (0, 0, 255)
                color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])
                self._draw_label_at(
                    painter, ref,
                    f"GT#{person_idx} occ={pct_str} teacher={tag}", color, False
                )

    def _draw_center_notice(self, painter: QPainter, text: str):
        """Draw a notice in the centre"""
        font = QFont()
        font.setPixelSize(22)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 200, 100)))
        fm = painter.fontMetrics()
        painter.drawText(
            QPointF((self.width() - fm.horizontalAdvance(text)) / 2, self.height() / 2),
            text,
        )

    def _draw_body_bbox(self, painter: QPainter, bbox: list, color: QColor, line_width: int, is_ignored: bool):
        """Draw a body bbox"""
        x, y, w, h = bbox
        p1 = self._img_to_widget(x, y)
        p2 = self._img_to_widget(x + w, y + h)

        pen = QPen(color, line_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(p1, p2))

        # Cross when ignored
        if is_ignored:
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(p1, p2)
            painter.drawLine(QPointF(p2.x(), p1.y()), QPointF(p1.x(), p2.y()))

    def _draw_head_bbox(self, painter: QPainter, bbox: list, color: QColor, line_width: int):
        """Draw a head bbox (dashed)"""
        x, y, w, h = bbox
        p1 = self._img_to_widget(x, y)
        p2 = self._img_to_widget(x + w, y + h)

        pen = QPen(color, self.LINE_WIDTH_HEAD_BBOX)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern(self.DASH_PATTERN)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(p1, p2))

    def _draw_person_label(self, painter: QPainter, bbox: list, person_idx: int, color: QColor, is_ignored: bool, is_selected: bool):
        """Draw a person label (with matching info)"""
        x, y, w, h = bbox
        p1 = self._img_to_widget(x, y)

        # Matching info
        if person_idx in self._gt_to_ins_map:
            ins_idx = self._gt_to_ins_map[person_idx]
            match_info = f" -> Ins#{ins_idx}"
        elif self._inference_masks:  # inference available but no match
            match_info = " -> X"
        else:  # no inference result
            match_info = ""

        label = f"GT#{person_idx}{match_info}" + (" (Ignored)" if is_ignored else "")
        font = QFont()
        font_size = self.FONT_SIZE_LABEL + 4 if is_selected else self.FONT_SIZE_LABEL
        font.setPixelSize(font_size)
        font.setBold(is_selected)
        painter.setFont(font)

        fm = painter.fontMetrics()
        label_width = fm.horizontalAdvance(label)
        label_height = fm.height()

        label_x = p1.x()
        label_y = p1.y() - 4

        # Label background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawRect(QRectF(
            label_x - 2,
            label_y - label_height,
            label_width + 4,
            label_height + 4
        ))

        # Label text
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(QPointF(label_x, label_y), label)

    def _draw_keypoints(self, painter: QPainter, keypoints: list, colors: list = None) -> tuple:
        """Draw keypoints

        Visibility (COCO):
        - v=0: out of frame -> cross
        - v=1: occluded -> hollow circle
        - v=2: visible -> filled circle

        Args:
            colors: keypoint colours (BGR); head colours by default
        """
        if colors is None:
            colors = KEYPOINT_COLORS_BGR
        n_kpts = len(keypoints) // 3
        pts = []
        vis = []

        for i in range(n_kpts):
            x, y, v = keypoints[i*3:(i+1)*3]
            pt = self._img_to_widget(x, y)
            pts.append(pt)
            vis.append(int(v))

            color_bgr = colors[i % len(colors)]
            color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])

            if v == 2:
                # Visible: filled circle + white outline
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(pt, self.CIRCLE_OUTLINE, self.CIRCLE_OUTLINE)

                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(pt, self.CIRCLE_RADIUS, self.CIRCLE_RADIUS)
            elif v == 1:
                # Occluded: hollow circle + coloured outline
                painter.setPen(QPen(QColor(180, 180, 180), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(pt, self.CIRCLE_OUTLINE, self.CIRCLE_OUTLINE)

                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(pt, self.CIRCLE_RADIUS, self.CIRCLE_RADIUS)
            else:
                # Out of frame (v=0): cross
                r = self.CIRCLE_RADIUS
                px, py = int(pt.x()), int(pt.y())
                painter.setPen(QPen(color, 2))
                painter.drawLine(px - r, py - r, px + r, py + r)
                painter.drawLine(px - r, py + r, px + r, py - r)

        return pts, vis

    def _draw_skeleton(self, painter: QPainter, pts: list, vis: list, skeleton: list = None):
        """Draw the skeleton

        Visibility (COCO):
        - both v>=1 (occluded or visible): solid line
        - either v=0 (out of frame): dashed line

        Args:
            skeleton: skeleton definition; head skeleton (config.SKELETON) by default
        """
        if skeleton is None:
            skeleton = SKELETON
        n_kpts = len(pts)

        for s1, s2 in skeleton:
            if s1 < n_kpts and s2 < n_kpts:
                p1 = pts[s1]
                p2 = pts[s2]

                if vis[s1] >= 1 and vis[s2] >= 1:
                    # Both in frame: solid
                    pen = QPen(QColor(180, 180, 180), self.LINE_WIDTH_SKELETON)
                    painter.setPen(pen)
                    painter.drawLine(p1, p2)
                else:
                    # Either out of frame: dashed
                    pen = QPen(QColor(100, 100, 100), self.LINE_WIDTH_SKELETON)
                    pen.setStyle(Qt.DashLine)
                    pen.setDashPattern([4, 4])
                    painter.setPen(pen)
                    painter.drawLine(p1, p2)

    def _draw_mask(self, painter: QPainter, rle: dict, color_bgr: tuple, alpha: int = 76):
        """Draw an RLE mask semi-transparently (default alpha 0.3)"""
        self._draw_mask_array(painter, _decode_rle(rle), color_bgr, alpha)

    def _draw_mask_array(self, painter: QPainter, binary_mask: np.ndarray,
                         color_bgr: tuple, alpha: int = 76):
        """Draw a binary mask array semi-transparently

        Masks at a different resolution than the image (e.g. teacher-resolution
        part masks) are stretched over the whole image.
        """
        h, w = binary_mask.shape
        if h == 0 or w == 0:
            return

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = color_bgr[2]  # R
        mask_rgba[:, :, 1] = color_bgr[1]  # G
        mask_rgba[:, :, 2] = color_bgr[0]  # B
        mask_rgba[:, :, 3] = ((binary_mask > 0) * alpha).astype(np.uint8)

        self._draw_rgba(painter, mask_rgba)

    def _draw_rgba(self, painter: QPainter, rgba: np.ndarray):
        """Draw an RGBA array stretched over the whole image"""
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg.copy())

        # Map to the full image (0,0)-(img_w, img_h) regardless of the array resolution
        img_w, img_h = self._image_size
        p1 = self._img_to_widget(0, 0)
        p2 = self._img_to_widget(img_w, img_h)
        painter.drawPixmap(QRectF(p1, p2).toRect(), pixmap)

    def _draw_inference_masks(self, painter: QPainter):
        """Draw the predicted masks (with matching info)

        Matched masks: greenish
        Unmatched masks: reddish
        """
        if not self._inference_masks:
            return

        img_w, img_h = self._image_size
        if img_w == 0 or img_h == 0:
            return

        # Mask colours: matched = greenish palette, unmatched = red
        INS_COLORS = [
            (100, 200, 150),
            (150, 200, 100),
            (100, 180, 200),
            (200, 180, 100),
            (150, 150, 200),
        ]
        UNMATCHED_COLOR = (200, 80, 80)

        for i, mask in enumerate(self._inference_masks):
            if mask is None or mask.shape != (img_h, img_w):
                continue

            binary_mask = (mask > 0).astype(np.uint8)
            if binary_mask.max() == 0:
                continue

            is_matched = False
            if i < len(self._match_results):
                is_matched = self._match_results[i].matched

            if is_matched:
                color = INS_COLORS[i % len(INS_COLORS)]
                alpha = 120  # 0.47
            else:
                color = UNMATCHED_COLOR
                alpha = 100  # 0.39

            # RGBA image
            mask_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            mask_rgba[:, :, 0] = color[0]  # R
            mask_rgba[:, :, 1] = color[1]  # G
            mask_rgba[:, :, 2] = color[2]  # B
            mask_rgba[:, :, 3] = (binary_mask * alpha).astype(np.uint8)

            qimg = QImage(mask_rgba.data, img_w, img_h, img_w * 4, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg.copy())

            p1 = self._img_to_widget(0, 0)
            p2 = self._img_to_widget(img_w, img_h)
            target_rect = QRectF(p1, p2)

            painter.drawPixmap(target_rect.toRect(), pixmap)

        # Per-mask labels
        self._draw_inference_labels(painter)

        # Indicator while the inference layer is shown
        font = QFont()
        font.setPixelSize(24)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(QPointF(self.width() - 180, 30), "[INS MASK ON]")

    def _draw_inference_labels(self, painter: QPainter):
        """Draw the labels (matching info) of the predicted masks"""
        img_w, img_h = self._image_size
        if img_w == 0 or img_h == 0:
            return

        font = QFont()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)

        for i, mask in enumerate(self._inference_masks):
            if mask is None or mask.shape != (img_h, img_w):
                continue

            binary_mask = (mask > 0).astype(np.uint8)
            if binary_mask.max() == 0:
                continue

            # Mask centroid
            ys, xs = np.where(binary_mask > 0)
            if len(xs) == 0:
                continue
            center_x = int(xs.mean())
            center_y = int(ys.mean())

            # Label text
            if i < len(self._match_results):
                result = self._match_results[i]
                if result.matched:
                    label = f"Ins#{i} -> GT#{result.gt_idx}"
                    detail = f"IoU:{result.iou:.2f} Cov:{result.coverage:.2f}"
                    text_color = QColor(200, 255, 200)  # light green
                else:
                    label = f"Ins#{i} -> X"
                    detail = f"IoU:{result.iou:.2f} Cov:{result.coverage:.2f}" if result.iou > 0 else ""
                    text_color = QColor(255, 180, 180)  # light red
            else:
                label = f"Ins#{i}"
                detail = ""
                text_color = QColor(255, 255, 255)

            pos = self._img_to_widget(center_x, center_y)

            # Background
            fm = painter.fontMetrics()
            label_width = max(fm.horizontalAdvance(label), fm.horizontalAdvance(detail) if detail else 0)
            label_height = fm.height() * (2 if detail else 1)

            bg_rect = QRectF(
                pos.x() - label_width / 2 - 4,
                pos.y() - label_height / 2 - 2,
                label_width + 8,
                label_height + 4
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
            painter.drawRoundedRect(bg_rect, 4, 4)

            # Text
            painter.setPen(QPen(text_color))
            text_y = pos.y() - (fm.height() / 2 if detail else 0) + fm.ascent() / 2
            painter.drawText(QPointF(pos.x() - fm.horizontalAdvance(label) / 2, text_y), label)
            if detail:
                text_y2 = text_y + fm.height()
                painter.drawText(QPointF(pos.x() - fm.horizontalAdvance(detail) / 2, text_y2), detail)

    def _draw_info_text(self, painter: QPainter):
        """Draw the info text"""
        if not self._info_lines:
            return

        y = 25
        line_height = 25

        for text, font_scale, color_bgr in self._info_lines:
            font = QFont()
            font_size = int(self.FONT_SIZE_INFO * font_scale / 0.5)  # 0.5 = reference scale
            font.setPixelSize(font_size)
            painter.setFont(font)

            # Shadow
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(QPointF(12, y + 2), text)

            # Text
            color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])
            painter.setPen(QPen(color))
            painter.drawText(QPointF(10, y), text)

            y += line_height

    def mousePressEvent(self, event):
        """On click, report image coordinates (padding included)

        - with predicted masks: toggle the inference layer
        - otherwise: emit the clicked coordinates
        """
        padded_rect, scale = self._get_padded_image_rect()
        if not padded_rect.isEmpty():
            img_x, img_y = self._widget_to_img(event.x(), event.y())
            # Clicks inside the padded area (including outside the image)
            pad = self.VIRTUAL_PADDING
            if -pad <= img_x < self._image_size[0] + pad and -pad <= img_y < self._image_size[1] + pad:
                if self._inference_masks:
                    self.toggle_inference_layer()
                else:
                    self.clicked.emit(img_x, img_y)
        super().mousePressEvent(event)


def build_info_lines(
    split: str = None,
    image_id: int = None,
    file_name: str = None,
    pkl_name: str = None,
    row_idx: int = None,
    scene_name: str = None,
    crop_size: str = None,
    width: int = None,
    height: int = None,
    angle: float = None,
    mirrored: bool = None,
    cx: float = None,
    cy: float = None,
    annotations: list = None,
    shortcuts: str = None,
) -> list:
    """Build the info text overlay lines

    Args:
        split: dataset split (train/validation)
        image_id: image ID
        file_name: file name
        pkl_name: pkl file name (test.py)
        row_idx: row in the pkl (test.py)
        scene_name: scene name (viewer.py)
        crop_size: crop size string (e.g. "640x480")
        width: image width (used without crop_size)
        height: image height (used without crop_size)
        angle: rotation angle (degrees)
        mirrored: whether the image is mirrored
        cx: crop centre X
        cy: crop centre Y
        annotations: annotation list (person and ignore counts)
        shortcuts: keyboard shortcut help

    Returns:
        info_lines: [(text, font_scale, color_bgr), ...]
    """
    # Placeholder for unavailable values
    NA = "-"

    # Line 1: basic info
    line1_parts = []

    if split:
        line1_parts.append(f"Split: {split.upper()}")

    # PKL/Row (test.py) or Scene/ID (viewer.py)
    if pkl_name is not None:
        line1_parts.append(f"PKL: {pkl_name}")
    if row_idx is not None:
        line1_parts.append(f"Row: {row_idx}")
    if scene_name is not None:
        line1_parts.append(f"Scene: {scene_name}")
    if image_id is not None:
        line1_parts.append(f"ID: {image_id}")

    # Persons
    if annotations is not None:
        n_persons = len(annotations)
        valid_count = sum(1 for ann in annotations if not ann.get("ignore", False))
        ignore_count = n_persons - valid_count
        line1_parts.append(f"Persons: {n_persons} (valid={valid_count}, ignore={ignore_count})")

    line1 = " | ".join(line1_parts) if line1_parts else NA

    # Line 2: transform info
    line2_parts = []

    if crop_size:
        line2_parts.append(f"Size: {crop_size}")
    elif width is not None and height is not None:
        line2_parts.append(f"Size: {width}x{height}")

    if angle is not None:
        angle_text = f"Angle: {angle:.1f}°"
        if mirrored:
            angle_text += " [MIRRORED]"
        line2_parts.append(angle_text)
    elif mirrored:
        line2_parts.append("[MIRRORED]")

    if cx is not None and cy is not None:
        line2_parts.append(f"Center: ({cx:.0f}, {cy:.0f})")

    if file_name:
        line2_parts.append(f"File: {file_name}")

    line2 = " | ".join(line2_parts) if line2_parts else NA

    # Line 3: shortcuts
    if shortcuts is None:
        shortcuts = "[Enter/Space: Next] [Q: Quit]"

    return [
        (line1, 0.6, (255, 255, 255)),
        (line2, 0.5, (200, 200, 200)),
        (shortcuts, 0.5, (180, 180, 180)),
    ]


def update_3d_views(
    scroll_area,
    scroll_layout,
    person_3d_widgets: list,
    annotations: list,
    keypoint_names: list,
    skeleton: list,
    use_body: bool = False,
) -> list:
    """Rebuild the 3D views

    Args:
        scroll_area: QScrollArea (for the viewport height)
        scroll_layout: QVBoxLayout holding the widgets
        person_3d_widgets: existing Person3DWidget list (cleared)
        annotations: annotations (with keypoints_cam)
        keypoint_names: keypoint names (ignored when use_body=True)
        skeleton: skeleton (ignored when use_body=True)
        use_body: show keypoints_body_cam (31 body points) instead of the head

    Returns:
        new Person3DWidget list
    """
    from config import BODY_KEYPOINT_NAMES  # lazy import to avoid an import cycle

    # Remove the existing widgets
    for widget in person_3d_widgets:
        scroll_layout.removeWidget(widget)
        widget.deleteLater()
    person_3d_widgets.clear()

    # Height per person (50% of the viewport, at least 300px)
    viewport_height = scroll_area.viewport().height()
    per_person_height = max(300, int(viewport_height * 0.5))

    kpt_key = "keypoints_body_cam" if use_body else "keypoints_cam"
    names = BODY_KEYPOINT_NAMES if use_body else keypoint_names
    skel = BODY_SKELETON if use_body else skeleton
    colors = BODY_KEYPOINT_COLORS_RGB if use_body else None

    new_widgets = []

    for i, anno in enumerate(annotations):
        if anno.get(kpt_key) is None:
            continue

        widget = Person3DWidget(width=5, height=per_person_height / 100)
        widget.setMinimumHeight(per_person_height)
        widget.setMaximumHeight(per_person_height)

        widget.plot_keypoints(
            anno[kpt_key],
            names,
            skel,
            i,
            keypoint_colors=colors,
            show_labels=not use_body,  # 31 labels would overlap
        )

        scroll_layout.addWidget(widget)
        new_widgets.append(widget)

    # Spacer at the bottom
    scroll_layout.addStretch()

    return new_widgets


class Person3DWidget(FigureCanvas):
    """3D view of one person (rotatable with the mouse)"""

    def __init__(self, parent=None, width=4, height=3):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.elev = 0
        self.azim = -90
        self._dragging = False
        self._last_pos = None
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)

    def _on_press(self, event):
        if event.button == 1:
            self._dragging = True
            self._last_pos = (event.x, event.y)

    def _on_release(self, event):
        self._dragging = False
        self._last_pos = None

    def _on_motion(self, event):
        if self._dragging and self._last_pos and event.x and event.y:
            dx = event.x - self._last_pos[0]
            self.azim += dx * 0.5
            self._last_pos = (event.x, event.y)
            self._update_view()

    def _update_view(self):
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.draw()

    def plot_keypoints(
        self,
        keypoints_cam: list,
        keypoint_names: list,
        skeleton: list = None,
        person_idx: int = 0,
        keypoint_colors: list = None,
        show_labels: bool = True,
    ):
        """
        Plot unified-coordinate keypoints.

        Args:
            keypoints_cam: [x, y, z, v, ...] (4 per point; 44 for the head, 124 for the body)
            keypoint_names: keypoint names
            skeleton: skeleton (config.SKELETON by default)
            person_idx: person index (title)
            keypoint_colors: keypoint colours (config.KEYPOINT_COLORS_RGB by default)
            show_labels: draw a name label at each point
        """
        self.ax.clear()

        if skeleton is None:
            skeleton = SKELETON
        if keypoint_colors is None:
            keypoint_colors = KEYPOINT_COLORS_RGB

        stride = 4
        n_kpts = len(keypoints_cam) // stride

        points = []
        for i in range(n_kpts):
            points.append((
                keypoints_cam[i * stride],
                keypoints_cam[i * stride + 1],
                keypoints_cam[i * stride + 2],
            ))

        points = np.array(points)

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()

        margin = 0.05

        for i, (x, y, z) in enumerate(points):
            color = keypoint_colors[i % len(keypoint_colors)]
            self.ax.scatter([x], [y], [z], c=[color], s=50 if show_labels else 30)
            if show_labels:
                label = keypoint_names[i] if i < len(keypoint_names) else f"kpt{i}"
                self.ax.text(x, y, z, f" {label}", fontsize=7, color=color)

        for s1, s2 in skeleton:
            if s1 < n_kpts and s2 < n_kpts:
                x1, y1, z1 = points[s1]
                x2, y2, z2 = points[s2]
                self.ax.plot([x1, x2], [y1, y2], [z1, z2], 'gray', linewidth=1, alpha=0.7)

        grid_z_pos = z_min
        grid_x = np.linspace(x_min - margin, x_max + margin, 5)
        grid_y = np.linspace(y_min - margin, y_max + margin, 5)

        for gx in grid_x:
            self.ax.plot([gx, gx], [y_min - margin, y_max + margin], [grid_z_pos, grid_z_pos],
                        'lightgray', linewidth=0.5, alpha=0.5)
        for gy in grid_y:
            self.ax.plot([x_min - margin, x_max + margin], [gy, gy], [grid_z_pos, grid_z_pos],
                        'lightgray', linewidth=0.5, alpha=0.5)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2 + margin

        self.ax.set_xlim(x_center - max_range, x_center + max_range)
        self.ax.set_ylim(y_center - max_range, y_center + max_range)
        self.ax.set_zlim(z_center - max_range, z_center + max_range)

        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X (left/right)')
        self.ax.set_ylabel('Y (depth)')
        self.ax.set_zlabel('Z (up)')
        self.ax.set_title(f'Person #{person_idx}', fontsize=14, fontweight='bold', y=0.95)

        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.fig.subplots_adjust(top=0.92)
        self.draw()
