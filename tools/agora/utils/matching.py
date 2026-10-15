#!/usr/bin/env python3
"""
Mask matching utilities.

Matches GT (annotation) masks to RTMDet-Ins masks with the Hungarian algorithm.

- Besides IoU, handles RTMDet-Ins masks that cover only part of a person:
  the GT mask is amodal (whole body even when hidden), whereas RTMDet-Ins
  segments only the visible part.
- A high coverage (Intersection / Ins_area) counts as a match even when the
  IoU is low.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import (
    MATCHING_IOU_THRESHOLD,
    MATCHING_SCORE_THRESHOLD,
    MATCHING_MIN_COVERAGE,
)


@dataclass
class MatchResult:
    """Matching result"""

    gt_idx: int  # GT mask index (-1 = no match)
    ins_idx: int  # Ins mask index (-1 = no match)
    iou: float  # IoU
    coverage: float  # Intersection / Ins_area (how much of the Ins mask lies inside the GT)
    gt_recall: float  # Intersection / GT_area (how much of the GT the Ins mask covers)
    matched: bool  # whether the match holds


def decode_rle_mask(rle: dict) -> np.ndarray:
    """Decode an RLE mask

    Args:
        rle: RLE-encoded mask {'size': [h, w], 'counts': ...}

    Returns:
        binary mask (h, w)
    """
    if isinstance(rle["counts"], str):
        rle_copy = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    else:
        rle_copy = rle.copy()

    return mask_utils.decode(rle_copy)


def compute_mask_metrics(
    gt_mask: np.ndarray, ins_mask: np.ndarray
) -> Tuple[float, float, float, float]:
    """Metrics between two masks

    Args:
        gt_mask: GT mask (h, w), binary
        ins_mask: Ins mask (h, w), binary

    Returns:
        (iou, coverage, gt_recall, intersection_area)
        - iou: Intersection / Union
        - coverage: Intersection / Ins_area (precision of the Ins mask)
        - gt_recall: Intersection / GT_area (how much of the GT is covered)
        - intersection_area: intersection area (pixels)
    """
    gt_binary = (gt_mask > 0).astype(np.uint8)
    ins_binary = (ins_mask > 0).astype(np.uint8)

    intersection = np.logical_and(gt_binary, ins_binary).sum()
    gt_area = gt_binary.sum()
    ins_area = ins_binary.sum()
    union = gt_area + ins_area - intersection

    iou = intersection / union if union > 0 else 0.0

    # Coverage: fraction of the Ins mask inside the GT. High = what Ins detected
    # is correct (although it may cover only part of the GT)
    coverage = intersection / ins_area if ins_area > 0 else 0.0

    # GT recall: fraction of the GT covered by the Ins mask
    gt_recall = intersection / gt_area if gt_area > 0 else 0.0

    return iou, coverage, gt_recall, float(intersection)


def compute_matching_score(
    iou: float,
    coverage: float,
    gt_recall: float,
    iou_weight: float = 0.4,
    coverage_weight: float = 0.4,
    recall_weight: float = 0.2,
) -> float:
    """Combined matching score

    Combines IoU with coverage and GT recall so that an Ins mask covering only
    the visible part of the GT (low IoU, high coverage) can still match.

    Args:
        iou: IoU
        coverage: Intersection / Ins_area
        gt_recall: Intersection / GT_area
        iou_weight: IoU weight
        coverage_weight: coverage weight
        recall_weight: GT recall weight

    Returns:
        score in [0, 1]
    """
    return iou * iou_weight + coverage * coverage_weight + gt_recall * recall_weight


def match_masks(
    gt_masks: List[np.ndarray],
    ins_masks: List[np.ndarray],
    iou_threshold: float = MATCHING_IOU_THRESHOLD,
    score_threshold: float = MATCHING_SCORE_THRESHOLD,
    min_coverage: float = MATCHING_MIN_COVERAGE,
) -> List[MatchResult]:
    """Match GT masks to Ins masks

    Optimal assignment with the Hungarian algorithm.

    A pair is a match candidate if either holds:
    1. IoU >= iou_threshold
    2. coverage >= min_coverage and combined score >= score_threshold

    Condition 2 rescues Ins masks that cover only the visible part.

    Args:
        gt_masks: GT masks (decoded, each (h, w))
        ins_masks: Ins masks (each (h, w))
        iou_threshold: IoU threshold
        score_threshold: combined score threshold
        min_coverage: minimum coverage (rescue condition below the IoU threshold)

    Returns:
        one MatchResult per Ins mask
    """
    n_gt = len(gt_masks)
    n_ins = len(ins_masks)

    if n_ins == 0:
        return []

    if n_gt == 0:
        # No GT: no Ins mask can match
        return [
            MatchResult(
                gt_idx=-1,
                ins_idx=i,
                iou=0.0,
                coverage=0.0,
                gt_recall=0.0,
                matched=False,
            )
            for i in range(n_ins)
        ]

    # Cost matrix (the Hungarian algorithm minimizes, so the score is negated)
    # rows: Ins, columns: GT
    cost_matrix = np.zeros((n_ins, n_gt), dtype=np.float64)
    metrics_cache = {}  # (ins_idx, gt_idx) -> (iou, coverage, gt_recall)

    for i, ins_mask in enumerate(ins_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou, coverage, gt_recall, _ = compute_mask_metrics(gt_mask, ins_mask)
            metrics_cache[(i, j)] = (iou, coverage, gt_recall)

            is_candidate = False
            if iou >= iou_threshold:
                is_candidate = True
            elif coverage >= min_coverage:
                score = compute_matching_score(iou, coverage, gt_recall)
                if score >= score_threshold:
                    is_candidate = True

            if is_candidate:
                # Higher score = lower cost
                score = compute_matching_score(iou, coverage, gt_recall)
                cost_matrix[i, j] = -score
            else:
                # Not matchable: high cost
                cost_matrix[i, j] = 1.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results = []
    matched_gt_indices = set()

    for ins_idx in range(n_ins):
        # GT assigned to this Ins mask
        assignment_idx = np.where(row_ind == ins_idx)[0]

        if len(assignment_idx) > 0:
            gt_idx = col_ind[assignment_idx[0]]
            iou, coverage, gt_recall = metrics_cache[(ins_idx, gt_idx)]

            # Re-check the match conditions
            is_matched = False
            if iou >= iou_threshold:
                is_matched = True
            elif coverage >= min_coverage:
                score = compute_matching_score(iou, coverage, gt_recall)
                if score >= score_threshold:
                    is_matched = True

            if is_matched and cost_matrix[ins_idx, gt_idx] < 1.0:
                matched_gt_indices.add(gt_idx)
                results.append(
                    MatchResult(
                        gt_idx=gt_idx,
                        ins_idx=ins_idx,
                        iou=iou,
                        coverage=coverage,
                        gt_recall=gt_recall,
                        matched=True,
                    )
                )
            else:
                results.append(
                    MatchResult(
                        gt_idx=-1,
                        ins_idx=ins_idx,
                        iou=iou,
                        coverage=coverage,
                        gt_recall=gt_recall,
                        matched=False,
                    )
                )
        else:
            results.append(
                MatchResult(
                    gt_idx=-1,
                    ins_idx=ins_idx,
                    iou=0.0,
                    coverage=0.0,
                    gt_recall=0.0,
                    matched=False,
                )
            )

    return results


def get_unmatched_gt_indices(
    match_results: List[MatchResult], n_gt: int
) -> List[int]:
    """Indices of the GT masks that were not matched

    Args:
        match_results: matching results
        n_gt: number of GT masks

    Returns:
        unmatched GT indices
    """
    matched_gt = {r.gt_idx for r in match_results if r.matched}
    return [i for i in range(n_gt) if i not in matched_gt]
