"""MPS (Apple Silicon) NMS Workaround

NMS is not supported on the MPS backend, so it is run on the CPU and the result is moved
back to MPS.

Also makes mmengine InstanceData accept MPS index tensors (applied by patch_nms(), which
every tool running on MPS calls).
"""

from typing import Union

import torch
import torchvision.ops as tv_ops


_original_tv_nms = tv_ops.nms
_original_tv_batched_nms = tv_ops.batched_nms
_patched = False


def _patched_tv_nms(boxes, scores, iou_threshold):
    if boxes.device.type == 'mps':
        device = boxes.device
        keep = _original_tv_nms(boxes.cpu(), scores.cpu(), iou_threshold)
        return keep.to(device)
    return _original_tv_nms(boxes, scores, iou_threshold)


def _patched_tv_batched_nms(boxes, scores, idxs, iou_threshold):
    if boxes.device.type == 'mps':
        device = boxes.device
        keep = _original_tv_batched_nms(boxes.cpu(), scores.cpu(), idxs.cpu(), iou_threshold)
        return keep.to(device)
    return _original_tv_batched_nms(boxes, scores, idxs, iou_threshold)


class _MPSTensorMeta(type):
    """Metaclass that makes isinstance() match MPS tensors of a given dtype"""

    def __instancecheck__(cls, obj):
        return (isinstance(obj, torch.Tensor) and obj.device.type == 'mps'
                and obj.dtype == cls._dtype)


class _MPSBoolTensor(metaclass=_MPSTensorMeta):
    _dtype = torch.bool


class _MPSLongTensor(metaclass=_MPSTensorMeta):
    _dtype = torch.int64


def _patch_instance_data_index_types():
    """Make mmengine InstanceData.__getitem__ accept MPS index tensors

    mmengine defines BoolTypeTensor/LongTypeTensor as Unions of the legacy CPU/CUDA tensor
    types, so a bool mask on MPS fails the isinstance check with an AssertionError. This
    happens in the RTMDet-Ins post-processing (small-bbox removal `results[valid_mask]` with
    min_bbox_size). __getitem__ looks the module globals up on every call, so replacing them
    is enough.
    """
    try:
        import mmengine.structures.instance_data as instance_data
    except ImportError:
        return

    instance_data.BoolTypeTensor = Union[instance_data.BoolTypeTensor, _MPSBoolTensor]
    instance_data.LongTypeTensor = Union[instance_data.LongTypeTensor, _MPSLongTensor]
    instance_data.IndexType = Union[instance_data.IndexType, _MPSBoolTensor, _MPSLongTensor]


def patch_nms():
    """Patch the NMS functions for MPS (and the InstanceData index type check)"""
    global _patched
    if _patched:
        return

    _patch_instance_data_index_types()

    tv_ops.nms = _patched_tv_nms
    tv_ops.batched_nms = _patched_tv_batched_nms

    # Patch mmcv.ops as well
    try:
        import mmcv.ops as mmcv_ops
        _original_mmcv_nms = mmcv_ops.nms
        _original_mmcv_batched_nms = mmcv_ops.batched_nms

        def _patched_mmcv_nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
            if boxes.device.type == 'mps':
                device = boxes.device
                result = _original_mmcv_nms(
                    boxes.cpu(), scores.cpu(), iou_threshold,
                    offset=offset, score_threshold=score_threshold, max_num=max_num
                )
                if isinstance(result, tuple):
                    return tuple(r.to(device) if isinstance(r, torch.Tensor) else r for r in result)
                return result.to(device) if isinstance(result, torch.Tensor) else result
            return _original_mmcv_nms(boxes, scores, iou_threshold, offset=offset,
                                       score_threshold=score_threshold, max_num=max_num)

        def _patched_mmcv_batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
            if boxes.device.type == 'mps':
                device = boxes.device
                dets, keep = _original_mmcv_batched_nms(
                    boxes.cpu(), scores.cpu(), idxs.cpu(), nms_cfg, class_agnostic=class_agnostic
                )
                return dets.to(device), keep
            return _original_mmcv_batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=class_agnostic)

        mmcv_ops.nms = _patched_mmcv_nms
        mmcv_ops.batched_nms = _patched_mmcv_batched_nms
        print("MPS NMS workaround: patched mmcv.ops.nms and batched_nms")
    except ImportError:
        pass

    _patched = True
