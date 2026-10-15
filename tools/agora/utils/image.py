#!/usr/bin/env python3
"""
Image utilities.

Loading, saving, rotation / cropping and mask conversion.
Uses torch on CUDA when available for speed.
"""

from pathlib import Path
from typing import Optional

# Import torch before cv2 (avoids a DLL conflict on Windows)
import torch
import torch.nn.functional as F

import cv2
import numpy as np
from pycocotools import mask as mask_utils

# kornia is only checked for availability (GPU path is enabled with CUDA + kornia)
try:
    import kornia.geometry.transform  # noqa: F401
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


# ============================================================
# GPU image processor
# ============================================================

class GPUImageProcessor:
    """
    Image processing on the GPU.

    Keeps the source image on the GPU and runs many crop / rotation
    operations on it quickly.
    """

    def __init__(self, img: np.ndarray, device: str = "cuda"):
        """
        Args:
            img: source image (BGR, numpy array)
            device: device to use ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # BGR -> RGB, HWC -> CHW, float, to device
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0)
        self.img_tensor = self.img_tensor.to(self.device)

        self.img_h, self.img_w = img.shape[:2]

    def crop_rotated(
        self,
        cx: float,
        cy: float,
        width: int,
        height: int,
        angle_deg: float,
    ) -> Optional[np.ndarray]:
        """
        Cut out a rotated crop (GPU).

        Same definition as the CPU version:
        1. rotate the whole source image by angle_deg around (cx, cy) (no scaling)
        2. cut width x height with top-left (cx - width/2, cy - height/2)

        Args:
            cx, cy: crop centre (pixels)
            width, height: crop size (pixels)
            angle_deg: rotation angle (degrees, counter-clockwise positive)

        Returns:
            cropped image (BGR, numpy array) or None
        """
        if self.device == "cpu":
            return self._crop_rotated_cpu(cx, cy, width, height, angle_deg)

        try:
            # Output grid (u, v) in [0, width) x [0, height)
            u = torch.arange(width, device=self.device, dtype=torch.float32)
            v = torch.arange(height, device=self.device, dtype=torch.float32)
            grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (height, width)

            # Offset from the crop centre.
            # CPU version: sx = cx - width/2, cropped = rotated[sy:ey, sx:ex],
            # i.e. output (u, v) is rotated-image pixel (sx+u, sy+v)
            dx = grid_u - (width / 2)   # (height, width)
            dy = grid_v - (height / 2)  # (height, width)

            # Map back to source coordinates with the inverse rotation.
            # The inverse of cv2.getRotationMatrix2D((cx, cy), angle, 1.0) is a
            # rotation by -angle around (cx, cy):
            # src_x = cx + cos(a)*dx - sin(a)*dy
            # src_y = cy + sin(a)*dx + cos(a)*dy
            angle_rad = angle_deg * np.pi / 180
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            src_x = cx + cos_a * dx - sin_a * dy
            src_y = cy + sin_a * dx + cos_a * dy

            # Normalize to [-1, 1] for grid_sample
            # align_corners=True: pixel 0 -> -1, pixel (size-1) -> +1
            norm_x = 2.0 * src_x / (self.img_w - 1) - 1.0
            norm_y = 2.0 * src_y / (self.img_h - 1) - 1.0

            # Grid (1, height, width, 2)
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

            cropped = F.grid_sample(
                self.img_tensor, grid,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

            # CHW -> HWC, to CPU, RGB -> BGR
            result = cropped[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            return result

        except Exception:
            # Fall back to the CPU on error
            return self._crop_rotated_cpu(cx, cy, width, height, angle_deg)

    def _crop_rotated_cpu(
        self,
        cx: float,
        cy: float,
        width: int,
        height: int,
        angle_deg: float,
    ) -> Optional[np.ndarray]:
        """CPU rotated crop (fallback)"""
        img_rgb = self.img_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return crop_rotated_image_from_array(img_bgr, cx, cy, width, height, angle_deg)

    def release(self):
        """Release GPU memory"""
        del self.img_tensor
        if self.device == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# GPU availability
# ============================================================

_GPU_AVAILABLE = None


def _check_gpu_available() -> bool:
    """Whether GPU image processing is available"""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    _GPU_AVAILABLE = torch.cuda.is_available() and KORNIA_AVAILABLE
    return _GPU_AVAILABLE


def get_image_processor_name() -> str:
    """Name of the image processing backend in use"""
    if _check_gpu_available():
        return "kornia (GPU CUDA)"
    else:
        return "OpenCV (CPU)"


# ============================================================
# Basic image operations
# ============================================================

def crop_rotated_image_from_array(
    img: np.ndarray,
    cx: float,
    cy: float,
    width: int,
    height: int,
    angle_deg: float,
) -> Optional[np.ndarray]:
    """
    Cut out a rotated crop (CPU).

    Args:
        img: source image
        cx, cy: crop centre
        width, height: crop size
        angle_deg: rotation angle (degrees, counter-clockwise positive)

    Returns:
        cropped image, or None on failure
    """

    # Rotation matrix around the crop centre
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # The rotated image keeps the source size (before cropping)
    img_h, img_w = img.shape[:2]
    rotated = cv2.warpAffine(img, rotation_matrix, (img_w, img_h))

    # Cut width x height around the centre
    sx = int(cx - width / 2)
    sy = int(cy - height / 2)
    ex = sx + width
    ey = sy + height

    # Bounds check (the centre range already guarantees this; kept as a safeguard)
    sx = max(0, sx)
    sy = max(0, sy)
    ex = min(img_w, ex)
    ey = min(img_h, ey)

    cropped = rotated[sy:ey, sx:ex]

    if cropped.shape[0] != height or cropped.shape[1] != width:
        return None

    return cropped


# ============================================================
# JPEG saving (GPU encoder when available)
# ============================================================

_JPEG_ENCODER_CHECKED = False
_JPEG_ENCODER_NAME = "OpenCV (CPU)"
_JPEG_ENCODER_TYPE = "opencv"  # "torchvision_cuda", "simplejpeg", "opencv"


def _init_jpeg_encoder(debug: bool = False):
    """Select the fastest available JPEG encoder (lazy)"""
    global _JPEG_ENCODER_CHECKED, _JPEG_ENCODER_NAME, _JPEG_ENCODER_TYPE

    if _JPEG_ENCODER_CHECKED:
        return

    _JPEG_ENCODER_CHECKED = True

    # 1. torchvision CUDA encode_jpeg (fastest)
    try:
        import torchvision.io as tio
        if torch.cuda.is_available():
            test_img = torch.randint(0, 255, (3, 8, 8), dtype=torch.uint8, device='cuda')
            tio.encode_jpeg(test_img, quality=60)
            _JPEG_ENCODER_TYPE = "torchvision_cuda"
            _JPEG_ENCODER_NAME = "torchvision (GPU CUDA)"
            return
    except Exception as e:
        if debug:
            print(f"[DEBUG] torchvision CUDA failed: {type(e).__name__}: {e}")
        pass

    # 2. simplejpeg (libjpeg-turbo, SIMD)
    try:
        import simplejpeg
        test_img = np.zeros((8, 8, 3), dtype=np.uint8)
        simplejpeg.encode_jpeg(test_img, quality=60, colorspace='BGR')
        _JPEG_ENCODER_TYPE = "simplejpeg"
        _JPEG_ENCODER_NAME = "simplejpeg (CPU libjpeg-turbo)"
        return
    except Exception:
        pass

    # 3. Fallback: OpenCV
    _JPEG_ENCODER_TYPE = "opencv"
    _JPEG_ENCODER_NAME = "OpenCV (CPU)"


def save_image_jpeg(
    img: np.ndarray,
    dst_path: Path,
    quality: int = 60,
) -> bool:
    """
    Save an image as JPEG.

    Encoder priority:
    1. torchvision CUDA encode_jpeg (GPU, fastest)
    2. simplejpeg (libjpeg-turbo, CPU SIMD)
    3. OpenCV (CPU)

    Args:
        img: image (BGR)
        dst_path: destination path
        quality: JPEG quality (1-100, lower = smaller files; default 60)

    Returns:
        True on success
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    _init_jpeg_encoder()

    # 1. torchvision CUDA
    if _JPEG_ENCODER_TYPE == "torchvision_cuda":
        try:
            import torchvision.io as tio
            # BGR -> RGB, CHW
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).cuda()
            jpeg_data = tio.encode_jpeg(img_tensor, quality=quality)
            with open(dst_path, 'wb') as f:
                f.write(jpeg_data.cpu().numpy().tobytes())
            return True
        except Exception:
            pass

    # 2. simplejpeg
    if _JPEG_ENCODER_TYPE == "simplejpeg":
        try:
            import simplejpeg
            jpeg_data = simplejpeg.encode_jpeg(img, quality=quality, colorspace='BGR')
            with open(dst_path, 'wb') as f:
                f.write(jpeg_data)
            return True
        except Exception:
            pass

    # 3. Fallback: OpenCV
    cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return True


def get_jpeg_encoder_name() -> str:
    """Name of the JPEG encoder in use"""
    _init_jpeg_encoder()
    return _JPEG_ENCODER_NAME


# ============================================================
# Mirroring
# ============================================================

def mirror_image_array(img: np.ndarray) -> np.ndarray:
    """
    Flip an image horizontally (GPU when available).

    Args:
        img: image (BGR)

    Returns:
        flipped image
    """
    if _check_gpu_available():
        try:
            img_tensor = torch.from_numpy(img).cuda()
            flipped = torch.flip(img_tensor, dims=[1])  # flip the width axis
            return flipped.cpu().numpy()
        except Exception:
            pass

    # Fallback: CPU
    return cv2.flip(img, 1)  # 1 = horizontal


# ============================================================
# Masks
# ============================================================

def mask_to_binary_2d(mask: np.ndarray) -> np.ndarray:
    """Reduce a mask image to 2D (non-zero = person), BGRA safe

    OpenCV returns BGRA, so looking at a single channel would miss persons whose
    colour has a zero component there. The max over all colour channels detects
    the person region regardless of its colour.
    """
    if mask.ndim == 3:
        return mask[:, :, :3].max(axis=2)
    return mask


def load_mask_2d(mask_path: Path) -> Optional[np.ndarray]:
    """Load a mask PNG as 2D (non-zero = person region)"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    return mask_to_binary_2d(mask)


def _rotate_mask_gpu(
    mask_2d: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
) -> np.ndarray:
    """
    Rotate a mask on the GPU.

    Same result as the CPU version:
    cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0) + cv2.warpAffine
    """
    img_h, img_w = mask_2d.shape

    # To tensor (float32)
    mask_tensor = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

    # Output grid
    u = torch.arange(img_w, device='cuda', dtype=torch.float32)
    v = torch.arange(img_h, device='cuda', dtype=torch.float32)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (img_h, img_w)

    # Offset from the rotation centre
    dx = grid_u - cx
    dy = grid_v - cy

    # Inverse of the OpenCV warpAffine rotation
    angle_rad = angle_deg * np.pi / 180
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # src_x = cx + cos(a)*dx - sin(a)*dy
    # src_y = cy + sin(a)*dx + cos(a)*dy
    src_x = cx + cos_a * dx - sin_a * dy
    src_y = cy + sin_a * dx + cos_a * dy

    # Normalize to [-1, 1] for grid_sample
    norm_x = 2.0 * src_x / (img_w - 1) - 1.0
    norm_y = 2.0 * src_y / (img_h - 1) - 1.0

    # Grid (1, img_h, img_w, 2)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    # Nearest-neighbour sampling (mask)
    rotated = F.grid_sample(
        mask_tensor, grid,
        mode='nearest', padding_mode='zeros', align_corners=True
    )

    return rotated[0, 0].cpu().numpy().astype(np.uint8)


def compute_rotated_cropped_bbox_from_array(
    mask_2d: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    crop_width: int,
    crop_height: int,
) -> tuple[Optional[list], bool]:
    """
    Rotate a mask, compute its bbox and return it in crop coordinates.

    The bbox is computed on the whole rotated mask and then shifted into crop
    coordinates. It is not clamped: a person cut off by the crop gets a bbox
    extending outside the crop.

    Args:
        mask_2d: mask (source resolution, non-zero = person)
        cx, cy: crop centre (source image coordinates)
        angle_deg: rotation angle (degrees)
        crop_width, crop_height: crop size

    Returns:
        bbox: [x, y, width, height] (crop coordinates, may extend outside)
        is_clipped: True if the bbox extends outside the crop
    """
    img_h, img_w = mask_2d.shape[:2]

    # Rotate the mask (GPU or CPU)
    if _check_gpu_available() and angle_deg != 0:
        try:
            rotated_mask = _rotate_mask_gpu(mask_2d, cx, cy, angle_deg)
        except Exception:
            # Fallback: CPU
            rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
            rotated_mask = cv2.warpAffine(mask_2d, rotation_matrix, (img_w, img_h))
    elif angle_deg != 0:
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated_mask = cv2.warpAffine(mask_2d, rotation_matrix, (img_w, img_h))
    else:
        rotated_mask = mask_2d

    # Non-zero pixels of the whole rotated mask
    non_zero = cv2.findNonZero(rotated_mask)
    if non_zero is None:
        return None, False

    # Bbox in rotated-image coordinates
    orig_x, orig_y, w, h = cv2.boundingRect(non_zero)

    # To crop coordinates (crop top-left is (cx - crop_width/2, cy - crop_height/2))
    crop_offset_x = cx - crop_width / 2
    crop_offset_y = cy - crop_height / 2

    x = orig_x - crop_offset_x
    y = orig_y - crop_offset_y

    is_clipped = (x < 0 or y < 0 or x + w > crop_width or y + h > crop_height)

    return [float(x), float(y), float(w), float(h)], is_clipped


def compute_rotated_cropped_mask_rle_from_array(
    mask_2d: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    crop_width: int,
    crop_height: int,
) -> Optional[dict]:
    """
    Rotate and crop a mask and encode it as COCO RLE.

    The mask is clipped to the crop.

    Args:
        mask_2d: mask (source resolution, non-zero = person)
        cx, cy: crop centre (source image coordinates)
        angle_deg: rotation angle (degrees)
        crop_width, crop_height: crop size

    Returns:
        COCO RLE dict {"size": [h, w], "counts": ...}, or None if the crop is empty
    """
    img_h, img_w = mask_2d.shape[:2]

    # Rotate the mask (GPU or CPU)
    if _check_gpu_available() and angle_deg != 0:
        try:
            rotated_mask = _rotate_mask_gpu(mask_2d, cx, cy, angle_deg)
        except Exception:
            rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
            rotated_mask = cv2.warpAffine(mask_2d, rotation_matrix, (img_w, img_h))
    elif angle_deg != 0:
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated_mask = cv2.warpAffine(mask_2d, rotation_matrix, (img_w, img_h))
    else:
        rotated_mask = mask_2d

    # Crop window
    sx = int(cx - crop_width / 2)
    sy = int(cy - crop_height / 2)
    ex = sx + crop_width
    ey = sy + crop_height

    # Crop the mask (pixels outside the source image stay 0)
    cropped_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)

    # Overlap of the source image and the crop window
    src_sx = max(0, sx)
    src_sy = max(0, sy)
    src_ex = min(img_w, ex)
    src_ey = min(img_h, ey)

    dst_sx = src_sx - sx
    dst_sy = src_sy - sy
    dst_ex = dst_sx + (src_ex - src_sx)
    dst_ey = dst_sy + (src_ey - src_sy)

    if src_ex > src_sx and src_ey > src_sy:
        cropped_mask[dst_sy:dst_ey, dst_sx:dst_ex] = rotated_mask[src_sy:src_ey, src_sx:src_ex]

    if cv2.countNonZero(cropped_mask) == 0:
        return None

    binary_mask = (cropped_mask > 0).astype(np.uint8)

    # RLE encode (Fortran order)
    binary_mask_fortran = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(binary_mask_fortran)

    # counts: bytes -> str (JSON)
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def mirror_mask_rle(rle: dict, crop_width: int) -> dict:
    """
    Flip an RLE mask horizontally (GPU when available).

    Args:
        rle: COCO RLE dict
        crop_width: crop width (unused; the width is taken from the RLE)

    Returns:
        flipped RLE dict
    """
    # Decode
    if isinstance(rle["counts"], str):
        rle_copy = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    else:
        rle_copy = rle.copy()

    binary_mask = mask_utils.decode(rle_copy)

    # Flip (GPU or CPU)
    if _check_gpu_available():
        try:
            mask_tensor = torch.from_numpy(binary_mask).cuda()
            flipped_mask = torch.flip(mask_tensor, dims=[1]).cpu().numpy()
        except Exception:
            flipped_mask = cv2.flip(binary_mask, 1)
    else:
        flipped_mask = cv2.flip(binary_mask, 1)

    # Re-encode
    flipped_mask_fortran = np.asfortranarray(flipped_mask)
    new_rle = mask_utils.encode(flipped_mask_fortran)
    new_rle["counts"] = new_rle["counts"].decode("utf-8")

    return new_rle
