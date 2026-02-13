"""
Medical image loading and preprocessing utilities.
Includes CLAHE contrast enhancement and border/annotation removal.
"""

import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter


def load_image_from_path(image_path):
    """Load an image file and convert to RGB."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist.")
    return Image.open(image_path).convert("RGB")


def preprocess_medical_image(image, modality=None):
    """
    Apply medical-image-specific preprocessing.

    Args:
        image: PIL Image (RGB)
        modality: DICOM modality code (CR, DX, CT, MR, etc.)

    Returns:
        Preprocessed PIL Image (RGB)
    """
    img = image.convert("RGB")

    # Auto-contrast for X-ray images (CR, DX) — equivalent to simplified CLAHE
    if modality in ("CR", "DX", "MG"):
        img = ImageOps.autocontrast(img, cutoff=1)

    # Remove common border artifacts (letterbox bars, burned-in annotations)
    img = _crop_black_borders(img, threshold=10)

    return img


def _crop_black_borders(image, threshold=10, min_fraction=0.05):
    """
    Crop near-black borders from the image.
    Useful for removing DICOM letterbox bars and dark padding.

    Args:
        image: PIL Image (RGB)
        threshold: pixel intensity below this is considered "black"
        min_fraction: minimum fraction of image to crop on each side

    Returns:
        Cropped PIL Image
    """
    arr = np.array(image.convert("L"))
    h, w = arr.shape

    # Find rows/cols that are mostly dark
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)

    # Find first/last rows above threshold
    active_rows = np.where(row_means > threshold)[0]
    active_cols = np.where(col_means > threshold)[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        return image

    top = max(active_rows[0], 0)
    bottom = min(active_rows[-1] + 1, h)
    left = max(active_cols[0], 0)
    right = min(active_cols[-1] + 1, w)

    # Don't crop more than 20% from any side
    max_crop = 0.20
    top = min(top, int(h * max_crop))
    bottom = max(bottom, int(h * (1 - max_crop)))
    left = min(left, int(w * max_crop))
    right = max(right, int(w * (1 - max_crop)))

    if (right - left) < w * 0.5 or (bottom - top) < h * 0.5:
        return image  # Don't crop if result would be too small

    return image.crop((left, top, right, bottom))


def extract_roi_from_annotation(image, bbox):
    """
    Extract a region of interest from an annotated image.

    Args:
        image: PIL Image
        bbox: dict with keys 'x', 'y', 'width', 'height' (from Gradio ImageEditor)

    Returns:
        Cropped PIL Image of the ROI
    """
    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = int(bbox.get("width", image.width))
    h = int(bbox.get("height", image.height))

    x = max(0, min(x, image.width))
    y = max(0, min(y, image.height))
    right = min(x + w, image.width)
    bottom = min(y + h, image.height)

    return image.crop((x, y, right, bottom))
