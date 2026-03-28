"""
DICOM image handling module.
Supports reading DICOM files (X-ray, CT, MRI, Ultrasound),
extracting metadata, and converting to displayable images.
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


# Modality code to human-readable name mapping
MODALITY_MAP = {
    "CR": "Computed Radiography (X-Ray)",
    "DX": "Digital Radiography (X-Ray)",
    "CT": "Computed Tomography",
    "MR": "Magnetic Resonance Imaging",
    "US": "Ultrasound",
    "MG": "Mammography",
    "NM": "Nuclear Medicine",
    "PT": "PET Scan",
    "XA": "X-Ray Angiography",
    "RF": "Radiofluoroscopy",
    "OT": "Other",
}

# Body part mapping for physiological context
BODY_PART_MAP = {
    "CHEST": "胸部 (Chest)",
    "ABDOMEN": "腹部 (Abdomen)",
    "HEAD": "頭部 (Head)",
    "BRAIN": "腦部 (Brain)",
    "SPINE": "脊椎 (Spine)",
    "PELVIS": "骨盆 (Pelvis)",
    "EXTREMITY": "四肢 (Extremity)",
    "KNEE": "膝蓋 (Knee)",
    "HIP": "髖關節 (Hip)",
    "SHOULDER": "肩膀 (Shoulder)",
    "HAND": "手部 (Hand)",
    "FOOT": "足部 (Foot)",
    "NECK": "頸部 (Neck)",
    "HEART": "心臟 (Heart)",
    "LIVER": "肝臟 (Liver)",
    "KIDNEY": "腎臟 (Kidney)",
}


def check_pydicom_available():
    """Check if pydicom is installed."""
    if not HAS_PYDICOM:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install it with: pip install pydicom pylibjpeg pylibjpeg-libjpeg"
        )


def load_dicom_file(file_path):
    """
    Load a DICOM file and return the dataset.

    Args:
        file_path: Path to the DICOM file (.dcm)

    Returns:
        pydicom.Dataset object
    """
    check_pydicom_available()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    ds = pydicom.dcmread(file_path)
    return ds


def dicom_to_pil_image(ds, window_center=None, window_width=None):
    """
    Convert a DICOM dataset to a PIL Image for display.

    Handles windowing (VOI LUT), photometric interpretation,
    and normalization for proper display.

    Args:
        ds: pydicom.Dataset object
        window_center: Optional manual window center override
        window_width: Optional manual window width override

    Returns:
        PIL.Image in RGB mode
    """
    check_pydicom_available()

    pixel_array = ds.pixel_array.astype(np.float64)

    # Apply rescale slope/intercept if present (common in CT)
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    pixel_array = pixel_array * slope + intercept

    # Apply windowing
    if window_center is not None and window_width is not None:
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, min_val, max_val)
    else:
        try:
            pixel_array = apply_voi_lut(pixel_array, ds)
        except Exception:
            pass

    # Handle photometric interpretation
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        pixel_array = np.max(pixel_array) - pixel_array

    # Normalize to 0-255
    min_px = np.min(pixel_array)
    max_px = np.max(pixel_array)
    if max_px != min_px:
        pixel_array = (pixel_array - min_px) / (max_px - min_px) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)

    pixel_array = pixel_array.astype(np.uint8)

    # Handle multi-frame or 3D data (take middle slice)
    if len(pixel_array.shape) == 3 and pixel_array.shape[2] not in (3, 4):
        mid = pixel_array.shape[0] // 2
        pixel_array = pixel_array[mid]

    # Convert to PIL
    if len(pixel_array.shape) == 2:
        img = Image.fromarray(pixel_array, mode='L').convert('RGB')
    elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
        img = Image.fromarray(pixel_array, mode='RGB')
    else:
        img = Image.fromarray(pixel_array[:, :, 0], mode='L').convert('RGB')

    return img


def extract_dicom_metadata(ds):
    """
    Extract clinically relevant metadata from a DICOM dataset.

    Returns:
        dict with structured metadata
    """
    def safe_get(attr, default="N/A"):
        val = getattr(ds, attr, None)
        if val is None or str(val).strip() == "":
            return default
        return str(val).strip()

    modality_code = safe_get('Modality', 'Unknown')
    modality_name = MODALITY_MAP.get(modality_code, modality_code)

    body_part_code = safe_get('BodyPartExamined', 'Unknown')
    body_part_name = BODY_PART_MAP.get(body_part_code.upper(), body_part_code)

    # Parse study date
    study_date_raw = safe_get('StudyDate', '')
    study_date = study_date_raw
    if study_date_raw and study_date_raw != 'N/A' and len(study_date_raw) == 8:
        try:
            dt = datetime.strptime(study_date_raw, '%Y%m%d')
            study_date = dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

    metadata = {
        "patient_info": {
            "patient_name": safe_get('PatientName'),
            "patient_id": safe_get('PatientID'),
            "patient_age": safe_get('PatientAge'),
            "patient_sex": safe_get('PatientSex'),
            "patient_weight": safe_get('PatientWeight'),
        },
        "study_info": {
            "study_date": study_date,
            "study_description": safe_get('StudyDescription'),
            "study_id": safe_get('StudyID'),
            "accession_number": safe_get('AccessionNumber'),
            "referring_physician": safe_get('ReferringPhysicianName'),
            "institution": safe_get('InstitutionName'),
        },
        "series_info": {
            "modality": modality_code,
            "modality_name": modality_name,
            "series_description": safe_get('SeriesDescription'),
            "body_part": body_part_code,
            "body_part_name": body_part_name,
            "view_position": safe_get('ViewPosition'),
            "laterality": safe_get('Laterality'),
        },
        "image_info": {
            "rows": safe_get('Rows'),
            "columns": safe_get('Columns'),
            "bits_allocated": safe_get('BitsAllocated'),
            "bits_stored": safe_get('BitsStored'),
            "pixel_spacing": safe_get('PixelSpacing'),
            "slice_thickness": safe_get('SliceThickness'),
            "photometric_interpretation": safe_get('PhotometricInterpretation'),
            "window_center": safe_get('WindowCenter'),
            "window_width": safe_get('WindowWidth'),
        },
        "equipment_info": {
            "manufacturer": safe_get('Manufacturer'),
            "model_name": safe_get('ManufacturerModelName'),
            "station_name": safe_get('StationName'),
            "software_version": safe_get('SoftwareVersions'),
        },
    }

    return metadata


def format_metadata_display(metadata):
    """
    Format metadata dict into a readable display string (Traditional Chinese).
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  DICOM 影像資訊 / DICOM Image Information")
    lines.append("=" * 60)

    # Patient info
    p = metadata["patient_info"]
    lines.append("\n【病患資訊 Patient Info】")
    lines.append(f"  姓名 Name:       {p['patient_name']}")
    lines.append(f"  病歷號 ID:       {p['patient_id']}")
    lines.append(f"  年齡 Age:        {p['patient_age']}")
    lines.append(f"  性別 Sex:        {p['patient_sex']}")
    lines.append(f"  體重 Weight:     {p['patient_weight']}")

    # Study info
    s = metadata["study_info"]
    lines.append("\n【檢查資訊 Study Info】")
    lines.append(f"  檢查日期 Date:   {s['study_date']}")
    lines.append(f"  檢查描述 Desc:   {s['study_description']}")
    lines.append(f"  檢查號碼 ID:     {s['study_id']}")
    lines.append(f"  醫囑醫師 Ref:    {s['referring_physician']}")
    lines.append(f"  醫療機構 Inst:   {s['institution']}")

    # Series info
    sr = metadata["series_info"]
    lines.append("\n【序列資訊 Series Info】")
    lines.append(f"  影像模態 Modality: {sr['modality']} ({sr['modality_name']})")
    lines.append(f"  序列描述 Desc:     {sr['series_description']}")
    lines.append(f"  檢查部位 Body:     {sr['body_part']} ({sr['body_part_name']})")
    lines.append(f"  投射位置 View:     {sr['view_position']}")
    lines.append(f"  側別 Laterality:   {sr['laterality']}")

    # Image info
    im = metadata["image_info"]
    lines.append("\n【影像參數 Image Parameters】")
    lines.append(f"  影像大小 Size:     {im['rows']} x {im['columns']}")
    lines.append(f"  位元深度 Bits:     {im['bits_allocated']} / {im['bits_stored']}")
    lines.append(f"  像素間距 Spacing:  {im['pixel_spacing']}")
    lines.append(f"  切片厚度 Slice:    {im['slice_thickness']}")
    lines.append(f"  窗位 WC / 窗寬 WW: {im['window_center']} / {im['window_width']}")

    # Equipment info
    eq = metadata["equipment_info"]
    lines.append("\n【設備資訊 Equipment Info】")
    lines.append(f"  製造商 Manufacturer: {eq['manufacturer']}")
    lines.append(f"  型號 Model:          {eq['model_name']}")
    lines.append(f"  工作站 Station:      {eq['station_name']}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def get_ct_window_presets():
    """Return common CT windowing presets."""
    return {
        "Lung (肺窗)": {"center": -600, "width": 1500},
        "Mediastinum (縱膈腔窗)": {"center": 40, "width": 350},
        "Bone (骨窗)": {"center": 400, "width": 1800},
        "Brain (腦窗)": {"center": 40, "width": 80},
        "Subdural (硬膜下窗)": {"center": 75, "width": 215},
        "Stroke (中風窗)": {"center": 32, "width": 8},
        "Abdomen (腹部窗)": {"center": 60, "width": 400},
        "Liver (肝臟窗)": {"center": 70, "width": 150},
        "Soft Tissue (軟組織窗)": {"center": 50, "width": 350},
    }


def load_dicom_series(directory_path):
    """
    Load all DICOM files from a directory as a series.

    Args:
        directory_path: Path to directory containing .dcm files

    Returns:
        list of (slice_location, pydicom.Dataset) tuples sorted by position
    """
    check_pydicom_available()

    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    datasets = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            ds = pydicom.dcmread(filepath)
            loc = float(getattr(ds, 'SliceLocation', 0) or 0)
            datasets.append((loc, ds))
        except Exception:
            continue

    datasets.sort(key=lambda x: x[0])
    return datasets
