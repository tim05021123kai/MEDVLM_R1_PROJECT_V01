"""
Phase 2: DICOM Integration Tests
Requires: pip install pydicom pydicom-data numpy Pillow

Run: python tests/test_dicom_integration.py
"""

import sys
import os
import traceback
import tempfile
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---- Check dependencies ----
try:
    import pydicom
    from pydicom.data import get_testdata_file
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("WARNING: pydicom not installed. Run: pip install pydicom pydicom-data")

try:
    import numpy as np
    from PIL import Image
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False
    print("WARNING: numpy/Pillow not installed.")

# ---- Mock heavy deps we don't need for DICOM tests ----
import types

def _mock(name, attrs=None):
    mod = types.ModuleType(name)
    mod.__file__ = f"<mock:{name}>"
    mod.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod

for mod_name in [
    "torch", "torch.cuda", "torch.backends", "torch.backends.mps",
    "torch.nn", "torch.nn.functional",
    "transformers", "qwen_vl_utils", "modelscope",
    "gradio", "pynetdicom", "pynetdicom.sop_class",
    "reportlab", "reportlab.lib", "reportlab.lib.pagesizes",
    "reportlab.lib.units", "reportlab.lib.styles", "reportlab.lib.enums",
    "reportlab.pdfgen", "reportlab.pdfgen.canvas", "reportlab.platypus",
]:
    if mod_name not in sys.modules:
        attrs = {}
        if mod_name == "torch":
            attrs = {
                "float16": "float16", "float32": "float32",
                "Tensor": type("Tensor", (), {}),
                "no_grad": lambda: type("ctx", (), {
                    "__enter__": lambda s: s, "__exit__": lambda s,*a: None})(),
            }
            m = _mock(mod_name, attrs)
            m.cuda = _mock("torch.cuda", {"is_available": lambda: False})
            m.backends = _mock("torch.backends")
            m.backends.mps = _mock("torch.backends.mps", {"is_available": lambda: False})
            nn = _mock("torch.nn", {
                "Conv2d": type("Conv2d", (), {}),
                "Module": type("Module", (), {}),
                "LayerNorm": type("LayerNorm", (), {}),
            })
            nn.functional = _mock("torch.nn.functional")
            m.nn = nn
            sys.modules[mod_name] = m
            sys.modules["torch.cuda"] = m.cuda
            sys.modules["torch.backends"] = m.backends
            sys.modules["torch.backends.mps"] = m.backends.mps
            sys.modules["torch.nn"] = nn
            sys.modules["torch.nn.functional"] = nn.functional
            continue
        if mod_name == "transformers":
            attrs = {
                "Qwen2VLForConditionalGeneration": type("Q", (), {
                    "from_pretrained": staticmethod(lambda *a, **k: None)}),
                "AutoProcessor": type("A", (), {
                    "from_pretrained": staticmethod(lambda *a, **k: None)}),
                "TextIteratorStreamer": type("S", (), {}),
            }
        elif mod_name == "reportlab.lib.pagesizes":
            attrs = {"A4": (595, 842)}
        elif mod_name == "reportlab.lib.units":
            attrs = {"mm": 2.835}
        elif mod_name == "reportlab.lib.enums":
            attrs = {"TA_LEFT": 0}
        elif mod_name == "qwen_vl_utils":
            attrs = {"process_vision_info": lambda msgs: (None, None)}
        elif mod_name == "gradio":
            class _FC:
                def __init__(self, *a, **k): pass
                def change(self, *a, **k): return self
                def click(self, *a, **k): return self
                def then(self, *a, **k): return self
            class _FB:
                def __init__(self, *a, **k): pass
                def __enter__(self): return self
                def __exit__(self, *a): pass
                def launch(self, *a, **k): pass
            attrs = {
                "Blocks": _FB, "Tabs": _FB, "Tab": _FB, "Row": _FB,
                "Column": _FB, "Accordion": _FB,
                "State": lambda val=None, *a, **k: type("S", (), {"value": val})(),
                "Image": _FC, "Textbox": _FC, "Number": _FC, "Button": _FC,
                "File": _FC, "Dropdown": _FC, "Radio": _FC, "Checkbox": _FC,
                "Slider": _FC, "Markdown": lambda *a, **k: None,
                "HTML": lambda *a, **k: None, "update": lambda **k: k,
            }
        sys.modules[mod_name] = _mock(mod_name, attrs)

# ---- Test runner ----
PASS = 0
FAIL = 0
SKIP = 0
ERRORS = []

def test(name, fn, requires_pydicom=True):
    global PASS, FAIL, SKIP
    if requires_pydicom and not HAS_PYDICOM:
        SKIP += 1
        print(f"  SKIP: {name} (pydicom not installed)")
        return
    if not HAS_IMAGING:
        SKIP += 1
        print(f"  SKIP: {name} (numpy/Pillow not installed)")
        return
    try:
        fn()
        PASS += 1
        print(f"  PASS: {name}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL: {name} -> {e}")
        ERRORS.append((name, str(e), traceback.format_exc()))

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ===================================================================
# D-01 ~ D-05: DICOM Loading
# ===================================================================
section("DICOM Loading Tests")

def test_load_ct():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    assert ds is not None, "Dataset should not be None"
    img = dicom_to_pil_image(ds)
    assert img is not None, "Image should not be None"
    assert isinstance(img, Image.Image), "Should return PIL Image"
    assert img.size[0] > 0 and img.size[1] > 0, "Image should have positive dimensions"

test("D-01 Load CT DICOM", test_load_ct)

def test_load_mr():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("MR_small.dcm")
    ds = load_dicom_file(path)
    assert ds is not None
    img = dicom_to_pil_image(ds)
    assert img is not None
    assert isinstance(img, Image.Image)

test("D-02 Load MR DICOM", test_load_mr)

def test_load_invalid():
    from dicom.dicom_handler import load_dicom_file
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
        f.write(b"this is not a dicom file")
        f.flush()
        try:
            result = load_dicom_file(f.name)
            # Should either return None/error or raise exception
            if result is not None:
                ds, img, meta = result
                # If it returns a tuple, ds or img may be None
        except Exception:
            pass  # Expected - invalid DICOM should raise or return gracefully
    os.unlink(f.name)

test("D-04 Load invalid file (graceful)", test_load_invalid)

def test_load_nonexistent():
    from dicom.dicom_handler import load_dicom_file
    try:
        result = load_dicom_file("/nonexistent/path/test.dcm")
    except Exception:
        pass  # Expected

test("D-05 Load nonexistent file", test_load_nonexistent)


# ===================================================================
# D-06 ~ D-07: Metadata Extraction
# ===================================================================
section("Metadata Extraction Tests")

def test_metadata_complete():
    from dicom.dicom_handler import load_dicom_file, extract_dicom_metadata
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    meta = extract_dicom_metadata(ds)
    assert "patient_info" in meta, "Should have patient_info"
    assert "study_info" in meta, "Should have study_info"
    assert "series_info" in meta, "Should have series_info"

test("D-06 Extract complete metadata", test_metadata_complete)

def test_metadata_display():
    from dicom.dicom_handler import load_dicom_file, extract_dicom_metadata, format_metadata_display
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    meta = extract_dicom_metadata(ds)
    display = format_metadata_display(meta)
    assert isinstance(display, str)
    assert len(display) > 10, "Display should have meaningful content"

test("D-07 Format metadata display", test_metadata_display)


# ===================================================================
# W-01 ~ W-06: CT Windowing
# ===================================================================
section("CT Windowing Tests")

def test_window_lung():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img = dicom_to_pil_image(ds, window_center=-600, window_width=1500)
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0

test("W-01 Lung window (-600/1500)", test_window_lung)

def test_window_brain():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img = dicom_to_pil_image(ds, window_center=40, window_width=80)
    assert isinstance(img, Image.Image)

test("W-02 Brain window (40/80)", test_window_brain)

def test_window_bone():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img = dicom_to_pil_image(ds, window_center=400, window_width=1800)
    assert isinstance(img, Image.Image)

test("W-03 Bone window (400/1800)", test_window_bone)

def test_window_soft_tissue():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img = dicom_to_pil_image(ds, window_center=40, window_width=400)
    assert isinstance(img, Image.Image)

test("W-04 Soft tissue window (40/400)", test_window_soft_tissue)

def test_window_custom():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img = dicom_to_pil_image(ds, window_center=0, window_width=2000)
    assert isinstance(img, Image.Image)

test("W-06 Custom window (0/2000)", test_window_custom)

def test_window_default():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    # No windowing params -> auto VOI-LUT
    img = dicom_to_pil_image(ds)
    assert isinstance(img, Image.Image)

test("W-07 Default windowing (auto)", test_window_default)

def test_different_windows_differ():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    lung = dicom_to_pil_image(ds, window_center=-600, window_width=1500)
    bone = dicom_to_pil_image(ds, window_center=400, window_width=1800)
    # Different windows should produce different images
    lung_arr = np.array(lung)
    bone_arr = np.array(bone)
    assert not np.array_equal(lung_arr, bone_arr), "Different windows should produce different images"

test("W-08 Different windows produce different results", test_different_windows_differ)


# ===================================================================
# P-01 ~ P-06: De-identification with real DICOM
# ===================================================================
section("De-identification Tests (Real DICOM)")

def test_phi_detection():
    from dicom.dicom_handler import load_dicom_file
    from utils.dicom_deidentify import get_phi_summary
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    summary = get_phi_summary(ds)
    assert isinstance(summary, str)
    # CT_small.dcm should have some PHI fields
    assert len(summary) > 0

test("P-01 PHI detection on real DICOM", test_phi_detection)

def test_deidentify_real_dicom():
    import copy
    from dicom.dicom_handler import load_dicom_file
    from utils.dicom_deidentify import deidentify_dicom
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    ds_copy = copy.deepcopy(ds)
    deidentify_dicom(ds_copy, keep_age=True, keep_sex=True)
    # Check PatientName is anonymized
    if hasattr(ds_copy, "PatientName"):
        assert str(ds_copy.PatientName) == "ANONYMOUS", \
            f"PatientName should be ANONYMOUS, got {ds_copy.PatientName}"

test("P-02 De-identify real DICOM", test_deidentify_real_dicom)

def test_deidentify_preserves_image():
    import copy
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    from utils.dicom_deidentify import deidentify_dicom
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img_before = dicom_to_pil_image(ds)
    ds_copy = copy.deepcopy(ds)
    deidentify_dicom(ds_copy)
    img_after = dicom_to_pil_image(ds_copy)
    # Pixel data should be unchanged
    assert np.array_equal(np.array(img_before), np.array(img_after)), \
        "De-identification should not modify pixel data"

test("P-03 De-identify preserves pixel data", test_deidentify_preserves_image)

def test_deidentify_idempotent():
    import copy
    from dicom.dicom_handler import load_dicom_file
    from utils.dicom_deidentify import deidentify_dicom, get_phi_summary
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    ds_copy = copy.deepcopy(ds)
    deidentify_dicom(ds_copy)
    summary1 = get_phi_summary(ds_copy)
    deidentify_dicom(ds_copy)  # Run again
    summary2 = get_phi_summary(ds_copy)
    # Should be the same after double de-identification
    assert summary1 == summary2, "De-identification should be idempotent"

test("P-06 De-identify idempotent", test_deidentify_idempotent)


# ===================================================================
# I-01 ~ I-04: Image Preprocessing with real DICOM
# ===================================================================
section("Image Preprocessing Tests (Real DICOM)")

def test_preprocess_ct():
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    from utils.image_utils import preprocess_medical_image
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img = dicom_to_pil_image(ds)
    result = preprocess_medical_image(img, modality="CT")
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"

test("I-01 Preprocess CT image", test_preprocess_ct)

def test_preprocess_adds_contrast_xray():
    from utils.image_utils import preprocess_medical_image
    # Create a low-contrast grayscale image (simulating X-ray)
    arr = np.random.randint(100, 150, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    result = preprocess_medical_image(img, modality="CR")
    result_arr = np.array(result)
    # After autocontrast, range should be wider
    assert result_arr.max() > arr.max() or result_arr.min() < arr.min(), \
        "AutoContrast should expand dynamic range"

test("I-02 X-ray autocontrast enhancement", test_preprocess_adds_contrast_xray)


# ===================================================================
# E-01 ~ E-03: Export with real metadata
# ===================================================================
section("Export Tests (with DICOM metadata)")

def test_export_fhir_with_real_meta():
    from dicom.dicom_handler import load_dicom_file, extract_dicom_metadata, dicom_to_pil_image
    from utils.export import export_report_to_fhir
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    meta = extract_dicom_metadata(ds)
    with tempfile.TemporaryDirectory() as d:
        fpath = export_report_to_fhir(
            "AI analysis: Normal CT scan",
            metadata=meta,
            clinical_history="Routine checkup",
            output_dir=d
        )
        assert os.path.exists(fpath)
        with open(fpath) as f:
            data = json.load(f)
        assert data["resourceType"] == "DiagnosticReport"
        assert data["status"] == "preliminary"

test("E-01 FHIR export with real DICOM metadata", test_export_fhir_with_real_meta)

def test_export_text_with_report():
    from report.report_generator import RadiologyReportGenerator
    from utils.export import export_report_to_text
    gen = RadiologyReportGenerator()
    report = gen.generate_report(
        "## FINDINGS\nNo acute abnormality.\n## IMPRESSION\nNormal.",
        backend_name="Ollama (qwen2-vl)",
        confidence=0.92
    )
    with tempfile.TemporaryDirectory() as d:
        path = export_report_to_text(report, output_dir=d)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "FINDINGS" in content
        assert "92%" in content
        assert "Ollama" in content

test("E-02 Text export with structured report", test_export_text_with_report)


# ===================================================================
# APP: End-to-end DICOM workflow in app.py
# ===================================================================
section("App.py DICOM Workflow Tests")

def test_app_dicom_upload():
    """Test the full DICOM upload workflow through app.py"""
    from dicom.dicom_handler import load_dicom_file, dicom_to_pil_image
    from app import handle_standard_image_upload
    path = get_testdata_file("CT_small.dcm")
    ds = load_dicom_file(path)
    img = dicom_to_pil_image(ds)
    state = {"current_image": None, "dicom_ds": None, "dicom_metadata": None}
    preview, status, new_state = handle_standard_image_upload(img, state)
    assert new_state["current_image"] is not None
    assert isinstance(preview, Image.Image)

test("APP-01 DICOM → PIL → app upload workflow", test_app_dicom_upload)

def test_app_ct_window_presets():
    from dicom.dicom_handler import get_ct_window_presets
    presets = get_ct_window_presets()
    assert len(presets) >= 9, f"Should have >= 9 presets, got {len(presets)}"
    # Preset names include Chinese, e.g. "Lung (肺窗)"
    expected_keywords = ["Lung", "Brain", "Bone", "Soft Tissue", "Liver",
                         "Mediastinum", "Stroke", "Abdomen", "Subdural"]
    for keyword in expected_keywords:
        found = any(keyword in name for name in presets.keys())
        assert found, f"No preset containing '{keyword}' found in {list(presets.keys())}"
    for name, vals in presets.items():
        assert "center" in vals, f"Preset '{name}' missing 'center'"
        assert "width" in vals, f"Preset '{name}' missing 'width'"

test("APP-02 All CT window presets defined", test_app_ct_window_presets)


# ===================================================================
# Summary
# ===================================================================
section("PHASE 2 DICOM INTEGRATION TEST SUMMARY")
total = PASS + FAIL + SKIP
print(f"\n  Total:   {total}")
print(f"  Passed:  {PASS}")
print(f"  Failed:  {FAIL}")
print(f"  Skipped: {SKIP}")

if ERRORS:
    print(f"\n  {'='*60}")
    print(f"  FAILED TESTS DETAIL:")
    print(f"  {'='*60}")
    for name, err, tb in ERRORS:
        print(f"\n  [{name}]")
        print(f"  Error: {err}")
        for line in tb.strip().split('\n')[-5:]:
            print(f"    {line}")

print()
sys.exit(0 if FAIL == 0 else 1)
