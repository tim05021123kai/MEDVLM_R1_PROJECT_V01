"""
Regression test suite for MedVLM-R1 Medical Image AI Viewer v2.
Tests all modules for import errors, interface correctness, and functional regression.
Mocks heavy dependencies (torch, transformers, gradio) that aren't in CI.
"""

import sys
import os
import traceback
import json
import tempfile
import types

# ---- Setup project root on path ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---- Mock heavy dependencies not available in test env ----
def _create_mock_module(name, attrs=None):
    mod = types.ModuleType(name)
    mod.__file__ = f"<mock:{name}>"
    mod.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


# Mock torch
if "torch" not in sys.modules:
    torch_mock = _create_mock_module("torch")
    torch_mock.float16 = "float16"
    torch_mock.float32 = "float32"
    torch_mock.no_grad = lambda: type("ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s,*a: None})()
    torch_mock.is_grad_enabled = lambda: False
    torch_mock.set_grad_enabled = lambda v: None
    torch_mock.cuda = _create_mock_module("torch.cuda")
    torch_mock.cuda.is_available = lambda: False
    torch_mock.backends = _create_mock_module("torch.backends")
    torch_mock.backends.mps = _create_mock_module("torch.backends.mps")
    torch_mock.backends.mps.is_available = lambda: False
    torch_mock.Tensor = type("Tensor", (), {})
    # nn sub-modules
    nn_mock = _create_mock_module("torch.nn")
    nn_mock.Conv2d = type("Conv2d", (), {})
    nn_mock.LayerNorm = type("LayerNorm", (), {})
    nn_mock.Module = type("Module", (), {})
    nn_mock.functional = _create_mock_module("torch.nn.functional")
    nn_mock.functional.relu = lambda x: x
    nn_mock.functional.interpolate = lambda *a, **k: None
    nn_mock.functional.pad = lambda *a, **k: None
    torch_mock.nn = nn_mock
    sys.modules["torch"] = torch_mock
    sys.modules["torch.cuda"] = torch_mock.cuda
    sys.modules["torch.backends"] = torch_mock.backends
    sys.modules["torch.backends.mps"] = torch_mock.backends.mps
    sys.modules["torch.nn"] = nn_mock
    sys.modules["torch.nn.functional"] = nn_mock.functional

# Mock transformers
if "transformers" not in sys.modules:
    tf_mock = _create_mock_module("transformers")
    tf_mock.Qwen2VLForConditionalGeneration = type("Qwen2VL", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
    tf_mock.AutoProcessor = type("AutoProc", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
    tf_mock.TextIteratorStreamer = type("Streamer", (), {})
    sys.modules["transformers"] = tf_mock

# Mock qwen_vl_utils
if "qwen_vl_utils" not in sys.modules:
    qvl_mock = _create_mock_module("qwen_vl_utils")
    qvl_mock.process_vision_info = lambda msgs: (None, None)
    sys.modules["qwen_vl_utils"] = qvl_mock

# Mock modelscope
if "modelscope" not in sys.modules:
    sys.modules["modelscope"] = _create_mock_module("modelscope")

# Mock gradio
if "gradio" not in sys.modules:
    gr_mock = _create_mock_module("gradio")
    # Minimal stubs for Gradio components
    class _FakeComponent:
        def __init__(self, *a, **k): pass
        def change(self, *a, **k): return self
        def click(self, *a, **k): return self
        def then(self, *a, **k): return self
    class _FakeBlocks:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def launch(self, *a, **k): pass
    class _FakeTabs(_FakeBlocks): pass
    class _FakeTab(_FakeBlocks):
        def __init__(self, *a, **k): pass
    class _FakeRow(_FakeBlocks): pass
    class _FakeColumn(_FakeBlocks): pass
    class _FakeAccordion(_FakeBlocks): pass
    class _FakeState:
        def __init__(self, val=None, *a, **k): self.value = val

    gr_mock.Blocks = _FakeBlocks
    gr_mock.Tabs = _FakeTabs
    gr_mock.Tab = _FakeTab
    gr_mock.Row = _FakeRow
    gr_mock.Column = _FakeColumn
    gr_mock.Accordion = _FakeAccordion
    gr_mock.State = _FakeState
    gr_mock.Image = _FakeComponent
    gr_mock.Textbox = _FakeComponent
    gr_mock.Number = _FakeComponent
    gr_mock.Button = _FakeComponent
    gr_mock.File = _FakeComponent
    gr_mock.Dropdown = _FakeComponent
    gr_mock.Radio = _FakeComponent
    gr_mock.Checkbox = _FakeComponent
    gr_mock.Slider = _FakeComponent
    gr_mock.Markdown = lambda *a, **k: None
    gr_mock.HTML = lambda *a, **k: None
    gr_mock.update = lambda **k: k
    sys.modules["gradio"] = gr_mock

# Mock pydicom
if "pydicom" not in sys.modules:
    pydicom_mock = _create_mock_module("pydicom")
    pydicom_mock.dcmread = lambda *a, **k: None
    pdu_mock = _create_mock_module("pydicom.pixel_data_handlers")
    pdu_util = _create_mock_module("pydicom.pixel_data_handlers.util")
    pdu_util.apply_voi_lut = lambda *a, **k: None
    sys.modules["pydicom"] = pydicom_mock
    sys.modules["pydicom.pixel_data_handlers"] = pdu_mock
    sys.modules["pydicom.pixel_data_handlers.util"] = pdu_util

# Mock pynetdicom
if "pynetdicom" not in sys.modules:
    pnd_mock = _create_mock_module("pynetdicom")
    pnd_mock.AE = type("AE", (), {"__init__": lambda *a, **k: None})
    pnd_mock.StoragePresentationContexts = []
    sys.modules["pynetdicom"] = pnd_mock
    sys.modules["pynetdicom.sop_class"] = _create_mock_module("pynetdicom.sop_class")

# Mock reportlab
if "reportlab" not in sys.modules:
    sys.modules["reportlab"] = _create_mock_module("reportlab")
    sys.modules["reportlab.lib"] = _create_mock_module("reportlab.lib")
    sys.modules["reportlab.lib.pagesizes"] = _create_mock_module("reportlab.lib.pagesizes", {"A4": (595, 842)})
    sys.modules["reportlab.lib.units"] = _create_mock_module("reportlab.lib.units", {"mm": 2.835})
    sys.modules["reportlab.lib.styles"] = _create_mock_module("reportlab.lib.styles")
    sys.modules["reportlab.lib.enums"] = _create_mock_module("reportlab.lib.enums", {"TA_LEFT": 0})
    sys.modules["reportlab.pdfgen"] = _create_mock_module("reportlab.pdfgen")
    sys.modules["reportlab.pdfgen.canvas"] = _create_mock_module("reportlab.pdfgen.canvas")
    sys.modules["reportlab.platypus"] = _create_mock_module("reportlab.platypus")

# ---- Now we can import ----
import numpy as np
from PIL import Image

# ==================================================================
PASS = 0
FAIL = 0
ERRORS = []


def test(name, fn):
    global PASS, FAIL
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
# 1. Import Tests
# ===================================================================

section("1. Module Import Tests")

test("import utils.cache", lambda: __import__("utils.cache"))
test("import utils.audit_logger", lambda: __import__("utils.audit_logger"))
test("import utils.export", lambda: __import__("utils.export"))
test("import utils.dicom_deidentify", lambda: __import__("utils.dicom_deidentify"))
test("import utils.image_utils", lambda: __import__("utils.image_utils"))
test("import utils.prompt_utils", lambda: __import__("utils.prompt_utils"))
test("import utils.physio_params", lambda: __import__("utils.physio_params"))
test("import model.ollama_client", lambda: __import__("model.ollama_client"))
test("import model.medvlm_loader", lambda: __import__("model.medvlm_loader"))
test("import inference.run_inference", lambda: __import__("inference.run_inference"))
test("import report.report_generator", lambda: __import__("report.report_generator"))
test("import gradcam.gradcam_engine", lambda: __import__("gradcam.gradcam_engine"))
test("import dicom.dicom_handler", lambda: __import__("dicom.dicom_handler"))
test("import dicom.pacs_client", lambda: __import__("dicom.pacs_client"))
test("import app", lambda: __import__("app"))


# ===================================================================
# 2. Cache & History Tests
# ===================================================================

section("2. Cache & History Tests")

def test_cache_basic():
    from utils.cache import InferenceCache
    cache = InferenceCache(max_size=3)
    img = Image.new("RGB", (100, 100), "red")
    assert cache.get(img, "test", "ollama") is None
    cache.put(img, "test", "result1", "ollama")
    assert cache.get(img, "test", "ollama") == "result1"
    assert cache.size == 1
    assert cache.get(img, "test2", "ollama") is None  # diff prompt

test("InferenceCache basic", test_cache_basic)

def test_cache_backends_separate():
    from utils.cache import InferenceCache
    cache = InferenceCache()
    img = Image.new("RGB", (50, 50), "green")
    cache.put(img, "q", "res_ollama", "ollama")
    cache.put(img, "q", "res_medvlm", "medvlm-r1")
    assert cache.get(img, "q", "ollama") == "res_ollama"
    assert cache.get(img, "q", "medvlm-r1") == "res_medvlm"

test("InferenceCache separate backends", test_cache_backends_separate)

def test_cache_eviction():
    from utils.cache import InferenceCache
    cache = InferenceCache(max_size=3)
    for i in range(10):
        cache.put(Image.new("RGB", (10+i, 10), "blue"), f"p{i}", f"r{i}", "o")
    assert cache.size == 3

test("InferenceCache LRU eviction", test_cache_eviction)

def test_history():
    from utils.cache import AnalysisHistory
    h = AnalysisHistory(max_entries=5)
    assert h.count == 0
    h.add_entry("img.jpg", "prompt", "result", "ollama", confidence=0.85)
    assert h.count == 1
    display = h.format_history_display()
    assert "ollama" in display
    assert "85%" in display
    for i in range(10):
        h.add_entry(f"img{i}", "p", "r", "o")
    assert h.count == 5

test("AnalysisHistory", test_history)

def test_history_clear():
    from utils.cache import AnalysisHistory
    h = AnalysisHistory()
    h.add_entry("img", "p", "r", "o")
    h.clear()
    assert h.count == 0

test("AnalysisHistory clear", test_history_clear)


# ===================================================================
# 3. Audit Logger Tests
# ===================================================================

section("3. Audit Logger Tests")

def test_audit_write_read():
    from utils.audit_logger import AuditLogger
    with tempfile.TemporaryDirectory() as d:
        logger = AuditLogger(log_dir=d)
        logger.log_analysis(
            backend="ollama", model_name="qwen3-vl", image_source="upload",
            prompt_summary="test", result_summary="result", confidence=0.9,
            dicom_metadata={"series_info": {"modality": "CT", "body_part": "CHEST"}}
        )
        logs = logger.get_recent_logs()
        assert len(logs) == 1
        assert logs[0]["backend"] == "ollama"
        assert logs[0]["imaging_metadata"]["modality"] == "CT"
        assert "patient_name" not in json.dumps(logs[0])

test("AuditLogger write/read", test_audit_write_read)

def test_audit_no_crash_on_missing():
    from utils.audit_logger import AuditLogger
    logger = AuditLogger(log_dir="/tmp/nonexistent_test_audit_abc123")
    logs = logger.get_recent_logs()
    assert isinstance(logs, list)

test("AuditLogger graceful on missing file", test_audit_no_crash_on_missing)


# ===================================================================
# 4. Export Tests
# ===================================================================

section("4. Export Tests")

def test_export_text():
    from utils.export import export_report_to_text
    with tempfile.TemporaryDirectory() as d:
        path = export_report_to_text("Test content", output_dir=d)
        assert os.path.exists(path)
        with open(path) as f:
            assert "Test content" in f.read()

test("export_report_to_text", test_export_text)

def test_export_fhir():
    from utils.export import export_report_to_fhir
    with tempfile.TemporaryDirectory() as d:
        path = export_report_to_fhir(
            "AI analysis", metadata={"patient_info": {"patient_name": "T"},
                                      "series_info": {"modality": "CT"}},
            output_dir=d)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["resourceType"] == "DiagnosticReport"

test("export_report_to_fhir", test_export_fhir)


# ===================================================================
# 5. De-identification Tests
# ===================================================================

section("5. De-identification Tests")

def test_deidentify_metadata():
    from utils.dicom_deidentify import deidentify_metadata
    meta = {
        "patient_info": {"patient_name": "John", "patient_id": "123", "patient_age": "65Y", "patient_sex": "M"},
        "study_info": {"referring_physician": "Dr. X", "institution": "Hospital", "study_date": "2024-01-01"},
        "series_info": {"modality": "CT", "body_part": "CHEST"},
    }
    s = deidentify_metadata(meta)
    assert s["patient_info"]["patient_name"] == "ANONYMOUS"
    assert s["patient_info"]["patient_id"] == "ANONYMOUS"
    assert s["patient_info"]["patient_age"] == "65Y"
    assert s["series_info"]["modality"] == "CT"

test("deidentify_metadata", test_deidentify_metadata)


# ===================================================================
# 6. Image Utils Tests
# ===================================================================

section("6. Image Utils Tests")

def test_preprocess_basic():
    from utils.image_utils import preprocess_medical_image
    img = Image.new("RGB", (200, 200), "gray")
    result = preprocess_medical_image(img, modality="CR")
    assert result.mode == "RGB"

test("preprocess_medical_image basic", test_preprocess_basic)

def test_preprocess_borders():
    from utils.image_utils import preprocess_medical_image
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[30:170, 30:170, :] = 128
    img = Image.fromarray(arr)
    result = preprocess_medical_image(img)
    assert result.size[0] <= 200

test("preprocess_medical_image border crop", test_preprocess_borders)

def test_extract_roi():
    from utils.image_utils import extract_roi_from_annotation
    img = Image.new("RGB", (400, 400), "blue")
    roi = extract_roi_from_annotation(img, {"x": 50, "y": 50, "width": 100, "height": 100})
    assert roi.size == (100, 100)

test("extract_roi_from_annotation", test_extract_roi)

def test_extract_roi_clamp():
    from utils.image_utils import extract_roi_from_annotation
    img = Image.new("RGB", (100, 100), "red")
    roi = extract_roi_from_annotation(img, {"x": 80, "y": 80, "width": 200, "height": 200})
    assert roi.size[0] <= 100
    assert roi.size[1] <= 100

test("extract_roi_from_annotation clamp", test_extract_roi_clamp)


# ===================================================================
# 7. Prompt Utils Tests
# ===================================================================

section("7. Prompt Utils Tests")

def test_build_prompt_medical():
    from utils.prompt_utils import build_prompt
    p = build_prompt("fracture?", "medical")
    assert "## FINDINGS" in p
    assert "## IMPRESSION" in p

test("build_prompt medical → structured", test_build_prompt_medical)

def test_build_prompt_simple():
    from utils.prompt_utils import build_prompt
    p = build_prompt("fracture?", "simple")
    assert "fracture" in p
    assert "## FINDINGS" not in p

test("build_prompt simple", test_build_prompt_simple)

def test_cot_medvlm():
    from utils.prompt_utils import build_cot_prompt
    p = build_cot_prompt("Analyze", backend="medvlm-r1")
    assert "<think>" in p

test("build_cot_prompt medvlm uses <think>", test_cot_medvlm)

def test_cot_ollama():
    from utils.prompt_utils import build_cot_prompt
    p = build_cot_prompt("Analyze", backend="ollama")
    assert "<think>" not in p
    assert "step by step" in p.lower()

test("build_cot_prompt ollama natural CoT", test_cot_ollama)

def test_comparison_prompt():
    from utils.prompt_utils import build_comparison_prompt
    p = build_comparison_prompt("post-surgery")
    assert "PRIOR" in p and "CURRENT" in p

test("build_comparison_prompt", test_comparison_prompt)

def test_fewshot_chest():
    from utils.prompt_utils import get_fewshot_for_body_part
    assert "Cardiothoracic" in get_fewshot_for_body_part("CHEST")

test("get_fewshot CHEST", test_fewshot_chest)

def test_fewshot_unknown():
    from utils.prompt_utils import get_fewshot_for_body_part
    assert get_fewshot_for_body_part("FINGER") == ""

test("get_fewshot unknown → empty", test_fewshot_unknown)


# ===================================================================
# 8. Physio Params Tests
# ===================================================================

section("8. Physio Params Tests")

def test_context_basic():
    from utils.physio_params import get_context_for_ai_prompt
    ctx = get_context_for_ai_prompt("CHEST", "CR")
    assert len(ctx) > 0

test("get_context_for_ai_prompt CHEST", test_context_basic)

def test_context_pediatric():
    from utils.physio_params import get_context_for_ai_prompt
    ctx = get_context_for_ai_prompt("CHEST", "CR", age_str="005Y", sex="M")
    assert "Pediatric" in ctx

test("get_context pediatric age adjustment", test_context_pediatric)

def test_context_elderly_female_spine():
    from utils.physio_params import get_context_for_ai_prompt
    ctx = get_context_for_ai_prompt("SPINE", "CR", age_str="070Y", sex="F")
    assert "osteoporosis" in ctx.lower() or "Degenerative" in ctx

test("get_context elderly female spine", test_context_elderly_female_spine)

def test_context_ct_lab():
    from utils.physio_params import get_context_for_ai_prompt
    ctx = get_context_for_ai_prompt("ABDOMEN", "CT")
    assert "lab" in ctx.lower() or "Creatinine" in ctx or "renal" in ctx.lower()

test("get_context CT includes lab refs", test_context_ct_lab)

def test_parse_age():
    from utils.physio_params import _parse_age
    assert _parse_age("065Y") == 65
    assert abs(_parse_age("003M") - 0.25) < 0.01
    assert _parse_age(None) is None
    assert _parse_age("N/A") is None

test("_parse_age", test_parse_age)

def test_age_notes_empty():
    from utils.physio_params import get_age_adjusted_notes
    assert len(get_age_adjusted_notes("CHEST", None, None)) == 0

test("get_age_adjusted_notes None → empty", test_age_notes_empty)

def test_format_reference():
    from utils.physio_params import format_reference_params
    result = format_reference_params("CHEST")
    assert "CHEST" in result or "Chest" in result or "chest" in result

test("format_reference_params CHEST", test_format_reference)

def test_format_lab():
    from utils.physio_params import format_lab_references
    result = format_lab_references()
    assert len(result) > 0

test("format_lab_references", test_format_lab)


# ===================================================================
# 9. Report Generator Tests
# ===================================================================

section("9. Report Generator Tests")

def test_report_basic():
    from report.report_generator import RadiologyReportGenerator
    gen = RadiologyReportGenerator()
    r = gen.generate_report("Normal findings.", backend_name="Ollama (qwen3-vl)")
    assert "RADIOLOGY REPORT" in r
    assert "Ollama" in r
    assert "DISCLAIMER" in r

test("report basic", test_report_basic)

def test_report_with_confidence():
    from report.report_generator import RadiologyReportGenerator
    gen = RadiologyReportGenerator()
    r = gen.generate_report("Normal.", confidence=0.87, backend_name="Test")
    assert "87%" in r

test("report with confidence", test_report_with_confidence)

def test_report_section_extraction():
    from report.report_generator import RadiologyReportGenerator
    gen = RadiologyReportGenerator()
    r = gen.generate_report(
        "## FINDINGS\nConsolidation\n## IMPRESSION\nPneumonia\n## RECOMMENDATIONS\n6 weeks")
    assert "Pneumonia" in r
    assert "6 weeks" in r

test("report section extraction", test_report_section_extraction)

def test_radiology_prompt():
    from report.report_generator import generate_ai_radiology_prompt
    p = generate_ai_radiology_prompt(
        metadata={"patient_info": {"patient_age": "65Y", "patient_sex": "F"},
                   "series_info": {"modality": "CT"}},
        clinical_history="headache")
    assert "65Y" in p
    assert "headache" in p

test("generate_ai_radiology_prompt", test_radiology_prompt)


# ===================================================================
# 10. Gradcam Engine Tests
# ===================================================================

section("10. Gradcam Engine Tests")

def test_normalize_cam():
    from gradcam.gradcam_engine import _normalize_cam
    cam = np.array([[0.0, 0.5], [1.0, 0.25]])
    r = _normalize_cam(cam)
    assert r.dtype == np.uint8 and r.max() == 255 and r.min() == 0

test("_normalize_cam", test_normalize_cam)

def test_normalize_cam_zeros():
    from gradcam.gradcam_engine import _normalize_cam
    assert _normalize_cam(np.zeros((5, 5))).max() == 0

test("_normalize_cam zeros", test_normalize_cam_zeros)

def test_colormap():
    from gradcam.gradcam_engine import _apply_colormap
    h = np.array([[0, 128, 255]], dtype=np.uint8)
    c = _apply_colormap(h)
    assert c.shape == (1, 3, 3) and c.dtype == np.uint8

test("_apply_colormap", test_colormap)

def test_overlay():
    from gradcam.gradcam_engine import _create_overlay
    img = Image.new("RGB", (100, 100), "gray")
    hm = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    o = _create_overlay(img, hm)
    assert o.size == (100, 100) and o.mode == "RGB"

test("_create_overlay", test_overlay)

def test_overlay_none():
    from gradcam.gradcam_engine import _create_overlay
    img = Image.new("RGB", (50, 50), "red")
    assert _create_overlay(img, None) == img

test("_create_overlay None heatmap", test_overlay_none)

def test_fallback():
    from gradcam.gradcam_engine import _generate_fallback_attention
    r = _generate_fallback_attention(Image.new("RGB", (200, 200), "white"))
    assert r.size == (200, 200)

test("_generate_fallback_attention", test_fallback)

def test_side_by_side():
    from gradcam.gradcam_engine import create_side_by_side
    a = Image.new("RGB", (100, 100), "red")
    b = Image.new("RGB", (100, 100), "blue")
    r = create_side_by_side(a, b)
    assert r.width == 210 and r.height == 100

test("create_side_by_side", test_side_by_side)

def test_side_by_side_none():
    from gradcam.gradcam_engine import create_side_by_side
    a = Image.new("RGB", (100, 100), "red")
    assert create_side_by_side(a, None) == a

test("create_side_by_side None", test_side_by_side_none)


# ===================================================================
# 11. Ollama Client Tests
# ===================================================================

section("11. Ollama Client Tests")

def test_b64():
    import base64
    from model.ollama_client import pil_image_to_base64
    img = Image.new("RGB", (10, 10), "red")
    b = pil_image_to_base64(img)
    assert base64.b64decode(b)[:4] == b'\x89PNG'

test("pil_image_to_base64", test_b64)

def test_conn_graceful():
    from model.ollama_client import check_ollama_connection
    ok, msg = check_ollama_connection("http://localhost:99999")
    assert not ok and isinstance(msg, str)

test("check_ollama_connection graceful fail", test_conn_graceful)

def test_list_graceful():
    from model.ollama_client import list_ollama_models
    assert isinstance(list_ollama_models("http://localhost:99999"), list)

test("list_ollama_models graceful fail", test_list_graceful)


# ===================================================================
# 12. App.py Function Tests
# ===================================================================

section("12. App.py Function Tests")

def test_set_backend_ollama():
    from app import set_backend, _backend_cfg
    r = set_backend("ollama", "qwen3-vl:30b", "http://localhost:11434")
    assert "Ollama" in r
    assert _backend_cfg["active"] == "ollama"
    assert _backend_cfg["ollama_model"] == "qwen3-vl:30b"

test("set_backend ollama", test_set_backend_ollama)

def test_set_backend_medvlm():
    from app import set_backend, _backend_cfg
    r = set_backend("medvlm-r1", "", "")
    assert "MedVLM" in r
    assert _backend_cfg["active"] == "medvlm-r1"

test("set_backend medvlm-r1", test_set_backend_medvlm)

def test_upload_image():
    from app import handle_standard_image_upload
    state = {"current_image": None, "dicom_ds": None, "dicom_metadata": None}
    preview, status, s = handle_standard_image_upload(Image.new("RGB", (50, 50)), state)
    assert preview is not None and s["current_image"] is not None

test("handle_standard_image_upload", test_upload_image)

def test_upload_none():
    from app import handle_standard_image_upload
    state = {"current_image": None, "dicom_ds": None, "dicom_metadata": None}
    _, status, _ = handle_standard_image_upload(None, state)
    assert "upload" in status.lower()

test("handle_standard_image_upload None", test_upload_none)

def test_check_phi_no_dicom():
    from app import check_phi
    assert "No DICOM" in check_phi({"dicom_ds": None})

test("check_phi no DICOM", test_check_phi_no_dicom)

def test_ct_window_no_dicom():
    from app import apply_ct_window
    _, msg = apply_ct_window("Brain", {"dicom_ds": None, "current_image": None})
    assert "DICOM" in msg or "Load" in msg

test("apply_ct_window no DICOM", test_ct_window_no_dicom)

def test_comparison_no_images():
    from app import run_comparison
    r = run_comparison(None, None, "", {})
    assert "upload" in r.lower() or "Please" in r

test("run_comparison no images", test_comparison_no_images)

def test_export_pdf_no_data():
    from app import export_pdf
    assert "No report" in export_pdf({"last_report": ""})

test("export_pdf empty", test_export_pdf_no_data)

def test_export_fhir_no_data():
    from app import export_fhir
    assert "No analysis" in export_fhir({"last_result": "", "dicom_metadata": None})

test("export_fhir empty", test_export_fhir_no_data)

def test_show_history():
    from app import show_history
    assert isinstance(show_history(), str)

test("show_history", test_show_history)

def test_show_audit():
    from app import show_audit_log
    assert isinstance(show_audit_log(), str)

test("show_audit_log", test_show_audit)

def test_clear_cache():
    from app import clear_cache
    assert "Cache" in clear_cache() or "cleared" in clear_cache().lower()

test("clear_cache", test_clear_cache)


# ===================================================================
# 13. DICOM Handler Tests
# ===================================================================

section("13. DICOM Handler Tests")

def test_ct_presets():
    from dicom.dicom_handler import get_ct_window_presets
    p = get_ct_window_presets()
    assert len(p) >= 9
    for v in p.values():
        assert "center" in v and "width" in v

test("get_ct_window_presets", test_ct_presets)

def test_maps():
    from dicom.dicom_handler import BODY_PART_MAP, MODALITY_MAP
    assert "CHEST" in BODY_PART_MAP and "CT" in MODALITY_MAP

test("BODY_PART_MAP / MODALITY_MAP", test_maps)


# ===================================================================
# Summary
# ===================================================================

section("REGRESSION TEST SUMMARY")
print(f"\n  Total:  {PASS + FAIL}")
print(f"  Passed: {PASS}")
print(f"  Failed: {FAIL}")

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
