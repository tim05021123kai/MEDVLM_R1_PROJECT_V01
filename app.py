"""
MedVLM-R1 Medical Image AI Viewer — v2
Full-featured medical image analysis system:
- DICOM import (file + PACS), CT windowing, DICOM de-identification
- Dual backend: Ollama (local VLM) + MedVLM-R1 (HuggingFace)
- Streaming inference output (token-by-token)
- Backend-aware CoT prompts, structured output, few-shot examples
- Medical image preprocessing (auto-contrast, border crop)
- Multi-image comparison (prior vs current)
- Confidence scoring
- Grad-CAM / Attention Rollout / Perturbation Saliency
- Structured report generation with PDF + FHIR export
- LRU cache + analysis history
- Audit trail logging
- Gradio Session State (multi-user safe)
"""

import gradio as gr
import numpy as np
import os
import time
from PIL import Image

from model.medvlm_loader import load_medvlm_model
from model.ollama_client import (
    check_ollama_connection,
    list_ollama_models,
    DEFAULT_OLLAMA_URL,
)
from utils.prompt_utils import (
    build_prompt, build_cot_prompt, build_comparison_prompt,
    get_fewshot_for_body_part,
)
from utils.physio_params import (
    format_reference_params,
    format_lab_references,
    get_context_for_ai_prompt,
    IMAGING_REFERENCE,
)
from utils.image_utils import preprocess_medical_image
from utils.cache import InferenceCache, AnalysisHistory
from utils.audit_logger import AuditLogger
from utils.export import export_report_to_pdf, export_report_to_fhir
from inference.run_inference import (
    generate_answer, generate_answer_stream,
    generate_answer_ollama, generate_answer_ollama_stream,
    generate_multi_image_comparison,
)
from report.report_generator import RadiologyReportGenerator, generate_ai_radiology_prompt
from gradcam.gradcam_engine import (
    generate_gradcam_visualization,
    perturbation_saliency,
    create_side_by_side,
)

# Conditional imports
try:
    from dicom.dicom_handler import (
        load_dicom_file, dicom_to_pil_image, extract_dicom_metadata,
        format_metadata_display, get_ct_window_presets, HAS_PYDICOM,
    )
except ImportError:
    HAS_PYDICOM = False

try:
    from dicom.pacs_client import PACSClient, PACSConfig
    HAS_PACS = True
except ImportError:
    HAS_PACS = False

try:
    from utils.dicom_deidentify import deidentify_dicom, deidentify_metadata, get_phi_summary
    HAS_DEIDENT = True
except ImportError:
    HAS_DEIDENT = False

# ---------------------------------------------------------------------------
# Singletons (thread-safe, read-mostly)
# ---------------------------------------------------------------------------
_model_cache = {"model": None, "processor": None}
report_gen = RadiologyReportGenerator()
inference_cache = InferenceCache(max_size=100)
analysis_history = AnalysisHistory(max_entries=200)
audit_logger = AuditLogger()

# Backend config (mutable from UI)
_backend_cfg = {
    "active": "ollama",
    "ollama_model": "qwen3-vl",
    "ollama_url": DEFAULT_OLLAMA_URL,
}


def _load_model():
    """Lazy-load MedVLM-R1 (cached globally)."""
    if _model_cache["model"] is None:
        print("Loading MedVLM-R1 model...")
        m, p = load_medvlm_model()
        _model_cache["model"] = m
        _model_cache["processor"] = p
        print("Model loaded!")
    return _model_cache["model"], _model_cache["processor"]


# ===================================================================
# Backend helpers
# ===================================================================

def test_ollama_connection(url):
    url = url.strip() or DEFAULT_OLLAMA_URL
    _backend_cfg["ollama_url"] = url
    connected, msg = check_ollama_connection(url)
    if connected:
        models = list_ollama_models(url)
        model_list = "\n".join(f"  - {m}" for m in models) if models else "  (none)"
        return f"Connected to {url}\n\nAvailable models:\n{model_list}"
    return f"Connection failed: {msg}"


def refresh_ollama_models(url):
    url = url.strip() or DEFAULT_OLLAMA_URL
    models = list_ollama_models(url)
    return gr.update(choices=models, value=models[0] if models else "")


def set_backend(choice, selected_model, url):
    _backend_cfg["active"] = choice
    if selected_model:
        _backend_cfg["ollama_model"] = selected_model
    if url:
        _backend_cfg["ollama_url"] = url.strip()
    if choice == "ollama":
        return f"Backend: Ollama ({_backend_cfg['ollama_model']}) @ {_backend_cfg['ollama_url']}"
    return "Backend: MedVLM-R1 (HuggingFace)"


def _run_inference(image, prompt, stream=False):
    """Run inference with the active backend. Returns string or yields chunks."""
    backend = _backend_cfg["active"]
    if backend == "ollama":
        if stream:
            return generate_answer_ollama_stream(
                image, prompt,
                model_name=_backend_cfg["ollama_model"],
                base_url=_backend_cfg["ollama_url"],
            )
        return generate_answer_ollama(
            image, prompt,
            model_name=_backend_cfg["ollama_model"],
            base_url=_backend_cfg["ollama_url"],
        )
    else:
        m, p = _load_model()
        if stream:
            return generate_answer_stream(m, p, image, prompt)
        return generate_answer(m, p, image, prompt)


def _run_inference_with_fallback(image, prompt):
    """Try active backend; on failure, try the other backend."""
    try:
        return _run_inference(image, prompt, stream=False)
    except Exception as first_err:
        alt = "medvlm-r1" if _backend_cfg["active"] == "ollama" else "ollama"
        try:
            if alt == "ollama":
                return generate_answer_ollama(
                    image, prompt,
                    model_name=_backend_cfg["ollama_model"],
                    base_url=_backend_cfg["ollama_url"],
                )
            else:
                m, p = _load_model()
                return generate_answer(m, p, image, prompt)
        except Exception:
            raise first_err  # Re-raise original if fallback also fails


# ===================================================================
# Tab 1 — Image Import
# ===================================================================

def handle_standard_image_upload(image, state):
    if image is None:
        return None, "Please upload an image", state
    state["current_image"] = image
    state["dicom_ds"] = None
    state["dicom_metadata"] = None
    return image, "Image loaded", state


def handle_dicom_upload(file_obj, auto_deident, state):
    if not HAS_PYDICOM:
        return None, "pydicom not installed", "", state
    if file_obj is None:
        return None, "Upload a .dcm file", "", state
    try:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        ds = load_dicom_file(path)

        if auto_deident and HAS_DEIDENT:
            ds = deidentify_dicom(ds)

        metadata = extract_dicom_metadata(ds)
        pil_image = dicom_to_pil_image(ds)

        # Medical preprocessing
        modality = metadata.get("series_info", {}).get("modality", "")
        pil_image = preprocess_medical_image(pil_image, modality=modality)

        state["current_image"] = pil_image
        state["dicom_ds"] = ds
        state["dicom_metadata"] = metadata

        return pil_image, "DICOM loaded", format_metadata_display(metadata), state
    except Exception as e:
        return None, f"DICOM error: {e}", "", state


def apply_ct_window(preset_name, state):
    ds = state.get("dicom_ds")
    if ds is None:
        return state.get("current_image"), "Load a DICOM first"
    presets = get_ct_window_presets()
    if preset_name not in presets:
        return state.get("current_image"), f"Unknown preset: {preset_name}"
    p = presets[preset_name]
    try:
        img = dicom_to_pil_image(ds, window_center=p["center"], window_width=p["width"])
        state["current_image"] = img
        return img, f"Applied {preset_name}"
    except Exception as e:
        return state.get("current_image"), str(e)


def apply_custom_window(wc, ww, state):
    ds = state.get("dicom_ds")
    if ds is None:
        return state.get("current_image"), "Load a DICOM first"
    try:
        img = dicom_to_pil_image(ds, window_center=float(wc), window_width=float(ww))
        state["current_image"] = img
        return img, f"Custom WC={wc}, WW={ww}"
    except Exception as e:
        return state.get("current_image"), str(e)


# ===================================================================
# Tab 2 — PACS
# ===================================================================
pacs_client = None
pacs_query_results = []


def pacs_connect(host, port, ae_title, peer_ae_title):
    global pacs_client
    if not HAS_PACS:
        return "pynetdicom not installed"
    try:
        config = PACSConfig(host=host.strip(), port=int(port),
                            ae_title=ae_title.strip(), peer_ae_title=peer_ae_title.strip())
        pacs_client = PACSClient(config)
        ok, msg = pacs_client.verify_connection()
        return pacs_client.get_connection_status_text(ok, msg)
    except Exception as e:
        return str(e)


def pacs_query(pn, pid, sd, mod):
    global pacs_query_results
    if pacs_client is None:
        return "Connect to PACS first"
    try:
        results = pacs_client.query_studies(
            patient_name=pn.strip(), patient_id=pid.strip(),
            study_date=sd.strip(), modality=mod.strip() if mod else "")
        pacs_query_results = results
        return pacs_client.format_query_results_table(results)
    except Exception as e:
        return str(e)


def pacs_retrieve(idx, state):
    global pacs_query_results
    if pacs_client is None:
        return None, "Connect first", "", state
    try:
        i = int(idx) - 1
        if i < 0 or i >= len(pacs_query_results):
            return None, f"Enter 1-{len(pacs_query_results)}", "", state
        result = pacs_query_results[i]
        ok, files, msg = pacs_client.retrieve_study(result.study_instance_uid)
        if ok and files:
            ds = load_dicom_file(files[0])
            metadata = extract_dicom_metadata(ds)
            img = dicom_to_pil_image(ds)
            state["current_image"] = img
            state["dicom_ds"] = ds
            state["dicom_metadata"] = metadata
            return img, msg, format_metadata_display(metadata), state
        return None, msg, "", state
    except Exception as e:
        return None, str(e), "", state


# ===================================================================
# Tab 3 — AI Virtual Radiologist (streaming)
# ===================================================================

def run_ai_analysis(image, question, analysis_type, clinical_history,
                    body_part_override, gen_report, enable_fewshot, state):
    """Streaming AI analysis with cache, history, audit, and error recovery."""
    work_image = image if image is not None else state.get("current_image")
    if work_image is None:
        yield "Please upload an image first", "", "", state
        return

    if not question.strip():
        question = "Please analyze this medical image and describe your findings"

    # Check cache
    cached = inference_cache.get(work_image, question, _backend_cfg["active"])
    if cached:
        yield cached, "", "(cached result)", state
        return

    backend = _backend_cfg["active"]
    model_label = (f"Ollama ({_backend_cfg['ollama_model']})"
                   if backend == "ollama" else "MedVLM-R1")

    yield f"Analyzing with {model_label}...", "", "", state

    # Determine body part and metadata
    meta = state.get("dicom_metadata")
    body_part = body_part_override.strip().upper() if body_part_override.strip() else None
    modality = ""
    age_str = None
    sex = None
    if meta:
        series = meta.get("series_info", {})
        patient = meta.get("patient_info", {})
        if not body_part:
            body_part = series.get("body_part", "")
        modality = series.get("modality", "")
        age_str = patient.get("patient_age")
        sex = patient.get("patient_sex")

    # Physiological context with age/sex adjustment
    physio_context = ""
    if body_part:
        physio_context = get_context_for_ai_prompt(body_part, modality, age_str, sex)

    # Build prompt
    if analysis_type == "AI Radiology Report":
        prompt = generate_ai_radiology_prompt(
            metadata=meta, clinical_history=clinical_history,
            physio_context=physio_context,
        )
    elif analysis_type == "Chain-of-Thought":
        prompt = build_cot_prompt(question, backend=backend)
    elif analysis_type == "Simple":
        prompt = build_prompt(question, template_type="simple")
    else:
        prompt = build_prompt(question, template_type="medical")
        if physio_context:
            prompt += f"\n\nReference parameters:\n{physio_context}"

    # Add few-shot example if enabled
    if enable_fewshot and body_part:
        fewshot = get_fewshot_for_body_part(body_part)
        if fewshot:
            prompt = fewshot + "\nNow analyze the provided image:\n" + prompt

    # Streaming inference with error recovery
    accumulated = ""
    try:
        stream = _run_inference(work_image, prompt, stream=True)
        for chunk in stream:
            accumulated += chunk
            yield accumulated, "", "", state
    except Exception as e:
        # Fallback to other backend
        try:
            accumulated = _run_inference_with_fallback(work_image, prompt)
            model_label += " (fallback)"
        except Exception as e2:
            yield f"Error: {e2}", "", "", state
            return

    # Cache the result
    inference_cache.put(work_image, question, accumulated, backend)

    # Physio display
    physio_display = format_reference_params(body_part) if body_part else ""

    # Report generation
    report_text = ""
    if gen_report:
        report_text = report_gen.generate_report(
            ai_analysis=accumulated,
            metadata=meta,
            physio_context=physio_context,
            clinical_history=clinical_history,
            backend_name=model_label,
        )

    # History + Audit
    analysis_history.add_entry(
        image_name="uploaded", prompt=question,
        result=accumulated, backend=model_label,
    )
    audit_logger.log_analysis(
        backend=backend, model_name=_backend_cfg.get("ollama_model", "medvlm-r1"),
        image_source="upload", prompt_summary=question,
        result_summary=accumulated[:300], dicom_metadata=meta,
    )

    yield accumulated, physio_display, report_text, state


# ===================================================================
# Tab 3b — Multi-Image Comparison
# ===================================================================

def run_comparison(prior_image, current_image, clinical_context, state):
    if prior_image is None or current_image is None:
        return "Please upload both prior and current images"
    prompt = build_comparison_prompt(clinical_context)
    try:
        result = generate_multi_image_comparison(
            images=[prior_image, current_image], prompt=prompt,
            model_name=_backend_cfg["ollama_model"],
            base_url=_backend_cfg["ollama_url"],
        )
        return result
    except Exception as e:
        return f"Comparison error: {e}"


# ===================================================================
# Tab 4 — Interpretability
# ===================================================================

def run_gradcam(image, prompt_text, method, state):
    try:
        m, p = _load_model()
        work_image = image if image is not None else state.get("current_image")
        if work_image is None:
            return None, None, "Upload an image first"
        if not prompt_text.strip():
            prompt_text = "Analyze this medical image"
        overlay, description = generate_gradcam_visualization(
            m, p, work_image, prompt_text, method=method)
        side = create_side_by_side(work_image, overlay)
        return overlay, side, description
    except Exception as e:
        return None, None, f"Error: {e}"


def run_perturbation_saliency(image, prompt_text, grid_size, state):
    """Perturbation saliency — works with Ollama too."""
    work_image = image if image is not None else state.get("current_image")
    if work_image is None:
        return None, None, "Upload an image first"
    if not prompt_text.strip():
        prompt_text = "Analyze this medical image"

    def inference_fn(img, pmt):
        return _run_inference_with_fallback(img, pmt)

    try:
        _, overlay = perturbation_saliency(
            work_image, prompt_text, inference_fn, grid_size=int(grid_size))
        side = create_side_by_side(work_image, overlay)
        return overlay, side, (
            f"Perturbation Saliency ({int(grid_size)}x{int(grid_size)} grid):\n"
            f"  Backend: {_backend_cfg['active']}\n"
            f"  Red/yellow = regions critical to the analysis\n"
            f"  Gray masking was applied to each grid cell\n"
            f"  Response change was measured to determine importance."
        )
    except Exception as e:
        return None, None, f"Error: {e}"


# ===================================================================
# Tab 5 — Reference
# ===================================================================

def show_body_part_reference(key):
    return format_reference_params(key)

def show_lab_references():
    return format_lab_references()


# ===================================================================
# Tab 6 — History / Export / Audit
# ===================================================================

def show_history():
    return analysis_history.format_history_display()


def show_audit_log():
    return audit_logger.format_log_display()


def export_pdf(state):
    report = state.get("last_report", "")
    if not report:
        return "No report to export"
    path = export_report_to_pdf(report)
    return f"Exported to: {path}"


def export_fhir(state):
    result = state.get("last_result", "")
    meta = state.get("dicom_metadata")
    if not result:
        return "No analysis to export"
    path = export_report_to_fhir(result, metadata=meta)
    return f"Exported to: {path}"


def clear_cache():
    inference_cache.clear()
    return f"Cache cleared (was {inference_cache.size} entries)"


# ===================================================================
# DICOM De-identification
# ===================================================================

def check_phi(state):
    ds = state.get("dicom_ds")
    if ds is None:
        return "No DICOM loaded"
    if not HAS_DEIDENT:
        return "De-identification module not available"
    return get_phi_summary(ds)


def run_deidentify(state):
    ds = state.get("dicom_ds")
    if ds is None:
        return "No DICOM loaded", state
    if not HAS_DEIDENT:
        return "Module not available", state
    ds = deidentify_dicom(ds)
    meta = extract_dicom_metadata(ds)
    state["dicom_ds"] = ds
    state["dicom_metadata"] = meta
    return "De-identification complete. PHI removed.", state


# ===================================================================
# GUI Construction
# ===================================================================

def create_interface():
    css = """
    .gradio-container { max-width: 1400px !important; }
    .medical-header {
        text-align: center;
        background: linear-gradient(135deg, #1a3a4a 0%, #2c5364 100%);
        color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px;
    }
    .medical-header h1 { color: white; margin: 0; }
    .medical-header p { color: #b0d4e8; margin: 5px 0 0; }
    .report-output { font-family: 'Courier New', monospace; white-space: pre-wrap; }
    """

    ct_presets = list(get_ct_window_presets().keys()) if HAS_PYDICOM else []
    initial_ollama_models = list_ollama_models(DEFAULT_OLLAMA_URL)

    with gr.Blocks(css=css, title="MedVLM-R1 Medical Image AI Viewer v2") as interface:

        # Session state
        session = gr.State({
            "current_image": None,
            "dicom_ds": None,
            "dicom_metadata": None,
            "last_report": "",
            "last_result": "",
        })

        # Header
        gr.HTML("""
        <div class="medical-header">
            <h1>MedVLM-R1 Medical Image AI Viewer v2</h1>
            <p>DICOM | AI Radiologist | Streaming | Multi-Image | Grad-CAM | Ollama</p>
            <p style="font-size:12px; margin-top:8px; color:#ffcccc;">
                For educational and research purposes only. Not for clinical decision-making.
            </p>
        </div>
        """)

        # Backend selector
        with gr.Accordion("Model Backend Settings", open=True):
            with gr.Row():
                backend_radio = gr.Radio(
                    ["ollama", "medvlm-r1"], value="ollama", label="AI Backend")
                ollama_url = gr.Textbox(
                    label="Ollama URL", value=DEFAULT_OLLAMA_URL)
                ollama_model_dd = gr.Dropdown(
                    choices=initial_ollama_models,
                    value=initial_ollama_models[0] if initial_ollama_models else "",
                    label="Ollama Model", allow_custom_value=True)
            with gr.Row():
                ollama_test_btn = gr.Button("Test Connection", variant="secondary")
                ollama_refresh_btn = gr.Button("Refresh Models", variant="secondary", size="sm")
                backend_apply_btn = gr.Button("Apply", variant="primary")
                backend_status = gr.Textbox(
                    label="Active Backend", interactive=False,
                    value=f"Backend: Ollama ({initial_ollama_models[0]})" if initial_ollama_models else "Backend: Ollama")
            conn_info = gr.Textbox(label="Connection Info", interactive=False, lines=4, visible=False)

            ollama_test_btn.click(test_ollama_connection, [ollama_url], [conn_info]).then(
                lambda: gr.update(visible=True), outputs=[conn_info])
            ollama_refresh_btn.click(refresh_ollama_models, [ollama_url], [ollama_model_dd])
            backend_apply_btn.click(set_backend, [backend_radio, ollama_model_dd, ollama_url], [backend_status])

        with gr.Tabs():

            # ==================== TAB 1: Image Import ====================
            with gr.Tab("1. Image Import"):
                with gr.Row():
                    with gr.Column():
                        std_img = gr.Image(label="Upload (JPG/PNG)", type="pil", height=350)
                        std_status = gr.Textbox(label="Status", interactive=False)
                        gr.Markdown("---")
                        dicom_file = gr.File(label="DICOM (.dcm)", file_types=[".dcm", ".dicom"])
                        auto_deident = gr.Checkbox(label="Auto de-identify PHI on upload", value=False)
                        dicom_status = gr.Textbox(label="DICOM Status", interactive=False)
                    with gr.Column():
                        preview = gr.Image(label="Preview", height=400, interactive=False)
                        meta_display = gr.Textbox(label="Metadata", lines=12, interactive=False, show_copy_button=True)

                with gr.Accordion("CT Window Presets", open=False):
                    with gr.Row():
                        w_preset = gr.Dropdown(choices=ct_presets, label="Preset")
                        apply_preset = gr.Button("Apply Preset")
                    with gr.Row():
                        c_wc = gr.Number(label="WC", value=40)
                        c_ww = gr.Number(label="WW", value=350)
                        apply_custom = gr.Button("Apply Custom")

                with gr.Accordion("DICOM De-identification", open=False):
                    phi_check_btn = gr.Button("Check PHI Fields")
                    phi_result = gr.Textbox(label="PHI Summary", interactive=False, lines=6)
                    deident_btn = gr.Button("Remove PHI", variant="stop")
                    deident_status = gr.Textbox(label="De-ID Status", interactive=False)

                std_img.change(handle_standard_image_upload, [std_img, session], [preview, std_status, session])
                dicom_file.change(handle_dicom_upload, [dicom_file, auto_deident, session],
                                  [preview, dicom_status, meta_display, session])
                apply_preset.click(apply_ct_window, [w_preset, session], [preview, dicom_status])
                apply_custom.click(apply_custom_window, [c_wc, c_ww, session], [preview, dicom_status])
                phi_check_btn.click(check_phi, [session], [phi_result])
                deident_btn.click(run_deidentify, [session], [deident_status, session])

            # ==================== TAB 2: PACS ====================
            with gr.Tab("2. PACS Network"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Connection")
                        p_host = gr.Textbox(label="Host", value="127.0.0.1")
                        p_port = gr.Number(label="Port", value=11112, precision=0)
                        p_ae = gr.Textbox(label="Local AE", value="MEDVLM_SCU")
                        p_peer = gr.Textbox(label="PACS AE", value="PACS_SCP")
                        p_conn_btn = gr.Button("Test", variant="primary")
                        p_conn_st = gr.Textbox(label="Status", interactive=False, lines=4)
                    with gr.Column():
                        gr.Markdown("#### Query")
                        qpn = gr.Textbox(label="Patient Name", placeholder="*")
                        qpid = gr.Textbox(label="Patient ID")
                        qsd = gr.Textbox(label="Study Date", placeholder="YYYYMMDD")
                        qmod = gr.Dropdown(["", "CR", "DX", "CT", "MR", "US", "MG"], label="Modality")
                        p_query_btn = gr.Button("Search", variant="primary")
                p_results = gr.Textbox(label="Results", lines=10, interactive=False, show_copy_button=True)
                with gr.Row():
                    ret_idx = gr.Number(label="Study #", value=1, precision=0)
                    p_ret_btn = gr.Button("Retrieve", variant="primary")
                with gr.Row():
                    p_preview = gr.Image(label="Retrieved", height=400, interactive=False)
                    p_meta = gr.Textbox(label="Metadata", lines=10, interactive=False)
                p_ret_st = gr.Textbox(label="Retrieve Status", interactive=False)

                p_conn_btn.click(pacs_connect, [p_host, p_port, p_ae, p_peer], [p_conn_st])
                p_query_btn.click(pacs_query, [qpn, qpid, qsd, qmod], [p_results])
                p_ret_btn.click(pacs_retrieve, [ret_idx, session], [p_preview, p_ret_st, p_meta, session])

            # ==================== TAB 3: AI Radiologist ====================
            with gr.Tab("3. AI Radiologist"):
                gr.Markdown("### AI Virtual Radiologist — streaming analysis with structured reports")
                with gr.Row():
                    with gr.Column():
                        ai_img = gr.Image(label="Image", type="pil", height=350)
                        ai_q = gr.Textbox(label="Question", lines=3,
                                          value="Please analyze this medical image and describe your findings")
                        ai_type = gr.Radio(
                            ["Standard", "Simple", "Chain-of-Thought", "AI Radiology Report"],
                            value="AI Radiology Report", label="Mode")
                        ai_hist = gr.Textbox(label="Clinical History", placeholder="e.g., cough for 2 weeks", lines=2)
                        ai_bp = gr.Textbox(label="Body Part Override", placeholder="CHEST, HEAD, ABDOMEN...")
                        with gr.Row():
                            ai_report_cb = gr.Checkbox(label="Generate Report", value=True)
                            ai_fewshot_cb = gr.Checkbox(label="Include Few-shot Example", value=True)
                        with gr.Row():
                            ai_run = gr.Button("Analyze", variant="primary", size="lg")
                            ai_clear = gr.Button("Clear")

                        gr.Markdown("#### Examples")
                        for eq in ["What abnormalities do you see?",
                                   "Describe the location and features of any lesions",
                                   "What is the likely diagnosis?",
                                   "What further workup do you recommend?"]:
                            gr.Button(eq, size="sm").click(lambda q=eq: q, outputs=ai_q)

                    with gr.Column():
                        ai_result = gr.Textbox(label="AI Analysis (streaming)", lines=15, show_copy_button=True)
                        ai_physio = gr.Textbox(label="Reference Parameters", lines=8, show_copy_button=True)

                with gr.Accordion("Structured Report", open=True):
                    ai_report = gr.Textbox(label="Report", lines=25, show_copy_button=True, elem_classes=["report-output"])

                ai_run.click(
                    run_ai_analysis,
                    [ai_img, ai_q, ai_type, ai_hist, ai_bp, ai_report_cb, ai_fewshot_cb, session],
                    [ai_result, ai_physio, ai_report, session])
                ai_clear.click(lambda: (None, "", "", "", ""),
                               outputs=[ai_img, ai_result, ai_physio, ai_report, ai_q])

            # ==================== TAB 3b: Multi-Image Comparison ====================
            with gr.Tab("3b. Comparison"):
                gr.Markdown("### Multi-Image Comparison (Prior vs Current)")
                gr.Markdown("*Requires Ollama with a vision model that supports multiple images (e.g. qwen3-vl)*")
                with gr.Row():
                    comp_prior = gr.Image(label="Prior Study", type="pil", height=300)
                    comp_current = gr.Image(label="Current Study", type="pil", height=300)
                comp_ctx = gr.Textbox(label="Clinical Context", placeholder="e.g., follow-up after treatment")
                comp_btn = gr.Button("Compare", variant="primary")
                comp_result = gr.Textbox(label="Comparison Result", lines=20, show_copy_button=True)

                comp_btn.click(run_comparison, [comp_prior, comp_current, comp_ctx, session], [comp_result])

            # ==================== TAB 4: Interpretability ====================
            with gr.Tab("4. Interpretability"):
                gr.Markdown("### Model Attention Visualization")
                with gr.Tabs():
                    with gr.Tab("Grad-CAM / Attention Rollout"):
                        gr.Markdown("*Requires MedVLM-R1 model loaded*")
                        with gr.Row():
                            with gr.Column():
                                gc_img = gr.Image(label="Image", type="pil", height=350)
                                gc_prompt = gr.Textbox(label="Prompt", value="Analyze this medical image", lines=2)
                                gc_method = gr.Radio(
                                    ["auto", "attention_rollout", "gradcam"],
                                    value="auto", label="Method")
                                gc_run = gr.Button("Generate", variant="primary")
                            with gr.Column():
                                gc_overlay = gr.Image(label="Heatmap", height=350, interactive=False)
                        gc_side = gr.Image(label="Side-by-Side", height=400, interactive=False)
                        gc_desc = gr.Textbox(label="Description", lines=8, interactive=False, show_copy_button=True)
                        gc_run.click(run_gradcam, [gc_img, gc_prompt, gc_method, session],
                                     [gc_overlay, gc_side, gc_desc])

                    with gr.Tab("Perturbation Saliency"):
                        gr.Markdown("*Works with ANY backend (Ollama included) — slower but model-agnostic*")
                        with gr.Row():
                            with gr.Column():
                                ps_img = gr.Image(label="Image", type="pil", height=350)
                                ps_prompt = gr.Textbox(label="Prompt", value="Analyze this medical image", lines=2)
                                ps_grid = gr.Slider(3, 10, value=5, step=1, label="Grid Size (NxN)")
                                ps_run = gr.Button("Generate Saliency Map", variant="primary")
                            with gr.Column():
                                ps_overlay = gr.Image(label="Saliency", height=350, interactive=False)
                        ps_side = gr.Image(label="Side-by-Side", height=400, interactive=False)
                        ps_desc = gr.Textbox(label="Description", lines=6, interactive=False, show_copy_button=True)
                        ps_run.click(run_perturbation_saliency, [ps_img, ps_prompt, ps_grid, session],
                                     [ps_overlay, ps_side, ps_desc])

                gr.Markdown("""
                **Interpretation guide:**
                - **Red/Yellow** = High attention (model focused here)
                - **Blue/Green** = Low attention
                - Verify the AI examines clinically relevant regions
                """)

            # ==================== TAB 5: Reference ====================
            with gr.Tab("5. Reference"):
                with gr.Row():
                    with gr.Column():
                        ref_bp = gr.Dropdown(choices=list(IMAGING_REFERENCE.keys()),
                                             label="Body Part", value="CHEST")
                        ref_show = gr.Button("Show Parameters", variant="primary")
                        ref_lab = gr.Button("Show Lab References")
                    with gr.Column(scale=2):
                        ref_display = gr.Textbox(label="Reference", lines=25, interactive=False, show_copy_button=True)
                ref_show.click(show_body_part_reference, [ref_bp], [ref_display])
                ref_lab.click(show_lab_references, outputs=[ref_display])

            # ==================== TAB 6: History / Export / Audit ====================
            with gr.Tab("6. History & Export"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Analysis History")
                        hist_btn = gr.Button("Show History")
                        hist_display = gr.Textbox(label="History", lines=20, interactive=False, show_copy_button=True)
                        hist_btn.click(show_history, outputs=[hist_display])

                    with gr.Column():
                        gr.Markdown("#### Export")
                        pdf_btn = gr.Button("Export Report as PDF")
                        pdf_status = gr.Textbox(label="PDF Export", interactive=False)
                        fhir_btn = gr.Button("Export as FHIR JSON")
                        fhir_status = gr.Textbox(label="FHIR Export", interactive=False)
                        pdf_btn.click(export_pdf, [session], [pdf_status])
                        fhir_btn.click(export_fhir, [session], [fhir_status])

                with gr.Accordion("Audit Trail", open=False):
                    audit_btn = gr.Button("Show Audit Log")
                    audit_display = gr.Textbox(label="Audit Log", lines=15, interactive=False, show_copy_button=True)
                    audit_btn.click(show_audit_log, outputs=[audit_display])

                with gr.Accordion("Cache Management", open=False):
                    cache_btn = gr.Button("Clear Inference Cache")
                    cache_status = gr.Textbox(label="Cache", interactive=False)
                    cache_btn.click(clear_cache, outputs=[cache_status])

        gr.Markdown("""
        ---
        **MedVLM-R1 Medical Image AI Viewer v2** | Ollama + MedVLM-R1 | For educational & research purposes only
        """)

    return interface


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("  MedVLM-R1 Medical Image AI Viewer v2")
    print("=" * 60)

    connected, msg = check_ollama_connection(DEFAULT_OLLAMA_URL)
    if connected:
        models = list_ollama_models(DEFAULT_OLLAMA_URL)
        print(f"Ollama connected. Models: {models}")
    else:
        print(f"Ollama not available: {msg}")

    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1", server_port=7860,
        share=False, debug=True, show_error=True,
    )


if __name__ == "__main__":
    main()
