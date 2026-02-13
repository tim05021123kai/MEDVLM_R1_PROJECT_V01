"""
MedVLM-R1 Medical Image AI Viewer
A comprehensive medical image analysis system with:
- DICOM image import (from PACS or manual upload)
- Virtual AI radiologist for structured radiology report generation
- Physiological parameter references
- Grad-CAM interpretability visualization
- User-friendly tabbed Gradio GUI
"""

import gradio as gr
import torch
import numpy as np
import os
from PIL import Image

from model.medvlm_loader import load_medvlm_model
from utils.prompt_utils import build_prompt, build_cot_prompt
from utils.physio_params import (
    format_reference_params,
    format_lab_references,
    get_context_for_ai_prompt,
    IMAGING_REFERENCE,
)
from inference.run_inference import generate_answer
from report.report_generator import RadiologyReportGenerator, generate_ai_radiology_prompt
from gradcam.gradcam_engine import (
    generate_gradcam_visualization,
    create_side_by_side,
)

# Conditional DICOM imports
try:
    from dicom.dicom_handler import (
        load_dicom_file,
        dicom_to_pil_image,
        extract_dicom_metadata,
        format_metadata_display,
        get_ct_window_presets,
        HAS_PYDICOM,
    )
except ImportError:
    HAS_PYDICOM = False

try:
    from dicom.pacs_client import PACSClient, PACSConfig
    HAS_PACS = True
except ImportError:
    HAS_PACS = False

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
processor = None
current_dicom_ds = None
current_dicom_metadata = None
current_image = None
report_gen = RadiologyReportGenerator()
pacs_client = None
pacs_query_results = []


def load_model():
    """Load the MedVLM-R1 model (loaded once, cached globally)."""
    global model, processor
    if model is None:
        print("正在載入 MedVLM-R1 模型...")
        model, processor = load_medvlm_model()
        print("模型載入完成！")
    return model, processor


# ===================================================================
# Tab 1 – Image Import (DICOM + standard images)
# ===================================================================

def handle_standard_image_upload(image):
    """Handle standard image upload (JPG/PNG)."""
    global current_image, current_dicom_ds, current_dicom_metadata
    if image is None:
        return None, "請上傳影像 (Please upload an image)"
    current_image = image
    current_dicom_ds = None
    current_dicom_metadata = None
    return image, "影像載入成功 (Image loaded successfully)"


def handle_dicom_upload(file_obj):
    """Handle DICOM file upload."""
    global current_image, current_dicom_ds, current_dicom_metadata

    if not HAS_PYDICOM:
        return (
            None,
            "pydicom 未安裝。請執行: pip install pydicom pylibjpeg pylibjpeg-libjpeg",
            "N/A",
        )

    if file_obj is None:
        return None, "請上傳 DICOM 檔案 (.dcm)", ""

    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        ds = load_dicom_file(file_path)
        current_dicom_ds = ds

        metadata = extract_dicom_metadata(ds)
        current_dicom_metadata = metadata
        metadata_text = format_metadata_display(metadata)

        pil_image = dicom_to_pil_image(ds)
        current_image = pil_image

        return pil_image, "DICOM 檔案載入成功！", metadata_text

    except Exception as e:
        return None, f"DICOM 載入錯誤: {str(e)}", ""


def apply_ct_window(preset_name):
    """Apply a CT windowing preset to the current DICOM image."""
    global current_image, current_dicom_ds

    if current_dicom_ds is None:
        return current_image, "請先載入 DICOM 檔案"

    if not HAS_PYDICOM:
        return current_image, "pydicom 未安裝"

    presets = get_ct_window_presets()
    if preset_name not in presets:
        return current_image, f"未知的窗位預設: {preset_name}"

    preset = presets[preset_name]
    try:
        pil_image = dicom_to_pil_image(
            current_dicom_ds,
            window_center=preset["center"],
            window_width=preset["width"],
        )
        current_image = pil_image
        return pil_image, f"已套用 {preset_name} (WC={preset['center']}, WW={preset['width']})"
    except Exception as e:
        return current_image, f"窗位調整錯誤: {str(e)}"


def apply_custom_window(wc, ww):
    """Apply custom window center/width."""
    global current_image, current_dicom_ds

    if current_dicom_ds is None:
        return current_image, "請先載入 DICOM 檔案"

    if not HAS_PYDICOM:
        return current_image, "pydicom 未安裝"

    try:
        pil_image = dicom_to_pil_image(
            current_dicom_ds,
            window_center=float(wc),
            window_width=float(ww),
        )
        current_image = pil_image
        return pil_image, f"自訂窗位: WC={wc}, WW={ww}"
    except Exception as e:
        return current_image, f"窗位調整錯誤: {str(e)}"


# ===================================================================
# Tab 2 – PACS Query & Retrieve
# ===================================================================

def pacs_connect(host, port, ae_title, peer_ae_title):
    """Connect to a PACS server and verify."""
    global pacs_client

    if not HAS_PACS:
        return "pynetdicom 未安裝。請執行: pip install pynetdicom"

    try:
        config = PACSConfig(
            host=host.strip(),
            port=int(port),
            ae_title=ae_title.strip(),
            peer_ae_title=peer_ae_title.strip(),
        )
        pacs_client = PACSClient(config)
        success, message = pacs_client.verify_connection()
        return pacs_client.get_connection_status_text(success, message)
    except Exception as e:
        return f"Connection failed: {str(e)}"


def pacs_query(patient_name, patient_id, study_date, modality):
    """Query PACS for studies."""
    global pacs_client, pacs_query_results

    if pacs_client is None:
        return "請先連線到 PACS 伺服器"

    try:
        results = pacs_client.query_studies(
            patient_name=patient_name.strip(),
            patient_id=patient_id.strip(),
            study_date=study_date.strip(),
            modality=modality.strip() if modality else "",
        )
        pacs_query_results = results
        return pacs_client.format_query_results_table(results)
    except Exception as e:
        return f"查詢錯誤: {str(e)}"


def pacs_retrieve(study_index):
    """Retrieve a study from PACS by index."""
    global pacs_client, pacs_query_results, current_image, current_dicom_ds, current_dicom_metadata

    if not HAS_PYDICOM:
        return None, "pydicom 未安裝", ""

    if pacs_client is None:
        return None, "請先連線到 PACS 伺服器", ""

    try:
        idx = int(study_index) - 1
        if idx < 0 or idx >= len(pacs_query_results):
            return None, f"無效的編號，請輸入 1-{len(pacs_query_results)}", ""

        result = pacs_query_results[idx]
        success, files, message = pacs_client.retrieve_study(
            result.study_instance_uid
        )

        if success and files:
            ds = load_dicom_file(files[0])
            current_dicom_ds = ds
            metadata = extract_dicom_metadata(ds)
            current_dicom_metadata = metadata
            pil_image = dicom_to_pil_image(ds)
            current_image = pil_image
            return pil_image, message, format_metadata_display(metadata)
        else:
            return None, message, ""

    except Exception as e:
        return None, f"接收錯誤: {str(e)}", ""


# ===================================================================
# Tab 3 – AI Virtual Radiologist
# ===================================================================

def run_ai_analysis(image, question, analysis_type, clinical_history,
                    body_part_override, generate_report_flag):
    """
    Run AI analysis on the current image with optional report generation.
    Yields intermediate status updates for the Gradio interface.
    """
    global current_image, current_dicom_metadata

    try:
        m, p = load_model()

        work_image = image if image is not None else current_image
        if work_image is None:
            yield "請先上傳影像", "", ""
            return

        if not question.strip():
            question = "請分析這張醫學影像並描述你的發現"

        yield "正在分析影像，請稍候...", "", ""

        # Determine body part for physiological context
        body_part = body_part_override.strip().upper() if body_part_override.strip() else None
        if body_part is None and current_dicom_metadata:
            body_part = current_dicom_metadata.get("series_info", {}).get("body_part", "")

        modality = ""
        if current_dicom_metadata:
            modality = current_dicom_metadata.get("series_info", {}).get("modality", "")

        # Get physiological context
        physio_context = ""
        if body_part:
            physio_context = get_context_for_ai_prompt(body_part, modality)

        # Build prompt based on analysis type
        if analysis_type == "AI放射科報告":
            prompt = generate_ai_radiology_prompt(
                metadata=current_dicom_metadata,
                clinical_history=clinical_history,
                physio_context=physio_context,
            )
        elif analysis_type == "鏈式思考":
            prompt = build_cot_prompt(question)
        elif analysis_type == "簡單分析":
            prompt = build_prompt(question, template_type="simple")
        else:
            prompt = build_prompt(question, template_type="medical")
            if physio_context:
                prompt += f"\n\n相關生理參考參數:\n{physio_context}"

        # Generate AI analysis
        ai_result = generate_answer(m, p, work_image, prompt)

        # Physio reference display
        physio_display = ""
        if body_part:
            physio_display = format_reference_params(body_part)

        # Generate structured report if requested
        report_text = ""
        if generate_report_flag:
            report_text = report_gen.generate_report(
                ai_analysis=ai_result,
                metadata=current_dicom_metadata,
                physio_context=physio_context,
                clinical_history=clinical_history,
            )

        yield ai_result, physio_display, report_text

    except Exception as e:
        yield f"錯誤: {str(e)}", "", ""


# ===================================================================
# Tab 4 – Grad-CAM Interpretability
# ===================================================================

def run_gradcam(image, prompt_text):
    """Generate Grad-CAM visualization."""
    global current_image

    try:
        m, p = load_model()

        work_image = image if image is not None else current_image
        if work_image is None:
            return None, None, "請先上傳影像"

        if not prompt_text.strip():
            prompt_text = "請分析這張醫學影像"

        overlay, description = generate_gradcam_visualization(
            m, p, work_image, prompt_text
        )

        side_by_side = create_side_by_side(work_image, overlay)

        return overlay, side_by_side, description

    except Exception as e:
        return None, None, f"Grad-CAM 錯誤: {str(e)}"


# ===================================================================
# Tab 5 – Reference Parameters
# ===================================================================

def show_body_part_reference(body_part_key):
    """Show reference parameters for a selected body part."""
    return format_reference_params(body_part_key)


def show_lab_references():
    """Show lab reference values."""
    return format_lab_references()


# ===================================================================
# GUI Construction
# ===================================================================

def create_interface():
    """Create the full Gradio interface with all tabs."""

    css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .medical-header {
        text-align: center;
        background: linear-gradient(135deg, #1a3a4a 0%, #2c5364 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .medical-header h1 { color: white; margin: 0; }
    .medical-header p { color: #b0d4e8; margin: 5px 0 0; }
    .report-output {
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        line-height: 1.5;
    }
    """

    # Determine CT window preset choices
    ct_presets = []
    if HAS_PYDICOM:
        ct_presets = list(get_ct_window_presets().keys())

    with gr.Blocks(css=css, title="MedVLM-R1 Medical Image AI Viewer") as interface:

        # ---- Header ----
        gr.HTML("""
        <div class="medical-header">
            <h1>MedVLM-R1 Medical Image AI Viewer</h1>
            <p>DICOM Import | AI Radiologist | Grad-CAM Interpretability</p>
            <p style="font-size:12px; margin-top:8px; color:#ffcccc;">
                For educational and research purposes only. Not for clinical decision-making.
            </p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ======================================================
            # TAB 1 – Image Import
            # ======================================================
            with gr.Tab("1. Image Import", id="tab_import"):
                gr.Markdown("### Import medical images from files (DICOM, JPG, PNG) or PACS network")

                with gr.Row():
                    # Left – upload area
                    with gr.Column(scale=1):
                        gr.Markdown("#### Standard Image (JPG/PNG)")
                        std_image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=350,
                        )
                        std_upload_status = gr.Textbox(
                            label="Status", interactive=False, lines=1
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### DICOM File (.dcm)")
                        dicom_file_input = gr.File(
                            label="Upload DICOM (.dcm)",
                            file_types=[".dcm", ".dicom", ".DCM"],
                        )
                        dicom_upload_status = gr.Textbox(
                            label="DICOM Status", interactive=False, lines=1
                        )

                    # Right – preview + metadata
                    with gr.Column(scale=1):
                        gr.Markdown("#### Image Preview")
                        preview_image = gr.Image(
                            label="Preview", height=400, interactive=False
                        )

                        gr.Markdown("#### DICOM Metadata")
                        metadata_display = gr.Textbox(
                            label="Metadata",
                            lines=12,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True,
                        )

                # CT Windowing controls
                with gr.Accordion("CT Window Presets", open=False):
                    with gr.Row():
                        window_preset = gr.Dropdown(
                            choices=ct_presets,
                            label="Window Preset",
                        )
                        apply_preset_btn = gr.Button("Apply Preset", variant="secondary")
                    with gr.Row():
                        custom_wc = gr.Number(label="Window Center (WC)", value=40)
                        custom_ww = gr.Number(label="Window Width (WW)", value=350)
                        apply_custom_btn = gr.Button("Apply Custom", variant="secondary")

                # Event bindings – Image Import
                std_image_input.change(
                    handle_standard_image_upload,
                    inputs=[std_image_input],
                    outputs=[preview_image, std_upload_status],
                )

                dicom_file_input.change(
                    handle_dicom_upload,
                    inputs=[dicom_file_input],
                    outputs=[preview_image, dicom_upload_status, metadata_display],
                )

                apply_preset_btn.click(
                    apply_ct_window,
                    inputs=[window_preset],
                    outputs=[preview_image, dicom_upload_status],
                )
                apply_custom_btn.click(
                    apply_custom_window,
                    inputs=[custom_wc, custom_ww],
                    outputs=[preview_image, dicom_upload_status],
                )

            # ======================================================
            # TAB 2 – PACS Network
            # ======================================================
            with gr.Tab("2. PACS Network", id="tab_pacs"):
                gr.Markdown("### Query and retrieve images from a PACS server via DICOM networking")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Connection Settings")
                        pacs_host = gr.Textbox(label="PACS Host", value="127.0.0.1")
                        pacs_port = gr.Number(label="PACS Port", value=11112, precision=0)
                        pacs_ae = gr.Textbox(label="Local AE Title", value="MEDVLM_SCU")
                        pacs_peer_ae = gr.Textbox(label="PACS AE Title", value="PACS_SCP")
                        pacs_connect_btn = gr.Button("Test Connection", variant="primary")
                        pacs_connect_status = gr.Textbox(
                            label="Connection Status",
                            interactive=False,
                            lines=4,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### Query Studies")
                        q_patient_name = gr.Textbox(label="Patient Name (supports *)", placeholder="*")
                        q_patient_id = gr.Textbox(label="Patient ID", placeholder="")
                        q_study_date = gr.Textbox(
                            label="Study Date (YYYYMMDD or range)",
                            placeholder="20240101-20241231",
                        )
                        q_modality = gr.Dropdown(
                            choices=["", "CR", "DX", "CT", "MR", "US", "MG", "NM", "PT"],
                            label="Modality",
                        )
                        pacs_query_btn = gr.Button("Search", variant="primary")

                pacs_results_display = gr.Textbox(
                    label="Query Results",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                )

                with gr.Row():
                    retrieve_index = gr.Number(
                        label="Study # to Retrieve",
                        value=1,
                        precision=0,
                    )
                    pacs_retrieve_btn = gr.Button("Retrieve", variant="primary")

                with gr.Row():
                    pacs_preview = gr.Image(label="Retrieved Image", height=400, interactive=False)
                    pacs_metadata = gr.Textbox(
                        label="Retrieved Metadata", lines=10, interactive=False
                    )
                pacs_retrieve_status = gr.Textbox(label="Retrieve Status", interactive=False)

                # PACS event bindings
                pacs_connect_btn.click(
                    pacs_connect,
                    inputs=[pacs_host, pacs_port, pacs_ae, pacs_peer_ae],
                    outputs=[pacs_connect_status],
                )
                pacs_query_btn.click(
                    pacs_query,
                    inputs=[q_patient_name, q_patient_id, q_study_date, q_modality],
                    outputs=[pacs_results_display],
                )
                pacs_retrieve_btn.click(
                    pacs_retrieve,
                    inputs=[retrieve_index],
                    outputs=[pacs_preview, pacs_retrieve_status, pacs_metadata],
                )

            # ======================================================
            # TAB 3 – AI Virtual Radiologist
            # ======================================================
            with gr.Tab("3. AI Radiologist", id="tab_ai"):
                gr.Markdown("### Virtual AI Radiologist — generates structured radiology reports")

                with gr.Row():
                    with gr.Column(scale=1):
                        ai_image_input = gr.Image(
                            label="Image (or use imported image from Tab 1)",
                            type="pil",
                            height=350,
                        )
                        ai_question = gr.Textbox(
                            label="Question / Prompt",
                            placeholder="What abnormalities do you see in this image?",
                            lines=3,
                            value="請分析這張醫學影像並描述你的發現",
                        )
                        ai_analysis_type = gr.Radio(
                            choices=["標準分析", "簡單分析", "鏈式思考", "AI放射科報告"],
                            value="AI放射科報告",
                            label="Analysis Mode",
                        )
                        ai_clinical_history = gr.Textbox(
                            label="Clinical History (optional)",
                            placeholder="e.g., cough for 2 weeks, fever",
                            lines=2,
                        )
                        ai_body_part = gr.Textbox(
                            label="Body Part Override (auto-detected from DICOM)",
                            placeholder="CHEST, HEAD, ABDOMEN, SPINE ...",
                        )
                        ai_gen_report = gr.Checkbox(
                            label="Generate Structured Report",
                            value=True,
                        )

                        with gr.Row():
                            ai_run_btn = gr.Button("Analyze", variant="primary", size="lg")
                            ai_clear_btn = gr.Button("Clear", variant="secondary")

                        # Example questions
                        gr.Markdown("#### Example Questions")
                        example_qs = [
                            "這張影像有什麼異常？",
                            "請描述病變的位置和特徵",
                            "這可能是什麼疾病？",
                            "建議做哪些進一步檢查？",
                        ]
                        for eq in example_qs:
                            gr.Button(eq, size="sm").click(
                                lambda q=eq: q, outputs=ai_question
                            )

                    with gr.Column(scale=1):
                        ai_result_text = gr.Textbox(
                            label="AI Analysis Result",
                            lines=15,
                            max_lines=25,
                            show_copy_button=True,
                        )
                        ai_physio_text = gr.Textbox(
                            label="Reference Parameters",
                            lines=8,
                            max_lines=15,
                            show_copy_button=True,
                        )

                with gr.Accordion("Structured Radiology Report", open=True):
                    ai_report_text = gr.Textbox(
                        label="Report",
                        lines=25,
                        max_lines=50,
                        show_copy_button=True,
                        elem_classes=["report-output"],
                    )

                # AI tab event bindings
                ai_run_btn.click(
                    run_ai_analysis,
                    inputs=[
                        ai_image_input, ai_question, ai_analysis_type,
                        ai_clinical_history, ai_body_part, ai_gen_report,
                    ],
                    outputs=[ai_result_text, ai_physio_text, ai_report_text],
                )

                def clear_ai():
                    return None, "", "", "", ""

                ai_clear_btn.click(
                    clear_ai,
                    outputs=[ai_image_input, ai_result_text, ai_physio_text,
                             ai_report_text, ai_question],
                )

            # ======================================================
            # TAB 4 – Grad-CAM
            # ======================================================
            with gr.Tab("4. Grad-CAM", id="tab_gradcam"):
                gr.Markdown(
                    "### Grad-CAM Visualization — understand which image regions the AI focused on"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gc_image_input = gr.Image(
                            label="Image (or use imported image from Tab 1)",
                            type="pil",
                            height=350,
                        )
                        gc_prompt = gr.Textbox(
                            label="Analysis Prompt",
                            value="請分析這張醫學影像並描述你的發現",
                            lines=2,
                        )
                        gc_run_btn = gr.Button(
                            "Generate Grad-CAM", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        gc_overlay = gr.Image(
                            label="Grad-CAM Heatmap Overlay",
                            height=350,
                            interactive=False,
                        )

                gc_side_by_side = gr.Image(
                    label="Side-by-Side: Original vs Grad-CAM",
                    height=400,
                    interactive=False,
                )
                gc_description = gr.Textbox(
                    label="Grad-CAM Analysis Description",
                    lines=8,
                    interactive=False,
                    show_copy_button=True,
                )

                gr.Markdown("""
                **How to interpret:**
                - **Red / Yellow** = High attention areas — the AI focused most here
                - **Blue / Green** = Low attention areas
                - Helps verify the AI is looking at clinically relevant regions
                """)

                gc_run_btn.click(
                    run_gradcam,
                    inputs=[gc_image_input, gc_prompt],
                    outputs=[gc_overlay, gc_side_by_side, gc_description],
                )

            # ======================================================
            # TAB 5 – Reference Parameters
            # ======================================================
            with gr.Tab("5. Reference", id="tab_reference"):
                gr.Markdown("### Physiological & Imaging Reference Parameters")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Select Body Part")
                        ref_body_part = gr.Dropdown(
                            choices=list(IMAGING_REFERENCE.keys()),
                            label="Body Part",
                            value="CHEST",
                        )
                        ref_show_btn = gr.Button("Show Parameters", variant="primary")
                        ref_lab_btn = gr.Button("Show Lab References", variant="secondary")

                    with gr.Column(scale=2):
                        ref_display = gr.Textbox(
                            label="Reference Parameters",
                            lines=25,
                            max_lines=40,
                            show_copy_button=True,
                            interactive=False,
                        )

                ref_show_btn.click(
                    show_body_part_reference,
                    inputs=[ref_body_part],
                    outputs=[ref_display],
                )
                ref_lab_btn.click(
                    show_lab_references,
                    outputs=[ref_display],
                )

        # ---- Footer ----
        gr.Markdown("""
        ---
        **MedVLM-R1 Medical Image AI Viewer** | Model: JZPeterPan/MedVLM-R1
        | For educational & research purposes only
        """)

    return interface


# ===================================================================
# Main entry point
# ===================================================================

def main():
    """Launch the application."""
    print("=" * 60)
    print("  MedVLM-R1 Medical Image AI Viewer")
    print("  Starting Gradio interface...")
    print("=" * 60)

    # Pre-load model
    try:
        load_model()
        print("Model pre-loaded successfully!")
    except Exception as e:
        print(f"Model pre-load failed: {e}")
        print("Model will be loaded on first use.")

    interface = create_interface()

    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
