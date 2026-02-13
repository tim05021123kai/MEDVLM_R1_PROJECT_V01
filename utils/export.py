"""
Report export module.
Supports PDF export (plain text fallback) and FHIR DiagnosticReport JSON.
"""

import json
import os
from datetime import datetime


def export_report_to_text(report_text, output_dir="./exports"):
    """Export report as a plain text file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)
    return filepath


def export_report_to_pdf(report_text, output_dir="./exports"):
    """
    Export report as PDF. Tries reportlab first, falls back to text.

    Returns:
        filepath to the exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_LEFT

        filepath = os.path.join(output_dir, filename + ".pdf")
        doc = SimpleDocTemplate(filepath, pagesize=A4,
                                leftMargin=20 * mm, rightMargin=20 * mm,
                                topMargin=20 * mm, bottomMargin=20 * mm)
        styles = getSampleStyleSheet()
        style_normal = styles["Normal"]
        style_normal.fontName = "Helvetica"
        style_normal.fontSize = 9
        style_normal.leading = 13

        style_title = styles["Heading1"]
        style_title.fontSize = 14

        story = []
        story.append(Paragraph("Radiology Report", style_title))
        story.append(Spacer(1, 5 * mm))

        for line in report_text.split("\n"):
            if line.strip():
                # Escape HTML special chars
                safe = (line.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;"))
                story.append(Paragraph(safe, style_normal))
            else:
                story.append(Spacer(1, 3 * mm))

        doc.build(story)
        return filepath

    except ImportError:
        # Fallback to plain text
        filepath = os.path.join(output_dir, filename + ".txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)
        return filepath


def export_report_to_fhir(ai_analysis, metadata=None, clinical_history="",
                          output_dir="./exports"):
    """
    Export report as FHIR R4 DiagnosticReport JSON.

    Returns:
        filepath to the exported JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"fhir_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    patient = {}
    series = {}
    if metadata:
        patient = metadata.get("patient_info", {})
        series = metadata.get("series_info", {})

    # Build FHIR DiagnosticReport resource
    fhir_report = {
        "resourceType": "DiagnosticReport",
        "status": "preliminary",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic imaging study"
                }
            ],
            "text": f"{series.get('modality', 'Imaging')} - {series.get('body_part', 'Unknown')}"
        },
        "subject": {
            "display": patient.get("patient_name", "Anonymous")
        },
        "effectiveDateTime": now,
        "issued": now,
        "performer": [
            {
                "display": "MedVLM-R1 AI Virtual Radiologist"
            }
        ],
        "conclusion": ai_analysis[:500] if ai_analysis else "",
        "presentedForm": [
            {
                "contentType": "text/plain",
                "data": None,
                "title": "Full AI Analysis"
            }
        ],
    }

    if clinical_history:
        fhir_report["clinicalNote"] = clinical_history

    # Add modality info as imagingStudy reference
    if series.get("modality"):
        fhir_report["imagingStudy"] = [
            {
                "display": f"{series.get('modality_name', series.get('modality', ''))} - {series.get('body_part_name', series.get('body_part', ''))}"
            }
        ]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(fhir_report, f, indent=2, ensure_ascii=False)

    return filepath
