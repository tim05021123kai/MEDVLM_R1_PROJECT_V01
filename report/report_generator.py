"""
Radiology report generator module.
Generates structured radiology reports following standard clinical formats,
incorporating AI analysis results, DICOM metadata, and physiological parameters.
"""

from datetime import datetime


REPORT_TEMPLATES = {
    "standard": {
        "name": "標準放射報告 Standard Radiology Report",
        "sections": [
            "header",
            "patient_info",
            "clinical_info",
            "technique",
            "comparison",
            "findings",
            "impression",
            "recommendation",
            "signature",
        ],
    },
    "structured": {
        "name": "結構化報告 Structured Report",
        "sections": [
            "header",
            "patient_info",
            "clinical_info",
            "technique",
            "comparison",
            "findings_by_organ",
            "impression",
            "recommendation",
            "bi_rads_or_scoring",
            "signature",
        ],
    },
    "emergency": {
        "name": "急診報告 Emergency Report",
        "sections": [
            "header",
            "patient_info",
            "critical_findings",
            "findings",
            "impression",
            "notification",
        ],
    },
}


class RadiologyReportGenerator:
    """
    Generates structured radiology reports integrating AI analysis results
    with clinical data and imaging metadata.
    """

    def __init__(self):
        self.report_counter = 0

    def generate_report(self, ai_analysis, metadata=None, physio_context=None,
                        clinical_history="", template_type="standard",
                        comparison_studies="", additional_notes="",
                        backend_name="MedVLM-R1", confidence=None):
        """
        Generate a complete radiology report.

        Args:
            ai_analysis: The AI-generated analysis text
            metadata: DICOM metadata dict (from extract_dicom_metadata)
            physio_context: Physiological reference parameters string
            clinical_history: Clinical history / reason for exam
            template_type: Report template type
            comparison_studies: Prior studies for comparison
            additional_notes: Additional clinical notes

        Returns:
            Formatted report string
        """
        self.report_counter += 1
        now = datetime.now()

        # Extract info from metadata
        patient = {}
        study = {}
        series = {}
        image = {}
        if metadata:
            patient = metadata.get("patient_info", {})
            study = metadata.get("study_info", {})
            series = metadata.get("series_info", {})
            image = metadata.get("image_info", {})

        report_lines = []

        # Header
        report_lines.extend(self._build_header(now, series, study))

        # Patient information
        report_lines.extend(self._build_patient_section(patient, study))

        # Clinical information
        report_lines.extend(self._build_clinical_section(
            clinical_history, series, study
        ))

        # Technique
        report_lines.extend(self._build_technique_section(series, image))

        # Comparison
        if template_type != "emergency":
            report_lines.extend(self._build_comparison_section(comparison_studies))

        # Reference parameters
        if physio_context:
            report_lines.extend(self._build_reference_section(physio_context))

        # Findings (AI analysis)
        if template_type == "emergency":
            report_lines.extend(self._build_critical_findings(ai_analysis))
        else:
            report_lines.extend(self._build_findings_section(ai_analysis))

        # Impression
        report_lines.extend(self._build_impression_section(ai_analysis))

        # Recommendations
        if template_type != "emergency":
            report_lines.extend(self._build_recommendation_section(ai_analysis))

        # Additional notes
        if additional_notes:
            report_lines.append("\n【附加說明 Additional Notes】")
            report_lines.append("-" * 50)
            report_lines.append(additional_notes)

        # Confidence
        if confidence is not None:
            report_lines.append(f"【AI Confidence: {confidence:.0%}】")
            report_lines.append("")

        # Signature
        report_lines.extend(self._build_signature(now, backend_name))

        # Disclaimer
        report_lines.extend(self._build_disclaimer())

        return "\n".join(report_lines)

    def _build_header(self, now, series, study):
        lines = []
        lines.append("╔" + "═" * 58 + "╗")
        lines.append("║" + "  放射科影像報告 RADIOLOGY REPORT".center(58) + "║")
        lines.append("╚" + "═" * 58 + "╝")
        lines.append("")

        modality_name = series.get("modality_name", series.get("modality", "N/A"))
        body_part = series.get("body_part_name", series.get("body_part", "N/A"))
        lines.append(f"  報告編號 Report #:  MedVLM-{now.strftime('%Y%m%d')}-{self.report_counter:04d}")
        lines.append(f"  報告日期 Date:      {now.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"  影像模態 Modality:  {modality_name}")
        lines.append(f"  檢查部位 Body Part: {body_part}")
        lines.append("")
        return lines

    def _build_patient_section(self, patient, study):
        lines = []
        lines.append("【病患資訊 Patient Information】")
        lines.append("-" * 50)
        lines.append(f"  姓名 Name:       {patient.get('patient_name', 'N/A')}")
        lines.append(f"  病歷號 ID:       {patient.get('patient_id', 'N/A')}")
        lines.append(f"  年齡 Age:        {patient.get('patient_age', 'N/A')}")
        lines.append(f"  性別 Sex:        {patient.get('patient_sex', 'N/A')}")
        lines.append(f"  醫囑醫師 Ref:    {study.get('referring_physician', 'N/A')}")
        lines.append(f"  檢查日期 Date:   {study.get('study_date', 'N/A')}")
        lines.append("")
        return lines

    def _build_clinical_section(self, clinical_history, series, study):
        lines = []
        lines.append("【臨床資訊 Clinical Information】")
        lines.append("-" * 50)

        history = clinical_history.strip() if clinical_history else "未提供 (Not provided)"
        lines.append(f"  臨床病史 History: {history}")

        exam_desc = study.get("study_description", "N/A")
        lines.append(f"  檢查描述 Exam:    {exam_desc}")
        lines.append("")
        return lines

    def _build_technique_section(self, series, image):
        lines = []
        lines.append("【檢查技術 Technique】")
        lines.append("-" * 50)

        modality = series.get("modality", "N/A")
        view = series.get("view_position", "N/A")
        desc = series.get("series_description", "N/A")
        size = f"{image.get('rows', 'N/A')} x {image.get('columns', 'N/A')}"
        spacing = image.get("pixel_spacing", "N/A")
        thickness = image.get("slice_thickness", "N/A")

        lines.append(f"  影像模態 Modality:   {modality}")
        lines.append(f"  投射位置 View:       {view}")
        lines.append(f"  序列描述 Series:     {desc}")
        lines.append(f"  影像大小 Size:       {size}")
        lines.append(f"  像素間距 Spacing:    {spacing}")
        if thickness != "N/A":
            lines.append(f"  切片厚度 Thickness:  {thickness}")
        lines.append("")
        return lines

    def _build_comparison_section(self, comparison_studies):
        lines = []
        lines.append("【比較研究 Comparison】")
        lines.append("-" * 50)
        comp = comparison_studies.strip() if comparison_studies else "無可比較之先前影像 (No prior studies available for comparison)"
        lines.append(f"  {comp}")
        lines.append("")
        return lines

    def _build_reference_section(self, physio_context):
        lines = []
        lines.append("【生理參考參數 Physiological Reference Parameters】")
        lines.append("-" * 50)
        for line in physio_context.split("\n"):
            lines.append(f"  {line}")
        lines.append("")
        return lines

    def _build_findings_section(self, ai_analysis):
        lines = []
        lines.append("【影像發現 Findings】")
        lines.append("-" * 50)

        # If AI analysis contains structured sections, preserve them
        if ai_analysis:
            for line in ai_analysis.split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("  AI分析結果尚未產生 (AI analysis pending)")

        lines.append("")
        return lines

    def _build_critical_findings(self, ai_analysis):
        lines = []
        lines.append("⚠️ 【緊急發現 CRITICAL FINDINGS】 ⚠️")
        lines.append("=" * 50)

        if ai_analysis:
            for line in ai_analysis.split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("  待分析 (Pending analysis)")

        lines.append("")
        return lines

    def _build_impression_section(self, ai_analysis):
        lines = []
        lines.append("【影像診斷 Impression】")
        lines.append("-" * 50)

        # Try to extract impression from AI analysis
        impression = self._extract_section(ai_analysis, [
            "impression", "診斷", "印象", "結論", "conclusion",
            "final", "最終", "answer", "答案"
        ])

        if impression:
            for line in impression.split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("  請參考上方影像發現之綜合分析")
            lines.append("  (Please refer to the findings section above)")

        lines.append("")
        return lines

    def _build_recommendation_section(self, ai_analysis):
        lines = []
        lines.append("【建議 Recommendations】")
        lines.append("-" * 50)

        recommendation = self._extract_section(ai_analysis, [
            "建議", "recommendation", "follow-up", "追蹤",
            "進一步", "further", "後續"
        ])

        if recommendation:
            for line in recommendation.split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("  1. 請結合臨床症狀進行綜合判斷")
            lines.append("  2. 如有疑問，建議專科醫師會診")
            lines.append("  3. 必要時安排追蹤檢查")

        lines.append("")
        return lines

    def _build_signature(self, now, backend_name="MedVLM-R1"):
        lines = []
        lines.append("【簽章 Signature】")
        lines.append("-" * 50)
        lines.append(f"  報告產生系統: AI 虛擬放射科醫師")
        lines.append(f"  報告時間:     {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  AI Model:     {backend_name}")
        lines.append("")
        return lines

    def _build_disclaimer(self):
        lines = []
        lines.append("┌" + "─" * 58 + "┐")
        lines.append("│" + "  ⚠️  免責聲明 DISCLAIMER".center(58) + "│")
        lines.append("│" + "".center(58) + "│")
        lines.append("│" + "  本報告由 AI 系統自動產生，僅供教育及研究參考。".center(46) + "│")
        lines.append("│" + "  不可作為正式臨床診斷依據。所有醫療決策".center(46) + "│")
        lines.append("│" + "  應由合格醫療專業人員做出。".center(46) + "│")
        lines.append("│" + "".center(58) + "│")
        lines.append("│" + "  This AI-generated report is for educational/research".center(58) + "│")
        lines.append("│" + "  purposes only. Not for clinical decision-making.".center(58) + "│")
        lines.append("└" + "─" * 58 + "┘")
        return lines

    def _extract_section(self, text, keywords):
        """Try to extract a specific section from AI analysis text."""
        if not text:
            return None

        text_lower = text.lower()
        for keyword in keywords:
            kw_lower = keyword.lower()
            idx = text_lower.find(kw_lower)
            if idx != -1:
                # Find the line with the keyword and take subsequent content
                remaining = text[idx:]
                # Skip the header line
                lines = remaining.split("\n")
                if len(lines) > 1:
                    content_lines = []
                    for line in lines[1:]:
                        stripped = line.strip()
                        # Stop at next section header
                        if stripped and (
                            stripped.startswith("【") or
                            stripped.startswith("##") or
                            stripped.startswith("===")
                        ):
                            break
                        if stripped:
                            content_lines.append(stripped)
                    if content_lines:
                        return "\n".join(content_lines[:10])
        return None


def generate_ai_radiology_prompt(metadata=None, clinical_history="",
                                  physio_context="", language="zh-tw"):
    """
    Build an enhanced prompt for the AI virtual radiologist that includes
    clinical context and reference parameters.

    Args:
        metadata: DICOM metadata dict
        clinical_history: Patient clinical history
        physio_context: Physiological reference parameters
        language: Language for the report (zh-tw or en)

    Returns:
        Enhanced prompt string for the AI model
    """
    parts = []

    parts.append(
        "你是一位經驗豐富的虛擬 AI 放射科醫師，具備深厚的影像判讀專業知識。"
        "請根據提供的醫學影像進行系統性、全面性的分析，並產生結構化的放射報告。"
    )

    if metadata:
        series = metadata.get("series_info", {})
        patient = metadata.get("patient_info", {})

        parts.append(f"\n病患資訊: 年齡 {patient.get('patient_age', 'N/A')}, "
                     f"性別 {patient.get('patient_sex', 'N/A')}")
        parts.append(f"影像模態: {series.get('modality_name', series.get('modality', 'N/A'))}")
        parts.append(f"檢查部位: {series.get('body_part_name', series.get('body_part', 'N/A'))}")
        parts.append(f"投射位置: {series.get('view_position', 'N/A')}")

    if clinical_history:
        parts.append(f"\n臨床病史: {clinical_history}")

    if physio_context:
        parts.append(f"\n相關生理參考參數:\n{physio_context}")

    parts.append("\n請依照以下結構進行報告:")
    parts.append("1. 影像品質評估 (Image Quality Assessment)")
    parts.append("2. 系統性影像發現 (Systematic Findings)")
    parts.append("   - 依解剖結構逐一描述所見")
    parts.append("   - 對比正常參考值指出異常")
    parts.append("   - 描述異常的位置、大小、形態、密度特徵")
    parts.append("3. 影像診斷 / 印象 (Impression)")
    parts.append("   - 列出主要診斷（依可能性排序）")
    parts.append("   - 列出鑑別診斷")
    parts.append("4. 建議 (Recommendations)")
    parts.append("   - 建議追蹤或進一步檢查")
    parts.append("   - 臨床相關建議")
    parts.append("\n請保持專業、客觀、謹慎的態度。如有不確定之處，請明確說明。")

    return "\n".join(parts)
