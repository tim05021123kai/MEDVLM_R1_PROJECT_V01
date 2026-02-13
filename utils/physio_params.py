"""
Physiological parameters reference module.
Provides normal reference ranges for vital signs, lab values,
and imaging-specific parameters organized by body region and modality.
"""


# Normal vital sign ranges
VITAL_SIGNS = {
    "heart_rate": {
        "name": "心率 Heart Rate",
        "unit": "bpm",
        "adult_normal": "60-100",
        "child_normal": "70-120",
        "infant_normal": "100-160",
    },
    "blood_pressure_systolic": {
        "name": "收縮壓 Systolic BP",
        "unit": "mmHg",
        "adult_normal": "90-120",
        "hypertension_stage1": "130-139",
        "hypertension_stage2": "≥140",
    },
    "blood_pressure_diastolic": {
        "name": "舒張壓 Diastolic BP",
        "unit": "mmHg",
        "adult_normal": "60-80",
        "hypertension_stage1": "80-89",
        "hypertension_stage2": "≥90",
    },
    "respiratory_rate": {
        "name": "呼吸速率 Respiratory Rate",
        "unit": "breaths/min",
        "adult_normal": "12-20",
        "child_normal": "20-30",
    },
    "temperature": {
        "name": "體溫 Body Temperature",
        "unit": "°C",
        "normal": "36.1-37.2",
        "fever": "≥37.5",
    },
    "spo2": {
        "name": "血氧飽和度 SpO2",
        "unit": "%",
        "normal": "95-100",
        "hypoxia": "<90",
    },
}


# Imaging-specific reference parameters by body region
IMAGING_REFERENCE = {
    "CHEST": {
        "name": "胸部 Chest",
        "parameters": {
            "cardiothoracic_ratio": {
                "name": "心胸比 Cardiothoracic Ratio (CTR)",
                "normal": "< 0.50",
                "borderline": "0.50 - 0.55",
                "abnormal": "> 0.55",
                "note": "心臟擴大的重要指標; CTR > 0.5 通常提示心臟擴大",
            },
            "tracheal_width": {
                "name": "氣管寬度 Tracheal Width",
                "unit": "mm",
                "normal": "15-25",
                "note": "測量於主動脈弓上方",
            },
            "aortic_knob_width": {
                "name": "主動脈弓寬度 Aortic Knob Width",
                "unit": "mm",
                "normal": "< 35",
                "abnormal": "≥ 40 提示主動脈擴張",
            },
            "costophrenic_angle": {
                "name": "肋膈角 Costophrenic Angle",
                "normal": "銳角 (Sharp)",
                "abnormal": "鈍角 (Blunted) - 提示肋膜積液",
            },
            "lung_fields": {
                "name": "肺野 Lung Fields",
                "normal": "透亮度正常, 無浸潤或結節",
                "note": "注意對比兩側肺野透亮度",
            },
            "mediastinum_width": {
                "name": "縱膈腔寬度 Mediastinal Width",
                "unit": "cm",
                "normal": "< 8 cm (PA view)",
                "abnormal": "> 8 cm 提示縱膈腔擴大",
            },
            "diaphragm_position": {
                "name": "橫膈位置 Diaphragm Position",
                "normal": "右側略高於左側 (1-2 cm)",
                "note": "右側橫膈通常位於第10後肋水平",
            },
        },
    },
    "HEAD": {
        "name": "頭部 Head / Brain",
        "parameters": {
            "midline_shift": {
                "name": "中線偏移 Midline Shift",
                "unit": "mm",
                "normal": "< 5",
                "significant": "≥ 5 mm 需考慮手術介入",
            },
            "ventricular_size": {
                "name": "腦室大小 Ventricular Size",
                "normal": "Evans ratio < 0.3",
                "hydrocephalus": "Evans ratio ≥ 0.3",
                "note": "Evans ratio = 最大前角寬度 / 同層面最大雙頂徑",
            },
            "gray_white_differentiation": {
                "name": "灰白質分界 Gray-White Matter Differentiation",
                "normal": "清晰可辨",
                "abnormal": "模糊或消失提示水腫或缺血",
            },
            "sulci_and_cisterns": {
                "name": "腦溝與腦池 Sulci and Cisterns",
                "normal": "正常可見",
                "abnormal": "消失提示腦水腫, 擴大提示腦萎縮",
            },
            "hounsfield_units": {
                "name": "CT值 Hounsfield Units",
                "values": {
                    "急性出血 Acute Blood": "50-70 HU",
                    "灰質 Gray Matter": "37-45 HU",
                    "白質 White Matter": "20-30 HU",
                    "腦脊液 CSF": "0-15 HU",
                    "骨骼 Bone": "700-3000 HU",
                    "空氣 Air": "-1000 HU",
                    "脂肪 Fat": "-50 to -100 HU",
                },
            },
        },
    },
    "ABDOMEN": {
        "name": "腹部 Abdomen",
        "parameters": {
            "liver_size": {
                "name": "肝臟大小 Liver Size",
                "unit": "cm",
                "normal": "MCL span < 15.5 cm",
                "hepatomegaly": "MCL span > 15.5 cm",
            },
            "liver_density": {
                "name": "肝臟密度 Liver Density (CT)",
                "unit": "HU",
                "normal": "肝臟密度 > 脾臟密度 (約 8-10 HU 差)",
                "fatty_liver": "肝臟密度 < 脾臟密度",
            },
            "spleen_size": {
                "name": "脾臟大小 Spleen Size",
                "unit": "cm",
                "normal": "長度 < 12 cm",
                "splenomegaly": "長度 ≥ 12 cm",
            },
            "kidney_size": {
                "name": "腎臟大小 Kidney Size",
                "unit": "cm",
                "adult_normal": "9-13 cm (長度)",
                "note": "右腎通常略小於左腎",
            },
            "aorta_diameter": {
                "name": "腹主動脈直徑 Abdominal Aorta Diameter",
                "unit": "cm",
                "normal": "< 3.0 cm",
                "aneurysm": "≥ 3.0 cm",
                "surgical": "≥ 5.5 cm 考慮手術",
            },
            "gallbladder_wall": {
                "name": "膽囊壁厚度 Gallbladder Wall Thickness",
                "unit": "mm",
                "normal": "< 3 mm",
                "abnormal": "≥ 3 mm 提示膽囊炎",
            },
            "cbd_diameter": {
                "name": "總膽管直徑 Common Bile Duct Diameter",
                "unit": "mm",
                "normal": "< 6 mm (< 8 mm if post-cholecystectomy)",
                "dilated": "≥ 6 mm 提示膽道阻塞",
            },
        },
    },
    "SPINE": {
        "name": "脊椎 Spine",
        "parameters": {
            "vertebral_body_height": {
                "name": "椎體高度 Vertebral Body Height",
                "normal": "前後高度比 ≥ 0.8",
                "compression_fracture": "前後高度比 < 0.8 或高度減少 ≥ 20%",
            },
            "disc_height": {
                "name": "椎間盤高度 Disc Height",
                "normal": "漸進性增加至 L4-L5",
                "abnormal": "局部減低提示退化性疾病",
            },
            "spinal_canal_diameter": {
                "name": "椎管直徑 Spinal Canal Diameter",
                "unit": "mm",
                "cervical_normal": "> 13 mm (頸椎)",
                "lumbar_normal": "> 15 mm (腰椎)",
                "stenosis": "頸椎 < 10 mm; 腰椎 < 12 mm",
            },
            "alignment": {
                "name": "脊椎排列 Spinal Alignment",
                "normal": "正常前凸/後凸曲線",
                "abnormal": "側彎、後凸加重或滑脫",
            },
        },
    },
    "MUSCULOSKELETAL": {
        "name": "骨骼肌肉系統 Musculoskeletal",
        "parameters": {
            "bone_density": {
                "name": "骨密度 Bone Density",
                "normal": "T-score ≥ -1.0",
                "osteopenia": "T-score -1.0 to -2.5",
                "osteoporosis": "T-score ≤ -2.5",
            },
            "joint_space": {
                "name": "關節間隙 Joint Space",
                "normal": "對稱且均勻",
                "abnormal": "狹窄提示退化性關節炎或發炎",
            },
            "soft_tissue": {
                "name": "軟組織 Soft Tissue",
                "normal": "無腫脹、鈣化或異常密度",
            },
        },
    },
}


# Lab value reference ranges relevant to radiology
LAB_REFERENCES = {
    "renal_function": {
        "name": "腎功能 Renal Function (contrast safety)",
        "parameters": {
            "creatinine": {
                "name": "肌酐 Creatinine",
                "unit": "mg/dL",
                "male_normal": "0.7-1.3",
                "female_normal": "0.6-1.1",
                "contrast_caution": "> 1.5 (顯影劑使用需謹慎)",
            },
            "gfr": {
                "name": "腎絲球過濾率 eGFR",
                "unit": "mL/min/1.73m²",
                "normal": "> 90",
                "mild_decrease": "60-89",
                "moderate_decrease": "30-59 (顯影劑使用需評估)",
                "severe": "< 30 (禁用含碘顯影劑)",
            },
            "bun": {
                "name": "血尿素氮 BUN",
                "unit": "mg/dL",
                "normal": "7-20",
            },
        },
    },
    "coagulation": {
        "name": "凝血功能 Coagulation (interventional procedures)",
        "parameters": {
            "pt_inr": {
                "name": "凝血酶原時間 PT/INR",
                "normal_inr": "0.8-1.2",
                "procedure_safe": "INR < 1.5 (介入手術安全)",
            },
            "platelet": {
                "name": "血小板 Platelet Count",
                "unit": "×10³/μL",
                "normal": "150-400",
                "procedure_caution": "< 50 (介入手術需謹慎)",
            },
        },
    },
    "thyroid": {
        "name": "甲狀腺功能 Thyroid (iodinated contrast)",
        "parameters": {
            "tsh": {
                "name": "促甲狀腺激素 TSH",
                "unit": "mIU/L",
                "normal": "0.4-4.0",
                "note": "甲狀腺功能異常者使用含碘顯影劑需特別注意",
            },
        },
    },
}


def get_reference_for_body_part(body_part_code):
    """
    Get imaging reference parameters for a given body part.

    Args:
        body_part_code: DICOM body part code (e.g., 'CHEST', 'HEAD')

    Returns:
        dict with reference parameters, or None if not found
    """
    code = body_part_code.upper() if body_part_code else ""

    # Map common DICOM body part codes to our reference keys
    mapping = {
        "CHEST": "CHEST",
        "THORAX": "CHEST",
        "LUNG": "CHEST",
        "HEAD": "HEAD",
        "BRAIN": "HEAD",
        "SKULL": "HEAD",
        "ABDOMEN": "ABDOMEN",
        "LIVER": "ABDOMEN",
        "KIDNEY": "ABDOMEN",
        "PELVIS": "ABDOMEN",
        "SPINE": "SPINE",
        "CSPINE": "SPINE",
        "TSPINE": "SPINE",
        "LSPINE": "SPINE",
        "KNEE": "MUSCULOSKELETAL",
        "HIP": "MUSCULOSKELETAL",
        "SHOULDER": "MUSCULOSKELETAL",
        "HAND": "MUSCULOSKELETAL",
        "FOOT": "MUSCULOSKELETAL",
        "EXTREMITY": "MUSCULOSKELETAL",
        "WRIST": "MUSCULOSKELETAL",
        "ANKLE": "MUSCULOSKELETAL",
        "ELBOW": "MUSCULOSKELETAL",
    }

    ref_key = mapping.get(code)
    if ref_key and ref_key in IMAGING_REFERENCE:
        return IMAGING_REFERENCE[ref_key]
    return None


def format_reference_params(body_part_code):
    """
    Format reference parameters for display in the GUI.

    Args:
        body_part_code: DICOM body part code

    Returns:
        Formatted string with reference parameters
    """
    ref = get_reference_for_body_part(body_part_code)
    if not ref:
        return f"暫無 '{body_part_code}' 的專用參考參數資料"

    lines = []
    lines.append(f"【{ref['name']} 影像參考參數】")
    lines.append("=" * 50)

    for key, param in ref["parameters"].items():
        lines.append(f"\n▸ {param['name']}")
        for k, v in param.items():
            if k == "name":
                continue
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    lines.append(f"    {sub_k}: {sub_v}")
            else:
                label = k.replace("_", " ").title()
                lines.append(f"    {label}: {v}")

    return "\n".join(lines)


def format_lab_references():
    """Format all lab references for display."""
    lines = []
    lines.append("【實驗室檢查參考值 Laboratory References】")
    lines.append("=" * 50)

    for category_key, category in LAB_REFERENCES.items():
        lines.append(f"\n◆ {category['name']}")
        lines.append("-" * 40)

        for param_key, param in category["parameters"].items():
            lines.append(f"\n  ▸ {param['name']}")
            for k, v in param.items():
                if k == "name":
                    continue
                label = k.replace("_", " ").title()
                lines.append(f"      {label}: {v}")

    return "\n".join(lines)


def get_context_for_ai_prompt(body_part_code, modality_code=""):
    """
    Generate a physiological context string to include in the AI prompt.
    This helps the AI provide more accurate analysis.

    Args:
        body_part_code: DICOM body part code
        modality_code: DICOM modality code

    Returns:
        String with relevant physiological context for the AI
    """
    context_parts = []

    ref = get_reference_for_body_part(body_part_code)
    if ref:
        context_parts.append(f"檢查部位: {ref['name']}")
        context_parts.append("相關參考參數:")
        for key, param in ref["parameters"].items():
            normal = param.get("normal", param.get("adult_normal", ""))
            if normal:
                context_parts.append(f"  - {param['name']}: 正常值 {normal}")

    if modality_code in ("CT",):
        context_parts.append("\n顯影劑相關實驗室檢查參考:")
        renal = LAB_REFERENCES.get("renal_function", {})
        for pk, pv in renal.get("parameters", {}).items():
            normal = pv.get("normal", pv.get("male_normal", ""))
            context_parts.append(f"  - {pv['name']}: {normal}")

    return "\n".join(context_parts)
