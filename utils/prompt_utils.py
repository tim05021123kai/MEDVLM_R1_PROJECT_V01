"""
Prompt engineering module with backend-aware templates,
structured output directives, and few-shot medical examples.
"""


# ---------------------------------------------------------------------------
# Few-shot examples by modality
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES = {
    "chest_xray": {
        "label": "Chest X-ray Example",
        "example": (
            "## FINDINGS\n"
            "Image Quality: PA upright view, adequate inspiration with 10 posterior ribs visible, "
            "no rotation noted.\n"
            "Heart: Cardiothoracic ratio approximately 0.48, within normal limits. "
            "No pericardial effusion.\n"
            "Mediastinum: Trachea is midline. Mediastinal width is normal. "
            "Aortic knob is within normal limits.\n"
            "Lungs: Bilateral lung fields are clear. No focal consolidation, "
            "pleural effusion, or pneumothorax. Costophrenic angles are sharp bilaterally.\n"
            "Bones: No acute fracture or lytic lesion.\n"
            "Soft tissues: Unremarkable.\n\n"
            "## IMPRESSION\n"
            "Normal chest radiograph. No acute cardiopulmonary process.\n\n"
            "## RECOMMENDATIONS\n"
            "No further imaging recommended at this time."
        ),
    },
    "brain_ct": {
        "label": "Brain CT Example",
        "example": (
            "## FINDINGS\n"
            "Image Quality: Non-contrast CT of the head, adequate study quality.\n"
            "Brain Parenchyma: No acute intracranial hemorrhage. "
            "Gray-white matter differentiation is preserved. "
            "No focal lesion or mass effect.\n"
            "Ventricles: Normal in size and configuration. Evans ratio < 0.3. "
            "No hydrocephalus.\n"
            "Midline: No midline shift.\n"
            "Extra-axial spaces: No epidural, subdural, or subarachnoid hemorrhage.\n"
            "Calvarium: No fracture.\n\n"
            "## IMPRESSION\n"
            "Normal non-contrast head CT. No acute intracranial pathology.\n\n"
            "## RECOMMENDATIONS\n"
            "Clinical correlation recommended. MRI if further evaluation needed."
        ),
    },
    "abdomen_ct": {
        "label": "Abdomen CT Example",
        "example": (
            "## FINDINGS\n"
            "Liver: Normal size (MCL span ~14 cm) and homogeneous attenuation. "
            "No focal hepatic lesion.\n"
            "Gallbladder: Normal wall thickness (<3 mm). No gallstones.\n"
            "Bile ducts: Common bile duct is not dilated (<6 mm).\n"
            "Spleen: Normal size (<12 cm). Homogeneous.\n"
            "Pancreas: Unremarkable.\n"
            "Kidneys: Bilateral normal size. No hydronephrosis or calculi.\n"
            "Aorta: Normal caliber (<3.0 cm). No aneurysm.\n"
            "Bowel: No obstruction or free air.\n"
            "Lymph nodes: No pathological lymphadenopathy.\n\n"
            "## IMPRESSION\n"
            "Normal CT abdomen. No acute intra-abdominal pathology.\n\n"
            "## RECOMMENDATIONS\n"
            "No further imaging recommended."
        ),
    },
}


# ---------------------------------------------------------------------------
# Structured output directive
# ---------------------------------------------------------------------------

STRUCTURED_OUTPUT_DIRECTIVE = """
Please structure your response using EXACTLY these section headers:

## FINDINGS
[Describe systematic observations organized by anatomical structure]

## IMPRESSION
[List diagnoses in order of likelihood, include differentials]

## RECOMMENDATIONS
[Follow-up studies, clinical correlation, or further actions]
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompt(question: str, template_type: str = "medical"):
    """Build prompt for medical image analysis."""
    if template_type == "medical":
        return (
            f"You are a professional medical image analysis AI assistant. "
            f"Carefully observe the provided medical image and provide a detailed, "
            f"accurate analysis based on the following question.\n\n"
            f"Question: {question}\n"
            f"{STRUCTURED_OUTPUT_DIRECTIVE}"
        )
    elif template_type == "simple":
        return f"Please analyze this medical image and answer: {question}"
    else:
        return question


def build_cot_prompt(question: str, backend: str = "medvlm-r1"):
    """
    Build chain-of-thought prompt, adapted for the active backend.
    MedVLM-R1 uses <think>/<answer> tags; Ollama models use natural CoT.
    """
    if backend == "medvlm-r1":
        return (
            f"{question}\n\n"
            f"Please analyze step by step:\n"
            f"<think>\n"
            f"Step 1: Describe the basic image information (type, view, technical quality)\n"
            f"Step 2: Systematically observe each anatomical structure\n"
            f"Step 3: Identify and describe any abnormal findings\n"
            f"Step 4: Analyze the characteristics and distribution of abnormalities\n"
            f"Step 5: Consider possible diagnoses and differentials\n"
            f"Step 6: Assess clinical significance and recommendations\n"
            f"</think>\n\n"
            f"Final answer: <answer>Provide your final diagnostic opinion and recommendations</answer>"
        )
    else:
        # For Ollama models (qwen3-vl, etc.) use natural language CoT
        return (
            f"{question}\n\n"
            f"Let me think through this step by step:\n\n"
            f"Step 1: Image type and quality assessment\n"
            f"Step 2: Systematic anatomical survey\n"
            f"Step 3: Abnormality identification\n"
            f"Step 4: Characterization of findings (location, size, morphology, density)\n"
            f"Step 5: Differential diagnosis (most likely to least likely)\n"
            f"Step 6: Clinical significance and recommendations\n\n"
            f"After analysis, summarize with:\n"
            f"{STRUCTURED_OUTPUT_DIRECTIVE}"
        )


def build_comparison_prompt(clinical_context: str = ""):
    """Build prompt for multi-image comparison (prior vs current)."""
    ctx = f"\nClinical context: {clinical_context}" if clinical_context else ""
    return (
        f"You are an experienced radiologist performing a comparison study. "
        f"Image 1 is the PRIOR study and Image 2 is the CURRENT study.{ctx}\n\n"
        f"Please analyze both images and provide:\n"
        f"{STRUCTURED_OUTPUT_DIRECTIVE}\n"
        f"In FINDINGS, specifically note:\n"
        f"- What has changed between the two studies\n"
        f"- New findings in the current study\n"
        f"- Resolved findings from the prior study\n"
        f"- Stable/unchanged findings"
    )


def get_fewshot_for_body_part(body_part_code: str) -> str:
    """Return a few-shot example string for the given body part."""
    code = (body_part_code or "").upper()
    mapping = {
        "CHEST": "chest_xray",
        "THORAX": "chest_xray",
        "LUNG": "chest_xray",
        "HEAD": "brain_ct",
        "BRAIN": "brain_ct",
        "SKULL": "brain_ct",
        "ABDOMEN": "abdomen_ct",
        "LIVER": "abdomen_ct",
        "KIDNEY": "abdomen_ct",
        "PELVIS": "abdomen_ct",
    }
    key = mapping.get(code)
    if key and key in FEWSHOT_EXAMPLES:
        ex = FEWSHOT_EXAMPLES[key]
        return f"\n--- Example Report ({ex['label']}) ---\n{ex['example']}\n--- End Example ---\n"
    return ""
