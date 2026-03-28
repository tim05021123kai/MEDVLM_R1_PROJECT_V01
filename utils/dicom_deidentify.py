"""
DICOM de-identification module.
Implements HIPAA Safe Harbor de-identification by removing/replacing
Protected Health Information (PHI) tags from DICOM datasets.
"""

# DICOM tags that contain PHI per HIPAA Safe Harbor
PHI_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "ReferringPhysicianName",
    "ReferringPhysicianTelephoneNumbers",
    "ReferringPhysicianAddress",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "PerformingPhysicianName",
    "OperatorsName",
    "NameOfPhysiciansReadingStudy",
    "RequestingPhysician",
    "ScheduledPerformingPhysicianName",
    "AccessionNumber",
    "StudyID",
    "RequestAttributesSequence",
]

# Tags to replace with anonymized values
REPLACE_TAGS = {
    "PatientName": "ANONYMOUS",
    "PatientID": "ANON000000",
    "ReferringPhysicianName": "ANONYMOUS",
    "InstitutionName": "ANONYMOUS_INSTITUTION",
}


def deidentify_dicom(ds, keep_age=True, keep_sex=True):
    """
    Remove PHI from a pydicom Dataset (in-place modification).

    Args:
        ds: pydicom.Dataset object
        keep_age: whether to keep PatientAge (useful for clinical context)
        keep_sex: whether to keep PatientSex (useful for clinical context)

    Returns:
        Modified pydicom.Dataset with PHI removed
    """
    for tag_name in PHI_TAGS:
        if hasattr(ds, tag_name):
            if tag_name in REPLACE_TAGS:
                setattr(ds, tag_name, REPLACE_TAGS[tag_name])
            else:
                delattr(ds, tag_name)

    # Optionally remove age and sex
    if not keep_age and hasattr(ds, "PatientAge"):
        delattr(ds, "PatientAge")
    if not keep_sex and hasattr(ds, "PatientSex"):
        delattr(ds, "PatientSex")

    # Remove dates that could be identifying
    for date_tag in ["StudyDate", "SeriesDate", "AcquisitionDate",
                     "ContentDate", "InstanceCreationDate"]:
        if hasattr(ds, date_tag):
            # Keep year only for clinical relevance
            val = str(getattr(ds, date_tag, ""))
            if len(val) >= 4:
                setattr(ds, date_tag, val[:4] + "0101")

    return ds


def deidentify_metadata(metadata):
    """
    Remove PHI from an extracted metadata dict (from extract_dicom_metadata).

    Args:
        metadata: dict from dicom_handler.extract_dicom_metadata()

    Returns:
        Sanitized metadata dict
    """
    sanitized = {}
    for section_key, section in metadata.items():
        if isinstance(section, dict):
            sanitized[section_key] = {}
            for k, v in section.items():
                if k in ("patient_name", "patient_id", "referring_physician",
                         "institution", "accession_number"):
                    sanitized[section_key][k] = "ANONYMOUS"
                else:
                    sanitized[section_key][k] = v
        else:
            sanitized[section_key] = section
    return sanitized


def get_phi_summary(ds):
    """List which PHI fields are present in a DICOM dataset."""
    present = []
    for tag_name in PHI_TAGS:
        if hasattr(ds, tag_name):
            val = str(getattr(ds, tag_name, ""))
            if val and val.strip() and val != "N/A":
                present.append(f"  {tag_name}: {val[:30]}...")
    if present:
        return f"PHI fields detected ({len(present)}):\n" + "\n".join(present)
    return "No PHI fields detected."
