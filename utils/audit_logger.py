"""
Audit trail logger for medical AI system.
Records who analyzed what image with which model and when.
Outputs to a local JSON-lines log file for compliance tracking.
"""

import json
import os
import time
from datetime import datetime


class AuditLogger:
    """Append-only JSON-lines audit log."""

    def __init__(self, log_dir="./audit_logs"):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(
            log_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

    def log_analysis(self, backend, model_name, image_source, prompt_summary,
                     result_summary, confidence=None, dicom_metadata=None):
        """
        Record an analysis event.

        Args:
            backend: 'ollama' or 'medvlm-r1'
            model_name: specific model identifier
            image_source: description of image origin (upload, PACS, etc.)
            prompt_summary: first 200 chars of the prompt
            result_summary: first 300 chars of the result
            confidence: optional confidence score
            dicom_metadata: optional dict of non-PHI metadata (modality, body part)
        """
        safe_metadata = {}
        if dicom_metadata:
            # Only log non-PHI fields
            series = dicom_metadata.get("series_info", {})
            safe_metadata = {
                "modality": series.get("modality", ""),
                "body_part": series.get("body_part", ""),
            }

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch": time.time(),
            "backend": backend,
            "model_name": model_name,
            "image_source": image_source,
            "prompt_summary": prompt_summary[:200],
            "result_summary": result_summary[:300],
            "confidence": confidence,
            "imaging_metadata": safe_metadata,
        }

        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Audit logging should never break the main workflow

    def get_recent_logs(self, n=20):
        """Read the last N log entries."""
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            entries = []
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
            return entries
        except FileNotFoundError:
            return []

    def format_log_display(self, n=20):
        """Format recent audit logs for display."""
        entries = self.get_recent_logs(n)
        if not entries:
            return "No audit log entries."

        lines = ["Audit Trail", "=" * 60]
        for entry in reversed(entries):
            ts = entry.get("timestamp", "?")
            lines.append(
                f"\n[{ts}] {entry.get('backend', '?')} / {entry.get('model_name', '?')}"
            )
            meta = entry.get("imaging_metadata", {})
            if meta:
                lines.append(f"  Modality: {meta.get('modality', 'N/A')} | Body Part: {meta.get('body_part', 'N/A')}")
            lines.append(f"  Prompt: {entry.get('prompt_summary', '')[:80]}...")
            conf = entry.get("confidence")
            if conf is not None:
                lines.append(f"  Confidence: {conf:.0%}")
        return "\n".join(lines)
