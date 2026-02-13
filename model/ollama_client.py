"""
Ollama model integration module.
Connects to a local Ollama server for vision-language model inference.
Supports qwen3-vl and other Ollama-hosted VLM models.
"""

import base64
import io
import json
import urllib.request
import urllib.error
from PIL import Image


DEFAULT_OLLAMA_URL = "http://localhost:11434"


def check_ollama_connection(base_url=DEFAULT_OLLAMA_URL):
    """
    Check if Ollama server is running and accessible.

    Returns:
        tuple: (is_connected: bool, message: str)
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, "Ollama 伺服器連線成功 (Connected)"
            return False, f"Ollama 回應異常: HTTP {resp.status}"
    except urllib.error.URLError as e:
        return False, f"無法連線 Ollama: {e.reason}"
    except Exception as e:
        return False, f"連線錯誤: {str(e)}"


def list_ollama_models(base_url=DEFAULT_OLLAMA_URL):
    """
    List all models available in the local Ollama instance.

    Returns:
        list of model name strings
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception:
        return []


def pil_image_to_base64(image):
    """Convert a PIL Image to base64 string for Ollama API."""
    buffer = io.BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_with_ollama(image, prompt, model_name="qwen3-vl",
                         base_url=DEFAULT_OLLAMA_URL, temperature=0.7,
                         max_tokens=2048):
    """
    Generate a response from an Ollama vision-language model.

    Args:
        image: PIL Image object
        prompt: Text prompt for the model
        model_name: Ollama model name (e.g. 'qwen3-vl', 'qwen3-vl:30b')
        base_url: Ollama server URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        str: Model response text
    """
    image_b64 = pil_image_to_base64(image)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            message = result.get("message", {})
            return message.get("content", "")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama API 錯誤 (HTTP {e.code}): {error_body}"
        )
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"無法連線 Ollama 伺服器 ({base_url}): {e.reason}\n"
            f"請確認 Ollama 正在運行: ollama serve"
        )
