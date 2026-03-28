"""
Ollama model integration module.
Connects to a local Ollama server for vision-language model inference.
Supports qwen3-vl and other Ollama-hosted VLM models.
Provides both blocking and streaming inference.
"""

import base64
import io
import json
import urllib.request
import urllib.error
from PIL import Image


DEFAULT_OLLAMA_URL = "http://localhost:11434"


def check_ollama_connection(base_url=DEFAULT_OLLAMA_URL):
    """Check if Ollama server is running and accessible."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, "Ollama connected"
            return False, f"Ollama HTTP {resp.status}"
    except urllib.error.URLError as e:
        return False, f"Cannot connect to Ollama: {e.reason}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def list_ollama_models(base_url=DEFAULT_OLLAMA_URL):
    """List all models available in the local Ollama instance."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pil_image_to_base64(image):
    """Convert a PIL Image to base64 string for Ollama API."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_with_ollama(image, prompt, model_name="qwen3-vl",
                         base_url=DEFAULT_OLLAMA_URL, temperature=0.7,
                         max_tokens=2048):
    """
    Generate a response from an Ollama vision-language model (blocking).

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
            return result.get("message", {}).get("content", "")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API error (HTTP {e.code}): {error_body}")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama ({base_url}): {e.reason}\n"
            f"Please ensure Ollama is running: ollama serve"
        )


def generate_with_ollama_stream(image, prompt, model_name="qwen3-vl",
                                base_url=DEFAULT_OLLAMA_URL, temperature=0.7,
                                max_tokens=2048):
    """
    Generate a response from an Ollama VLM with streaming (yields chunks).

    Yields:
        str: Incremental text chunks
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
        "stream": True,
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
        resp = urllib.request.urlopen(req, timeout=300)
        buffer = b""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        resp.close()
                        return
                except json.JSONDecodeError:
                    continue
        resp.close()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API error (HTTP {e.code}): {error_body}")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama ({base_url}): {e.reason}\n"
            f"Please ensure Ollama is running: ollama serve"
        )


def generate_with_ollama_multi_image(images, prompt, model_name="qwen3-vl",
                                     base_url=DEFAULT_OLLAMA_URL,
                                     temperature=0.7, max_tokens=2048):
    """
    Generate a response with multiple images (for comparison studies).

    Args:
        images: list of PIL Images
        prompt: text prompt
    Returns:
        str: Model response text
    """
    images_b64 = [pil_image_to_base64(img) for img in images]

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
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
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content", "")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API error (HTTP {e.code}): {error_body}")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama ({base_url}): {e.reason}"
        )
