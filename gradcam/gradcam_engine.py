"""
Interpretability module: Grad-CAM, Attention Rollout, and Perturbation Saliency.
Provides visual explanations for AI model predictions on medical images.
"""

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ===================================================================
# Grad-CAM
# ===================================================================

class GradCAM:
    """Grad-CAM implementation for vision-language models."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.activations = {}
        self.gradients = {}
        self._hooks = []

    def _find_target_layer(self):
        """Find the best target layer for Grad-CAM."""
        target_layer = None
        target_name = None

        for name, module in self.model.named_modules():
            if any(key in name.lower() for key in [
                'visual', 'vision', 'vit', 'encoder', 'patch_embed'
            ]):
                if hasattr(module, 'weight') or any(
                    hasattr(module, attr) for attr in ['out_proj', 'dense', 'fc']
                ):
                    target_layer = module
                    target_name = name

            if isinstance(module, (torch.nn.Conv2d, torch.nn.LayerNorm)):
                if 'visual' in name.lower() or 'vision' in name.lower():
                    target_layer = module
                    target_name = name

        if target_layer is None:
            for name, module in self.model.named_modules():
                if 'visual' in name.lower() or 'vision' in name.lower():
                    if hasattr(module, 'weight'):
                        target_layer = module
                        target_name = name

        return target_layer, target_name

    def register_hooks(self, target_layer=None):
        """Register forward and backward hooks on the target layer."""
        self.remove_hooks()

        if target_layer is None:
            target_layer, layer_name = self._find_target_layer()
            if target_layer is None:
                return False, "Could not find target layer"

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations['value'] = output[0].detach()
            else:
                self.activations['value'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients['value'] = grad_output[0].detach()
            else:
                self.gradients['value'] = grad_output.detach()

        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([fh, bh])

        return True, f"Hooks registered on: {type(target_layer).__name__}"

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.activations = {}
        self.gradients = {}

    def generate_heatmap(self, image, prompt, image_size=None):
        """Generate a Grad-CAM heatmap."""
        if not HAS_TORCH:
            return None, None, None

        if image_size is None:
            image_size = image.size

        try:
            success, msg = self.register_hooks()
            if not success:
                return None, _generate_fallback_attention(image), None

            from qwen_vl_utils import process_vision_info

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=text, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

            self.model.eval()
            was_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
                    inputs[key].requires_grad_(True)

            outputs = self.model(**inputs)
            logits = outputs.logits
            target = logits[:, -1, :].max()
            target.backward(retain_graph=False)

            torch.set_grad_enabled(was_grad_enabled)

            if 'value' in self.activations and 'value' in self.gradients:
                activations = self.activations['value']
                gradients = self.gradients['value']
                cam = self._compute_cam(activations, gradients, image_size)
                heatmap = _normalize_cam(cam)
                overlay = _create_overlay(image, heatmap)
                self.remove_hooks()
                return heatmap, overlay, cam
            else:
                self.remove_hooks()
                return None, _generate_fallback_attention(image), None

        except Exception as e:
            self.remove_hooks()
            print(f"Grad-CAM error: {e}")
            return None, _generate_fallback_attention(image), None

    def _compute_cam(self, activations, gradients, image_size):
        if len(gradients.shape) == 3:
            weights = gradients.mean(dim=1, keepdim=True)
            cam = (weights * activations).sum(dim=-1)
            seq_len = cam.shape[1]
            h = w = int(np.sqrt(seq_len))
            if h * w != seq_len:
                h = w = int(np.ceil(np.sqrt(seq_len)))
                cam = F.pad(cam, (0, h * w - seq_len))
            cam = cam.view(1, h, w)
        elif len(gradients.shape) == 4:
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1)
        else:
            cam = (gradients * activations).sum(dim=-1)
            if len(cam.shape) == 1:
                s = int(np.ceil(np.sqrt(cam.shape[0])))
                cam = F.pad(cam, (0, s * s - cam.shape[0]))
                cam = cam.view(s, s).unsqueeze(0)
            elif len(cam.shape) == 2:
                s = int(np.ceil(np.sqrt(cam.shape[1])))
                cam = F.pad(cam, (0, s * s - cam.shape[1]))
                cam = cam.view(1, s, s)

        cam = F.relu(cam)
        cam = cam.unsqueeze(0) if cam.dim() == 2 else cam
        if cam.dim() == 3:
            cam = cam.unsqueeze(0)

        w, h = image_size
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return cam


# ===================================================================
# Attention Rollout (better for ViT-based models like Qwen2-VL)
# ===================================================================

class AttentionRollout:
    """
    Attention Rollout aggregates attention weights across all layers
    of a Vision Transformer to produce a more accurate attention map.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._attention_maps = []
        self._hooks = []

    def _register_attention_hooks(self):
        """Hook into all self-attention layers to capture attention weights."""
        self._attention_maps = []
        self._hooks = []

        for name, module in self.model.named_modules():
            if not ('visual' in name.lower() or 'vision' in name.lower()):
                continue
            if any(key in name.lower() for key in ['attn', 'attention', 'self_attn']):
                if hasattr(module, 'forward'):
                    def make_hook(layer_name):
                        def hook(mod, inp, out):
                            if isinstance(out, tuple) and len(out) > 1:
                                attn_w = out[1]
                                if attn_w is not None and isinstance(attn_w, torch.Tensor):
                                    self._attention_maps.append(attn_w.detach().cpu())
                        return hook
                    h = module.register_forward_hook(make_hook(name))
                    self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate_rollout(self, image, prompt):
        """
        Generate attention rollout visualization.

        Returns:
            tuple: (heatmap_uint8, overlay_image)
        """
        if not HAS_TORCH:
            return None, _generate_fallback_attention(image)

        try:
            self._register_attention_hooks()

            from qwen_vl_utils import process_vision_info

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=text, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

            self.model.eval()
            with torch.no_grad():
                self.model(**inputs)

            self._remove_hooks()

            if not self._attention_maps:
                return None, _generate_fallback_attention(image)

            # Attention Rollout: multiply attention maps across layers
            rollout = None
            for attn in self._attention_maps:
                if attn.dim() == 4:
                    attn = attn.mean(dim=1)
                eye = torch.eye(attn.size(-1)).unsqueeze(0)
                attn = attn + eye
                attn = attn / attn.sum(dim=-1, keepdim=True)

                if rollout is None:
                    rollout = attn
                else:
                    if rollout.shape == attn.shape:
                        rollout = torch.bmm(attn, rollout)

            if rollout is None:
                return None, _generate_fallback_attention(image)

            # Take attention from CLS token to all other tokens
            cls_attention = rollout[0, 0, 1:]
            seq_len = cls_attention.shape[0]
            h = w = int(np.sqrt(seq_len))
            if h * w < seq_len:
                h = w = int(np.ceil(np.sqrt(seq_len)))
                padded = torch.zeros(h * w)
                padded[:seq_len] = cls_attention
                cls_attention = padded

            attn_map = cls_attention[:h * w].reshape(h, w).numpy()
            attn_map = _normalize_cam(attn_map)

            img_w, img_h = image.size
            attn_resized = np.array(
                Image.fromarray(attn_map).resize((img_w, img_h), Image.BILINEAR)
            )
            overlay = _create_overlay(image, attn_resized)
            return attn_resized, overlay

        except Exception as e:
            self._remove_hooks()
            print(f"Attention Rollout error: {e}")
            return None, _generate_fallback_attention(image)


# ===================================================================
# Perturbation-based Saliency (works with ANY model, including Ollama)
# ===================================================================

def perturbation_saliency(image, prompt, inference_fn, grid_size=7):
    """
    Perturbation-based saliency map. Works with any model backend
    (including Ollama) since it only needs the inference function.

    Args:
        image: PIL Image
        prompt: text prompt
        inference_fn: callable(image, prompt) -> str
        grid_size: number of grid cells per axis

    Returns:
        tuple: (saliency_uint8, overlay_image)
    """
    img_w, img_h = image.size

    base_response = inference_fn(image, prompt)
    base_words = set(base_response.lower().split())

    saliency = np.zeros((grid_size, grid_size), dtype=np.float32)
    cell_w = img_w / grid_size
    cell_h = img_h / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            masked = image.copy()
            draw = ImageDraw.Draw(masked)
            x0 = int(j * cell_w)
            y0 = int(i * cell_h)
            x1 = int((j + 1) * cell_w)
            y1 = int((i + 1) * cell_h)
            draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))

            masked_response = inference_fn(masked, prompt)
            masked_words = set(masked_response.lower().split())

            if base_words or masked_words:
                union = len(base_words | masked_words)
                intersection = len(base_words & masked_words)
                change = 1.0 - (intersection / max(union, 1))
            else:
                change = 0.0

            saliency[i, j] = change

    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    saliency_uint8 = (saliency * 255).astype(np.uint8)
    saliency_resized = np.array(
        Image.fromarray(saliency_uint8).resize((img_w, img_h), Image.BILINEAR)
    )

    overlay = _create_overlay(image, saliency_resized)
    return saliency_resized, overlay


# ===================================================================
# Shared helper functions
# ===================================================================

def _normalize_cam(cam):
    """Normalize CAM to 0-255 range."""
    if cam is None:
        return None
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min > 0:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return (cam * 255).astype(np.uint8)


def _apply_colormap(heatmap):
    """Apply JET colormap without matplotlib dependency."""
    h = heatmap.astype(np.float32) / 255.0
    result = np.zeros((*heatmap.shape, 3), dtype=np.float32)

    result[..., 2] = np.clip(
        np.where(h < 0.35, 1.0,
                 np.where(h < 0.65, (0.65 - h) / 0.3, 0.0)), 0, 1)
    result[..., 1] = np.clip(
        np.where(h < 0.15, 0.0,
                 np.where(h < 0.35, (h - 0.15) / 0.2,
                          np.where(h < 0.65, 1.0,
                                   np.where(h < 0.85, (0.85 - h) / 0.2, 0.0)))), 0, 1)
    result[..., 0] = np.clip(
        np.where(h < 0.35, 0.0,
                 np.where(h < 0.65, (h - 0.35) / 0.3, 1.0)), 0, 1)

    return (result * 255).astype(np.uint8)


def _create_overlay(original_image, heatmap, alpha=0.4):
    """Create an overlay of the heatmap on the original image."""
    if heatmap is None:
        return original_image

    w, h = original_image.size
    heatmap_resized = np.array(
        Image.fromarray(heatmap).resize((w, h), Image.BILINEAR)
    )
    colored_heatmap = _apply_colormap(heatmap_resized).astype(np.float32)
    original_np = np.array(original_image.convert('RGB')).astype(np.float32)

    overlay = original_np * (1 - alpha) + colored_heatmap * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _generate_fallback_attention(image):
    """Generate center-weighted Gaussian attention as fallback."""
    w, h = image.size
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    attention = np.exp(
        -((x_grid - cx) ** 2 / (2 * (w / 3) ** 2) +
          (y_grid - cy) ** 2 / (2 * (h / 3) ** 2))
    )
    attention = (attention / attention.max() * 200).astype(np.uint8)
    return _create_overlay(image, attention)


# ===================================================================
# High-level API functions
# ===================================================================

def generate_gradcam_visualization(model, processor, image, prompt,
                                   method="auto"):
    """
    High-level function to generate interpretability visualization.

    Args:
        method: 'gradcam', 'attention_rollout', or 'auto'

    Returns:
        tuple: (overlay_image, description)
    """
    overlay = None
    heatmap = None
    method_used = method

    if method in ("auto", "attention_rollout"):
        ar = AttentionRollout(model, processor)
        heatmap, overlay = ar.generate_rollout(image, prompt)
        if heatmap is not None:
            method_used = "attention_rollout"

    if heatmap is None and method in ("auto", "gradcam"):
        gradcam = GradCAM(model, processor)
        heatmap, overlay, _ = gradcam.generate_heatmap(image, prompt)
        if heatmap is not None:
            method_used = "gradcam"

    if heatmap is not None:
        high_attention = np.sum(heatmap > 178) / heatmap.size * 100
        medium_attention = np.sum((heatmap > 76) & (heatmap <= 178)) / heatmap.size * 100

        label = "Attention Rollout" if method_used == "attention_rollout" else "Grad-CAM"
        description = (
            f"{label} Attention Analysis:\n"
            f"  High attention regions: {high_attention:.1f}%\n"
            f"  Medium attention regions: {medium_attention:.1f}%\n"
            f"  Red/yellow = high attention, Blue = low attention\n\n"
            f"  Method: {label}\n"
            f"  This visualization helps verify the AI is examining\n"
            f"  clinically relevant anatomical regions."
        )
    else:
        overlay = _generate_fallback_attention(image)
        description = (
            "Approximate Attention Map (fallback):\n"
            "  Could not extract model attention weights.\n"
            "  Using center-weighted approximation.\n"
            "  For better results, try Perturbation Saliency."
        )

    return overlay, description


def create_side_by_side(original_image, gradcam_image, padding=10):
    """Create a side-by-side comparison of original and attention images."""
    if gradcam_image is None:
        return original_image

    h1 = original_image.height
    h2 = gradcam_image.height
    target_h = max(h1, h2)

    if h1 != target_h:
        ratio = target_h / h1
        original_image = original_image.resize(
            (int(original_image.width * ratio), target_h), Image.LANCZOS
        )
    if h2 != target_h:
        ratio = target_h / h2
        gradcam_image = gradcam_image.resize(
            (int(gradcam_image.width * ratio), target_h), Image.LANCZOS
        )

    total_w = original_image.width + padding + gradcam_image.width
    canvas = Image.new('RGB', (total_w, target_h), (40, 40, 40))
    canvas.paste(original_image, (0, 0))
    canvas.paste(gradcam_image, (original_image.width + padding, 0))
    return canvas
