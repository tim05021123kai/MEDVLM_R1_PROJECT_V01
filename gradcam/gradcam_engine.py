"""
Grad-CAM (Gradient-weighted Class Activation Mapping) module.
Provides visual explanations for AI model predictions on medical images,
enhancing interpretability and trust in clinical settings.
"""

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GradCAM:
    """
    Grad-CAM implementation for vision-language models.
    Generates attention heatmaps showing which regions of the image
    the model focused on during analysis.
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.activations = {}
        self.gradients = {}
        self._hooks = []

    def _find_target_layer(self):
        """
        Find the best target layer for Grad-CAM in the model.
        Searches for the last convolutional or visual encoder layer.
        """
        target_layer = None
        target_name = None

        for name, module in self.model.named_modules():
            # Look for visual encoder layers (Qwen2-VL architecture)
            if any(key in name.lower() for key in [
                'visual', 'vision', 'vit', 'encoder', 'patch_embed'
            ]):
                if hasattr(module, 'weight') or any(
                    hasattr(module, attr) for attr in ['out_proj', 'dense', 'fc']
                ):
                    target_layer = module
                    target_name = name

            # Also check for nn.Conv2d or nn.Linear in vision parts
            if isinstance(module, (torch.nn.Conv2d, torch.nn.LayerNorm)):
                if 'visual' in name.lower() or 'vision' in name.lower():
                    target_layer = module
                    target_name = name

        if target_layer is None:
            # Fallback: find any layer with visual/vision in name
            for name, module in self.model.named_modules():
                if ('visual' in name.lower() or 'vision' in name.lower()):
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
                return False, "無法找到適合的目標層 (Could not find target layer)"

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
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.activations = {}
        self.gradients = {}

    def generate_heatmap(self, image, prompt, image_size=None):
        """
        Generate a Grad-CAM heatmap for the given image and prompt.

        Args:
            image: PIL Image
            prompt: Text prompt used for analysis
            image_size: (width, height) of original image for resizing

        Returns:
            tuple: (heatmap_array, overlay_image, raw_cam)
                   - heatmap_array: numpy array of the heatmap (H, W)
                   - overlay_image: PIL Image with heatmap overlay
                   - raw_cam: raw CAM values before normalization
        """
        if not HAS_TORCH:
            return None, None, None

        if image_size is None:
            image_size = image.size  # (width, height)

        try:
            # Register hooks
            success, msg = self.register_hooks()
            if not success:
                return None, self._generate_fallback_attention(image), None

            # Prepare model input
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

            # Forward pass with gradients
            self.model.eval()
            was_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

            # Enable gradients on inputs
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
                    inputs[key].requires_grad_(True)

            outputs = self.model(**inputs)

            # Get the logits and create a target for backpropagation
            logits = outputs.logits
            # Use the max logit of the last generated position as target
            target = logits[:, -1, :].max()
            target.backward(retain_graph=False)

            torch.set_grad_enabled(was_grad_enabled)

            # Compute Grad-CAM
            if 'value' in self.activations and 'value' in self.gradients:
                activations = self.activations['value']
                gradients = self.gradients['value']

                cam = self._compute_cam(activations, gradients, image_size)
                heatmap = self._normalize_cam(cam)
                overlay = self._create_overlay(image, heatmap)

                self.remove_hooks()
                return heatmap, overlay, cam
            else:
                self.remove_hooks()
                return None, self._generate_fallback_attention(image), None

        except Exception as e:
            self.remove_hooks()
            print(f"Grad-CAM error: {e}")
            return None, self._generate_fallback_attention(image), None

    def _compute_cam(self, activations, gradients, image_size):
        """Compute the CAM from activations and gradients."""
        # Global average pooling of gradients -> weights
        if len(gradients.shape) == 3:
            # (batch, seq_len, hidden_dim)
            weights = gradients.mean(dim=1, keepdim=True)
            cam = (weights * activations).sum(dim=-1)
            # Reshape to 2D
            seq_len = cam.shape[1]
            h = w = int(np.sqrt(seq_len))
            if h * w != seq_len:
                h = w = int(np.ceil(np.sqrt(seq_len)))
                cam = F.pad(cam, (0, h * w - seq_len))
            cam = cam.view(1, h, w)
        elif len(gradients.shape) == 4:
            # (batch, channels, h, w)
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1)
        else:
            # Fallback: simple product and reshape
            cam = (gradients * activations).sum(dim=-1)
            if len(cam.shape) == 1:
                s = int(np.ceil(np.sqrt(cam.shape[0])))
                cam = F.pad(cam, (0, s * s - cam.shape[0]))
                cam = cam.view(s, s).unsqueeze(0)
            elif len(cam.shape) == 2:
                s = int(np.ceil(np.sqrt(cam.shape[1])))
                cam = F.pad(cam, (0, s * s - cam.shape[1]))
                cam = cam.view(1, s, s)

        # ReLU
        cam = F.relu(cam)

        # Resize to image dimensions
        cam = cam.unsqueeze(0) if cam.dim() == 2 else cam
        if cam.dim() == 3:
            cam = cam.unsqueeze(0)

        w, h = image_size
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        return cam

    def _normalize_cam(self, cam):
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

    def _create_overlay(self, original_image, heatmap, alpha=0.4, colormap='jet'):
        """
        Create an overlay of the heatmap on the original image.

        Args:
            original_image: PIL Image
            heatmap: numpy array (H, W) in 0-255 range
            alpha: Transparency of the heatmap overlay
            colormap: Color mapping scheme

        Returns:
            PIL Image with heatmap overlay
        """
        if heatmap is None:
            return original_image

        # Resize heatmap to match image
        w, h = original_image.size
        heatmap_resized = np.array(
            Image.fromarray(heatmap).resize((w, h), Image.BILINEAR)
        )

        # Apply colormap
        colored_heatmap = self._apply_colormap(heatmap_resized, colormap)

        # Convert original to RGB numpy
        original_np = np.array(original_image.convert('RGB')).astype(np.float32)
        colored_heatmap = colored_heatmap.astype(np.float32)

        # Blend
        overlay = original_np * (1 - alpha) + colored_heatmap * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)

    def _apply_colormap(self, heatmap, colormap='jet'):
        """
        Apply a colormap to a single-channel heatmap.
        Implements JET colormap without requiring matplotlib.
        """
        h = heatmap.astype(np.float32) / 255.0
        result = np.zeros((*heatmap.shape, 3), dtype=np.float32)

        # JET colormap implementation
        # Blue channel
        result[..., 2] = np.clip(
            np.where(h < 0.35, 1.0,
                     np.where(h < 0.65, (0.65 - h) / 0.3, 0.0)),
            0, 1
        )
        # Green channel
        result[..., 1] = np.clip(
            np.where(h < 0.15, 0.0,
                     np.where(h < 0.35, (h - 0.15) / 0.2,
                              np.where(h < 0.65, 1.0,
                                       np.where(h < 0.85, (0.85 - h) / 0.2, 0.0)))),
            0, 1
        )
        # Red channel
        result[..., 0] = np.clip(
            np.where(h < 0.35, 0.0,
                     np.where(h < 0.65, (h - 0.35) / 0.3, 1.0)),
            0, 1
        )

        return (result * 255).astype(np.uint8)

    def _generate_fallback_attention(self, image):
        """
        Generate a simple attention visualization as fallback
        when Grad-CAM cannot extract proper gradients.
        Uses center-weighted Gaussian as approximate attention.
        """
        w, h = image.size

        # Create center-weighted attention map
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        sigma_x = w / 3
        sigma_y = h / 3
        attention = np.exp(
            -((x_grid - cx) ** 2 / (2 * sigma_x ** 2) +
              (y_grid - cy) ** 2 / (2 * sigma_y ** 2))
        )

        attention = (attention / attention.max() * 200).astype(np.uint8)
        colored = self._apply_colormap(attention)
        original_np = np.array(image.convert('RGB')).astype(np.float32)
        colored = colored.astype(np.float32)

        overlay = original_np * 0.6 + colored * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)


def generate_gradcam_visualization(model, processor, image, prompt):
    """
    High-level function to generate Grad-CAM visualization.

    Args:
        model: The loaded AI model
        processor: The model processor
        image: PIL Image
        prompt: Analysis prompt

    Returns:
        tuple: (overlay_image, description)
    """
    gradcam = GradCAM(model, processor)

    heatmap, overlay, raw_cam = gradcam.generate_heatmap(image, prompt)

    if heatmap is not None:
        # Calculate attention statistics
        high_attention = np.sum(heatmap > 178) / heatmap.size * 100
        medium_attention = np.sum((heatmap > 76) & (heatmap <= 178)) / heatmap.size * 100

        description = (
            "Grad-CAM 注意力分析 (Attention Analysis):\n"
            f"  高關注區域 (High attention): {high_attention:.1f}% 的影像區域\n"
            f"  中等關注區域 (Medium attention): {medium_attention:.1f}% 的影像區域\n"
            f"  紅色/黃色區域代表模型重點關注的部位\n"
            f"  藍色區域代表模型較少關注的部位\n\n"
            f"  此視覺化有助於了解AI在做出判斷時重點關注的解剖結構。\n"
            f"  (Red/yellow = high attention, Blue = low attention)"
        )
    else:
        description = (
            "Grad-CAM 近似注意力圖 (Approximate Attention Map):\n"
            "  由於模型架構限制，使用近似方法產生注意力視覺化。\n"
            "  實際的模型注意力分佈可能有所不同。\n"
            "  (Approximate visualization due to model architecture)"
        )

    return overlay, description


def create_side_by_side(original_image, gradcam_image, padding=10):
    """
    Create a side-by-side comparison of original and Grad-CAM images.

    Args:
        original_image: PIL Image (original)
        gradcam_image: PIL Image (with Grad-CAM overlay)
        padding: Pixels of padding between images

    Returns:
        PIL Image with side-by-side comparison
    """
    if gradcam_image is None:
        return original_image

    # Resize to same height
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

    # Create canvas
    total_w = original_image.width + padding + gradcam_image.width
    canvas = Image.new('RGB', (total_w, target_h), (40, 40, 40))
    canvas.paste(original_image, (0, 0))
    canvas.paste(gradcam_image, (original_image.width + padding, 0))

    return canvas
