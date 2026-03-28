"""
Inference module supporting MedVLM-R1 and Ollama backends.
Provides both blocking and streaming generation, plus multi-image support.
"""

import threading
import torch
from qwen_vl_utils import process_vision_info


def generate_answer(model, processor, image, prompt, max_new_tokens=2048):
    """Generate answer using MedVLM-R1 (blocking)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generation_config = {
        "use_cache": True,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
    }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_config)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0] if output_text else ""


def generate_answer_stream(model, processor, image, prompt, max_new_tokens=2048):
    """
    Generate answer using MedVLM-R1 with streaming via TextIteratorStreamer.
    Yields incremental text chunks.
    """
    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        # Fallback to blocking if streamer not available
        yield generate_answer(model, processor, image, prompt, max_new_tokens)
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "use_cache": True,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for chunk in streamer:
        if chunk:
            yield chunk

    thread.join()


def generate_answer_ollama(image, prompt, model_name="qwen3-vl",
                           base_url="http://localhost:11434",
                           temperature=0.7, max_tokens=2048):
    """Generate answer using Ollama VLM (blocking)."""
    from model.ollama_client import generate_with_ollama
    return generate_with_ollama(
        image=image, prompt=prompt, model_name=model_name,
        base_url=base_url, temperature=temperature, max_tokens=max_tokens,
    )


def generate_answer_ollama_stream(image, prompt, model_name="qwen3-vl",
                                  base_url="http://localhost:11434",
                                  temperature=0.7, max_tokens=2048):
    """Generate answer using Ollama VLM with streaming (yields chunks)."""
    from model.ollama_client import generate_with_ollama_stream
    yield from generate_with_ollama_stream(
        image=image, prompt=prompt, model_name=model_name,
        base_url=base_url, temperature=temperature, max_tokens=max_tokens,
    )


def generate_multi_image_comparison(images, prompt, model_name="qwen3-vl",
                                    base_url="http://localhost:11434",
                                    temperature=0.7, max_tokens=2048):
    """Generate answer with multiple images for comparison studies."""
    from model.ollama_client import generate_with_ollama_multi_image
    return generate_with_ollama_multi_image(
        images=images, prompt=prompt, model_name=model_name,
        base_url=base_url, temperature=temperature, max_tokens=max_tokens,
    )


def compute_confidence_from_logits(model, processor, image, prompt,
                                   num_runs=3, temperature=0.8):
    """
    Estimate confidence by running inference multiple times with sampling
    and measuring output consistency.

    Returns:
        tuple: (primary_answer, confidence_score 0-1, all_answers)
    """
    answers = []
    for _ in range(num_runs):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=512, do_sample=True,
                temperature=temperature, use_cache=True,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], generated_ids)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        answers.append(decoded[0] if decoded else "")

    # Measure consistency via simple word overlap
    if not answers:
        return "", 0.0, []

    primary = answers[0]
    if len(answers) == 1:
        return primary, 1.0, answers

    primary_words = set(primary.lower().split())
    overlaps = []
    for ans in answers[1:]:
        ans_words = set(ans.lower().split())
        if primary_words or ans_words:
            overlap = len(primary_words & ans_words) / max(len(primary_words | ans_words), 1)
            overlaps.append(overlap)

    confidence = sum(overlaps) / len(overlaps) if overlaps else 1.0
    return primary, confidence, answers
