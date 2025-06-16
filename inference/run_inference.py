import torch
from qwen_vl_utils import process_vision_info

def generate_answer(model, processor, image, prompt):
    """使用 MedVLM-R1 生成回答"""
    
    # 構建消息格式
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    # 應用聊天模板
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 處理視覺信息
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 準備輸入
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 生成配置
    generation_config = {
        'use_cache': True,
        'max_new_tokens': 512,
        'do_sample': False,
        'temperature': 1.0,
    }
    
    # 生成回答
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_config)
    
    # 解碼輸出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""
