from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

def load_medvlm_model():
    """加載 MedVLM-R1 模型和處理器"""
    MODEL_PATH = 'JZPeterPan/MedVLM-R1'
    
    print("正在下載/加載模型，這可能需要幾分鐘...")
    
    # 加載模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # CPU 使用 float32
        device_map="cpu",
        trust_remote_code=True
    )
    
    # 加載處理器
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    return model, processor



