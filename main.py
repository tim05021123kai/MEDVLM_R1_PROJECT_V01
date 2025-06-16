from model.medvlm_loader import load_medvlm_model
from utils.image_utils import load_image_from_path
from utils.prompt_utils import build_prompt
from inference.run_inference import generate_answer

def main():
    print("正在加載 MedVLM-R1 模型...")
    
    try:
        # 加載模型和處理器
        model, processor = load_medvlm_model()
        print("模型加載成功！")
        
        # 加載圖像
        image_path = "data/example_image.jpg"
        print(f"正在加載圖像: {image_path}")
        image = load_image_from_path(image_path)
        
        # 構建提示
        prompt = build_prompt("請問這張胸部X光有什麼異常？")
        print(f"提示: {prompt}")
        
        # 生成回答
        print("正在生成回答...")
        answer = generate_answer(model, processor, image, prompt)
        
        print("\n" + "="*50)
        print("模型回答：")
        print(answer)
        print("="*50)
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()