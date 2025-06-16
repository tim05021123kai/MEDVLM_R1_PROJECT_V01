import gradio as gr
import torch
from PIL import Image
from model.medvlm_loader import load_medvlm_model
from utils.prompt_utils import build_prompt, build_cot_prompt
from inference.run_inference import generate_answer

# 全局變量存儲模型
model = None
processor = None

def load_model():
    """載入模型（只載入一次）"""
    global model, processor
    if model is None:
        print("正在載入 MedVLM-R1 模型...")
        model, processor = load_medvlm_model()
        print("模型載入完成！")
    return model, processor

def analyze_medical_image(image, question, analysis_type="標準分析"):
    """分析醫學影像"""
    try:
        # 確保模型已載入
        model, processor = load_model()
        
        # 檢查輸入
        if image is None:
            return "請上傳一張醫學影像"
        
        if not question.strip():
            question = "請分析這張醫學影像並描述你的發現"
        
        # 根據分析類型選擇提示
        if analysis_type == "鏈式思考":
            prompt = build_cot_prompt(question)
        elif analysis_type == "簡單分析":
            prompt = build_prompt(question, template_type="simple")
        else:  # 標準分析
            prompt = build_prompt(question, template_type="medical")
        
        # 生成回答
        answer = generate_answer(model, processor, image, prompt)
        
        return answer
        
    except Exception as e:
        return f"錯誤: {str(e)}"

def create_interface():
    """創建 Gradio 介面"""
    
    # 自定義 CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .medical-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="MedVLM-R1 醫學影像分析系統") as interface:
        
        gr.Markdown(
            """
            # 🏥 MedVLM-R1 醫學影像分析系統
            
            上傳醫學影像（X光、CT、MRI等），AI將提供專業的影像分析和診斷建議。
            
            ⚠️ **免責聲明**: 此系統僅供教育和研究用途，不能替代專業醫師的診斷。
            """,
            elem_classes="medical-title"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # 輸入區域
                gr.Markdown("## 📸 影像輸入")
                
                image_input = gr.Image(
                    label="上傳醫學影像",
                    type="pil",
                    height=400
                )
                
                question_input = gr.Textbox(
                    label="問題（可選）",
                    placeholder="請問這張影像有什麼異常？",
                    lines=3,
                    value="請問這張醫學影像有什麼異常？"
                )
                
                analysis_type = gr.Radio(
                    choices=["標準分析", "簡單分析", "鏈式思考"],
                    value="標準分析",
                    label="分析模式"
                )
                
                # 按鈕
                with gr.Row():
                    analyze_btn = gr.Button("🔍 開始分析", variant="primary")
                    clear_btn = gr.Button("🗑️ 清除", variant="secondary")
                
                # 預設問題範例
                gr.Markdown("### 💡 問題範例")
                example_questions = [
                    "這張影像有什麼異常？",
                    "請描述病變的位置和特徵",
                    "這可能是什麼疾病？",
                    "建議做哪些進一步檢查？",
                    "評估病變的嚴重程度"
                ]
                
                for i, question in enumerate(example_questions):
                    gr.Button(
                        question,
                        size="sm"
                    ).click(
                        lambda q=question: q,
                        outputs=question_input
                    )
            
            with gr.Column(scale=1):
                # 輸出區域
                gr.Markdown("## 📋 分析結果")
                
                output_text = gr.Textbox(
                    label="AI 分析結果",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
                
                # 分析狀態
                status_text = gr.Textbox(
                    label="狀態",
                    value="準備就緒",
                    interactive=False
                )
        
        # 事件處理
        def on_analyze(image, question, analysis_type):
            if image is None:
                return "請先上傳一張醫學影像", "錯誤: 未上傳影像"
            
            # 更新狀態
            yield "分析中，請稍候...", "正在分析影像..."
            
            # 執行分析
            result = analyze_medical_image(image, question, analysis_type)
            
            # 返回結果
            yield result, "分析完成"
        
        analyze_btn.click(
            on_analyze,
            inputs=[image_input, question_input, analysis_type],
            outputs=[output_text, status_text]
        )
        
        # 清除功能
        def clear_all():
            return None, "", "準備就緒"
        
        clear_btn.click(
            clear_all,
            outputs=[image_input, output_text, status_text]
        )
        
        # 範例
        gr.Markdown("### 📚 使用說明")
        gr.Markdown(
            """
            1. **上傳影像**: 支援 JPG、PNG 等格式的醫學影像
            2. **輸入問題**: 描述你想了解的具體問題（可選）
            3. **選擇模式**: 
               - 標準分析: 完整的醫學影像分析
               - 簡單分析: 簡潔的發現描述
               - 鏈式思考: 顯示詳細的推理過程
            4. **開始分析**: 點擊按鈕開始AI分析
            """
        )
    
    return interface

def main():
    """主函數"""
    print("正在啟動 MedVLM-R1 Gradio 介面...")
    
    # 預先載入模型（可選）
    try:
        load_model()
        print("模型預載入成功！")
    except Exception as e:
        print(f"模型預載入失敗: {e}")
        print("將在首次使用時載入模型")
    
    # 創建並啟動介面
    interface = create_interface()
    
    # 啟動服務
    interface.launch(
        server_name="127.0.0.1",  # 允許外部訪問
        server_port=7860,       # 預設端口
        share=False,            # 設為 True 可獲得公開鏈接
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()