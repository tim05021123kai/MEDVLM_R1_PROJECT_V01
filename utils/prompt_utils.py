def build_prompt(question: str, template_type: str = "medical"):
    """構建適合醫學影像分析的提示"""
    
    if template_type == "medical":
        return f"""你是一位專業的醫學影像分析AI助手。請仔細觀察提供的醫學影像，並根據以下問題提供詳細、準確的分析。

問題：{question}

請在回答中包含：
1. 影像類型和質量評估
2. 觀察到的具體發現
3. 可能的診斷建議
4. 建議的後續檢查或處理

請用中文回答，並保持專業和謹慎的態度。"""

    elif template_type == "simple":
        return f"請分析這張醫學影像並回答：{question}"
    
    else:
        return question

def build_cot_prompt(question: str):
    """構建鏈式思考提示，適用於複雜的醫學推理"""
    return f"""{question}

請按照以下步驟進行分析：
<think>
步驟1：描述影像的基本信息（類型、視角、技術參數等）
步驟2：系統性觀察各個解剖結構
步驟3：識別和描述任何異常發現
步驟4：分析異常的特徵和分布
步驟5：考慮可能的診斷和鑑別診斷
步驟6：評估臨床意義和建議
</think>

最終答案：<answer>在此提供最終的診斷意見和建議</answer>"""