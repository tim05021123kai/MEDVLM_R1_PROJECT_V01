# MedVLM-R1 Medical Image AI Viewer — 完整測試方案

> 版本: v2.0 | 日期: 2026-02-13

---

## 目錄

1. [測試架構總覽](#1-測試架構總覽)
2. [測試材料與下載網址](#2-測試材料與下載網址)
3. [Phase 1: 單元測試 (已完成)](#3-phase-1-單元測試)
4. [Phase 2: DICOM 整合測試](#4-phase-2-dicom-整合測試)
5. [Phase 3: AI 推論端對端測試](#5-phase-3-ai-推論端對端測試)
6. [Phase 4: PACS 網路測試](#6-phase-4-pacs-網路測試)
7. [Phase 5: GUI 驗收測試](#7-phase-5-gui-驗收測試)
8. [測試環境建置指南](#8-測試環境建置指南)

---

## 1. 測試架構總覽

```
┌─────────────────────────────────────────────────────────┐
│  Phase 5: GUI 驗收測試 (手動 + Gradio Client)            │
├─────────────────────────────────────────────────────────┤
│  Phase 4: PACS 網路測試 (Orthanc Docker)                 │
├─────────────────────────────────────────────────────────┤
│  Phase 3: AI 推論端對端 (Ollama + 真實影像)               │
├─────────────────────────────────────────────────────────┤
│  Phase 2: DICOM 整合測試 (真實 DICOM 檔案)               │
├─────────────────────────────────────────────────────────┤
│  Phase 1: 單元測試 73/73 PASS (Mock, 已完成)             │
└─────────────────────────────────────────────────────────┘
```

| Phase | 測試數量 | 需要真實資料 | 需要外部服務 | 自動化程度 |
|-------|---------|------------|------------|-----------|
| 1     | 73      | 否         | 否          | 100% 自動 |
| 2     | 25      | DICOM 檔案  | 否          | 100% 自動 |
| 3     | 18      | 醫學影像    | Ollama      | 90% 自動  |
| 4     | 12      | DICOM 檔案  | Orthanc     | 80% 自動  |
| 5     | 15      | 全部        | 全部        | 手動驗收   |

---

## 2. 測試材料與下載網址

### 2.1 小型 DICOM 範例檔 (免註冊、直接下載)

這些是最快取得的測試材料，適合開發階段反覆執行。

| # | 資源名稱 | 內容說明 | 格式 | 下載網址 |
|---|---------|---------|------|---------|
| 1 | **Rubo Medical DICOM Samples** | CT/MR/US/XA 多模態個別 DICOM 檔，極小檔案 | `.dcm` | https://www.rubomedical.com/dicom_files/ |
| 2 | **OsiriX DICOM Image Library** | 腦部腫瘤 CT、腹部 CTA、心臟 CT、全身 PET-CT 等完整系列 | `.dcm` | https://www.osirix-viewer.com/resources/dicom-image-library/ |
| 3 | **Zenodo 腹部 CT Phantom** | Siemens CT 腹部掃描，含雙能量(80/140kVp)和多種重建模式 | `.dcm` (ZIP) | https://zenodo.org/records/4461395 |
| 4 | **DICOM Library** | 用戶上傳、自動匿名化的 DICOM 資料集 | `.dcm` | https://www.dicomlibrary.com/ |
| 5 | **3Dicom Free DICOM Library** | 按解剖部位分類的匿名 DICOM 影像 | `.dcm` | https://3dicomviewer.com/dicom-library/ |
| 6 | **Aliza Medical Datasets** | GE/Philips/Siemens 多廠牌 CT 測試資料 | `.dcm` | https://www.aliza-dicom-viewer.com/download/datasets |
| 7 | **OFFIS DICOM Images** | DCMTK 開發者提供的標準測試影像 | `.dcm` | https://dicom.offis.de/download/images/ |
| 8 | **David Clunie DICOM Samples** | DICOM 標準編輯者提供的邊界案例檔案 | `.dcm` | http://www.dclunie.com/ |
| 9 | **Internet Archive Brain CT** | 腦部 CT DICOM (PCIR 公開資料) | `.dcm` | https://archive.org/details/9889023420030505CT |
| 10 | **Medimodel Samples** | 已匿名化的 CT/MRI DICOM (含牙科、獸醫) | `.dcm` | https://medimodel.com/sample-dicom-files/ |

### 2.2 程式化測試夾具 (開發者工具)

| # | 資源名稱 | 說明 | 安裝/使用方式 | 網址 |
|---|---------|------|-------------|------|
| 1 | **pydicom 內建測試資料** | `CT_small.dcm` 等小型 DICOM | `from pydicom.data import get_testdata_file` | https://github.com/pydicom/pydicom |
| 2 | **pydicom-data** | 更多測試 DICOM 資料集 | `pip install pydicom-data` | https://github.com/pydicom/pydicom-data |
| 3 | **robyoung/dicom-test-files** | 16x16 超小 DICOM，適合 CI/CD | Git clone | https://github.com/robyoung/dicom-test-files |
| 4 | **dicomgenerator** | Python 合成 DICOM 產生器 | `pip install dicomgenerator` | https://github.com/sjoerdk/dicomgenerator |
| 5 | **NEMA 官方測試資料** | DICOM 標準參考實作 | FTP 下載 | ftp://medical.nema.org/medical/Dicom/DataSets/ |

### 2.3 大型公開資料集 (AI 分析驗證)

| # | 資料集名稱 | 影像數量 | 部位 | 格式 | 大小 | 下載網址 |
|---|-----------|---------|------|------|------|---------|
| 1 | **NIH Chest X-ray14** | 112,120 張 | 胸部 | PNG + CSV | ~45GB (完整) | https://www.kaggle.com/datasets/nih-chest-xrays/data |
| 1b | NIH Chest X-ray (小型樣本) | ~5,600 張 | 胸部 | PNG | ~1GB | https://www.kaggle.com/datasets/nih-chest-xrays/sample |
| 1c | NIH Chest X-ray (224x224) | 112,120 張 | 胸部 | PNG | 2.5GB | https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0 |
| 1d | NIH Chest X-ray (HuggingFace) | 112,120 張 | 胸部 | PNG | ~45GB | https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset |
| 2 | **CheXpert (Stanford)** | 224,316 張 | 胸部 | JPG | ~11GB | https://aimi.stanford.edu/datasets/chexpert-chest-x-rays |
| 2b | CheXpert Plus (含 DICOM) | 223,462 組 | 胸部 | DICOM + Report | ~440GB | https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1 |
| 3 | **MIMIC-CXR (DICOM)** | 377,110 張 | 胸部 | DICOM + Report | ~4.7TB | https://physionet.org/content/mimic-cxr/2.1.0/ |
| 3b | MIMIC-CXR (JPG) | 377,110 張 | 胸部 | JPG + Report | ~65GB | https://physionet.org/content/mimic-cxr-jpg/2.1.0/ |
| 4 | **VinDr-CXR** | 100,000+ 張 | 胸部 | DICOM + BBox | ~30GB | https://physionet.org/content/vindr-cxr/1.0.0/ |
| 5 | **RSNA Pneumonia Detection** | ~30,000 張 | 胸部 | DICOM + BBox | ~3GB | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge |
| 6 | **RSNA 腦部出血偵測** | 874,035 張 | 腦部 CT | DICOM + Labels | ~180GB | https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data |
| 7 | **TCIA CT-ORG (多器官)** | 140 例 | 多部位 | DICOM + Seg | 數 GB | https://www.cancerimagingarchive.net/collection/ct-org/ |
| 8 | **TCIA LIDC-IDRI (肺部)** | 1,010 例 | 肺部 CT | DICOM + Annot | 數十 GB | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| 9 | **Open-i (NLM)** | 7,470 張 CXR | 胸部 | PNG + Report | 線上瀏覽 | https://openi.nlm.nih.gov/ |
| 10 | **NCI Imaging Data Commons** | 85+ TB | 全部位 | DICOM | 雲端 | https://datacommons.cancer.gov/repository/imaging-data-commons |

### 2.4 PACS 伺服器 (C-ECHO/C-FIND/C-MOVE 測試)

| # | 資源名稱 | 說明 | 下載/啟動方式 | 網址 |
|---|---------|------|-------------|------|
| 1 | **Orthanc DICOM Server** | 輕量開源 PACS，支援全部 DICOM 服務 | Docker 一鍵啟動 | https://www.orthanc-server.com/ |
| 1b | Orthanc Docker Image | | `docker run -p 4242:4242 -p 8042:8042 jodogne/orthanc` | https://hub.docker.com/r/jodogne/orthanc |
| 2 | **Orthanc Setup Samples** | 多種 Docker Compose 配置 (含 OHIF Viewer) | Git clone | https://github.com/orthanc-server/orthanc-setup-samples |
| 3 | **PyOrthanc** | Python Orthanc REST API 客戶端 | `pip install pyorthanc` | https://gacou54.github.io/pyorthanc/ |

---

## 3. Phase 1: 單元測試

**狀態: 已完成 73/73 PASS**

檔案: `tests/test_regression.py`

```bash
python tests/test_regression.py
```

| 測試區段 | 數量 | 狀態 |
|---------|------|------|
| 模組匯入 | 15 | PASS |
| Cache & History | 5 | PASS |
| Audit Logger | 2 | PASS |
| Export | 2 | PASS |
| De-identification | 1 | PASS |
| Image Utils | 4 | PASS |
| Prompt Utils | 7 | PASS |
| Physio Params | 8 | PASS |
| Report Generator | 4 | PASS |
| Gradcam Engine | 8 | PASS |
| Ollama Client | 3 | PASS |
| App.py Functions | 12 | PASS |
| DICOM Handler | 2 | PASS |

---

## 4. Phase 2: DICOM 整合測試

### 前置作業

```bash
# 下載測試 DICOM 檔案
mkdir -p test_data/dicom

# 方法 A: 使用 pydicom 內建資料 (最快)
pip install pydicom pydicom-data
python -c "
from pydicom.data import get_testdata_file
import shutil
for name in ['CT_small.dcm', 'MR_small.dcm', 'CT_small.dcm']:
    src = get_testdata_file(name)
    shutil.copy(src, 'test_data/dicom/')
    print(f'Copied: {name}')
"

# 方法 B: 從 Rubo Medical 下載
# 網址: https://www.rubomedical.com/dicom_files/
# 下載 CT, MR, US 各一個 .dcm 檔

# 方法 C: 從 OsiriX 下載完整系列
# 網址: https://www.osirix-viewer.com/resources/dicom-image-library/
# 下載: BRAINIX (腦 MRI), Abdomen CT
```

### 測試案例 (25 項)

#### 4.2.1 DICOM 載入測試 (8 項)

| # | 測試名稱 | 輸入 | 預期結果 |
|---|---------|------|---------|
| D-01 | 載入 CT DICOM | CT_small.dcm | 回傳 Dataset + PIL Image |
| D-02 | 載入 MR DICOM | MR_small.dcm | 回傳 Dataset + PIL Image |
| D-03 | 載入 CR/DX X-ray | 胸部 X-ray .dcm | 回傳 Dataset + PIL Image |
| D-04 | 載入不合法檔案 | random.txt | 回傳錯誤訊息，不 crash |
| D-05 | 載入損壞 DICOM | truncated .dcm | 回傳錯誤訊息 |
| D-06 | 提取完整 Metadata | CT with all fields | patient/study/series 完整 |
| D-07 | 提取不完整 Metadata | 缺少欄位的 DICOM | 缺少欄位顯示 "N/A" |
| D-08 | 載入 DICOM 系列目錄 | 資料夾含多片 DICOM | 按 SliceLocation 排序 |

#### 4.2.2 CT Windowing 測試 (7 項)

| # | 測試名稱 | 輸入 | 預期結果 |
|---|---------|------|---------|
| W-01 | Lung Window | CT + (-600, 1500) | 肺部組織清晰可見 |
| W-02 | Brain Window | CT + (40, 80) | 灰白質區分 |
| W-03 | Bone Window | CT + (400, 1800) | 骨骼結構可見 |
| W-04 | Soft Tissue Window | CT + (40, 400) | 軟組織對比 |
| W-05 | Liver Window | CT + (60, 160) | 肝臟 |
| W-06 | Custom Window | CT + (0, 2000) | 自訂範圍正常 |
| W-07 | Windowing 非 CT | MR DICOM | 回傳提示或降級處理 |

#### 4.2.3 De-identification 測試 (6 項)

| # | 測試名稱 | 輸入 | 預期結果 |
|---|---------|------|---------|
| P-01 | PHI 偵測 | 含完整 PHI 的 DICOM | 列出所有 PHI 欄位 |
| P-02 | 執行去識別化 | 含 PHI 的 DICOM | PatientName → ANONYMOUS |
| P-03 | 保留年齡/性別 | keep_age=True | Age/Sex 保留，Name 移除 |
| P-04 | 完全去識別化 | keep_age=False | 所有 PHI 移除 |
| P-05 | 日期處理 | StudyDate=20240115 | 僅保留年份 20240101 |
| P-06 | 重複去識別化 | 已匿名 DICOM | 不出錯，冪等操作 |

#### 4.2.4 影像前處理測試 (4 項)

| # | 測試名稱 | 輸入 | 預期結果 |
|---|---------|------|---------|
| I-01 | X-ray 自動對比 | CR modality image | AutoContrast 增強 |
| I-02 | 黑邊裁切 | 有 letterbox 的影像 | 裁除黑邊 |
| I-03 | ROI 擷取 | 影像 + bbox | 正確裁切區域 |
| I-04 | ROI 邊界 clamp | bbox 超出影像 | 不超出影像邊界 |

---

## 5. Phase 3: AI 推論端對端測試

### 前置作業

```bash
# 確保 Ollama 執行中
ollama serve &
# 確保已拉取視覺模型
ollama pull qwen2-vl    # 或 qwen3-vl

# 準備測試影像
# 方法 A: 從 NIH Chest X-ray 下載小型樣本
# 網址: https://www.kaggle.com/datasets/nih-chest-xrays/sample

# 方法 B: 從 Open-i 下載數張胸部 X-ray PNG
# 網址: https://openi.nlm.nih.gov/
# 搜尋 "chest x-ray normal" 並下載 3-5 張

# 方法 C: 從 RSNA Pneumonia 下載 DICOM
# 網址: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
```

### 測試案例 (18 項)

#### 5.1 Ollama 推論測試 (8 項)

| # | 測試名稱 | 輸入 | 預期結果 |
|---|---------|------|---------|
| O-01 | 基礎推論 (blocking) | 胸部 X-ray + "Analyze" | 回傳結構化分析報告 |
| O-02 | 串流推論 (streaming) | 胸部 X-ray + "Analyze" | 逐字元產出 |
| O-03 | Chain-of-Thought | 胸部 X-ray + CoT prompt | 包含推理步驟 |
| O-04 | 結構化輸出 | 胸部 X-ray + medical prompt | 含 FINDINGS/IMPRESSION/RECOMMENDATIONS |
| O-05 | Few-shot 胸部 | CHEST body part | 提示包含範例報告 |
| O-06 | 多影像比較 | 兩張 X-ray | 回傳比較分析 |
| O-07 | 連線失敗降級 | 停止 Ollama | 回傳友善錯誤訊息 |
| O-08 | 長文產出 | max_tokens=4096 | 完整長報告 |

#### 5.2 推論輔助功能 (6 項)

| # | 測試名稱 | 輸入 | 預期結果 |
|---|---------|------|---------|
| A-01 | 快取命中 | 相同 image+prompt 第二次 | 立即返回，不重新推論 |
| A-02 | 快取未命中 | 不同 prompt | 重新推論 |
| A-03 | 歷史記錄 | 執行 3 次分析 | history 有 3 筆 |
| A-04 | 審計日誌 | 執行分析 | audit_logs/ 有記錄 |
| A-05 | 後端切換 | ollama → medvlm-r1 | 正確切換，config 更新 |
| A-06 | 後端 fallback | 主後端失敗 | 自動切換備用後端 |

#### 5.3 報告產出 (4 項)

| # | 測試名稱 | 輸入 | 預期結果 |
|---|---------|------|---------|
| R-01 | 放射報告格式 | AI 分析結果 | 含 HEADER/DISCLAIMER/SIGNATURE |
| R-02 | PDF 匯出 | 報告文字 | 產生可開啟的 PDF |
| R-03 | FHIR 匯出 | 報告 + metadata | 產生合法 FHIR R4 JSON |
| R-04 | 文字匯出 | 報告文字 | 產生 .txt 檔 |

---

## 6. Phase 4: PACS 網路測試

### 前置作業

```bash
# 啟動 Orthanc DICOM Server (Docker)
docker run -d --name orthanc \
  -p 4242:4242 \
  -p 8042:8042 \
  -e ORTHANC__DICOM_ALWAYS_ALLOW_ECHO=true \
  -e ORTHANC__DICOM_ALWAYS_ALLOW_FIND=true \
  -e ORTHANC__DICOM_ALWAYS_ALLOW_GET=true \
  -e ORTHANC__DICOM_ALWAYS_ALLOW_STORE=true \
  jodogne/orthanc

# 等待啟動完成
sleep 5
curl http://localhost:8042/system

# 上傳測試 DICOM 至 Orthanc
# 方法 A: Web UI
# 開啟 http://localhost:8042 (帳號: orthanc / 密碼: orthanc)
# 點擊 Upload → 選擇 DICOM 檔案

# 方法 B: REST API
curl -X POST http://localhost:8042/instances \
  -H "Content-Type: application/dicom" \
  --data-binary @test_data/dicom/CT_small.dcm \
  -u orthanc:orthanc

# 方法 C: DICOM C-STORE (storescu)
# storescu localhost 4242 test_data/dicom/*.dcm
```

### 測試案例 (12 項)

| # | 測試名稱 | 操作 | 預期結果 |
|---|---------|------|---------|
| PACS-01 | C-ECHO | 連線 localhost:4242 | 回傳 SUCCESS |
| PACS-02 | C-ECHO 失敗 | 連線 localhost:9999 | 回傳錯誤，不 crash |
| PACS-03 | C-FIND Patient | 查詢 PatientName=* | 回傳結果列表 |
| PACS-04 | C-FIND Modality | 查詢 Modality=CT | 僅回傳 CT 結果 |
| PACS-05 | C-FIND Date Range | 查詢指定日期 | 正確過濾 |
| PACS-06 | C-FIND 空結果 | 查詢不存在的 Patient | 回傳空列表 |
| PACS-07 | C-MOVE Study | 取回一個 study | 檔案儲存至 retrieve_dir |
| PACS-08 | C-MOVE 失敗 | 取回不存在的 UID | 回傳錯誤訊息 |
| PACS-09 | 結果顯示 | query 後 format | 表格格式正確 |
| PACS-10 | AE Title 配置 | 自訂 AE Title | 正確使用自訂值 |
| PACS-11 | 連線狀態 | 驗證後顯示 | 狀態文字正確 |
| PACS-12 | 完整 workflow | ECHO→FIND→MOVE→載入 | 端對端完整流程 |

---

## 7. Phase 5: GUI 驗收測試

### 前置作業

```bash
# 啟動完整環境
ollama serve &              # AI 後端
docker start orthanc        # PACS 伺服器
python app.py               # 啟動 Gradio 介面
# 開啟瀏覽器 http://localhost:7860
```

### 驗收檢查清單 (15 項)

#### Tab 1: Image Import

| # | 測試項目 | 步驟 | 預期結果 |
|---|---------|------|---------|
| G-01 | 上傳 PNG/JPG | 點擊上傳 → 選擇檔案 | 預覽顯示、狀態更新 |
| G-02 | 上傳 DICOM | 點擊 DICOM Upload → 選擇 .dcm | 影像顯示 + metadata 表格 |
| G-03 | CT Windowing | 載入 CT → 選擇 Lung/Brain/Bone | 影像隨 Window 改變 |
| G-04 | PHI 檢查 | 載入含 PHI 的 DICOM → Check PHI | 列出 PHI 欄位 |
| G-05 | 去識別化 | 點擊 De-identify → 檢查 metadata | PHI 已移除 |

#### Tab 2: PACS

| # | 測試項目 | 步驟 | 預期結果 |
|---|---------|------|---------|
| G-06 | PACS 連線 | 輸入 localhost:4242 → Test | 顯示 "Connected" |
| G-07 | PACS 查詢 | 輸入查詢條件 → Search | 結果表格顯示 |
| G-08 | PACS 取回 | 選擇結果 → Retrieve | 影像載入至 Tab 1 |

#### Tab 3: AI Radiologist

| # | 測試項目 | 步驟 | 預期結果 |
|---|---------|------|---------|
| G-09 | AI 分析 (Ollama) | 上傳影像 → Run Analysis | 串流顯示分析結果 |
| G-10 | AI 報告產出 | 勾選 Generate Report → Run | 完整放射報告 |
| G-11 | 多影像比較 | 上傳 Prior + Current → Compare | 比較分析結果 |

#### Tab 4: Interpretability

| # | 測試項目 | 步驟 | 預期結果 |
|---|---------|------|---------|
| G-12 | Grad-CAM | 上傳影像 → Run Grad-CAM | 熱力圖疊加顯示 |
| G-13 | Perturbation | 上傳影像 → Run Saliency | 重要區域標示 |

#### Tab 5-7: Reference, History, Export

| # | 測試項目 | 步驟 | 預期結果 |
|---|---------|------|---------|
| G-14 | 匯出 PDF/FHIR | 分析後 → Export PDF/FHIR | 檔案下載 |
| G-15 | 歷史/審計 | 多次分析後 → View History | 顯示分析歷史 |

---

## 8. 測試環境建置指南

### 8.1 最小測試環境 (Phase 1-2)

```bash
# Python 套件
pip install numpy Pillow pydicom pydicom-data

# 執行單元測試
python tests/test_regression.py

# 執行 DICOM 整合測試 (Phase 2 測試腳本)
python tests/test_dicom_integration.py
```

### 8.2 AI 推論環境 (Phase 3)

```bash
# 安裝 Ollama
# 網址: https://ollama.com/download
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取視覺模型 (選一個)
ollama pull qwen2-vl           # 較小，推論快
ollama pull qwen2-vl:72b       # 較大，效果好
ollama pull llama3.2-vision     # Meta 的視覺模型

# 驗證 Ollama 運作
curl http://localhost:11434/api/tags
```

### 8.3 PACS 測試環境 (Phase 4)

```bash
# 安裝 Docker
# 網址: https://docs.docker.com/get-docker/

# 啟動 Orthanc
docker run -d --name orthanc \
  -p 4242:4242 \
  -p 8042:8042 \
  jodogne/orthanc

# Web UI: http://localhost:8042 (orthanc/orthanc)
# DICOM Port: 4242
# AE Title: ORTHANC
```

### 8.4 完整測試環境 (Phase 5)

```bash
# 全部套件
pip install -r requirements.txt

# 全部服務
ollama serve &
docker start orthanc

# 啟動 App
python app.py
```

---

## 推薦測試材料快速取得清單

按照**測試階段**，建議的最少測試材料:

### Phase 2 最少需要:

| 材料 | 來源 | 動作 |
|------|------|------|
| CT DICOM (1個) | pydicom 內建 `CT_small.dcm` | `pip install pydicom` |
| MR DICOM (1個) | pydicom 內建 `MR_small.dcm` | 同上 |
| 胸部 X-ray DICOM (1個) | Rubo Medical | https://www.rubomedical.com/dicom_files/ |

### Phase 3 最少需要:

| 材料 | 來源 | 動作 |
|------|------|------|
| 胸部 X-ray PNG (3張) | Open-i | https://openi.nlm.nih.gov/ 搜尋下載 |
| 腦部 CT PNG (2張) | 同上 | 搜尋 "brain CT" |
| Ollama + qwen2-vl | Ollama 官網 | `ollama pull qwen2-vl` |

### Phase 4 最少需要:

| 材料 | 來源 | 動作 |
|------|------|------|
| Orthanc Docker | Docker Hub | `docker run jodogne/orthanc` |
| 5個 DICOM 檔案 | Phase 2 已下載 | 上傳至 Orthanc |

---

## 附錄: 資料集版權與使用限制

| 資料集 | 授權 | 需要註冊 | 商用限制 |
|--------|------|---------|---------|
| pydicom test data | MIT | 否 | 無 |
| Rubo Medical | 教育用 | 否 | 僅教育研究 |
| OsiriX Library | CC | 否 | 依個別授權 |
| Zenodo Abdomen CT | CC-BY 4.0 | 否 | 標註出處即可 |
| NIH Chest X-ray14 | CC0 | Kaggle 帳號 | 無 |
| CheXpert | Stanford RUA | 是 | 研究用 |
| MIMIC-CXR | PhysioNet DUA | 是 (需認證) | 研究用 |
| RSNA Challenges | Kaggle | 是 | 競賽規則 |
| TCIA | CC-BY | 看各集合 | 標註出處 |
| Orthanc | GPLv3 | 否 | 開源 |
