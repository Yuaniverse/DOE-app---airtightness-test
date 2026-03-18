# 氣密測試 DOE 實驗系統

此專案為一個以 **Streamlit** 建立的 DOE（Design of Experiments）應用，提供：

- 模組一：DOE 實驗表格產生器
- 模組二：響應變數計算（Y1 / AUC / GapQ / Y4）
- 模組三：參數優選決策
- 模組四：界線判定與風險驗證
- 模組五：基礎統計分析與 CCD 補點輔助

## 部署到 Streamlit Community Cloud

### 1. 上傳到 GitHub

請將以下檔案與資料夾推到 GitHub repository：

- `app.py`
- `requirements.txt`
- `modules/`
- `utils/`
- `.streamlit/config.toml`
- `README.md`

> 不需要上傳 `.venv/`、`__pycache__/`。

### 2. 在 Streamlit Community Cloud 建立 App

登入後建立新 App，設定如下：

- **Repository**：你的 GitHub repo
- **Branch**：main（或你的部署分支）
- **Main file path**：`app.py`

### 3. 安裝相依套件

Streamlit Community Cloud 會自動讀取：

- `requirements.txt`

本專案的 `requirements.txt` 已精簡為「直接依賴」版本，以降低 Community Cloud 卡在 **Your app is in the oven** 的機率。
本專案不需要額外系統套件。

## 本機執行

```bash
python -m streamlit run app.py
```

## 注意事項

- 模組二需上傳由模組一匯出的 `.xlsx` 實驗表後，才能計算響應值。
- 本專案未使用本機絕對路徑，也未依賴私人憑證，可直接部署到 Streamlit Community Cloud。
- 若首次部署較久，通常是雲端環境正在安裝 `requirements.txt` 套件。
