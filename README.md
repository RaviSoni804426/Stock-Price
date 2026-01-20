
---
title: Stock Decision Assistant
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# 📈 Stock Decision Assistant

An AI-powered swing trading analysis tool for Indian Stocks (NSE).

## 🚀 Deployment Instructions

### 1. Create a New Space
Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
- **SDK**: Streamlit
- **Hardware**: CPU Basic (Free) is sufficient for inference.

### 2. Upload Files
Upload the entire contents of this folder to your Space.
Ensure the following structure is preserved:

```text
/
├── app.py                # Main application (Monolithic Streamlit)
├── requirements.txt      # Dependencies
├── config.py             # Configuration
├── models/               # ML Models
│   ├── predictor.py
│   └── saved/            # ⚠️ IMPORTANT: Upload your trained .pkl files here
├── data/                 # Data fetched
├── features/             # Feature Engineering
├── rules/                # Decision Rules
└── llm/                  # Explanation Logic
```

### 3. Models
The app needs trained models to work.
Running `python -m models.train` locally generates `.pkl` files in `models/saved/`.
**You must upload these `.pkl` files to the Space** manually (or via Git), or the app will show an error asking you to train them.

## ⚠️ Disclaimer
This tool is for **educational purposes only**. It does not constitute financial advice. The models provided here are for demonstration and research (Gym/Validation environment) and may not have predictive alpha in live markets.
# Stock-Price
