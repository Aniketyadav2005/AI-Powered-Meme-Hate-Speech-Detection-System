# 🛡️ Meme Hate Speech Detector — Upgraded

A powerful Streamlit web app that detects hate speech in memes using AI-powered image captioning, OCR text extraction, and a RoBERTa hate-speech classifier.

## ✨ New Features (v2)

| Feature | Description |
|---|---|
| **Confidence Scores** | Visual progress bars showing hate % and safe % for every image |
| **Multi-language OCR** | Detect text in English, French, German, Spanish, Hindi, Chinese, Arabic, Japanese, Korean |
| **Batch Upload** | Upload and analyse up to 20 images in one go with a progress bar |
| **Export to CSV** | Download results as a spreadsheet for further analysis |
| **Export to PDF** | Download a formatted PDF report with full analysis details |
| **Auto-blur Hateful Images** | Hateful memes are automatically blurred; original viewable on demand |
| **Red-highlighted Flagged Text** | OCR text from hateful memes is highlighted in red |
| **Upload History Dashboard** | Full session history with filter, metrics, and bulk export |
| **Dark Mode UI** | Polished dark-themed dashboard layout |

## 🧰 Technologies Used

- **Streamlit** — Web UI framework
- **Transformers (HuggingFace)** — BLIP captioning + RoBERTa hate-speech classifier
- **EasyOCR** — Multi-language text extraction
- **PyTorch** — Model inference backend
- **Pillow** — Image processing and blurring
- **ReportLab** — PDF report generation

## 📦 Installation

```sh
git clone https://github.com/your-repository.git
cd your-repository
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

```sh
streamlit run app.py
```

Open the local URL in your browser.

### Workflow
1. Go to **Analyse Images** tab
2. Select OCR languages from the sidebar (default: English)
3. Upload one or more meme images (JPG/PNG)
4. Click **Analyse All Images**
5. Review per-image results: caption, OCR text, confidence scores, verdict
6. Download **CSV** or **PDF** report
7. View session history in the **Upload History** tab

## 📁 File Structure

```
├── app.py              # Main application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🤖 Model Details

| Model | Purpose |
|---|---|
| [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) | Generates image captions |
| [facebook/roberta-hate-speech-dynabench-r4-target](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) | Classifies hate speech with confidence scores |
| EasyOCR | Extracts text from images (multi-language) |

## 📄 License

MIT License
