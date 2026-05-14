# AI-Powered-Meme-Hate-Speech-Detection-System

I recently developed an advanced AI-based Meme Hate Speech Detection System designed to identify and moderate harmful meme content using Natural Language Processing (NLP), Computer Vision, OCR, and Deep Learning technologies. The project focuses on solving real-world social media moderation challenges by automatically analysing meme images, extracting embedded text, understanding image context, and detecting hateful or toxic content with confidence scoring.

This application combines multiple AI techniques into a single intelligent moderation pipeline. The system first extracts text from uploaded meme images using multi-language OCR (Optical Character Recognition) powered by EasyOCR. It then generates contextual image captions using the BLIP Image Captioning model from Hugging Face Transformers. After combining the extracted OCR text and generated image caption, the system uses a RoBERTa-based Hate Speech Classification model to determine whether the meme contains hateful or harmful content.

The application supports multilingual OCR detection, allowing users to analyse memes written in multiple languages including English, Hindi, French, German, Spanish, Arabic, Japanese, Korean, and Chinese. This makes the system more practical for real-world global social media moderation use cases.

To improve usability and safety, the system automatically blurs harmful or hateful meme images while still allowing controlled access to the original image when required. The dashboard also highlights detected harmful text in red and provides visual confidence scores for both hateful and safe classifications using interactive progress indicators.

The project was built with a modern dark-themed interactive dashboard using Streamlit, enabling users to upload and analyse multiple meme images simultaneously through batch processing. The application includes an upload history dashboard where users can review previous analyses, filter hateful or safe content, monitor analytics, and export complete analysis reports.

Additionally, the system supports:
✅ Batch image analysis
✅ Multi-language OCR support
✅ AI-generated image captioning
✅ Hate speech classification with confidence scores
✅ Automatic harmful image blurring
✅ PDF report generation
✅ CSV export functionality
✅ Upload history tracking and analytics
✅ Interactive Streamlit dashboard UI

🛠️ Technologies & Tools Used
Python
Streamlit
Hugging Face Transformers
PyTorch
EasyOCR
NLP (Natural Language Processing)
Computer Vision
BLIP Image Captioning Model
RoBERTa Hate Speech Classifier
Pillow (Image Processing)
NumPy
ReportLab (PDF Generation)
🤖 AI Models Used
Salesforce/blip-image-captioning-base
facebook/roberta-hate-speech-dynabench-r4-target
EasyOCR
📌 Project Highlights

This project demonstrates the practical implementation of:

Deep Learning
AI-based Content Moderation
Multimodal AI Systems
NLP + Computer Vision Integration
Real-time AI Inference
Interactive AI Web Applications
Responsible AI & Safety Systems

Through this project, I gained hands-on experience in AI model integration, image processing, OCR pipelines, transformer-based NLP models, model inference optimization, report generation, and building production-style AI dashboards.

I am continuously exploring AI, Machine Learning, NLP, and Computer Vision technologies to build impactful real-world applications and improve my expertise in intelligent systems development.
