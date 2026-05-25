---
title: Serene Wellness Bot
emoji: 🌿
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: mit
---
# 🌿 Serene: AI Wellness Companion

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)
![Supabase](https://img.shields.io/badge/Supabase-Database-green?logo=supabase)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)

> **Live Demo:** [Serene Wellness Bot on Hugging Face Spaces](https://huggingface.co/spaces/BlackMangoo/serene-wellness-bot)

**Serene** is an empathetic, AI-powered mental health support chatbot built as part of the DevelopersHub Corporation AI/ML Engineering Internship. It leverages fine-tuned Large Language Models, real-time emotion detection, and Retrieval-Augmented Generation (RAG) to provide safe, context-aware, and evidence-based wellness guidance.

---

## ✨ Key Features

* **🧠 Fine-Tuned Empathy:** Powered by `GPT-Neo-125M`, fine-tuned using LoRA on the *EmpatheticDialogues* dataset to generate supportive and conversational responses.
* **🎭 Real-Time Emotion Detection:** Uses a local `DistilRoBERTa` sequence classification model to instantly analyze the user's input and dynamically categorize their emotional state (e.g., Neutral, Joy, Sadness, Anger).
* **📚 Evidence-Based RAG Pipeline:** Integrates a local FAISS vector database with `SentenceTransformers`. When distress is detected, it retrieves clinically-backed wellness exercises from a curated knowledge base and injects them into the model's context.
* **🛡️ Crisis Management System:** Employs regex-based safety checks to detect crisis phrases (e.g., suicide, self-harm). It instantly overrides standard generation to display a high-visibility banner with global crisis hotlines.
* **💾 Persistent Memory:** Uses **Supabase (PostgreSQL)** to securely log conversation sessions, user messages, AI responses, and detected emotions, allowing for seamless session continuation.
* **💎 Premium UI:** Built with Streamlit, featuring a calming glassmorphism aesthetic, real-time emotion badges, and a responsive sidebar.

---

## 🏗️ System Architecture & Advantages

### 1. Fine-Tuning with LoRA (Low-Rank Adaptation)
Instead of fine-tuning the entire GPT-Neo model (which requires massive GPU clusters), we used **LoRA/PEFT**. 
* **Advantage:** LoRA freezes the pre-trained weights and injects trainable rank decomposition matrices into each layer. This reduced trainable parameters by over 95%, allowing the model to be trained quickly and efficiently on a free Google Colab T4 GPU without sacrificing quality.

### 2. Retrieval-Augmented Generation (RAG)
LLMs are prone to hallucinating facts. To ensure the bot gives safe psychological advice, we implemented a RAG pipeline.
* **Advantage:** By embedding a curated `knowledge_base.json` using `FAISS` and `all-MiniLM-L6-v2`, the chatbot fetches exact coping strategies based on semantic similarity. The model acts as a friendly wrapper around hard, factual data rather than inventing its own medical advice.

### 3. Edge Emotion Detection
We process emotions locally in the same runtime using a compressed `DistilRoBERTa` model rather than calling external APIs.
* **Advantage:** Eliminates network latency, ensures 100% data privacy for the user's emotional state, and acts as a lightweight gatekeeper to trigger the RAG pipeline only when necessary.

### 4. Serverless Deployment (Hugging Face Spaces)
The application is hosted on Hugging Face Spaces using a Dockerized Streamlit environment.
* **Advantage:** Provides a free, globally accessible, and highly scalable GPU/CPU environment designed specifically for Machine Learning demos. It handles dependency management and CI/CD directly via Git.

---

## 📁 Project Structure

```text
mental-health-support-chatbot/
├── app.py                             ← Main Streamlit application & UI
├── requirements.txt                   ← Production dependencies
├── README.md                          ← Project documentation
├── .env                               ← Environment variables (Supabase, HF)
├── notebooks/
│   └── fine_tuning_colab.ipynb        ← Training script (GPT-Neo + LoRA)
├── src/
│   ├── inference.py                   ← Model loading & text generation logic
│   ├── emotion.py                     ← DistilRoBERTa emotion & crisis detection
│   ├── rag.py                         ← FAISS vector DB & context retrieval
│   ├── memory.py                      ← Supabase DB connection & CRUD operations
│   └── config.py                      ← Centralized hyperparameters
└── data/
    └── knowledge_base.json            ← Curated wellness tips for RAG
```

---

## 🚀 Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/blackmangoo/mental-health-support-chatbot.git
   cd mental-health-support-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Create a `.env` file in the root directory:
   ```env
   SUPABASE_URL="your_supabase_url"
   SUPABASE_ANON_KEY="your_supabase_anon_key"
   ```

4. **Launch the App:**
   ```bash
   streamlit run app.py
   ```

---

## ⚠️ Ethical Considerations

> **IMPORTANT:** This chatbot is an experimental AI project for educational purposes. It is **NOT** a substitute for professional mental health care, therapy, or medical diagnosis.

* The bot is programmed to detect high-risk keywords and explicitly redirect users to human professionals.
* RAG content is sourced from general wellness guidelines (mindfulness, grounding techniques) rather than clinical interventions.

### Crisis Resources
* **National Suicide Prevention Lifeline (US):** 988
* **Crisis Text Line:** Text HOME to 741741
* **International Resources:** [Find A Helpline](https://findahelpline.com/)

---
*Built as part of DevelopersHub Corporation AI/ML Engineering Internship*