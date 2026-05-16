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
# 🧠 Mental Health Support Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

> A fine-tuned chatbot providing empathetic, supportive responses for emotional wellness.

## 🎯 Objective

Fine-tune a small language model on the EmpatheticDialogues dataset to create a chatbot that provides **supportive, empathetic responses** for stress, anxiety, and emotional wellness.

## 📚 What You'll Learn

- **Transfer learning & fine-tuning** (what it does to a model)
- **HuggingFace Transformers** ecosystem
- **Dataset preparation** for conversational AI
- **Training loop** (epochs, loss, learning rate)
- **LoRA/QLoRA** for efficient fine-tuning (less GPU needed)
- **Model evaluation** (perplexity, qualitative evaluation)
- **Ethical AI** considerations for mental health applications
- **Google Colab** for GPU training

## 🧠 Concepts to Revise Before Starting

| Concept | Resource |
|---------|----------|
| Transfer Learning | [StatQuest Video](https://www.youtube.com/watch?v=yofjFQddwHE) |
| Fine-Tuning LLMs | [HuggingFace Fine-Tuning Guide](https://huggingface.co/docs/transformers/training) |
| Tokenization | [HuggingFace Tokenizers](https://www.youtube.com/watch?v=VFp38yj8h3A) |
| LoRA | [LoRA Explained](https://www.youtube.com/watch?v=PXWYUTMt-AU) |
| Training Loop | [PyTorch Training Basics](https://www.youtube.com/watch?v=c36lUUr864M) |
| EmpatheticDialogues | [Paper](https://arxiv.org/abs/1811.00207) |
| Google Colab + GPU | [Colab Guide](https://colab.research.google.com/) |

## 📁 Project Structure

```
mental-health-support-chatbot/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── notebooks/
│   ├── fine_tuning.ipynb              ← Run on Google Colab (GPU)
│   └── evaluation.ipynb               ← Test & evaluate the model
├── src/
│   ├── __init__.py
│   ├── data_prep.py                   ← Prepare EmpatheticDialogues
│   ├── train.py                       ← Fine-tuning script
│   ├── inference.py                   ← Generate responses
│   ├── evaluate.py                    ← Evaluation metrics
│   └── config.py                      ← Model & training config
├── data/                               ← Processed dataset
├── models/                             ← Fine-tuned model weights
├── results/                            ← Training logs, plots
└── app/
    └── streamlit_app.py                ← Empathetic chat UI
```

## 🚀 Step-by-Step Implementation Guide

### Step 1: Setup Environment (Local)
```bash
conda create -n mental-health python=3.11 -y
conda activate mental-health
pip install -r requirements.txt
```

### Step 2: Setup Google Colab (for Training)
1. Open Google Colab
2. Go to Runtime → Change runtime type → T4 GPU
3. Install dependencies:
```python
!pip install transformers datasets accelerate peft torch
```
4. Mount Google Drive (to save model)

### Step 3: Dataset Preparation (`src/data_prep.py`)
- Load EmpatheticDialogues from HuggingFace datasets:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("empathetic_dialogues")
  ```
- Understand the dataset structure (context, utterance, emotion)
- Format into conversation pairs:
  - Input: "User: [context/utterance]"
  - Output: "Assistant: [empathetic response]"
- Tokenize using the model's tokenizer
- Split into train/validation sets

### Step 4: Model Selection & Configuration (`src/config.py`)
- Choose base model:
  - **DistilGPT2** (small, fast, good for learning) ← Start here
  - **GPT-Neo 125M** (slightly larger, better quality)
  - **Mistral-7B** (best quality, needs QLoRA — bonus)
- Define training hyperparameters:
  - Learning rate: 2e-5
  - Batch size: 8 (adjust for GPU memory)
  - Epochs: 3-5
  - Max length: 256 tokens

### Step 5: Fine-Tuning (`src/train.py` + `notebooks/fine_tuning.ipynb`)
- Load pre-trained model and tokenizer
- Apply LoRA if using larger model
- Use HuggingFace `Trainer` API:
  ```python
  from transformers import Trainer, TrainingArguments
  ```
- Track training loss per epoch
- Save model checkpoints
- Save final model to Google Drive → download locally

### Step 6: Evaluation (`src/evaluate.py` + `notebooks/evaluation.ipynb`)
- Quantitative:
  - Calculate perplexity on validation set
  - Compare before vs after fine-tuning
- Qualitative:
  - Test with emotional scenarios:
    - "I'm feeling really anxious about my exam"
    - "I had a terrible day at work"
    - "I feel like nobody understands me"
  - Rate responses on: empathy, relevance, safety
- Create a comparison table: base model vs fine-tuned

### Step 7: Inference (`src/inference.py`)
- Load fine-tuned model
- Generate responses with appropriate parameters:
  - temperature=0.7 (warm but not random)
  - top_p=0.9
  - repetition_penalty=1.2
- Add post-processing:
  - Ensure response isn't too long
  - Add safety disclaimer if discussing serious topics

### Step 8: Streamlit App (`app/streamlit_app.py`)
- Calming UI design:
  - Soft pastel colors (light blues, lavenders)
  - Rounded message bubbles
  - Gentle animations
  - Breathing exercise suggestion in sidebar
- Chat interface with message history
- Disclaimer banner: "This is not a replacement for professional help"
- Crisis hotline numbers displayed prominently
- Option to clear conversation

### Step 9: Training Visualization
- Plot training loss curve
- Plot validation loss curve
- Compare base vs fine-tuned response quality
- Save all plots to `results/`

### Step 10: Docker
```bash
docker build -t mental-health-chatbot .
docker run -p 8501:8501 mental-health-chatbot
```

## 🎯 Extra Challenges (Bonus Learning)

- [ ] Fine-tune with QLoRA on Mistral-7B for better responses
- [ ] Add emotion detection (classify user's emotion before responding)
- [ ] Implement conversation summarization
- [ ] Add guided meditation/breathing exercise feature
- [ ] Compare DistilGPT2 vs GPT-Neo fine-tuned results
- [ ] Create a training metrics dashboard

## 📊 Results

### Training Metrics
| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|-------------------|-------------------|
| Perplexity | - | - |
| Response Quality (1-5) | - | - |

### Response Examples
| User Input | Base Model Response | Fine-Tuned Response |
|-----------|-------------------|-------------------|
| "I'm stressed about exams" | - | - |
| "I feel lonely" | - | - |

### Screenshots
<!-- Add screenshots of your Streamlit app, training curves, etc. -->

## ⚠️ Ethical Considerations

> **IMPORTANT:** This chatbot is for educational/supportive purposes ONLY. It is NOT a substitute for professional mental health care.

- The bot should never attempt to diagnose mental health conditions
- Crisis resources should always be visible
- Users should be encouraged to seek professional help
- The model may generate inappropriate responses — always include safety disclaimers

### Crisis Resources
- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741
- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

## 🔗 Links

- **Base Model:** [DistilGPT2](https://huggingface.co/distilgpt2)
- **Dataset:** [EmpatheticDialogues](https://huggingface.co/datasets/empathetic_dialogues)
- **Internship:** DevelopersHub Corporation AI/ML Engineering

---
*Built as part of DevelopersHub Corporation AI/ML Engineering Internship*