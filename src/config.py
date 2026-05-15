import os
from dotenv import load_dotenv

load_dotenv()

# =============================================
# Model Configuration
# =============================================
# This is the HuggingFace repo where your fine-tuned model will live.
# After you run the Colab notebook, the model will be pushed here.
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "mental-health-gptneo")
HF_TOKEN = os.getenv("HF_TOKEN")

# The HuggingFace Hub model path (username/model-name)
# IMPORTANT: Update HF_USERNAME after creating HF account
HF_USERNAME = "your-hf-username"  # <-- Update this after Colab training
MODEL_ID = f"{HF_USERNAME}/{HF_MODEL_REPO}"

# Fallback: If fine-tuned model not yet available, use this base model
FALLBACK_MODEL_ID = "EleutherAI/gpt-neo-125m"

# Generation parameters for empathetic responses
GENERATION_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.75,         # Warm, slightly creative
    "top_p": 0.92,               # Nucleus sampling
    "repetition_penalty": 1.3,   # Avoid repeating phrases
    "do_sample": True,
}

# =============================================
# Supabase Configuration
# =============================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# =============================================
# RAG Configuration
# =============================================
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Lightweight, fast, local
RAG_TOP_K = 3                               # Number of context chunks to retrieve

# =============================================
# Emotion Detection Configuration
# =============================================
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# Crisis keywords that trigger emergency protocol
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", "self harm",
    "self-harm", "hurt myself", "don't want to live", "no reason to live",
    "cutting myself", "overdose", "better off dead"
]

# =============================================
# App Configuration
# =============================================
APP_TITLE = "Serene — Your Wellness Companion"
APP_ICON = "🌿"
MAX_HISTORY_TURNS = 8      # How many past turns to feed into the model context
