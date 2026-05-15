"""
emotion.py — Real-time emotion detection from user messages.

Uses a pre-trained DistilRoBERTa model fine-tuned on the GoEmotions dataset.
This runs locally with no API key needed.

Model: j-hartmann/emotion-english-distilroberta-base
Labels: anger, disgust, fear, joy, neutral, sadness, surprise
"""

from transformers import pipeline
from src.config import EMOTION_MODEL_ID, CRISIS_KEYWORDS
import streamlit as st

# Emotion → UI color mapping for dynamic theming
EMOTION_COLORS = {
    "joy":      {"bg": "rgba(134, 239, 172, 0.15)", "accent": "#22c55e", "label": "😊 Feeling Good"},
    "sadness":  {"bg": "rgba(147, 197, 253, 0.15)", "accent": "#60a5fa", "label": "😔 Feeling Low"},
    "fear":     {"bg": "rgba(196, 181, 253, 0.15)", "accent": "#a78bfa", "label": "😰 Feeling Anxious"},
    "anger":    {"bg": "rgba(252, 165, 165, 0.15)", "accent": "#f87171", "label": "😤 Feeling Frustrated"},
    "disgust":  {"bg": "rgba(253, 186, 116, 0.15)", "accent": "#fb923c", "label": "😣 Feeling Overwhelmed"},
    "surprise": {"bg": "rgba(253, 224, 71, 0.15)",  "accent": "#facc15", "label": "😲 Feeling Surprised"},
    "neutral":  {"bg": "rgba(148, 163, 184, 0.15)", "accent": "#94a3b8", "label": "😐 Neutral"},
}

@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    """Load the emotion classifier once and cache it for the session."""
    return pipeline(
        "text-classification",
        model=EMOTION_MODEL_ID,
        return_all_scores=False,
        device=-1,  # CPU; set to 0 for GPU
    )


def detect_emotion(text: str) -> tuple[str, float]:
    """
    Detect the dominant emotion in a piece of text.
    
    Returns:
        tuple: (emotion_label, confidence_score)
    """
    classifier = load_emotion_classifier()
    result = classifier(text[:512])[0]  # Truncate to model max length
    label = result["label"].lower()
    score = round(result["score"], 4)
    return label, score


def check_crisis(text: str) -> tuple[bool, str]:
    """
    Check if the user's message contains any crisis-level keywords.
    
    Returns:
        tuple: (is_crisis: bool, matched_phrase: str)
    """
    text_lower = text.lower()
    for phrase in CRISIS_KEYWORDS:
        if phrase in text_lower:
            return True, phrase
    return False, ""


def get_emotion_ui(emotion: str) -> dict:
    """Return the UI theme dict for a given emotion label."""
    return EMOTION_COLORS.get(emotion, EMOTION_COLORS["neutral"])
