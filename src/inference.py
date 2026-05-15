"""
inference.py — Response generation engine.

This module handles:
1. Loading the fine-tuned GPT-Neo model from HuggingFace Hub
2. Building the prompt with conversation history + RAG context
3. Generating empathetic responses with proper parameters

The prompt template conditions the model to respond with warmth
and emotional intelligence, matching the EmpatheticDialogues training style.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from src.config import MODEL_ID, FALLBACK_MODEL_ID, GENERATION_CONFIG


SYSTEM_PERSONA = """You are Serene, a warm, empathetic, and supportive wellness companion.
Your role is to provide emotional support, active listening, and evidence-based wellness guidance.
You never diagnose conditions or prescribe medication. You always encourage professional help for serious issues.
You speak with genuine warmth, validation, and hope. You never dismiss feelings.
"""


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """
    Load the fine-tuned model from HuggingFace Hub.
    Falls back to base GPT-Neo 125M if the fine-tuned model is not yet available.
    Cached for the entire Streamlit session.
    """
    # Try loading the fine-tuned model first
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        model_name = MODEL_ID
    except Exception:
        # Fallback to base model if fine-tuned not available yet
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL_ID, torch_dtype=torch.float32)
        model_name = FALLBACK_MODEL_ID
    
    model.eval()
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, model_name


def build_prompt(
    user_message: str,
    history: list[dict],
    rag_context: str,
    detected_emotion: str,
) -> str:
    """
    Build the full prompt for the model, incorporating:
    - System persona
    - RAG context (if available)
    - Conversation history (last N turns from Supabase)
    - Detected emotion (to condition the response style)
    - The new user message
    """
    prompt_parts = [SYSTEM_PERSONA]
    
    # Inject RAG context if available
    if rag_context:
        prompt_parts.append(f"\n[Wellness Knowledge]\n{rag_context}\n")
    
    # Emotion conditioning
    emotion_hint = {
        "sadness": "The user seems sad or low. Respond with extra warmth, validation, and gentle hope.",
        "fear": "The user seems anxious or fearful. Respond with grounding, calmness, and reassurance.",
        "anger": "The user seems frustrated or angry. Acknowledge their feelings without judgment first.",
        "joy": "The user seems positive. Reinforce and celebrate with them.",
    }.get(detected_emotion, "")
    
    if emotion_hint:
        prompt_parts.append(f"[Context: {emotion_hint}]")
    
    # Conversation history
    prompt_parts.append("\n[Conversation]")
    for turn in history[-6:]:  # Last 6 turns for context window efficiency
        role_label = "User" if turn["role"] == "user" else "Serene"
        prompt_parts.append(f"{role_label}: {turn['content']}")
    
    # New user message
    prompt_parts.append(f"User: {user_message}")
    prompt_parts.append("Serene:")
    
    return "\n".join(prompt_parts)


def generate_response(
    user_message: str,
    history: list[dict],
    rag_context: str = "",
    detected_emotion: str = "neutral",
) -> str:
    """
    Generate an empathetic response using the fine-tuned GPT-Neo model.
    
    Args:
        user_message: The new user input.
        history: List of past messages from Supabase.
        rag_context: Retrieved wellness context string.
        detected_emotion: Detected emotion label from emotion.py.
    
    Returns:
        str: The generated response text (cleaned).
    """
    model, tokenizer, _ = load_model_and_tokenizer()
    
    prompt = build_prompt(user_message, history, rag_context, detected_emotion)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
            do_sample=GENERATION_CONFIG["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the newly generated tokens (not the prompt)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up the response
    response = response.split("User:")[0].strip()  # Stop at next user turn if any
    response = response.split("Serene:")[0].strip()  # Stop at repeated marker
    
    if not response:
        response = "I hear you, and I'm here with you. Would you like to share more about how you're feeling?"
    
    return response
