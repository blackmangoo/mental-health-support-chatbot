"""
memory.py — Supabase-backed conversation memory.

This module handles all database operations for storing and retrieving
multi-turn conversation history from the Supabase PostgreSQL database.

Tables used:
  - sessions: tracks unique chat sessions (id, session_name, detected_mood)
  - messages: stores all conversation turns (role, content, emotion, timestamp)
  - crisis_flags: logs any crisis-level triggers for auditing
"""

import uuid
from datetime import datetime
from typing import Optional
from supabase import create_client, Client
from src.config import SUPABASE_URL, SUPABASE_ANON_KEY


def get_client() -> Client:
    """Initialize and return the Supabase client."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in your .env file."
        )
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def create_session(session_name: str = "New Conversation") -> str:
    """
    Create a new chat session in Supabase and return its UUID.
    This is called once when the user starts a new conversation.
    """
    client = get_client()
    result = client.table("sessions").insert({
        "session_name": session_name,
        "detected_mood": "neutral",
    }).execute()
    session_id = result.data[0]["id"]
    return session_id


def update_session_mood(session_id: str, mood: str) -> None:
    """Update the dominant mood detected in a session."""
    client = get_client()
    client.table("sessions").update({
        "detected_mood": mood,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", session_id).execute()


def save_message(
    session_id: str,
    role: str,
    content: str,
    emotion: str = "neutral",
    emotion_confidence: float = 0.0,
    rag_context_used: bool = False,
) -> str:
    """
    Save a single message to the messages table.
    Returns the new message's UUID.
    """
    client = get_client()
    result = client.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "emotion": emotion,
        "emotion_confidence": round(emotion_confidence, 4),
        "rag_context_used": rag_context_used,
    }).execute()
    return result.data[0]["id"]


def get_history(session_id: str, limit: int = 8) -> list[dict]:
    """
    Fetch the last `limit` messages from a session, ordered chronologically.
    Returns a list of dicts with 'role' and 'content' keys.
    """
    client = get_client()
    result = (
        client.table("messages")
        .select("role, content, emotion, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data


def flag_crisis(session_id: str, message_id: str, trigger_phrase: str) -> None:
    """
    Log a crisis flag for a message. This is an audit trail for safety.
    In a production system, this could also trigger a notification.
    """
    client = get_client()
    client.table("crisis_flags").insert({
        "session_id": session_id,
        "message_id": message_id,
        "trigger_phrase": trigger_phrase,
    }).execute()


def list_sessions(limit: int = 20) -> list[dict]:
    """
    List recent sessions for the sidebar session history panel.
    Returns session id, name, mood, and creation time.
    """
    client = get_client()
    result = (
        client.table("sessions")
        .select("id, session_name, detected_mood, created_at")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data
