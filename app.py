"""
streamlit_app.py — Serene: Mental Health Support Chatbot
Premium "Calm Waters" UI with emotion-aware theming, RAG, and Supabase memory.
"""

import streamlit as st
import sys, os, time, uuid

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import APP_TITLE, APP_ICON, MAX_HISTORY_TURNS
from src.emotion import detect_emotion, check_crisis, get_emotion_ui, EMOTION_COLORS
from src.rag import retrieve_context
from src.inference import generate_response, load_model_and_tokenizer
from src.memory import (
    create_session, save_message, get_history,
    update_session_mood, flag_crisis, list_sessions,
)

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Serene — Wellness Companion",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Theme ──────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    :root {
        --bg-primary: #0a0f1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --accent: #6366f1;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --border: rgba(99, 102, 241, 0.15);
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
    }

    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0f172a 50%, #1e1b4b 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        background: var(--bg-card) !important;
        backdrop-filter: blur(12px);
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        animation: fadeSlideIn 0.4s ease-out;
    }
    .stChatMessage p {
        color: var(--text-primary) !important;
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
    }

    /* ── Chat Input ── */
    .stChatInput > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px);
    }
    .stChatInput input, .stChatInput textarea {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px var(--accent-glow) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px var(--accent-glow) !important;
    }

    /* ── Emotion Pill ── */
    .emotion-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        backdrop-filter: blur(8px);
        animation: fadeSlideIn 0.3s ease-out;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* ── Crisis Banner ── */
    .crisis-banner {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: pulse-border 2s ease-in-out infinite;
    }
    .crisis-banner h3 { color: #fca5a5 !important; margin: 0 0 0.5rem 0; }
    .crisis-banner p { color: #fecaca !important; margin: 0.25rem 0; font-size: 0.9rem; }

    /* ── Hero Header ── */
    .hero-section {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero-section h1 {
        background: linear-gradient(135deg, #a5b4fc, #6366f1, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-section p {
        color: var(--text-secondary);
        font-size: 0.95rem;
    }

    /* ── Session Card ── */
    .session-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .session-card:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .session-card .name { color: #e2e8f0; font-weight: 500; font-size: 0.85rem; }
    .session-card .mood { color: #94a3b8; font-size: 0.75rem; }

    /* ── Animations ── */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-border {
        0%, 100% { border-color: rgba(239,68,68,0.4); }
        50% { border-color: rgba(239,68,68,0.8); }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

    /* ── Status indicator ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Hide default streamlit elements ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ─── Session State Init ─────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "emotion_confidence" not in st.session_state:
    st.session_state.emotion_confidence = 0.0
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🌿</div>
        <h2 style="background: linear-gradient(135deg, #a5b4fc, #6366f1);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 0.5rem 0 0.25rem;">Serene</h2>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">AI Wellness Companion</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # New conversation button
    if st.button("✨ New Conversation", use_container_width=True):
        try:
            session_id = create_session(f"Session {time.strftime('%b %d, %H:%M')}")
            st.session_state.session_id = session_id
            st.session_state.messages = []
            st.session_state.current_emotion = "neutral"
            st.session_state.message_count = 0
            st.rerun()
        except Exception as e:
            st.error(f"Database error: {e}")

    st.markdown("---")

    # Current emotion display
    emo_ui = get_emotion_ui(st.session_state.current_emotion)
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{emo_ui['label']}</div>
        <div class="label">Current Detected Emotion</div>
        <div style="margin-top: 8px; height: 4px; border-radius: 2px;
                    background: rgba(99,102,241,0.15);">
            <div style="height: 100%; width: {st.session_state.emotion_confidence*100:.0f}%;
                        background: {emo_ui['accent']}; border-radius: 2px;
                        transition: width 0.5s ease;"></div>
        </div>
        <div class="label" style="margin-top: 4px;">
            Confidence: {st.session_state.emotion_confidence*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Session stats
    st.markdown(f"""
    <div class="metric-card">
        <div class="value" style="font-size: 1.2rem;">{st.session_state.message_count}</div>
        <div class="label">Messages This Session</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Recent sessions
    st.markdown("##### 📋 Recent Sessions")
    try:
        sessions = list_sessions(limit=8)
        if sessions:
            for sess in sessions:
                mood_emoji = {"joy": "😊", "sadness": "😔", "fear": "😰",
                             "anger": "😤", "neutral": "😐"}.get(sess.get("detected_mood", ""), "💬")
                name = sess.get("session_name", "Untitled")
                st.markdown(f"""
                <div class="session-card">
                    <div class="name">{mood_emoji} {name}</div>
                    <div class="mood">{sess.get('detected_mood', 'neutral').title()}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #64748b; font-size: 0.8rem;">No sessions yet. Start a new conversation!</p>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<p style="color: #64748b; font-size: 0.8rem;">Click "New Conversation" to begin</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Tech stack info
    st.markdown("""
    <div style="padding: 0.75rem; background: rgba(99,102,241,0.06);
                border-radius: 12px; border: 1px solid rgba(99,102,241,0.1);">
        <p style="color: #94a3b8; font-size: 0.7rem; margin: 0;">
            <strong style="color: #a5b4fc;">Tech Stack</strong><br>
            🧠 GPT-Neo 125M (LoRA Fine-tuned)<br>
            🔍 FAISS + Sentence-Transformers RAG<br>
            💾 Supabase PostgreSQL Memory<br>
            🎭 DistilRoBERTa Emotion Detection
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Area ──────────────────────────────────────────────

# Auto-create session if none exists
if st.session_state.session_id is None:
    try:
        session_id = create_session(f"Session {time.strftime('%b %d, %H:%M')}")
        st.session_state.session_id = session_id
    except Exception:
        st.session_state.session_id = str(uuid.uuid4())

# Hero section (shown when no messages)
if not st.session_state.messages:
    st.markdown("""
    <div class="hero-section">
        <h1>🌿 Welcome to Serene</h1>
        <p>Your AI-powered wellness companion. I'm here to listen, support,<br>
        and guide you with empathy and evidence-based wellness techniques.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(3)
    features = [
        ("🎭", "Emotion-Aware", "Real-time emotion detection adapts my responses to how you're feeling."),
        ("🔍", "Evidence-Based", "RAG-powered retrieval provides clinically-backed wellness guidance."),
        ("🔒", "Safe & Private", "Crisis detection with instant helpline resources. Your safety comes first."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="min-height: 160px;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="value" style="font-size: 1rem;">{title}</div>
                <div class="label" style="font-size: 0.8rem; margin-top: 8px; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p style="text-align:center; color: #64748b; font-size: 0.85rem;">💬 Type a message below to start our conversation</p>', unsafe_allow_html=True)

# Load models with a nice spinner
if not st.session_state.models_loaded:
    with st.spinner("🧠 Loading AI models... (first time only, ~30s)"):
        try:
            _, _, model_name = load_model_and_tokenizer()
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

# ─── Display Chat History ───────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🌿" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        # Show emotion pill for user messages
        if msg["role"] == "user" and msg.get("emotion"):
            emo_ui = get_emotion_ui(msg["emotion"])
            st.markdown(f"""
            <div class="emotion-pill" style="background: {emo_ui['bg']}; border: 1px solid {emo_ui['accent']}40; color: {emo_ui['accent']};">
                {emo_ui['label']} ({msg.get('confidence', 0)*100:.0f}%)
            </div>
            """, unsafe_allow_html=True)

        st.markdown(msg["content"])

        # Show crisis banner if flagged
        if msg.get("crisis"):
            st.markdown("""
            <div class="crisis-banner">
                <h3>🚨 You Are Not Alone</h3>
                <p>If you're in crisis, please reach out for help:</p>
                <p><strong>📞 988 Suicide & Crisis Lifeline</strong> — Call or text 988</p>
                <p><strong>💬 Crisis Text Line</strong> — Text HOME to 741741</p>
                <p><strong>🇵🇰 Umang Helpline</strong> — 0317-4288665</p>
                <p style="margin-top: 0.75rem; color: #fde68a;">You matter. Help is available 24/7. 💛</p>
            </div>
            """, unsafe_allow_html=True)

        # Show RAG context indicator
        if msg["role"] == "assistant" and msg.get("rag_used"):
            st.markdown(f"""
            <div style="margin-top: 8px; padding: 4px 10px; background: rgba(34,197,94,0.08);
                        border: 1px solid rgba(34,197,94,0.2); border-radius: 8px;
                        display: inline-flex; align-items: center; gap: 4px;">
                <span style="color: #22c55e; font-size: 0.7rem;">🔍 Evidence-based guidance included</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Chat Input & Processing ───────────────────────────────
if user_input := st.chat_input("Share what's on your mind..."):
    # 1. Detect emotion
    emotion, confidence = detect_emotion(user_input)
    st.session_state.current_emotion = emotion
    st.session_state.emotion_confidence = confidence

    # 2. Check for crisis
    is_crisis, trigger = check_crisis(user_input)

    # 3. Add user message to state
    user_msg = {
        "role": "user",
        "content": user_input,
        "emotion": emotion,
        "confidence": confidence,
        "crisis": is_crisis,
    }
    st.session_state.messages.append(user_msg)
    st.session_state.message_count += 1

    # 4. Display user message
    with st.chat_message("user", avatar="👤"):
        emo_ui = get_emotion_ui(emotion)
        st.markdown(f"""
        <div class="emotion-pill" style="background: {emo_ui['bg']}; border: 1px solid {emo_ui['accent']}40; color: {emo_ui['accent']};">
            {emo_ui['label']} ({confidence*100:.0f}%)
        </div>
        """, unsafe_allow_html=True)
        st.markdown(user_input)

        if is_crisis:
            st.markdown("""
            <div class="crisis-banner">
                <h3>🚨 You Are Not Alone</h3>
                <p>If you're in crisis, please reach out for help:</p>
                <p><strong>📞 988 Suicide & Crisis Lifeline</strong> — Call or text 988</p>
                <p><strong>💬 Crisis Text Line</strong> — Text HOME to 741741</p>
                <p><strong>🇵🇰 Umang Helpline</strong> — 0317-4288665</p>
                <p style="margin-top: 0.75rem; color: #fde68a;">You matter. Help is available 24/7. 💛</p>
            </div>
            """, unsafe_allow_html=True)

    # 5. Save user message to Supabase
    try:
        msg_id = save_message(
            st.session_state.session_id, "user", user_input,
            emotion, confidence, False
        )
        if is_crisis:
            flag_crisis(st.session_state.session_id, msg_id, trigger)
        update_session_mood(st.session_state.session_id, emotion)
    except Exception:
        pass  # Don't block UX if DB save fails

    # 6. Generate AI response
    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner(""):
            # Retrieve RAG context
            _, rag_context = retrieve_context(user_input)
            rag_used = bool(rag_context)

            # Get conversation history from Supabase
            try:
                history = get_history(st.session_state.session_id, MAX_HISTORY_TURNS)
            except Exception:
                history = [{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages[-MAX_HISTORY_TURNS:]]

            # Generate response
            response = generate_response(
                user_message=user_input,
                history=history,
                rag_context=rag_context,
                detected_emotion=emotion,
            )

        st.markdown(response)

        if rag_used:
            st.markdown(f"""
            <div style="margin-top: 8px; padding: 4px 10px; background: rgba(34,197,94,0.08);
                        border: 1px solid rgba(34,197,94,0.2); border-radius: 8px;
                        display: inline-flex; align-items: center; gap: 4px;">
                <span style="color: #22c55e; font-size: 0.7rem;">🔍 Evidence-based guidance included</span>
            </div>
            """, unsafe_allow_html=True)

    # 7. Save assistant message
    assistant_msg = {
        "role": "assistant",
        "content": response,
        "rag_used": rag_used,
    }
    st.session_state.messages.append(assistant_msg)
    st.session_state.message_count += 1

    try:
        save_message(
            st.session_state.session_id, "assistant", response,
            emotion, confidence, rag_used
        )
    except Exception:
        pass

    st.rerun()
