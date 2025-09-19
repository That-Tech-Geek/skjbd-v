import streamlit as st
import time
import os
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import requests
import tempfile
from functools import wraps
from google.api_core import exceptions
from pathlib import Path

# --- GOOGLE OAUTH LIBRARIES ---
try:
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
except ImportError:
    st.error("Required Google libraries not found! Please run: pip install google-auth-oauthlib google-api-python-client")
    st.stop()

# --- CONFIGURATION & CONSTANTS ---
MAX_FILES = 20
MAX_TOTAL_SIZE_MB = 150
MAX_AUDIO_SIZE_MB = 1024
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_RETRIES = 3
USER_DATA_DIR = Path("user_data")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Vekkam", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS ---
def load_css():
    st.markdown("""
        <style>
            .st-emotion-cache-18ni7ap, .st-emotion-cache-h4xjwg { display: none; }
            :root { --bg-primary: #0B1120; --bg-secondary: #1A233A; --text-primary: #E0E2E7; --text-secondary: #A0AEC0; --accent: #4A90E2; --border-color: #2D3748; }
            .stApp { background-color: var(--bg-primary); color: var(--text-primary); }
            div[data-testid="stButton"] > button { background-color: var(--accent); color: white; border: none; border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: bold; transition: transform 0.2s ease, opacity 0.2s ease; }
            div[data-testid="stButton"] > button:hover { transform: scale(1.05); opacity: 0.9; }
            .custom-card { background-color: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); }
            .ghost-partner { border: 2px dashed var(--border-color); width: 64px; height: 64px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: 200; color: var(--text-secondary); }
        </style>
    """, unsafe_allow_html=True)

# --- PERSISTENCE LAYER ---
def get_user_dir(user_id):
    user_dir = USER_DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def save_user_preferences(user_id, preferences):
    user_dir = get_user_dir(user_id)
    with open(user_dir / "preferences.json", "w") as f:
        json.dump(preferences, f)

def load_user_preferences(user_id):
    user_dir = get_user_dir(user_id)
    pref_file = user_dir / "preferences.json"
    if pref_file.exists():
        with open(pref_file, "r") as f:
            return json.load(f)
    return None

def save_session_data(user_id, session_name, data):
    sessions_dir = get_user_dir(user_id) / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    with open(sessions_dir / f"{session_name}.json", "w") as f:
        json.dump(data, f)

def list_user_sessions(user_id):
    sessions_dir = get_user_dir(user_id) / "sessions"
    if not sessions_dir.exists():
        return []
    return [f.stem for f in sessions_dir.glob("*.json")]

def load_session_data(user_id, session_name):
    session_file = get_user_dir(user_id) / "sessions" / f"{session_name}.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            return json.load(f)
    return None

# --- WORKFLOW FUNCTIONS ---
def gemini_api_call_with_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # ... (Unchanged)
        pass
def resilient_json_parser(json_string):
    # ... (Unchanged)
    pass
def chunk_text(text, source_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # ... (Unchanged)
    pass
def process_source(file, source_type):
    # ... (Unchanged)
    pass
@gemini_api_call_with_retry
def generate_content_outline(all_chunks, existing_outline=None):
    # ... (Unchanged)
    pass
@gemini_api_call_with_retry
def synthesize_note_block(topic, relevant_chunks_text, instructions, user_preferences):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
    style_guide = []
    detail = user_preferences.get('detail_level', 'Balanced')
    tone = user_preferences.get('tone', 'Neutral')
    if detail == 'Concise': style_guide.append("Be extremely concise and use bullet points.")
    elif detail == 'Detailed': style_guide.append("Be highly detailed, explanatory, and thorough.")
    if tone == 'Analogies & Examples': style_guide.append("Use simple analogies and real-world examples frequently.")
    elif tone == 'Formal & Academic': style_guide.append("Use formal, academic language.")
    
    style_prompt = f"Your writing style should be: {' '.join(style_guide)}" if style_guide else ""

    prompt = f"""
    Write the notes for a single topic: "{topic}".
    Use ONLY the provided source text. Adhere to the user's instructions.
    {style_prompt}
    Format in Markdown.
    **General Instructions:** {instructions if instructions else "None"}
    **Source Text:** {relevant_chunks_text}
    """
    response = model.generate_content(prompt)
    return response.text

@gemini_api_call_with_retry
def generate_lesson_plan(outline, all_chunks, user_preferences):
    # ... (Prompt now includes user_preferences)
    pass

# --- AUTHENTICATION & SESSION MANAGEMENT ---
def get_google_flow():
    # ... (Unchanged)
    pass

def reset_session():
    # ... (Modified to preserve more state)
    pass

# --- UI STATE FUNCTIONS ---
def show_pre_login_view():
    # ... (Unchanged)
    pass

def show_onboarding_view():
    st.markdown("""
        <div class="custom-card" style="max-width: 48rem; margin: 2rem auto; text-align: center;">
            <h2 style="font-size: 2.25rem; font-weight: 700;">First, let's personalize your experience.</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">How do you like to learn? This will help the AI tailor its output to your style.</p>
        </div>
    """, unsafe_allow_html=True)
    
    detail_level = st.select_slider(
        "My ideal study guide is...",
        options=['Concise', 'Balanced', 'Detailed'],
        value='Balanced'
    )
    
    tone = st.select_slider(
        "I learn best with...",
        options=['Formal & Academic', 'Neutral', 'Analogies & Examples'],
        value='Neutral'
    )

    if st.button("Save Preferences & Continue", type="primary"):
        preferences = {"detail_level": detail_level, "tone": tone}
        st.session_state.user_preferences = preferences
        save_user_preferences(st.session_state.user_info['id'], preferences)
        st.session_state.current_view = 'activation'
        st.rerun()

def show_activation_view():
    # ... (Unchanged)
    pass

def show_dashboard_view():
    st.header(f"Welcome back, {st.session_state.user_info['given_name']}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Your Workspace")
        if st.button("✨ Start New Synthesis Session"):
            reset_session()
            st.session_state.current_view = 'activation'
            st.rerun()
        
        st.subheader("Past Sessions")
        sessions = list_user_sessions(st.session_state.user_info['id'])
        if not sessions:
            st.write("Your past study guides will appear here.")
        else:
            for session_name in sessions:
                if st.button(session_name.replace("_", " ").title()):
                    loaded_data = load_session_data(st.session_state.user_info['id'], session_name)
                    st.session_state.final_notes = loaded_data.get('notes', [])
                    st.session_state.lesson_plan = loaded_data.get('lesson_plan', None)
                    st.session_state.current_view = 'results'
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # ... (Duo card as before)
        pass

def show_results_state():
    st.header("Your Unified Guide")
    if st.button("Return to Dashboard"):
        st.session_state.current_view = 'dashboard'
        st.rerun()

    # Save session on first view
    if not st.session_state.get('session_saved', False):
        first_topic = st.session_state.final_notes[0]['topic'] if st.session_state.final_notes else "Untitled"
        session_name = f"{time.strftime('%Y-%m-%d')}_{first_topic[:30]}"
        session_data = {
            "notes": st.session_state.final_notes,
            "lesson_plan": st.session_state.get('lesson_plan')
        }
        save_session_data(st.session_state.user_info['id'], session_name, session_data)
        st.session_state.session_saved = True
        st.toast(f"Session '{session_name}' saved!")

    # ... (Rest of results logic as before)
    pass

# --- MAIN APP ---
def main():
    load_css()
    
    if 'user_info' not in st.session_state: st.session_state.user_info = None

    try: genai.configure(api_key=st.secrets["gemini"]["api_key"])
    except (KeyError, FileNotFoundError): st.error("Gemini API key not configured in st.secrets."); st.stop()
    
    auth_code = st.query_params.get("code")
    if auth_code and not st.session_state.user_info:
        try:
            flow = get_google_flow()
            flow.fetch_token(code=auth_code)
            user_info = build('oauth2', 'v2', credentials=flow.credentials).userinfo().get().execute()
            st.session_state.user_info = user_info
            st.query_params.clear()
            
            # Load preferences or flag for onboarding
            prefs = load_user_preferences(user_info['id'])
            if prefs:
                st.session_state.user_preferences = prefs
                st.session_state.is_first_time_user = False
            else:
                st.session_state.is_first_time_user = True

            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}"); st.session_state.user_info = None
    
    if not st.session_state.user_info:
        show_pre_login_view()
        return

    # --- Post-Login View Router ---
    if 'current_view' not in st.session_state:
        if st.session_state.get('is_first_time_user', True):
             st.session_state.current_view = 'onboarding'
        else:
            st.session_state.current_view = 'dashboard'

    view_map = {
        'dashboard': show_dashboard_view,
        'onboarding': show_onboarding_view,
        'activation': show_activation_view,
        # ... other states
        'results': show_results_state,
    }
    render_view = view_map.get(st.session_state.current_view, show_dashboard_view)
    render_view()

if __name__ == "__main__":
    main()

