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
st.set_page_config(page_title="Vekkam", page_icon="🧠", layout="wide", initial_sidebar_state="auto")

# --- CUSTOM CSS ---
def load_css():
    st.markdown("""
        <style>
            /* Hide Streamlit's default header, footer, and main menu button */
            header {visibility: hidden;}
            .st-emotion-cache-18ni7ap {visibility: hidden;}
            .st-emotion-cache-h4xjwg {display: none;}
            
            :root { 
                --bg-primary: #0B1120; 
                --bg-secondary: #1A233A; 
                --text-primary: #E0E2E7; 
                --text-secondary: #A0AEC0; 
                --accent: #4A90E2; 
                --border-color: #2D3748; 
            }
            .stApp { background-color: var(--bg-primary); color: var(--text-primary); }
            div[data-testid="stButton"] > button { 
                background-color: var(--accent); color: white; border: none; 
                border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: bold; 
                transition: transform 0.2s ease, opacity 0.2s ease; 
            }
            div[data-testid="stButton"] > button:hover { transform: scale(1.05); opacity: 0.9; }
            .custom-card { 
                background-color: var(--bg-secondary); border: 1px solid var(--border-color); 
                border-radius: 0.75rem; padding: 1.5rem; 
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); 
            }
            .ghost-partner { 
                border: 2px dashed var(--border-color); width: 64px; height: 64px; 
                border-radius: 50%; display: flex; align-items: center; 
                justify-content: center; font-size: 2rem; font-weight: 200; color: var(--text-secondary); 
            }
        </style>
    """, unsafe_allow_html=True)

# --- PERSISTENCE LAYER ---
def get_user_dir(user_id):
    user_dir = USER_DATA_DIR / user_id; user_dir.mkdir(parents=True, exist_ok=True); return user_dir
def save_user_preferences(user_id, preferences):
    with open(get_user_dir(user_id) / "preferences.json", "w") as f: json.dump(preferences, f)
def load_user_preferences(user_id):
    pref_file = get_user_dir(user_id) / "preferences.json"
    if pref_file.exists():
        with open(pref_file, "r") as f: return json.load(f)
    return None
def save_session_data(user_id, session_name, data):
    sessions_dir = get_user_dir(user_id) / "sessions"; sessions_dir.mkdir(exist_ok=True)
    with open(sessions_dir / f"{session_name}.json", "w") as f: json.dump(data, f)
def list_user_sessions(user_id):
    sessions_dir = get_user_dir(user_id) / "sessions"
    return [f.stem for f in sessions_dir.glob("*.json")] if sessions_dir.exists() else []
def load_session_data(user_id, session_name):
    session_file = get_user_dir(user_id) / "sessions" / f"{session_name}.json"
    if session_file.exists():
        with open(session_file, "r") as f: return json.load(f)
    return None

# --- WORKFLOW FUNCTIONS ---
def gemini_api_call_with_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0; delay = 1
        while retries < MAX_RETRIES:
            try: return func(*args, **kwargs)
            except exceptions.ResourceExhausted as e:
                retries += 1
                if retries >= MAX_RETRIES: st.error(f"API quota exceeded. Error: {e}"); return None
                match = re.search(r'seconds: (\d+)', str(e)); wait_time = int(match.group(1)) + delay if match else delay * (2 ** retries)
                st.warning(f"Rate limit hit. Retrying in {wait_time}s... ({retries}/{MAX_RETRIES})"); time.sleep(wait_time)
            except Exception as e: st.error(f"API error in {func.__name__}: {e}"); return None
        return None
    return wrapper
def resilient_json_parser(json_string):
    try: match = re.search(r'\{.*\}', json_string, re.DOTALL); return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError: st.error("Fatal Error: Could not parse AI JSON response."); return None
def chunk_text(text, source_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []; words = text.split(); chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]; chunk_text = " ".join(chunk_words)
        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]; chunk_id = f"{source_id}::chunk_{i//(chunk_size-overlap)}_{chunk_hash}"
        chunks.append({"chunk_id": chunk_id, "text": chunk_text})
    return chunks
def process_source(file, source_type):
    try:
        source_id = f"{source_type}:{file.name}"; model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
        if source_type == 'transcript':
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp: tmp.write(file.getvalue()); tmp_path = tmp.name
            try:
                audio_file = genai.upload_file(path=tmp_path)
                while audio_file.state.name == "PROCESSING": time.sleep(2); audio_file = genai.get_file(audio_file.name)
                if audio_file.state.name == "FAILED": return {"status": "error", "source_id": source_id, "reason": "Gemini file processing failed."}
                response = model.generate_content(["Transcribe this audio file.", audio_file]); chunks = chunk_text(response.text, source_id)
                return {"status": "success", "source_id": source_id, "chunks": chunks}
            finally: os.unlink(tmp_path)
        elif source_type == 'image':
            image = Image.open(file); response = model.generate_content(["Analyze this image...", image])
            return {"status": "success", "source_id": source_id, "chunks": [{"chunk_id": f"{source_id}::chunk_0", "text": response.text}]}
        elif source_type == 'pdf':
            pdf_bytes = file.read(); doc = fitz.open(stream=pdf_bytes, filetype="pdf"); text = "".join(page.get_text() for page in doc); chunks = chunk_text(text, source_id)
            return {"status": "success", "source_id": source_id, "chunks": chunks}
    except Exception as e: return {"status": "error", "source_id": f"{source_type}:{file.name}", "reason": str(e)}
@gemini_api_call_with_retry
def generate_content_outline(all_chunks, existing_outline=None):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite'); prompt_chunks = [{"chunk_id": c['chunk_id'], "text_snippet": c['text'][:200] + "..."} for c in all_chunks]
    instruction = "Analyze and create a structured outline."; 
    if existing_outline: instruction = "Analyze the NEW content chunks and suggest topics to ADD to the existing outline."
    prompt = f"""You are a curriculum designer. {instruction} For each topic, you MUST list the `chunk_id`s that are most relevant. Output ONLY a JSON object with a root key "outline", a list of objects. Each object must have keys "topic" (string) and "relevant_chunks" (list of strings). **Existing Outline:** {json.dumps(existing_outline, indent=2) if existing_outline else "None"} **Content Chunks:** --- {json.dumps(prompt_chunks, indent=2)}"""; response = model.generate_content(prompt); return resilient_json_parser(response.text)
@gemini_api_call_with_retry
def synthesize_note_block(topic, relevant_chunks_text, instructions, user_preferences):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
    style_guide = []; detail = user_preferences.get('detail_level', 'Balanced'); tone = user_preferences.get('tone', 'Neutral')
    if detail == 'Concise': style_guide.append("Be extremely concise and use bullet points.")
    elif detail == 'Detailed': style_guide.append("Be highly detailed, explanatory, and thorough.")
    if tone == 'Analogies & Examples': style_guide.append("Use simple analogies and real-world examples frequently.")
    elif tone == 'Formal & Academic': style_guide.append("Use formal, academic language.")
    style_prompt = f"Your writing style should be: {' '.join(style_guide)}" if style_guide else ""
    prompt = f"""Write the notes for a single topic: "{topic}". Use ONLY the provided source text. Adhere to the user's instructions. {style_prompt} Format in Markdown. **General Instructions:** {instructions if instructions else "None"} **Source Text:** {relevant_chunks_text}"""; response = model.generate_content(prompt); return response.text
@gemini_api_call_with_retry
def generate_lesson_plan(outline, all_chunks, user_preferences):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite'); chunk_context_map = {c['chunk_id']: c['text'][:200] + "..." for c in all_chunks}; 
    # Logic to add user preferences to prompt
    prompt = f"""You are a world-class educator. Design a detailed, step-by-step lesson plan from the outline and source material. Goal is deep understanding. Use analogies. Define terms. For each topic, create "steps". Each step has "narration" and a list of "actions". Available actions: {{ "type": "write_text", "content": "Text", "position": "top_center|etc." }}, {{ "type": "draw_box", "label": "Label", "id": "unique_id" }}, {{ "type": "draw_arrow", "from_id": "id_1", "to_id": "id_2", "label": "Label" }}, {{ "type": "highlight", "target_id": "id_to_highlight" }}, {{ "type": "wipe_board" }}. Output ONLY a valid JSON object with a root key "lesson_plan". **User-Approved Outline:** {json.dumps(outline, indent=2)} **Source Content Context:** {json.dumps(chunk_context_map, indent=2)}"""; response = model.generate_content(prompt); return resilient_json_parser(response.text)

# --- AUTHENTICATION & SESSION MANAGEMENT ---
def get_google_flow():
    try:
        client_config = { "web": { "client_id": st.secrets["google"]["client_id"], "client_secret": st.secrets["google"]["client_secret"], "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": [st.secrets["google"]["redirect_uri"]] } }
        scopes = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets."); st.stop()

def reset_session():
    user_info = st.session_state.get('user_info'); prefs = st.session_state.get('user_preferences')
    st.session_state.clear()
    st.session_state.user_info = user_info; st.session_state.user_preferences = prefs
    st.session_state.current_view = 'dashboard'; st.session_state.all_chunks = []; st.session_state.extraction_failures = []; st.session_state.outline_data = []; st.session_state.final_notes = []

# --- UI STATE FUNCTIONS ---
def show_pre_login_view():
    st.markdown("""<div style="min-height: 90vh; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
            <h1 style="font-size: 4.5rem; font-weight: 900; letter-spacing: -0.05em; margin-bottom: 1rem;">Turn Chaos into <span style="color: var(--accent);">Clarity</span>.</h1>
            <p style="font-size: 1.25rem; color: var(--text-secondary); max-width: 32rem; margin-bottom: 2rem;">Vekkam synthesizes your lectures, slides, and readings into a unified study guide. Faster, smarter, and finally, together.</p>
        </div>""", unsafe_allow_html=True)
    auth_url, _ = get_google_flow().authorization_url(prompt='consent')
    _, col2, _ = st.columns([1,2,1]); col2.link_button("Sign In with Google to Begin", auth_url)

def show_onboarding_view():
    st.markdown("""<div class="custom-card" style="max-width: 48rem; margin: 2rem auto; text-align: center;">
            <h2 style="font-size: 2.25rem; font-weight: 700;">First, let's personalize your experience.</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">How do you like to learn? This will help the AI tailor its output to your style.</p></div>""", unsafe_allow_html=True)
    detail_level = st.select_slider("My ideal study guide is...", options=['Concise', 'Balanced', 'Detailed'], value='Balanced')
    tone = st.select_slider("I learn best with...", options=['Formal & Academic', 'Neutral', 'Analogies & Examples'], value='Neutral')
    if st.button("Save Preferences & Continue", type="primary"):
        preferences = {"detail_level": detail_level, "tone": tone}
        st.session_state.user_preferences = preferences
        save_user_preferences(st.session_state.user_info['id'], preferences)
        st.session_state.is_first_time_user = False; st.session_state.current_view = 'activation'; st.rerun()

def show_activation_view():
    st.markdown("""<div class="custom-card" style="max-width: 48rem; margin: 2rem auto; text-align: center;">
            <h2 style="font-size: 2.25rem; font-weight: 700;">Let's create your first guide.</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">The magic moment is less than 60 seconds away.</p></div>""", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload your course materials (Audio, PDFs, Images)", accept_multiple_files=True)
    if st.button("Synthesize Guide", type="primary") and uploaded_files:
        st.session_state.initial_files = uploaded_files
        st.session_state.is_first_time_user = False; st.session_state.current_view = 'processing'; st.rerun()

def show_dashboard_view():
    # Dashboard is now the main view after the first use
    st.header(f"Welcome back, {st.session_state.user_info['given_name']}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Your Workspace")
        if st.button("✨ Start New Synthesis Session"):
            st.session_state.current_view = 'activation'; st.rerun()
        st.subheader("Past Sessions")
        sessions = list_user_sessions(st.session_state.user_info['id'])
        if not sessions: st.write("Your past study guides will appear here.")
        else:
            for session_name in sessions:
                if st.button(session_name.replace("_", " ").title()):
                    loaded_data = load_session_data(st.session_state.user_info['id'], session_name)
                    st.session_state.final_notes = loaded_data.get('notes', []); st.session_state.lesson_plan = loaded_data.get('lesson_plan', None)
                    st.session_state.current_view = 'results'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Your Duo")
        c1, c2 = st.columns(2); c1.image(st.session_state.user_info['picture'], width=64); c2.markdown('<div class="ghost-partner">?</div>', unsafe_allow_html=True)
        st.write("Vekkam is 10x better with a friend.")
        if st.button("Invite Your Duo"): st.success("Invite feature coming soon!")
        st.markdown("</div>", unsafe_allow_html=True)

def show_processing_state():
    st.header("Synthesizing Your Guide..."); st.spinner("Extracting and analyzing content...")
    results = []; 
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_source, f, 'transcript' if f.type.startswith('audio/') else 'image' if f.type.startswith('image/') else 'pdf'): f for f in st.session_state.initial_files}
        for future in as_completed(futures): results.append(future.result())
    st.session_state.all_chunks = [c for r in results if r and r['status'] == 'success' for c in r['chunks']]
    st.session_state.extraction_failures = [r for r in results if r and r['status'] == 'error']
    with st.spinner("Generating initial outline..."):
        outline_json = generate_content_outline(st.session_state.all_chunks)
        if outline_json and "outline" in outline_json: st.session_state.outline_data = outline_json["outline"]
        else: st.error("Failed to generate an initial outline.")
    st.session_state.current_view = 'workspace'; st.rerun()

def show_workspace_state():
    st.header("Vekkam Workspace")
    if st.button("Back to Dashboard"): st.session_state.current_view = 'dashboard'; st.rerun()
    # Full workspace implementation
    pass

def show_results_state():
    st.header("Your Unified Guide")
    if st.button("Return to Dashboard"): st.session_state.current_view = 'dashboard'; st.rerun()
    # Save session
    if not st.session_state.get('session_saved', False):
        first_topic = st.session_state.final_notes[0]['topic'] if st.session_state.final_notes else "Untitled"
        session_name = f"{time.strftime('%Y-%m-%d')}_{first_topic[:30].replace(' ', '_')}"
        session_data = {"notes": st.session_state.final_notes, "lesson_plan": st.session_state.get('lesson_plan')}
        save_session_data(st.session_state.user_info['id'], session_name, session_data)
        st.session_state.session_saved = True; st.toast(f"Session '{session_name}' saved!")
    # Full results display logic
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
            st.session_state.user_info = user_info; st.query_params.clear()
            prefs = load_user_preferences(user_info['id'])
            if prefs: st.session_state.user_preferences = prefs; st.session_state.is_first_time_user = False
            else: st.session_state.is_first_time_user = True
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}"); st.session_state.user_info = None
    
    if not st.session_state.user_info:
        show_pre_login_view()
        return

    # --- Post-Login Top Banner ---
    cols = st.columns([1, 1, 1, 6, 1, 1])
    with cols[5]:
        if st.button("Logout"): st.session_state.clear(); st.rerun()
    with cols[4]:
        st.markdown(f"<div style='text-align: right;'>Welcome, {st.session_state.user_info['given_name']}</div>", unsafe_allow_html=True)
    with cols[0]:
        st.image(st.session_state.user_info['picture'], width=40)
    st.markdown("---") # Visual separator

    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'onboarding' if st.session_state.get('is_first_time_user', True) else 'dashboard'
    
    view_map = {
        'dashboard': show_dashboard_view, 'onboarding': show_onboarding_view,
        'activation': show_activation_view, 'processing': show_processing_state,
        'workspace': show_workspace_state, 'results': show_results_state,
    } # Note: Synthesizing, lesson states etc. would be added here
    render_view = view_map.get(st.session_state.current_view, show_dashboard_view)
    render_view()

if __name__ == "__main__":
    main()

