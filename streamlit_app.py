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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Vekkam", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS TO INJECT THE UX/UI PHILOSOPHY ---
def load_css():
    st.markdown("""
        <style>
            /* Hide Streamlit's default header and footer */
            .st-emotion-cache-18ni7ap { display: none; }
            .st-emotion-cache-h4xjwg { display: none; }
            
            /* Theming variables */
            :root {
                --bg-primary: #0B1120;
                --bg-secondary: #1A233A;
                --text-primary: #E0E2E7;
                --text-secondary: #A0AEC0;
                --accent: #4A90E2;
                --border-color: #2D3748;
            }
            
            /* Main container styling */
            .stApp {
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }
            
            /* Custom button styling */
            div[data-testid="stButton"] > button {
                background-color: var(--accent);
                color: white;
                border: none;
                border-radius: 0.5rem;
                padding: 0.75rem 1.5rem;
                font-weight: bold;
                transition: transform 0.2s ease, opacity 0.2s ease;
            }
            div[data-testid="stButton"] > button:hover {
                transform: scale(1.05);
                opacity: 0.9;
            }
            
            /* Custom card styling */
            .custom-card {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 0.75rem;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            }
            
            /* Ghost partner card */
            .ghost-partner {
                border: 2px dashed var(--border-color);
                width: 64px;
                height: 64px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2rem;
                font-weight: 200;
                color: var(--text-secondary);
            }
        </style>
    """, unsafe_allow_html=True)

# --- WORKFLOW FUNCTIONS (UNCHANGED FROM PREVIOUS VERSION) ---
@gemini_api_call_with_retry
def generate_content_outline(all_chunks, existing_outline=None):
    # ... (function code is unchanged)
    pass
# ... (All other agentic and processing functions are here but omitted for brevity)
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
    instruction = "Analyze and create a structured outline."; prompt = f"""You are a curriculum designer. {instruction} For each topic, you MUST list the `chunk_id`s that are most relevant. Output ONLY a JSON object with a root key "outline", a list of objects. Each object must have keys "topic" (string) and "relevant_chunks" (list of strings). **Content Chunks:** --- {json.dumps(prompt_chunks, indent=2)}"""; response = model.generate_content(prompt); return resilient_json_parser(response.text)
@gemini_api_call_with_retry
def synthesize_note_block(topic, relevant_chunks_text, instructions):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite'); prompt = f"""Write the notes for a single topic: "{topic}". Use ONLY the provided source text. Adhere to the user's instructions. Format in Markdown. **Instructions:** {instructions if instructions else "None"} **Source Text:** {relevant_chunks_text}"""; response = model.generate_content(prompt); return response.text
@gemini_api_call_with_retry
def generate_lesson_plan(outline, all_chunks):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite'); chunk_context_map = {c['chunk_id']: c['text'][:200] + "..." for c in all_chunks}; prompt = f"""You are a world-class educator. Design a detailed, step-by-step lesson plan from the outline and source material. Goal is deep understanding. Use analogies. Define terms. For each topic, create "steps". Each step has "narration" and a list of "actions". Available actions: {{ "type": "write_text", "content": "Text", "position": "top_center|etc." }}, {{ "type": "draw_box", "label": "Label", "id": "unique_id" }}, {{ "type": "draw_arrow", "from_id": "id_1", "to_id": "id_2", "label": "Label" }}, {{ "type": "highlight", "target_id": "id_to_highlight" }}, {{ "type": "wipe_board" }}. Output ONLY a valid JSON object with a root key "lesson_plan". **User-Approved Outline:** {json.dumps(outline, indent=2)} **Source Content Context:** {json.dumps(chunk_context_map, indent=2)}"""; response = model.generate_content(prompt); return resilient_json_parser(response.text)

# --- AUTHENTICATION & SESSION MANAGEMENT ---
def get_google_flow():
    try:
        client_config = { "web": { "client_id": st.secrets["google"]["client_id"], "client_secret": st.secrets["google"]["client_secret"], "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": [st.secrets["google"]["redirect_uri"]] } }
        scopes = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets."); st.stop()

def reset_session():
    user_info = st.session_state.get('user_info')
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.current_view = 'dashboard'
    st.session_state.all_chunks = []; st.session_state.extraction_failures = []; st.session_state.outline_data = []; st.session_state.final_notes = []

# --- UI STATE FUNCTIONS ---
def show_pre_login_view():
    st.markdown("""
        <div style="min-height: 90vh; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
            <h1 style="font-size: 4.5rem; font-weight: 900; letter-spacing: -0.05em; margin-bottom: 1rem;">
                Turn Chaos into <span style="color: var(--accent);">Clarity</span>.
            </h1>
            <p style="font-size: 1.25rem; color: var(--text-secondary); max-width: 32rem; margin-bottom: 2rem;">
                Vekkam synthesizes your lectures, slides, and readings into a unified study guide. Faster, smarter, and finally, together.
            </p>
        </div>
    """, unsafe_allow_html=True)

def show_activation_view():
    st.markdown("""
        <div class="custom-card" style="max-width: 48rem; margin: 2rem auto; text-align: center;">
            <h2 style="font-size: 2.25rem; font-weight: 700;">Let's create your first guide.</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">The magic moment is less than 60 seconds away.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload your course materials (Audio, PDFs, Images)", accept_multiple_files=True)
    
    if st.button("Synthesize Guide", type="primary") and uploaded_files:
        st.session_state.initial_files = uploaded_files
        st.session_state.is_first_time_user = False # The magic moment has been triggered
        st.session_state.current_view = 'processing'
        st.rerun()

def show_dashboard_view():
    st.header(f"Welcome back, {st.session_state.user_info['given_name']}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Your Workspace")
        if st.button("✨ Start New Synthesis Session"):
            reset_session()
            st.session_state.current_view = 'activation' # Reuse activation view for uploads
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="custom-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
        st.subheader("Retention Habit: Daily Flashcard")
        st.info("Your daily quiz will appear here. Stay sharp.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Your Duo")
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(st.session_state.user_info['picture'], width=64)
        with c2:
            st.markdown('<div class="ghost-partner">?</div>', unsafe_allow_html=True)

        st.write("Vekkam is 10x better with a friend. Create shared 'Director's Cut' notes.")
        if st.button("Invite Your Duo"):
            st.success("Invite feature coming soon!")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="custom-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
        st.subheader("Your Progress")
        st.metric("Guides Created", "5")
        st.metric("Paired Sessions", "2")
        st.metric("Study Streak", "🔥 12 Days")
        st.markdown("</div>", unsafe_allow_html=True)

# ... (All the processing, workspace, and results states are needed for the full flow)
# For brevity, only the main router logic is shown, but these functions would be called.
def show_processing_state():
    # ...
    pass
def show_workspace_state():
    # ...
    pass
def show_results_state():
    # ...
    pass


# --- MAIN APP ---
def main():
    load_css()
    
    if 'user_info' not in st.session_state: st.session_state.user_info = None
    if 'is_first_time_user' not in st.session_state: st.session_state.is_first_time_user = True

    try: genai.configure(api_key=st.secrets["gemini"]["api_key"])
    except (KeyError, FileNotFoundError): st.error("Gemini API key not configured in st.secrets."); st.stop()
    
    flow = get_google_flow()
    auth_code = st.query_params.get("code")

    if auth_code and not st.session_state.user_info:
        try:
            flow.fetch_token(code=auth_code)
            user_info = build('oauth2', 'v2', credentials=flow.credentials).userinfo().get().execute()
            st.session_state.user_info = user_info
            st.query_params.clear(); st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}"); st.session_state.user_info = None
    
    if not st.session_state.user_info:
        show_pre_login_view()
        auth_url, _ = flow.authorization_url(prompt='consent')
        # A bit of a hack to center the button in the pre-login view
        _, col2, _ = st.columns([1,2,1])
        with col2:
            st.link_button("Sign In with Google to Begin", auth_url)
        return

    # --- Post-Login Sidebar ---
    user = st.session_state.user_info
    st.sidebar.image(user['picture'], width=60)
    st.sidebar.header(f"{user['given_name']}")
    if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()
    st.sidebar.divider()
    # (Sidebar content from previous versions can go here)

    # --- Post-Login View Router ---
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'activation' if st.session_state.is_first_time_user else 'dashboard'

    if st.session_state.current_view == 'dashboard':
        show_dashboard_view()
    elif st.session_state.current_view == 'activation':
        show_activation_view()
    elif st.session_state.current_view == 'processing':
        # This state would show a spinner and call the processing functions
        st.header("Synthesizing your guide...")
        st.spinner("This may take a moment...")
        # ... logic to call processing functions
    # ... other states like workspace, results etc.

if __name__ == "__main__":
    main()
