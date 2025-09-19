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
    is_first_time = st.session_state.get('is_first_time_user', True)
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.is_first_time_user = is_first_time
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
    auth_url, _ = get_google_flow().authorization_url(prompt='consent')
    _, col2, _ = st.columns([1,2,1])
    with col2: st.link_button("Sign In with Google to Begin", auth_url)

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
        st.session_state.is_first_time_user = False
        st.session_state.current_view = 'processing'
        st.rerun()

def show_dashboard_view():
    st.header(f"Welcome back, {st.session_state.user_info['given_name']}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Your Workspace")
        if st.button("✨ Start New Synthesis Session"):
            st.session_state.current_view = 'activation'
            st.rerun()
        # Placeholder for past guides
        st.write("Your past study guides will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Your Duo")
        c1, c2 = st.columns(2); c1.image(st.session_state.user_info['picture'], width=64); c2.markdown('<div class="ghost-partner">?</div>', unsafe_allow_html=True)
        st.write("Vekkam is 10x better with a friend.")
        if st.button("Invite Your Duo"): st.success("Invite feature coming soon!")
        st.markdown("</div>", unsafe_allow_html=True)

def show_processing_state():
    st.header("Synthesizing Your Guide...")
    with st.spinner("Extracting and analyzing content. This may take a moment..."):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_source, f, 'transcript' if f.type.startswith('audio/') else 'image' if f.type.startswith('image/') else 'pdf'): f for f in st.session_state.initial_files}
            for future in as_completed(futures): results.append(future.result())
        st.session_state.all_chunks = [c for r in results if r and r['status'] == 'success' for c in r['chunks']]
        st.session_state.extraction_failures = [r for r in results if r and r['status'] == 'error']
    
    with st.spinner("Generating initial outline..."):
        outline_json = generate_content_outline(st.session_state.all_chunks)
        if outline_json and "outline" in outline_json: st.session_state.outline_data = outline_json["outline"]
        else: st.error("Failed to generate an initial outline.")
    
    st.session_state.current_view = 'workspace'
    st.rerun()

def show_workspace_state():
    # This is a simplified version of the V7 workspace
    st.header("Vekkam Workspace")
    if st.button("Back to Dashboard"): st.session_state.current_view = 'dashboard'; st.rerun()
    if 'outline_data' in st.session_state and st.session_state.outline_data:
        initial_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
        st.session_state.editable_outline = st.text_area("Editable Outline:", value=initial_text, height=300)
        st.session_state.synthesis_instructions = st.text_area("Synthesis Instructions (Optional):", height=100)
        if st.button("Synthesize Notes", type="primary"):
            st.session_state.current_view = 'synthesizing'
            st.rerun()

def show_synthesizing_state():
    # This function would be implemented fully as per previous versions
    st.header("Finalizing your notes...")
    st.spinner("This will be quick...")
    # Simulate synthesis
    time.sleep(2)
    st.session_state.current_view = 'results'
    st.rerun()

def show_results_state():
    st.header("Your First Unified Guide!")
    st.success("Magic moment achieved! Here is your synthesized study guide.")
    st.markdown("---")
    # In a real flow, this would display the synthesized notes.
    st.write("This is where your notes, generated from the workspace, would appear.")
    st.markdown("---")
    if st.button("Return to Dashboard"):
        st.session_state.current_view = 'dashboard'
        st.rerun()

# --- MAIN APP ---
def main():
    load_css()
    
    if 'user_info' not in st.session_state: st.session_state.user_info = None
    if 'is_first_time_user' not in st.session_state: st.session_state.is_first_time_user = True

    try: genai.configure(api_key=st.secrets["gemini"]["api_key"])
    except (KeyError, FileNotFoundError): st.error("Gemini API key not configured in st.secrets."); st.stop()
    
    auth_code = st.query_params.get("code")
    if auth_code and not st.session_state.user_info:
        try:
            flow = get_google_flow()
            flow.fetch_token(code=auth_code)
            user_info = build('oauth2', 'v2', credentials=flow.credentials).userinfo().get().execute()
            st.session_state.user_info = user_info
            st.query_params.clear(); st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}"); st.session_state.user_info = None
    
    if not st.session_state.user_info:
        show_pre_login_view()
        return

    # --- Post-Login Sidebar ---
    user = st.session_state.user_info
    st.sidebar.image(user['picture'], width=60)
    st.sidebar.header(f"{user['given_name']}")
    if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()
    st.sidebar.divider()

    # --- Post-Login View Router ---
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'activation' if st.session_state.is_first_time_user else 'dashboard'

    view_map = {
        'dashboard': show_dashboard_view,
        'activation': show_activation_view,
        'processing': show_processing_state,
        'workspace': show_workspace_state,
        'synthesizing': show_synthesizing_state,
        'results': show_results_state,
    }
    render_view = view_map.get(st.session_state.current_view, show_dashboard_view)
    render_view()

if __name__ == "__main__":
    main()

