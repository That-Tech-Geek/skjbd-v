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
st.set_page_config(page_title="Vekkam Engine", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

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
            div[data-testid="stButton"] > button, .stDownloadButton > button, .stLinkButton > a { 
                background-color: var(--accent); color: white !important; border: none; 
                border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: bold; 
                transition: transform 0.2s ease, opacity 0.2s ease; 
                text-decoration: none;
            }
            div[data-testid="stButton"] > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover { 
                transform: scale(1.05); opacity: 0.9;
            }
            .custom-card { 
                background-color: var(--bg-secondary); border: 1px solid var(--border-color); 
                border-radius: 0.75rem; padding: 1.5rem; 
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); 
            }
        </style>
    """, unsafe_allow_html=True)


# --- EXPONENTIAL BACKOFF DECORATOR ---
def gemini_api_call_with_retry(func):
    """Decorator to handle Gemini API rate limiting with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        delay = 1
        while retries < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except exceptions.ResourceExhausted as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    st.error(f"API quota exceeded after multiple retries. Please check your Google Cloud billing plan. Error: {e}")
                    return None
                
                match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', str(e))
                if match:
                    wait_time = int(match.group(1)) + delay
                else:
                    wait_time = delay * (2 ** retries)

                st.warning(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
            except Exception as e:
                st.error(f"An unexpected API error occurred in {func.__name__}: {e}")
                return None
        return None
    return wrapper

# --- API SELF-DIAGNOSIS & UTILITIES ---
def resilient_json_parser(json_string):
    try:
        match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if match: return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        st.error("Fatal Error: Could not parse a critical AI JSON response."); return None

def chunk_text(text, source_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        chunk_id = f"{source_id}::chunk_{i//(chunk_size-overlap)}_{chunk_hash}"
        chunks.append({"chunk_id": chunk_id, "text": chunk_text})
    return chunks

# --- CONTENT PROCESSING ---
def process_source(file, source_type):
    try:
        source_id = f"{source_type}:{file.name}"
        model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
        if source_type == 'transcript':
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            try:
                audio_file = genai.upload_file(path=tmp_path)
                while audio_file.state.name == "PROCESSING":
                    time.sleep(2)
                    audio_file = genai.get_file(audio_file.name)
                if audio_file.state.name == "FAILED":
                    return {"status": "error", "source_id": source_id, "reason": "Gemini file processing failed."}
                response = model.generate_content(["Transcribe this audio file.", audio_file])
                chunks = chunk_text(response.text, source_id)
                return {"status": "success", "source_id": source_id, "chunks": chunks}
            finally:
                os.unlink(tmp_path)
        elif source_type == 'image':
            image = Image.open(file)
            response = model.generate_content(["Analyze this image...", image])
            return {"status": "success", "source_id": source_id, "chunks": [{"chunk_id": f"{source_id}::chunk_0", "text": response.text}]}
        elif source_type == 'pdf':
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            chunks = chunk_text(text, source_id)
            return {"status": "success", "source_id": source_id, "chunks": chunks}
    except Exception as e:
        return {"status": "error", "source_id": f"{source_type}:{file.name}", "reason": str(e)}

# --- AGENTIC WORKFLOW FUNCTIONS ---
@gemini_api_call_with_retry
def generate_content_outline(all_chunks, existing_outline=None):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
    prompt_chunks = [{"chunk_id": c['chunk_id'], "text_snippet": c['text'][:200] + "..."} for c in all_chunks]
    instruction = "Analyze the content chunks and create a structured outline."
    if existing_outline:
        instruction = "Analyze the NEW content chunks and suggest topics to ADD to the existing outline."
    prompt = f"""
    You are a curriculum designer. {instruction}
    For each topic, you MUST list the `chunk_id`s that are most relevant.
    Output ONLY a JSON object with a root key "outline", a list of objects. Each object must have keys "topic" (string) and "relevant_chunks" (list of strings).
    **Existing Outline (for context):**
    {json.dumps(existing_outline, indent=2) if existing_outline else "None"}
    **Content Chunks:**
    ---
    {json.dumps(prompt_chunks, indent=2)}
    """
    response = model.generate_content(prompt)
    return resilient_json_parser(response.text)

@gemini_api_call_with_retry
def synthesize_note_block(topic, relevant_chunks_text, instructions):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
    prompt = f"""
    Write the notes for a single topic: "{topic}".
    Use ONLY the provided source text. Adhere to the user's instructions. Format in Markdown.
    **Instructions:** {instructions if instructions else "None"}
    **Source Text:** {relevant_chunks_text}
    """
    response = model.generate_content(prompt)
    return response.text

@gemini_api_call_with_retry
def generate_lesson_plan(outline, all_chunks):
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
    chunk_context_map = {c['chunk_id']: c['text'][:200] + "..." for c in all_chunks}
    prompt = f"""
    You are a world-class educator with the explanatory power of Sal Khan and the visual clarity of Kurzgesagt. 
    Design a detailed, step-by-step lesson plan based on the provided outline and source material.
    The goal is deep, intuitive understanding. Build from first principles. Use analogies. Define all terms.
    For each topic in the outline, create a list of "steps". Each step must have "narration" and a list of "actions".
    Available actions:
    - {{ "type": "write_text", "content": "Text to write", "position": "top_center|middle_left|etc." }}
    - {{ "type": "draw_box", "label": "Box Label", "id": "unique_id_for_this_box" }}
    - {{ "type": "draw_arrow", "from_id": "box_id_1", "to_id": "box_id_2", "label": "Arrow Label" }}
    - {{ "type": "highlight", "target_id": "box_or_text_id_to_highlight" }}
    - {{ "type": "wipe_board" }}
    Output ONLY a valid JSON object with a root key "lesson_plan".

    **User-Approved Outline:**
    {json.dumps(outline, indent=2)}

    **Source Content Context:**
    {json.dumps(chunk_context_map, indent=2)}
    """
    response = model.generate_content(prompt)
    return resilient_json_parser(response.text)

# --- AUTHENTICATION & SESSION MANAGEMENT ---
def get_google_flow():
    try:
        client_config = {
            "web": { "client_id": st.secrets["google"]["client_id"], "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [st.secrets["google"]["redirect_uri"]],
            }}
        scopes = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets."); st.stop()

def reset_session(tool_choice):
    user_info = st.session_state.get('user_info')
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.tool_choice = tool_choice
    st.session_state.current_state = 'upload'
    st.session_state.all_chunks = []
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []

# --- UI STATE FUNCTIONS ---
def show_pre_login_view(flow):
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
    auth_url, _ = flow.authorization_url(prompt='consent')
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.link_button("Sign In with Google to Begin", auth_url)

def show_upload_state():
    st.header("Note & Lesson Engine: Upload")
    uploaded_files = st.file_uploader("Select files", accept_multiple_files=True, type=['mp3', 'm4a', 'wav', 'png', 'jpg', 'pdf'])
    if st.button("Process Files", type="primary") and uploaded_files:
        st.session_state.initial_files = uploaded_files
        st.session_state.current_state = 'processing'
        st.rerun()

def show_processing_state():
    st.header("Initial Processing...")
    with st.spinner("Extracting content from all files..."):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_source, f, 'transcript' if f.type.startswith('audio/') else 'image' if f.type.startswith('image/') else 'pdf'): f for f in st.session_state.initial_files}
            for future in as_completed(futures): results.append(future.result())
        st.session_state.all_chunks.extend([c for r in results if r and r['status'] == 'success' for c in r['chunks']])
        st.session_state.extraction_failures.extend([r for r in results if r and r['status'] == 'error'])
    st.session_state.current_state = 'workspace'
    st.rerun()

def show_workspace_state():
    st.header("Vekkam Workspace")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Controls & Outline")
        if st.button("Generate / Regenerate Full Outline"):
            with st.spinner("AI is analyzing all content..."):
                outline_json = generate_content_outline(st.session_state.all_chunks)
                if outline_json and "outline" in outline_json: st.session_state.outline_data = outline_json["outline"]
                else: st.error("Failed to generate outline.")
        
        if 'outline_data' in st.session_state and st.session_state.outline_data:
            initial_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
            st.session_state.editable_outline = st.text_area("Editable Outline:", value=initial_text, height=300)
            st.session_state.synthesis_instructions = st.text_area("Synthesis Instructions (Optional):", height=100)
            if st.button("Synthesize Notes", type="primary"):
                st.session_state.current_state = 'synthesizing'
                st.rerun()
    with col2:
        st.subheader("Source Explorer")
        with st.expander("Add More Files"):
            new_files = st.file_uploader("Upload more files", accept_multiple_files=True, key=f"uploader_{int(time.time())}")
            if new_files:
                st.info("File adding logic to be implemented.")

def show_synthesizing_state():
    st.header("Synthesizing Note Blocks...")
    st.session_state.final_notes = []
    topics = [line.strip() for line in st.session_state.editable_outline.split('\n') if line.strip()]
    chunks_map = {c['chunk_id']: c['text'] for c in st.session_state.all_chunks}
    outline_map = {item['topic']: item.get('relevant_chunks', []) for item in st.session_state.outline_data}
    bar = st.progress(0, "Starting synthesis...")
    for i, topic in enumerate(topics):
        bar.progress((i + 1) / len(topics), f"Synthesizing: {topic}")
        chunk_ids = outline_map.get(topic, [])
        text = "\n\n---\n\n".join([chunks_map.get(cid, "") for cid in chunk_ids])
        content = synthesize_note_block(topic, text, st.session_state.synthesis_instructions)
        st.session_state.final_notes.append({"topic": topic, "content": content, "source_chunks": chunk_ids})
    st.session_state.current_state = 'results'
    st.rerun()

def show_results_state():
    st.header("Your Unified Notes")
    if st.button("Start New Note Session"): reset_session(st.session_state.tool_choice); st.rerun()
    if st.button("Back to Workspace"): st.session_state.current_state = 'workspace'; st.rerun()
    st.subheader("Next Step: Create a Lesson")
    if st.button("Create Lesson Plan", type="primary"):
        st.session_state.current_state = 'generating_lesson'
        st.rerun()
    for i, block in enumerate(st.session_state.final_notes):
        st.subheader(block['topic'])
        st.markdown(block['content'])
        if st.button("Regenerate this block", key=f"regen_{i}"):
            st.info("Block regeneration logic to be implemented.")

def show_generating_lesson_state():
    st.header("Building Your Lesson...")
    with st.spinner("AI is designing your lesson plan..."):
        plan_json = generate_lesson_plan(st.session_state.outline_data, st.session_state.all_chunks)
        if plan_json and "lesson_plan" in plan_json:
            st.session_state.lesson_plan = plan_json["lesson_plan"]
            st.session_state.current_state = 'review_lesson'
            st.rerun()
        else:
            st.error("Failed to generate lesson plan."); st.session_state.current_state = 'results'; st.rerun()

def show_review_lesson_state():
    st.header("Review Your Lesson Plan")
    st.write("This is the DNA of your video. Edit the JSON directly before playback.")
    plan_str = json.dumps(st.session_state.lesson_plan, indent=2)
    edited_plan = st.text_area("Editable Lesson Plan (JSON):", value=plan_str, height=600)
    if st.button("Play Lesson", type="primary"):
        try:
            final_plan = json.loads(edited_plan)
            st.success("Lesson plan is valid! Triggering playback engine...")
            st.json(final_plan)
        except json.JSONDecodeError:
            st.error("Edited text is not valid JSON.")

def show_mock_test_placeholder():
    st.header("Mock Test Generator")
    st.image("https://placehold.co/800x400/1A233A/E0E2E7?text=Coming+Soon", use_container_width=True)
    st.write("This feature is under construction.")
    st.info("Planned workflow: Syllabus Upload -> Topic Extraction -> Question Bank Generation -> Test Assembly -> CV-based Grading.")

# --- MAIN APP ---
def main():
    load_css()
    if 'user_info' not in st.session_state: st.session_state.user_info = None
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
        show_pre_login_view(flow)
        return

    # Post-Login: Show sidebar
    st.sidebar.image(st.session_state.user_info['picture'], width=80)
    st.sidebar.subheader(f"Welcome, {st.session_state.user_info['given_name']}")
    if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()
    st.sidebar.divider()

    tool_choice = st.sidebar.radio("Select a Tool", ("Note & Lesson Engine", "Mock Test Generator"), key='tool_choice')
    
    if 'last_tool_choice' not in st.session_state: st.session_state.last_tool_choice = tool_choice
    if st.session_state.last_tool_choice != tool_choice:
        reset_session(tool_choice); st.session_state.last_tool_choice = tool_choice; st.rerun()

    if tool_choice == "Note & Lesson Engine":
        if 'current_state' not in st.session_state: reset_session(tool_choice)
        state_map = { 'upload': show_upload_state, 'processing': show_processing_state, 'workspace': show_workspace_state,
                      'synthesizing': show_synthesizing_state, 'results': show_results_state, 'generating_lesson': show_generating_lesson_state,
                      'review_lesson': show_review_lesson_state }
        state_function = state_map.get(st.session_state.current_state, show_upload_state)
        state_function()
    elif tool_choice == "Mock Test Generator":
        show_mock_test_placeholder()

if __name__ == "__main__":
    main()

