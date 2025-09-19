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

# --- GOOGLE OAUTH LIBRARIES ---
try:
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
except ImportError:
    st.error("""
        **Required Google libraries not found!**
        Please install the necessary packages using pip:
        ```bash
        pip install google-auth-oauthlib google-api-python-client
        ```
        The app cannot continue without these dependencies. Please install and refresh.
    """)
    st.stop()

# --- CONFIGURATION & CONSTANTS ---
MAX_FILES = 20
MAX_TOTAL_SIZE_MB = 150
MAX_AUDIO_SIZE_MB = 1024 # Gemini has a larger limit, but good to have a cap.
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50 # words

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vekkam Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API SELF-DIAGNOSIS & UTILITIES ---
def check_gemini_api():
    try:
        genai.get_model('models/gemini-2.5-flash-lite')
        return "Valid"
    except Exception as e:
        st.sidebar.error(f"Gemini API Key in secrets is invalid: {e}")
        return "Invalid"

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

# --- CONTENT PROCESSING & AGENTIC WORKFLOW ---
def process_source(file, source_type):
    try:
        source_id = f"{source_type}:{file.name}"
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

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

def generate_content_outline(all_chunks, existing_outline=None):
    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        prompt_chunks = [{"chunk_id": c['chunk_id'], "text_snippet": c['text'][:200] + "..."} for c in all_chunks]
        instruction = "Analyze the content chunks and create a structured outline."
        if existing_outline:
            instruction = f"Analyze the NEW content chunks and suggest topics to ADD to the existing outline."
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
    except Exception as e:
        st.error(f"Outline Generation Error: {e}"); return None

def synthesize_note_block(topic, relevant_chunks_text, instructions):
    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        prompt = f"""
        Write the notes for a single topic: "{topic}".
        Use ONLY the provided source text. Adhere to the user's instructions. Format in Markdown.
        **Instructions:** {instructions if instructions else "None"}
        **Source Text:** {relevant_chunks_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error synthesizing this block: {e}"

# --- AUTHENTICATION SETUP ---
def get_google_flow():
    try:
        client_config = {
            "web": {
                "client_id": st.secrets["google"]["client_id"],
                "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [st.secrets["google"]["redirect_uri"]],
            }
        }
        scopes = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets.")
        st.stop()

# --- SESSION STATE & RESET ---
def reset_session():
    for key in list(st.session_state.keys()):
        if key not in ['user_info']:
            del st.session_state[key]
    st.session_state.current_state = 'upload'
    st.session_state.all_chunks = []
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []

# --- MAIN APP LOGIC ---
def main():
    st.sidebar.title("Vekkam Engine")

    # Initialize session state & configure Gemini
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
    except (KeyError, FileNotFoundError):
        st.error("Gemini API key is not configured in st.secrets.")
        st.stop()

    flow = get_google_flow()
    query_params = st.query_params
    auth_code = query_params.get("code")

    if auth_code and not st.session_state.user_info:
        try:
            flow.fetch_token(code=auth_code)
            credentials = flow.credentials
            user_info_service = build('oauth2', 'v2', credentials=credentials)
            user_info = user_info_service.userinfo().get().execute()
            st.session_state.user_info = user_info
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to fetch token or user info: {e}")
            st.session_state.user_info = None
            
    # --- Authentication Gate ---
    if not st.session_state.user_info:
        st.title("Welcome to Vekkam")
        st.write("Sign in with Google to start synthesizing knowledge.")
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.link_button("Sign in with Google", auth_url)
        return

    # --- Post-Login App ---
    user = st.session_state.user_info
    st.sidebar.image(user['picture'], width=80)
    st.sidebar.subheader(f"Welcome, {user['given_name']}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("API Status")
    st.sidebar.write(f"Gemini 1.5 Pro: **{check_gemini_api()}**")

    if 'current_state' not in st.session_state:
        reset_session()
    
    # --- CORE APP VIEWS ROUTER ---
    if st.session_state.current_state == 'upload':
        show_upload_state()
    elif st.session_state.current_state == 'workspace':
        show_workspace_state()
    elif st.session_state.current_state == 'synthesizing':
        show_synthesizing_state()
    elif st.session_state.current_state == 'results':
        show_results_state()

def show_upload_state():
    st.header("Upload Your Sources")
    uploaded_files = st.file_uploader("Select files", accept_multiple_files=True, type=['mp3', 'm4a', 'wav', 'png', 'jpg', 'pdf'])
    if st.button("Process Files", type="primary") and uploaded_files:
        with st.spinner("Processing initial files... This can take a moment."):
            process_files_and_chunks(uploaded_files)
        st.session_state.current_state = 'workspace'
        st.rerun()

def process_files_and_chunks(files_to_process):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_source, f, 'transcript' if f.type.startswith('audio/') else 'image' if f.type.startswith('image/') else 'pdf'): f for f in files_to_process}
        for future in as_completed(futures):
            results.append(future.result())
    
    new_chunks = []
    for r in [res for res in results if res and res['status'] == 'success']:
        new_chunks.extend(r['chunks'])
    st.session_state.all_chunks.extend(new_chunks)
    st.session_state.extraction_failures.extend([r for r in results if r and r['status'] == 'error'])
    return new_chunks

def show_workspace_state():
    st.header("Vekkam Workspace")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Controls & Outline")
        if st.button("Generate / Regenerate Full Outline"):
            with st.spinner("AI is analyzing all content..."):
                outline_json = generate_content_outline(st.session_state.all_chunks)
                if outline_json and "outline" in outline_json:
                    st.session_state.outline_data = outline_json["outline"]
                else: st.error("Failed to generate outline.")
        
        if 'outline_data' in st.session_state and st.session_state.outline_data:
            initial_outline_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
            st.session_state.editable_outline = st.text_area("Editable Outline:", value=initial_outline_text, height=300)
            st.session_state.synthesis_instructions = st.text_area("Synthesis Instructions (Optional):", height=100)
            if st.button("Synthesize Notes", type="primary"):
                st.session_state.current_state = 'synthesizing'
                st.rerun()
    
    with col2:
        st.subheader("Source Explorer")
        with st.expander("Add More Files"):
            new_files = st.file_uploader("Upload more files", accept_multiple_files=True, key=f"uploader_{int(time.time())}")
            if new_files:
                new_chunks = process_files_and_chunks(new_files)
                with st.spinner("AI is suggesting new topics..."):
                    update_json = generate_content_outline(new_chunks, existing_outline=st.session_state.get('outline_data', []))
                    if update_json and "outline" in update_json:
                        st.session_state.outline_data.extend(update_json["outline"])
                        st.success(f"Added {len(update_json['outline'])} new topic(s)!")
                        st.rerun()

        if st.session_state.get('all_chunks'):
            with st.expander("Explore All Content Chunks", expanded=False):
                for i, chunk in enumerate(st.session_state.all_chunks):
                    st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}`")
                    st.text_area("", chunk['text'], height=100, key=f"chunk_viewer_{i}")

def show_synthesizing_state():
    st.header("Synthesizing Note Blocks...")
    st.session_state.final_notes = []
    outline_topics = [line.strip() for line in st.session_state.editable_outline.split('\n') if line.strip()]
    all_chunks_map = {chunk['chunk_id']: chunk['text'] for chunk in st.session_state.all_chunks}
    original_outline_map = {item['topic']: item.get('relevant_chunks', []) for item in st.session_state.outline_data}

    progress_bar = st.progress(0, "Starting synthesis...")
    for i, topic in enumerate(outline_topics):
        progress_bar.progress((i + 1) / len(outline_topics), f"Synthesizing: {topic}")
        relevant_chunk_ids = original_outline_map.get(topic, [])
        relevant_chunks_text = "\n\n---\n\n".join([all_chunks_map.get(cid, "") for cid in relevant_chunk_ids])
        content = synthesize_note_block(topic, relevant_chunks_text, st.session_state.synthesis_instructions)
        st.session_state.final_notes.append({"topic": topic, "content": content, "source_chunks": relevant_chunk_ids})
    
    st.session_state.current_state = 'results'
    st.rerun()

def show_results_state():
    st.header("Your Unified Notes")
    if st.button("Back to Workspace"):
        st.session_state.current_state = 'workspace'
        st.rerun()
    if st.button("Start New Session"):
        reset_session()
        st.rerun()

    for i, note_block in enumerate(st.session_state.final_notes):
        st.subheader(note_block['topic'])
        st.markdown(note_block['content'])
        if st.button("Regenerate this block", key=f"regen_{i}"):
            with st.spinner("Regenerating block..."):
                all_chunks_map = {chunk['chunk_id']: chunk['text'] for chunk in st.session_state.all_chunks}
                relevant_chunks_text = "\n\n---\n\n".join([all_chunks_map.get(cid, "") for cid in note_block['source_chunks']])
                new_content = synthesize_note_block(note_block['topic'], relevant_chunks_text, st.session_state.synthesis_instructions)
                st.session_state.final_notes[i]['content'] = new_content
                st.rerun()
        with st.expander("View Source Chunks for this Block"):
            st.json(note_block['source_chunks'])

if __name__ == "__main__":
    main()

