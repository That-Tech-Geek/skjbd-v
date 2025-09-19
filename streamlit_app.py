import streamlit as st
import time
import os
from openai import OpenAI
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import asyncio

# --- GOOGLE OAUTH LIBRARY ---
# This app requires a third-party library for Google authentication.
# If you see an error below, please install it from your terminal using the EXACT command:
# pip install streamlit-google-oauth
try:
    from streamlit_google_oauth import StGoogleOauth
except ImportError:
    st.error("""
        **Required library not found!**

        The Google OAuth library is not installed. Please install it using the correct package name:
        
        ```bash
        pip install streamlit-google-oauth
        ```

        The app cannot continue without this dependency. Please install it and refresh the page.
    """)
    st.stop()

# --- CONFIGURATION & CONSTANTS ---
MAX_FILES = 20
MAX_TOTAL_SIZE_MB = 150
MAX_AUDIO_SIZE_MB = 25
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50 # words

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Vekkam Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API SELF-DIAGNOSIS & UTILITIES ---
def check_openai_api(api_key):
    if not api_key: return "Empty"
    try: OpenAI(api_key=api_key).models.list(); return "Valid"
    except Exception: return "Invalid"

def check_gemini_api(api_key):
    if not api_key: return "Empty"
    try: genai.configure(api_key=api_key); genai.get_model('models/gemini-1.5-pro-latest'); return "Valid"
    except Exception: return "Invalid"

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
def process_source(file, api_key, source_type):
    """Generic processor for all file types."""
    try:
        source_id = f"{source_type}:{file.name}"
        if source_type == 'transcript':
            client = OpenAI(api_key=api_key)
            transcript = client.audio.transcriptions.create(model="whisper-1", file=file)
            chunks = chunk_text(transcript.text, source_id)
            return {"status": "success", "source_id": source_id, "chunks": chunks}
        elif source_type == 'image':
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
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

def generate_content_outline(all_chunks, api_key, existing_outline=None):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        prompt_chunks = [{"chunk_id": c['chunk_id'], "text_snippet": c['text'][:200] + "..."} for c in all_chunks]
        instruction = "Analyze the following content chunks and create a structured outline."
        if existing_outline:
            instruction = f"Analyze the following NEW content chunks and suggest topics to ADD to the existing outline provided."
        prompt = f"""
        You are a curriculum designer. {instruction}
        For each topic, you MUST list the `chunk_id`s that are most relevant to it.
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

def synthesize_note_block(topic, relevant_chunks_text, instructions, api_key):
    try:
        genai.configure(api_key=api_key)
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
try:
    CLIENT_ID = st.secrets["google_oauth"]["client_id"]
    CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
    REDIRECT_URI = st.secrets["google_oauth"]["redirect_uri"]
except (KeyError, FileNotFoundError):
    st.error("OAuth credentials are not configured. Please add them to your .streamlit/secrets.toml file.")
    st.stop()

# --- SESSION STATE & RESET ---
def reset_session():
    for key in list(st.session_state.keys()):
        if key not in ['user_info']: # Keep user info
            del st.session_state[key]
    st.session_state.current_state = 'upload'
    st.session_state.all_chunks = []
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []

# --- MAIN APP LOGIC ---
async def main():
    st.sidebar.title("Vekkam Engine")

    # Initialize session state variables
    if 'current_state' not in st.session_state:
        st.session_state.current_state = 'login'
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

    # --- Authentication Flow ---
    if st.session_state.user_info is None:
        st.title("Welcome to Vekkam")
        st.write("Sign in with Google to start synthesizing knowledge.")
        login_button_placeholder = st.empty()
        with login_button_placeholder:
            user_info = await StGoogleOauth(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI).get_user_info()

        if user_info:
            st.session_state.user_info = user_info
            st.session_state.current_state = 'upload'
            login_button_placeholder.empty()
            st.rerun()
        return

    # --- Post-Login Sidebar ---
    user = st.session_state.user_info
    st.sidebar.image(user['picture'], width=80)
    st.sidebar.subheader(f"Welcome, {user['given_name']}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.divider()
    
    # API Key Management
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

    if st.sidebar.button("Check API Keys"):
        st.session_state.openai_status = check_openai_api(openai_api_key)
        st.session_state.gemini_status = check_gemini_api(gemini_api_key)
    
    st.sidebar.write(f"OpenAI Key: **{st.session_state.get('openai_status', 'Unknown')}**")
    st.sidebar.write(f"Gemini 1.5 Key: **{st.session_state.get('gemini_status', 'Unknown')}**")

    # Initialize state for new sessions
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []
        st.session_state.extraction_failures = []
        st.session_state.outline_data = []
        st.session_state.final_notes = []

    # --- CORE APP VIEWS ROUTER ---
    if st.session_state.current_state == 'upload':
        show_upload_state(openai_api_key, gemini_api_key)
    elif st.session_state.current_state == 'workspace':
        show_workspace_state(openai_api_key, gemini_api_key)
    elif st.session_state.current_state == 'synthesizing':
        show_synthesizing_state(gemini_api_key)
    elif st.session_state.current_state == 'results':
        show_results_state(gemini_api_key)

def show_upload_state(openai_api_key, gemini_api_key):
    st.header("Upload Your Sources")
    st.write("Start by uploading audio, images, and PDFs. The engine will process them and take you to the workspace.")
    
    uploaded_files = st.file_uploader(
        "Select files", 
        accept_multiple_files=True, 
        type=['mp3', 'm4a', 'wav', 'png', 'jpg', 'pdf']
    )

    if st.button("Process Files", type="primary") and uploaded_files:
        if not (openai_api_key and gemini_api_key):
            st.error("Please enter both OpenAI and Gemini API keys in the sidebar before processing.")
            return
        with st.spinner("Processing initial files... This can take a moment."):
            process_files_and_chunks(uploaded_files, openai_api_key, gemini_api_key)
        st.session_state.current_state = 'workspace'
        st.rerun()

def process_files_and_chunks(files_to_process, openai_api_key, gemini_api_key):
    """Utility to run extraction in parallel."""
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {}
        for f in files_to_process:
            if f.type.startswith('audio/'):
                futures[executor.submit(process_source, f, openai_api_key, 'transcript')] = f
            elif f.type.startswith('image/'):
                futures[executor.submit(process_source, f, gemini_api_key, 'image')] = f
            elif f.type == 'application/pdf':
                futures[executor.submit(process_source, f, gemini_api_key, 'pdf')] = f
        
        for future in as_completed(futures):
            results.append(future.result())
    
    new_chunks = []
    for r in [res for res in results if res and res['status'] == 'success']:
        new_chunks.extend(r['chunks'])
    
    st.session_state.all_chunks.extend(new_chunks)
    st.session_state.extraction_failures.extend([r for r in results if r and r['status'] == 'error'])

def show_workspace_state(openai_api_key, gemini_api_key):
    st.header("Vekkam Workspace")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Controls & Outline")
        if st.button("Generate / Regenerate Full Outline"):
            with st.spinner("AI is analyzing all content..."):
                outline_json = generate_content_outline(st.session_state.all_chunks, gemini_api_key)
                if outline_json and "outline" in outline_json:
                    st.session_state.outline_data = outline_json["outline"]
                else: st.error("Failed to generate outline.")
        
        if 'outline_data' in st.session_state and st.session_state.outline_data:
            initial_outline_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
            st.session_state.editable_outline = st.text_area(
                "Editable Outline:", value=initial_outline_text, height=300)
            st.session_state.synthesis_instructions = st.text_area("Synthesis Instructions (Optional):", height=100)
            
            if st.button("Synthesize Notes", type="primary"):
                st.session_state.current_state = 'synthesizing'
                st.rerun()
    
    with col2:
        st.subheader("Source Explorer")
        # Add more files iteratively
        with st.expander("Add More Files"):
            new_files = st.file_uploader("Upload more files", accept_multiple_files=True, key=f"uploader_{int(time.time())}")
            if new_files:
                with st.spinner("Processing new files..."):
                    new_chunks = process_files_and_chunks(new_files, openai_api_key, gemini_api_key)
                with st.spinner("AI is suggesting new topics..."):
                    update_json = generate_content_outline(new_chunks, gemini_api_key, existing_outline=st.session_state.get('outline_data', []))
                    if update_json and "outline" in update_json:
                        st.session_state.outline_data.extend(update_json["outline"])
                        st.success(f"Added {len(update_json['outline'])} new topic(s)!")
                        st.rerun()

        # Display source chunks
        if st.session_state.get('all_chunks'):
            with st.expander("Explore All Content Chunks"):
                for i, chunk in enumerate(st.session_state.all_chunks):
                    st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}`")
                    st.text_area("", chunk['text'], height=100, key=f"chunk_viewer_{i}")

def show_synthesizing_state(gemini_api_key):
    st.header("Synthesizing Note Blocks...")
    st.session_state.final_notes = []
    
    outline_topics = [line.strip() for line in st.session_state.editable_outline.split('\n') if line.strip()]
    all_chunks_map = {chunk['chunk_id']: chunk['text'] for chunk in st.session_state.all_chunks}
    original_outline_map = {item['topic']: item['relevant_chunks'] for item in st.session_state.outline_data}

    progress_bar = st.progress(0, "Starting synthesis...")
    for i, topic in enumerate(outline_topics):
        progress_bar.progress((i + 1) / len(outline_topics), f"Synthesizing: {topic}")
        relevant_chunk_ids = original_outline_map.get(topic, [])
        relevant_chunks_text = "\n\n---\n\n".join([all_chunks_map.get(cid, "") for cid in relevant_chunk_ids])
        
        content = synthesize_note_block(topic, relevant_chunks_text, st.session_state.synthesis_instructions, gemini_api_key)
        
        st.session_state.final_notes.append({"topic": topic, "content": content, "source_chunks": relevant_chunk_ids})
    
    st.session_state.current_state = 'results'
    st.rerun()

def show_results_state(gemini_api_key):
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
                new_content = synthesize_note_block(note_block['topic'], relevant_chunks_text, st.session_state.synthesis_instructions, gemini_api_key)
                st.session_state.final_notes[i]['content'] = new_content
                st.rerun()

        with st.expander("View Source Chunks for this Block"):
            st.json(note_block['source_chunks'])


if __name__ == "__main__":
    asyncio.run(main()
