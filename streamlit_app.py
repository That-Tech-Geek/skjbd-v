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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vekkam | Your AI Study Partner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS & STYLING ---
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            background-color: #0E1117; /* Streamlit dark theme bg */
        }

        /* Hide Streamlit Header/Footer */
        #MainMenu, .stDeployButton, footer {
            visibility: hidden;
        }
        
        /* Main App Title */
        h1 {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
        }

        /* Pre-Login Hero Section */
        .hero {
            text-align: center;
            padding: 4rem 1rem;
        }
        .hero h1 {
            background: -webkit-linear-gradient(45deg, #ff8a00, #e52e71);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
        }
        .hero p {
            max-width: 600px;
            margin: 1rem auto;
            font-size: 1.1rem;
            color: #b0b3b8;
        }

        /* Custom Google Sign-in Button */
        .google-btn-container {
            display: flex;
            justify-content: center;
            margin-top: 2rem;
        }
        .google-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 12px 24px;
            border-radius: 8px;
            background-color: #4285F4;
            color: white;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .google-btn:hover {
            background-color: #357ae8;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
            color: white;
        }
        .google-btn img {
            width: 20px;
            margin-right: 12px;
        }

        /* Features Section */
        .features {
            display: flex;
            justify-content: center;
            gap: 2rem;
            padding: 3rem 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        .feature-card {
            background-color: #1a1a2e;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #2a2a3e;
            flex: 1;
        }
        .feature-card .icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        .feature-card h3 {
            color: #e0e2e7;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .feature-card p {
            color: #b0b3b8;
            font-size: 0.95rem;
        }
        
        /* Post-Login UI */
        .stButton>button {
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .st-emotion-cache-1fttcpj { /* Primary button style */
            background-color: #4285F4;
        }

    </style>
    """, unsafe_allow_html=True)

# --- API SELF-DIAGNOSIS & UTILITIES ---
def check_gemini_api():
    try:
        genai.get_model('models/gemini-1.5-pro-latest')
        return "✅ Valid"
    except Exception:
        return "❌ Invalid"

def resilient_json_parser(json_string):
    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```', json_string, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if match: return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        st.error("Fatal Error: Could not parse a critical AI JSON response."); return None

def chunk_text(text, source_id, chunk_size=500, overlap=50):
    if not text: return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text_val = " ".join(chunk_words)
        chunk_hash = hashlib.md5(chunk_text_val.encode()).hexdigest()[:8]
        chunk_id = f"{source_id}::chunk_{i//(chunk_size-overlap)}_{chunk_hash}"
        chunks.append({"chunk_id": chunk_id, "text": chunk_text_val})
    return chunks

# --- CONTENT PROCESSING & AGENTIC WORKFLOW ---
def process_source(file, source_type):
    try:
        source_id = f"{source_type}:{file.name}"
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

        if source_type == 'transcript':
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            try:
                audio_file = genai.upload_file(path=tmp_path)
                response = model.generate_content(
                    ["Transcribe this audio, paying close attention to faint speech and filtering out background noise.", audio_file],
                    request_options={"timeout": 600}
                )
                genai.delete_file(audio_file.name)
                chunks = chunk_text(response.text, source_id)
                return {"status": "success", "source_id": source_id, "chunks": chunks}
            finally:
                os.unlink(tmp_path)
        elif source_type == 'image':
            image = Image.open(file)
            response = model.generate_content(["Extract all text and describe the contents of this image for study purposes.", image])
            return {"status": "success", "source_id": source_id, "chunks": [{"chunk_id": f"{source_id}::chunk_0", "text": response.text}]}
        elif source_type == 'pdf':
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            chunks = chunk_text(text, source_id)
            return {"status": "success", "source_id": source_id, "chunks": chunks}
    except Exception as e:
        return {"status": "error", "source_id": f"{source_type}:{file.name}", "reason": str(e)}

@st.cache_data(ttl=3600)
def generate_content_outline(_all_chunks, existing_outline=None):
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        prompt_chunks = [{"chunk_id": c['chunk_id'], "text_snippet": c['text'][:200] + "..."} for c in _all_chunks]
        instruction = "Analyze these content chunks and generate a detailed, logical topic outline for a study guide. Be comprehensive."
        if existing_outline:
            instruction = f"Analyze the NEW content chunks and suggest topics to ADD to the existing outline."
        prompt = f"""
        You are a curriculum designer. {instruction}
        For each topic, you MUST list the `chunk_id`s that are most relevant.
        Output ONLY a JSON object formatted exactly like this: ```json{{"outline": [{{"topic": "string", "relevant_chunks": ["list of strings"]}}]}}```
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

@st.cache_data(ttl=3600)
def synthesize_note_block(_topic, _relevant_chunks_text, _instructions):
    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        prompt = f"""
        Act as a world-class educator. Synthesize a comprehensive note block for the topic: "{_topic}".
        Use ONLY the provided source text. Format the output in clean, readable Markdown.
        Explain concepts from first principles. Use analogies if helpful. Define key terms.
        **User Instructions:** {_instructions if _instructions else "Default: Create clear, concise, and comprehensive notes."}
        **Source Text:**
        ---
        {_relevant_chunks_text}
        """
        response = model.generate_content(prompt, request_options={"timeout": 300})
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
    user_info = st.session_state.get('user_info', None)
    st.session_state.clear()
    if user_info:
        st.session_state.user_info = user_info
    st.session_state.current_state = 'upload'
    st.session_state.all_chunks = []
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []

# --- UI VIEWS ---

def show_pre_login_page(auth_url):
    st.markdown("""
        <div class="hero">
            <h1>Stop juggling tabs. Start understanding.</h1>
            <p>ChatGPT gives you answers. Vekkam gives you understanding. Turn your messy lecture recordings, scribbled notes, and textbook PDFs into a single, unified study guide that actually helps you learn.</p>
            <div class="google-btn-container">
    """, unsafe_allow_html=True)
    
    # The button must be created with Streamlit to work
    if st.button("🚀 Get Started for Free with Google"):
        st.switch_page(auth_url)

    st.markdown("""
            </div>
        </div>
        <div class="features">
            <div class="feature-card">
                <div class="icon">🎤</div>
                <h3>Ambient Capture</h3>
                <p>Drop in anything—lecture audio, messy PDFs, even photos of a whiteboard. Vekkam intelligently extracts the core information from all your sources.</p>
            </div>
            <div class="feature-card">
                <div class="icon">✨</div>
                <h3>AI Synthesis</h3>
                <p>Go beyond simple summaries. Our AI builds a structured outline and synthesizes deep, intuitive notes on every topic, just like a world-class educator.</p>
            </div>
            <div class="feature-card">
                <div class="icon">🤝</div>
                <h3>Duo Mode</h3>
                <p>Learn better, together. Sync notes with a partner to create the ultimate 'Director's Cut' study guide. The most effective way to prep for exams. (Coming Soon!)</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_main_app():
    user = st.session_state.user_info
    with st.sidebar:
        st.image(user['picture'], width=80)
        st.subheader(f"Welcome, {user['given_name']}")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.subheader("API Status")
        st.write(f"Gemini 1.5 Pro: **{check_gemini_api()}**")
        st.divider()
        if st.button("Start New Session", use_container_width=True):
            reset_session()
            st.rerun()

    if 'current_state' not in st.session_state:
        reset_session()
    
    state_map = {
        'upload': show_upload_state,
        'workspace': show_workspace_state,
        'synthesizing': show_synthesizing_state,
        'results': show_results_state,
    }
    state_function = state_map.get(st.session_state.current_state, show_upload_state)
    state_function()


def show_upload_state():
    st.header("1. Upload Your Sources")
    st.info("Upload lecture audio, PDFs, and images of your notes. The more context you provide, the better your study guide will be.")
    uploaded_files = st.file_uploader(
        "Select files (MP3, WAV, PDF, PNG, JPG)",
        accept_multiple_files=True,
        type=['mp3', 'm4a', 'wav', 'png', 'jpg', 'pdf']
    )
    if st.button("✨ Process & Generate Outline", type="primary", use_container_width=True) and uploaded_files:
        with st.spinner("Step 1/2: Processing files... This may take a minute for large audio."):
            process_files_and_chunks(uploaded_files)
        with st.spinner("Step 2/2: AI is analyzing content and building your outline..."):
            outline_json = generate_content_outline(st.session_state.all_chunks)
            if outline_json and "outline" in outline_json:
                st.session_state.outline_data = outline_json["outline"]
                st.session_state.current_state = 'workspace'
                st.rerun()
            else:
                st.error("Failed to generate an outline from the provided files. Please try with different or clearer sources.")

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
    st.header("2. Refine Your Study Plan")
    st.info("Your content has been structured into a topic outline. You can edit, add, or remove topics before generating the final notes.")
    
    if 'outline_data' in st.session_state and st.session_state.outline_data:
        initial_outline_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
        st.session_state.editable_outline = st.text_area(
            "**Editable Outline:**",
            value=initial_outline_text,
            height=300,
            help="One topic per line. Add or remove topics as needed."
        )
        st.session_state.synthesis_instructions = st.text_area(
            "**Special Instructions (Optional):**",
            placeholder="e.g., 'Focus on key formulas', 'Explain this for a beginner', 'Create flashcard-style Q&A'",
            height=100
        )
        if st.button("📝 Synthesize My Notes", type="primary", use_container_width=True):
            st.session_state.current_state = 'synthesizing'
            st.rerun()

def show_synthesizing_state():
    st.header("Building Your Study Guide...")
    st.session_state.final_notes = []
    outline_topics = [line.strip() for line in st.session_state.editable_outline.split('\n') if line.strip()]
    all_chunks_map = {chunk['chunk_id']: chunk['text'] for chunk in st.session_state.all_chunks}
    original_outline_map = {item['topic']: item.get('relevant_chunks', []) for item in st.session_state.outline_data}

    progress_bar = st.progress(0, "Initializing synthesis...")
    status_text = st.empty()
    
    for i, topic in enumerate(outline_topics):
        progress_bar.progress((i + 1) / len(outline_topics))
        status_text.text(f"Synthesizing: {topic}")
        relevant_chunk_ids = original_outline_map.get(topic, [])
        relevant_chunks_text = "\n\n---\n\n".join([all_chunks_map.get(cid, "") for cid in relevant_chunk_ids])
        
        # Use caching to speed up re-runs
        content = synthesize_note_block(topic, relevant_chunks_text, st.session_state.synthesis_instructions)
        st.session_state.final_notes.append({"topic": topic, "content": content, "source_chunks": relevant_chunk_ids})
    
    st.session_state.current_state = 'results'
    st.rerun()

def show_results_state():
    st.header("✅ Your Unified Study Guide is Ready!")
    st.info("Review your notes below. Each topic can be expanded. You can regenerate any section that isn't perfect.")
    
    if st.button("⬅️ Back to Workspace"):
        st.session_state.current_state = 'workspace'
        st.rerun()

    for i, note_block in enumerate(st.session_state.final_notes):
        with st.expander(f"**{note_block['topic']}**", expanded=i==0):
            st.markdown(note_block['content'])
            if st.button("Regenerate this block", key=f"regen_{i}"):
                with st.spinner("Regenerating block..."):
                    all_chunks_map = {chunk['chunk_id']: chunk['text'] for chunk in st.session_state.all_chunks}
                    relevant_chunks_text = "\n\n---\n\n".join([all_chunks_map.get(cid, "") for cid in note_block['source_chunks']])
                    new_content = synthesize_note_block(note_block['topic'], relevant_chunks_text, st.session_state.synthesis_instructions)
                    st.session_state.final_notes[i]['content'] = new_content
                    st.rerun()


# --- MAIN APP ROUTER ---
def main():
    load_css()
    
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
            st.error(f"Authentication failed: {e}")
            st.session_state.user_info = None
            
    if st.session_state.user_info:
        show_main_app()
    else:
        auth_url, _ = flow.authorization_url(prompt='consent')
        show_pre_login_page(auth_url)

if __name__ == "__main__":
    main()
