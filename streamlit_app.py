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
st.set_page_config(page_title="Vekkam Engine", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

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
def check_gemini_api():
    try: genai.get_model('models/gemini-1.5-flash'); return "Valid"
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

# --- CONTENT PROCESSING ---
def process_source(file, source_type):
    try:
        source_id = f"{source_type}:{file.name}"
        model = genai.GenerativeModel('models/gemini-1.5-flash')
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
                response = model.generate_content(["Transcribe this noisy classroom audio recording. Prioritize capturing all speech, even if faint, over background noise and echo.", audio_file])
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
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    prompt_chunks = [{"chunk_id": c['chunk_id'], "text_snippet": c['text'][:200] + "..."} for c in all_chunks if c.get('text') and len(c['text'].split()) > 10]
    
    if not prompt_chunks:
        st.error("Could not find enough content to generate an outline. Please check your uploaded files.")
        return None

    instruction = "Analyze the content chunks and create a structured, logical topic outline. The topics should flow from foundational concepts to more advanced ones."
    if existing_outline:
        instruction = "Analyze the NEW content chunks and suggest topics to ADD to the existing outline. Maintain a logical flow."
    
    prompt = f"""
    You are a master curriculum designer. Your task is to create a coherent and comprehensive study outline from fragmented pieces of text.
    {instruction}
    For each topic, you MUST list the `chunk_id`s that are most relevant to that topic. Ensure every chunk is used if possible, but prioritize relevance.
    Do not invent topics not supported by the text. Base the outline STRICTLY on the provided content.
    Output ONLY a valid JSON object with a root key "outline", which is a list of objects. Each object must have keys "topic" (string) and "relevant_chunks" (a list of string chunk_ids).

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
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    prompt = f"""
    You are a world-class note-taker. Synthesize a detailed, clear, and well-structured note block for a single topic: "{topic}".
    Your entire response MUST be based STRICTLY and ONLY on the provided source text. Do not introduce any external information.
    Adhere to the user's instructions for formatting and style. Format the output in Markdown.

    **User Instructions:** {instructions if instructions else "Default: Create clear, concise, well-structured notes."}

    **Source Text (Use only this):**
    ---
    {relevant_chunks_text}
    ---
    """
    response = model.generate_content(prompt)
    return response.text

@gemini_api_call_with_retry
def generate_lesson_plan(outline, all_chunks):
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    chunk_context_map = {c['chunk_id']: c['text'][:200] + "..." for c in all_chunks}
    prompt = f"""
    You are a world-class educator. Design a detailed, step-by-step lesson plan based on the provided outline and source material.
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
    # Preserve user info and tool choice, clear everything else
    user_info = st.session_state.get('user_info')
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.tool_choice = tool_choice
    st.session_state.current_state = 'upload'
    st.session_state.all_chunks = []
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []

# --- NEW: ADVANCED PRE-LOGIN LANDING PAGE ---
def show_landing_page(auth_url):
    """Displays the new, feature-rich landing page."""
    
    st.markdown("""
        <style>
            /* --- General Styles --- */
            .main > div {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .stApp {
                background-color: #0F172A; /* Dark blue-gray background */
            }
            h1, h2, h3, p, .stMarkdown {
                color: #E2E8F0; /* Light gray text */
                text-align: center; /* Center all text by default */
            }
            .stButton > a { /* Target the link inside the button */
                width: 100%;
                text-align: center;
            }
            
            /* --- Hide Streamlit Header --- */
            header[data-testid="stHeader"] {
                display: none !important;
                visibility: hidden !important;
            }
            /* Adjust top padding for main content after hiding header */
            .main .block-container {
                padding-top: 2rem;
                /* --- FIX: ADD FLEXBOX CENTERING --- */
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            /* --- Specific Element Styles --- */
            .title {
                font-size: 3.5rem;
                font-weight: 700;
                line-height: 1.2;
                background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
            }
            .subtitle {
                font-size: 1.25rem;
                color: #94A3B8;
                max-width: 650px;
                margin: 0 auto 2rem auto;
            }
            .login-button-container {
                display: flex;
                justify-content: center;
                margin-bottom: 4rem;
            }
            .section-title {
                font-size: 2.5rem;
                font-weight: 600;
                margin-top: 5rem;
                margin-bottom: 3rem;
            }

            /* --- Comparison Table --- */
            .comparison-table {
                width: 100%; max-width: 900px; margin: 2rem auto;
                border-collapse: collapse; border-radius: 8px; overflow: hidden;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }
            .comparison-table th, .comparison-table td {
                padding: 1.25rem 1rem; border-bottom: 1px solid #334155;
            }
            .comparison-table th { background-color: #1E293B; font-size: 1.1rem; color: #F8FAFC; }
            .comparison-table td { background-color: #0F172A; color: #CBD5E1; }
            .comparison-table .feature-col { text-align: left; font-weight: 600; }
            .comparison-table .vekkam-col { background-color: rgba(30, 58, 138, 0.5); color: #E0E7FF; }
            .tick { color: #4ADE80; font-size: 1.5rem; font-weight: bold; }
            .cross { color: #F87171; font-size: 1.5rem; font-weight: bold; }
            
            /* --- How-It-Works & Who-Is-It-For Sections --- */
            .card {
                background-color: #1E293B; padding: 2rem; border-radius: 12px;
                border: 1px solid #334155; height: 100%;
            }
            .card .icon { font-size: 3rem; }
            .card h3 { font-size: 1.5rem; margin-top: 1rem; margin-bottom: 0.5rem; color: #F8FAFC; }
            .card p { color: #94A3B8; font-size: 1rem; line-height: 1.6; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">Stop Juggling Tabs. Start Understanding.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">General AI chatbots give you answers. Vekkam gives you a workflow. We turn your chaotic lecture recordings, messy notes, and dense PDFs into a single, unified study guide—a feat impossible for generic tools.</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-button-container">', unsafe_allow_html=True)
        st.link_button("Get Started - Sign in with Google", auth_url, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-title">The Right Tool for the Job</h2>', unsafe_allow_html=True)
    st.markdown("""
        <table class="comparison-table">
            <thead>
                <tr>
                    <th class="feature-col">Feature</th>
                    <th class="vekkam-col">Vekkam</th>
                    <th>ChatGPT / Gemini</th>
                </tr>
            </thead>
            <tbody>
                <tr><td class="feature-col">Multi-Source Input (Audio, PDF, Images)</td><td class="vekkam-col"><span class="tick">✔</span><br>Built-in</td><td><span class="cross">✖</span><br>Requires separate tools & copy-pasting</td></tr>
                <tr><td class="feature-col">Context-Aware Synthesis</td><td class="vekkam-col"><span class="tick">✔</span><br>Only uses <u>your</u> provided material</td><td><span class="cross">✖</span><br>Can drift and pull in irrelevant web data</td></tr>
                <tr><td class="feature-col">Unified Study Guide Output</td><td class="vekkam-col"><span class="tick">✔</span><br>One-click coherent output from all sources</td><td><span class="cross">✖</span><br>Manual summarization and compilation needed</td></tr>
                <tr><td class="feature-col">Purpose-Built Study Workflow</td><td class="vekkam-col"><span class="tick">✔</span><br>Designed for students from the ground up</td><td><span class="cross">✖</span><br>General purpose, not optimized for study</td></tr>
                <tr><td class="feature-col">Noise-Robust Audio Transcription</td><td class="vekkam-col"><span class="tick">✔</span><br>Fine-tuned for messy classroom audio</td><td><span class="cross">✖</span><br>Struggles with background noise & faint speech</td></tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">No Black Box. Just a Smarter Workflow.</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown('<div class="card"><div class="icon">📥</div><h3>1. Ingest & Deconstruct</h3><p>You upload everything—lecture recordings, PDFs, handwritten notes. Our first AI agent standardizes and breaks it all down into thousands of context-rich, searchable text chunks.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><div class="icon">🔗</div><h3>2. Connect & Outline</h3><p>A specialized curriculum agent analyzes these chunks, identifying core themes, key concepts, and the logical flow of information to propose a structured, editable study outline for your approval.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><div class="icon">📝</div><h3>3. Synthesize & Generate</h3><p>Once you approve the outline, a final agent writes your study guide, topic by topic. Crucially, it uses <strong>only the text chunks from your material</strong>, ensuring zero drift or hallucination.</p></div>', unsafe_allow_html=True)

    # --- NEW "WHO IS THIS FOR?" SECTION ---
    st.markdown('<h2 class="section-title">Built for the Modern Student</h2>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3, gap="large")
    with col4:
        st.markdown('<div class="card"><div class="icon">🤯</div><h3>The Overwhelmed</h3><p>For those juggling multiple complex subjects. Vekkam finds the signal in the noise, connecting dots between lecture slides, textbook chapters, and class discussions automatically.</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="card"><div class="icon">✍️</div><h3>The Diligent</h3><p>For the meticulous note-taker who wants more. Combine your handwritten notes (as images) with official materials to create a "director\'s cut" study guide that has every angle covered.</p></div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="card"><div class="icon">🗺️</div><h3>The Big-Picture Thinker</h3><p>For the student who needs to see the map before the journey. Vekkam excels at creating a high-level structure first, so you can dive into the details with a clear understanding of how everything fits together.</p></div>', unsafe_allow_html=True)


# --- UI STATE FUNCTIONS for NOTE & LESSON ENGINE ---
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
                else: st.error("Failed to generate outline. The AI couldn't structure the provided content. Try adding more context-rich files.")
        
        if 'outline_data' in st.session_state and st.session_state.outline_data:
            initial_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
            st.session_state.editable_outline = st.text_area("Editable Outline:", value=initial_text, height=300)
            st.session_state.synthesis_instructions = st.text_area("Synthesis Instructions (Optional):", height=100, placeholder="e.g., 'Explain this like I'm 15' or 'Focus on key formulas'")
            if st.button("Synthesize Notes", type="primary"):
                st.session_state.current_state = 'synthesizing'
                st.rerun()
    with col2:
        st.subheader("Source Explorer")
        if st.session_state.get('extraction_failures'):
            with st.expander("⚠️ Processing Errors", expanded=True):
                for failure in st.session_state.extraction_failures:
                    st.error(f"**{failure['source_id']}**: {failure['reason']}")

        with st.expander("Add More Files"):
            new_files = st.file_uploader("Upload more files", accept_multiple_files=True, key=f"uploader_{int(time.time())}")
            if new_files:
                st.info("File adding logic to be implemented.")

def show_synthesizing_state():
    st.header("Synthesizing Note Blocks...")
    st.session_state.final_notes = []
    topics = [line.strip() for line in st.session_state.editable_outline.split('\n') if line.strip()]
    chunks_map = {c['chunk_id']: c['text'] for c in st.session_state.all_chunks}
    
    original_outline_map = {item['topic']: item.get('relevant_chunks', []) for item in st.session_state.outline_data}
    
    bar = st.progress(0, "Starting synthesis...")
    for i, topic in enumerate(topics):
        bar.progress((i + 1) / len(topics), f"Synthesizing: {topic}")
        matched_chunks = original_outline_map.get(topic, [])
        if not matched_chunks:
            for original_topic, chunk_ids in original_outline_map.items():
                if topic in original_topic:
                    matched_chunks = chunk_ids
                    break
        
        text_to_synthesize = "\n\n---\n\n".join([chunks_map.get(cid, "") for cid in matched_chunks if cid in chunks_map])
        
        if not text_to_synthesize.strip():
            content = "Could not find source text for this topic. It might have been edited or removed."
        else:
            content = synthesize_note_block(topic, text_to_synthesize, st.session_state.synthesis_instructions)
            
        st.session_state.final_notes.append({"topic": topic, "content": content, "source_chunks": matched_chunks})

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

# --- UI STATE FUNCTIONS for MOCK TEST GENERATOR ---
def show_mock_test_placeholder():
    st.header("Mock Test Generator")
    st.image("https://placehold.co/800x400/1A233A/E0E2E7?text=Coming+Soon", use_column_width=True)
    st.write("This feature is under construction. The architecture for generating mock tests based on syllabus content, Bloom's Taxonomy, and a professor persona will be built here.")
    st.info("The planned workflow includes: Syllabus Upload -> Topic Extraction -> Question Bank Generation -> Test Assembly -> CV-based Grading.")


# --- MAIN APP ---
def main():
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
    
    # Pre-Login: Show the new landing page
    if not st.session_state.user_info:
        # Hide sidebar on landing page
        st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
        auth_url, _ = flow.authorization_url(prompt='consent')
        show_landing_page(auth_url) # <<<--- THIS IS THE ONLY CHANGE
        return

    # Post-Login: Show sidebar and run app
    st.sidebar.title("Vekkam Engine")
    user = st.session_state.user_info
    st.sidebar.image(user['picture'], width=80)
    st.sidebar.subheader(f"Welcome, {user['given_name']}")
    if st.sidebar.button("Logout"): 
        st.session_state.clear()
        st.rerun()
    st.sidebar.divider()

    tool_choice = st.sidebar.radio("Select a Tool", ("Note & Lesson Engine", "Mock Test Generator"), key='tool_choice')
    
    if 'last_tool_choice' not in st.session_state: st.session_state.last_tool_choice = tool_choice
    if st.session_state.last_tool_choice != tool_choice:
        reset_session(tool_choice)
        st.session_state.last_tool_choice = tool_choice
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("API Status")
    st.sidebar.write(f"Gemini: **{check_gemini_api()}**")

    if tool_choice == "Note & Lesson Engine":
        if 'current_state' not in st.session_state: reset_session(tool_choice)
        state_map = { 'upload': show_upload_state, 'processing': show_processing_state, 'workspace': show_workspace_state,
                        'synthesizing': show_synthesizing_state, 'results': show_results_state, 'generating_lesson': show_generating_lesson_state,
                        'review_lesson': show_review_lesson_state, }
        state_function = state_map.get(st.session_state.current_state, show_upload_state)
        state_function()
    elif tool_choice == "Mock Test Generator":
        show_mock_test_placeholder()


if __name__ == "__main__":
    main()
