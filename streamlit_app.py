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
import uuid

# --- GOOGLE OAUTH & API LIBRARIES ---
try:
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
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
DATA_DIR = Path("user_data")
DATA_DIR.mkdir(exist_ok=True)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Vekkam Engine", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- MOCK KNOWLEDGE GENOME DATA (FOR GENESIS MODULE) ---
ECON_101_GENOME = {
  "subject": "Econ 101",
  "version": "1.0",
  "nodes": [
    {
      "gene_id": "ECON101_SCARCITY",
      "gene_name": "Scarcity",
      "difficulty": 1,
      "content_alleles": [
        {"type": "text", "content": "Scarcity refers to the basic economic problem, the gap between limited – that is, scarce – resources and theoretically limitless wants. This situation requires people to make decisions about how to allocate resources in an efficient way, in order to satisfy as many of their wants as possible."},
        {"type": "video", "url": "https://www.youtube.com/watch?v=yoVc_S_gd_0"}
      ]
    },
    {
      "gene_id": "ECON101_OPPCOST",
      "gene_name": "Opportunity Cost",
      "difficulty": 2,
      "content_alleles": [
          {"type": "text", "content": "Opportunity cost is the potential forgone profit from a missed opportunity—the result of choosing one alternative and forgoing another. In short, it’s what you give up when you make a decision. The formula is simply the difference between the expected return of each option. Expected Return = (Probability of Gain x Potential Gain) - (Probability of Loss x Potential Loss)."},
          {"type": "video", "url": "https://www.youtube.com/watch?v=PSU-SA-Fv_M"}
      ]
    },
    {
      "gene_id": "ECON101_SND",
      "gene_name": "Supply and Demand",
      "difficulty": 3,
      "content_alleles": [
          {"type": "text", "content": "Supply and demand is a model of microeconomics. It describes how a price is formed in a market economy. In a competitive market, the unit price for a particular good will vary until it settles at a point where the quantity demanded by consumers (at the current price) will equal the quantity supplied by producers (at the current price), resulting in an economic equilibrium for price and quantity."},
          {"type": "video", "url": "https://www.youtube.com/watch?v=9QSWLmyGpYc"}
      ]
    }
  ],
  "edges": [
    {"from": "ECON101_SCARCITY", "to": "ECON101_OPPCOST"},
    {"from": "ECON101_OPPCOST", "to": "ECON101_SND"}
  ]
}


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

# --- PERSISTENT DATA STORAGE ---
def get_user_data_path(user_id):
    """Generates a secure filepath for a user data."""
    safe_filename = hashlib.md5(user_id.encode()).hexdigest() + ".json"
    return DATA_DIR / safe_filename

def load_user_data(user_id):
    """Loads a user session history from a JSON file."""
    filepath = get_user_data_path(user_id)
    if filepath.exists():
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"sessions": []}
    return {"sessions": []}

def save_user_data(user_id, data):
    """Saves a user session history to a JSON file."""
    filepath = get_user_data_path(user_id)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def save_session_to_history(user_id, final_notes):
    """Saves the full note content from a completed session for a user."""
    user_data = load_user_data(user_id)
    session_title = final_notes[0]['topic'] if final_notes else "Untitled Session"
    new_session = {
        "id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "title": session_title,
        "notes": final_notes
    }
    user_data["sessions"].insert(0, new_session)
    save_user_data(user_id, user_data)

# --- API SELF-DIAGNOSIS & UTILITIES ---
def check_gemini_api():
    try:
        genai.get_model('models/gemini-2.-flash-lite')
        return "Valid"
    except Exception as e:
        st.sidebar.error(f"Gemini API Key in secrets is invalid: {e}")
        return "Invalid"

def resilient_json_parser(json_string):
    try:
        # Relaxed JSON parsing to find JSON object within a string that might contain other text
        match = re.search(r'```(json)?\s*(\{.*?\})\s*```', json_string, re.DOTALL)
        if match:
            return json.loads(match.group(2))
        
        match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        
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
        model_name = 'gemini-2.-flash-lite-vision' if source_type in ['image', 'pdf'] else 'gemini-2.-flash-lite'
        model = genai.GenerativeModel(model_name)
        if source_type == 'transcript':
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            try:
                audio_file = genai.upload_file(path=tmp_path)
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
    model = genai.GenerativeModel('gemini-2.-flash-lite')
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
    model = genai.GenerativeModel('gemini-2.-flash-lite')
    prompt = f"""
    You are a world-class note-taker. Synthesize a detailed, clear, and well-structured note block for a single topic: {topic}.
    Your entire response MUST be based STRICTLY and ONLY on the provided source text. Do not introduce any external information.
    Adhere to the user instructions for formatting and style. Format the output in Markdown.

    **User Instructions:** {instructions if instructions else "Default: Create clear, concise, well-structured notes."}

    **Source Text (Use only this):**
    ---
    {relevant_chunks_text}
    ---
    """
    response = model.generate_content(prompt)
    return response.text

@gemini_api_call_with_retry
def answer_from_context(query, context):
    """Answers a user query based ONLY on the provided context."""
    model = genai.GenerativeModel('gemini-2.-flash-lite')
    prompt = f"""
    You are a helpful study assistant. Your task is to answer the user question based strictly and exclusively on the provided study material context.
    Do not use any external knowledge. If the answer is not in the context, clearly state that the information is not available in the provided materials.

    **User Question:**
    {query}

    **Study Material Context:**
    ---
    {context}
    ---
    """
    response = model.generate_content(prompt)
    return response.text

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
    # Preserve context across tool switches if it exists
    all_chunks = st.session_state.get('all_chunks', [])
    
    st.session_state.clear()
    
    st.session_state.user_info = user_info
    st.session_state.tool_choice = tool_choice
    st.session_state.all_chunks = all_chunks # Restore the context
    
    # Reset specific tool states
    st.session_state.current_state = 'upload'
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []


# --- LANDING PAGE ---
def show_landing_page(auth_url):
    """Displays the AARRR-framework-based landing page."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            .main { background-color: #FFFFFF; font-family: 'Inter', sans-serif; }
            .hero-container { padding: 4rem 1rem; text-align: center; }
            .hero-title { font-size: 3.5rem; font-weight: 700; background: -webkit-linear-gradient(45deg, #004080, #007BFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; }
            .hero-subtitle { font-size: 1.25rem; color: #555; max-width: 700px; margin: 0 auto 2rem auto; line-height: 1.6; }
            .how-it-works-card { padding: 1.5rem; text-align: center; }
            .how-it-works-card .step-number { display: inline-block; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; background-color: #E6F2FF; color: #007BFF; font-weight: 700; font-size: 1.2rem; margin-bottom: 1rem; }
            .comparison-table-premium { width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #E0E0E0; }
            .comparison-table-premium th, .comparison-table-premium td { padding: 1.2rem 1rem; text-align: left; border-bottom: 1px solid #E0E0E0; }
            .comparison-table-premium th { background-color: #F8F9FA; color: #333; font-weight: 600; }
            .comparison-table-premium tbody tr:last-child td { border-bottom: none; }
            .comparison-table-premium .check { color: #1E90FF; font-weight: bold; text-align: center; font-size: 1.2rem; }
            .comparison-table-premium .cross { color: #B0B0B0; font-weight: bold; text-align: center; font-size: 1.2rem; }
            .cta-button a { font-size: 1.1rem !important; font-weight: 600 !important; padding: 0.8rem 2rem !important; border-radius: 8px !important; background-image: linear-gradient(45deg, #007BFF, #0056b3) !important; border: none !important; transition: transform 0.2s, box-shadow 0.2s !important; }
            .cta-button a:hover { transform: scale(1.05); box-shadow: 0 6px 15px rgba(0, 123, 255, 0.3); }
            .section-header { text-align: center; color: #004080; font-weight: 700; margin-top: 4rem; margin-bottom: 2rem; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">From Classroom Chaos to Concept Clarity</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Stop drowning in disorganized notes and endless lectures. Vekkam transforms your course materials into a powerful, interactive knowledge base that helps you study smarter, not harder.</p>', unsafe_allow_html=True)
        st.link_button("Activate Your Smart Study Hub", auth_url, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<h2 class="section-header">Your Path to Mastery in 3 Simple Steps</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("""<div class="how-it-works-card"><div class="step-number">1</div><h3>Aggregate Your Materials</h3><p>Upload everything—audio lectures, textbook chapters, slide decks, and even whiteboard photos. Consolidate your entire syllabus in one place.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="how-it-works-card"><div class="step-number">2</div><h3>Synthesize & Understand</h3><p>Vekkam's AI analyzes and structures your content, creating a unified set of clear, editable notes. See the connections you never knew existed.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="how-it-works-card"><div class="step-number">3</div><h3>Query, Test & Master</h3><p>Chat with your personal AI tutor and generate mock tests directly from your notes. Turn passive knowledge into active, exam-ready expertise.</p></div>""", unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">A Knowledge Base That Grows With You</h2>', unsafe_allow_html=True)
    st.markdown("""Vekkam isn't just for one-time cramming. Every session you create builds upon the last, creating a personal, searchable library of your entire academic career. Your Personal TA becomes more intelligent about your curriculum over time, making it an indispensable tool for finals, comprehensive exams, and lifelong learning.""")
    st.markdown('<h2 class="section-header">The Unfair Advantage Over Other Tools</h2>', unsafe_allow_html=True)
    st.markdown("""
        <table class="comparison-table-premium">
            <thead><tr><th>Capability</th><th>Vekkam</th><th>ChatGPT / Gemini</th><th>Turbolearn</th><th>Perplexity</th></tr></thead>
            <tbody>
                <tr><td><strong>Multi-Modal Synthesis (Audio, PDF, IMG)</strong></td><td class="check">✔</td><td class="cross">Partial</td><td class="cross">YouTube Only</td><td class="cross">URL/Text Only</td></tr>
                <tr><td><strong>Chat With <u>Your</u> Content Only</strong></td><td class="check">✔</td><td class="cross">✖ (General)</td><td class="check">✔</td><td class="cross">✖ (Web Search)</td></tr>
                <tr><td><strong>Integrated Mock Test Generator</strong></td><td class="check">✔</td><td class="cross">✖</td><td class="cross">✖</td><td class="cross">✖</td></tr>
                <tr><td><strong>Builds a Persistent Knowledge Base</strong></td><td class="check">✔</td><td class="cross">✖ (Chat History)</td><td class="check">✔</td><td class="cross">✖</td></tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header" style="margin-top:2rem;">Ready to Stop Studying Harder and Start Studying Smarter?</h2>', unsafe_allow_html=True)
        st.link_button("Get Started for Free", auth_url, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.session_state.all_chunks = []
        st.session_state.all_chunks.extend([c for r in results if r and r['status'] == 'success' for c in r['chunks']])
        st.session_state.extraction_failures = [r for r in results if r and r['status'] == 'error']
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
                    matched_chunks = chunk_ids; break
        text_to_synthesize = "\n\n---\n\n".join([chunks_map.get(cid, "") for cid in matched_chunks if cid in chunks_map])
        content = "Could not find source text for this topic." if not text_to_synthesize.strip() else synthesize_note_block(topic, text_to_synthesize, st.session_state.synthesis_instructions)
        st.session_state.final_notes.append({"topic": topic, "content": content, "source_chunks": matched_chunks})
    if st.session_state.get('user_info') and st.session_state.final_notes:
        user_id = st.session_state.user_info.get('id') or st.session_state.user_info.get('email')
        save_session_to_history(user_id, st.session_state.final_notes)
    st.session_state.current_state = 'results'
    st.rerun()

def show_results_state():
    st.header("Your Unified Notes")
    col_actions1, col_actions2, _ = st.columns([1, 1, 3])
    with col_actions1:
        if st.button("Go to Workspace"): st.session_state.current_state = 'workspace'; st.rerun()
    with col_actions2:
        if st.button("Start New Session"): reset_session(st.session_state.tool_choice); st.rerun()
    st.divider()
    if 'selected_note_index' not in st.session_state: st.session_state.selected_note_index = None
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.subheader("Topics")
        for i, block in enumerate(st.session_state.final_notes):
            if st.button(block['topic'], key=f"topic_{i}", use_container_width=True):
                st.session_state.selected_note_index = i
    with col2:
        st.subheader("Content Viewer")
        if st.session_state.selected_note_index is not None:
            selected_note = st.session_state.final_notes[st.session_state.selected_note_index]
            tab1, tab2 = st.tabs(["Formatted Output", "Source Chunks"])
            with tab1:
                st.markdown(f"### {selected_note['topic']}")
                st.markdown(selected_note['content'])
            with tab2:
                st.markdown("These are the raw text chunks used to generate the note.")
                st.code('\n\n'.join(selected_note['source_chunks']))
        else:
            st.info("👆 Select a topic from the left to view its details.")
    st.divider()
    st.subheader("Communicate with these Notes")
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about the notes you just generated..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                current_notes_context = "\n\n".join([note['content'] for note in st.session_state.final_notes])
                response = answer_from_context(prompt, current_notes_context)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- UI STATE FUNCTION for PERSONAL TA ---
def show_personal_ta_ui():
    st.header("🎓 Your Personal TA")
    st.markdown("Ask questions and get answers based on the knowledge from all your past study sessions.")
    user_id = st.session_state.user_info.get('id') or st.session_state.user_info.get('email')
    user_data = load_user_data(user_id)
    if not user_data or not user_data["sessions"]:
        st.warning("You don't have any saved study sessions yet. Create some notes first to power up your TA!")
        return
    if "ta_messages" not in st.session_state: st.session_state.ta_messages = []
    for message in st.session_state.ta_messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask your Personal TA..."):
        st.session_state.ta_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Consulting your past notes..."):
                full_context = [f"Topic: {note['topic']}\nContent: {note['content']}" for session in user_data["sessions"] for note in session["notes"]]
                context_str = "\n\n---\n\n".join(full_context)
                response = answer_from_context(prompt, context_str)
                st.markdown(response)
        st.session_state.ta_messages.append({"role": "assistant", "content": response})

# --- UI STATE FUNCTIONS for MOCK TEST GENERATOR ---
def show_mock_test_generator():
    """Main function to handle the multi-stage mock test generation and execution."""
    st.header("📝 Mock Test Generator")

    # Initialize session state variables for the entire test flow
    if 'test_stage' not in st.session_state: st.session_state.test_stage = 'start'
    if 'syllabus' not in st.session_state: st.session_state.syllabus = ""
    if 'questions' not in st.session_state: st.session_state.questions = {}
    if 'user_answers' not in st.session_state: st.session_state.user_answers = {}
    if 'score' not in st.session_state: st.session_state.score = {}
    if 'feedback' not in st.session_state: st.session_state.feedback = {}

    # State machine to render the correct UI for each stage
    stage = st.session_state.test_stage
    
    # Define a map for stages to functions for cleaner routing
    stage_map = {
        'start': render_syllabus_input,
        'mcq_generating': lambda: render_generating_questions('mcq', 'vsa_generating', 10),
        'mcq_test': render_mcq_test,
        'mcq_results': lambda: render_mcq_results(7, 'vsa_generating'),
        'vsa_generating': lambda: render_generating_questions('vsa', 'vsa_test', 10),
        'vsa_test': lambda: render_subjective_test('vsa', 'vsa_results'),
        'vsa_results': lambda: render_subjective_results('vsa', 75, 'sa_generating'),
        'sa_generating': lambda: render_generating_questions('sa', 'sa_test', 5),
        'sa_test': lambda: render_subjective_test('sa', 'sa_results'),
        'sa_results': lambda: render_subjective_results('sa', 80, 'la_generating'),
        'la_generating': lambda: render_generating_questions('la', 'la_test', 3),
        'la_test': lambda: render_subjective_test('la', 'la_results'),
        'la_results': lambda: render_subjective_results('la', 90, 'final_results'),
        'final_results': render_final_results
    }
    
    # Execute the function corresponding to the current stage
    if stage in stage_map:
        stage_map[stage]()
    else: # Fallback
        st.session_state.test_stage = 'start'
        st.rerun()

def render_syllabus_input():
    st.subheader("Step 1: Provide Your Syllabus")
    st.write("Paste the syllabus or topic outline you want to be tested on. The more detail you provide, the better the questions will be.")
    syllabus_text = st.text_area("Syllabus / Topics", height=250, key="syllabus_input_area")
    if st.button("Generate My Test", type="primary"):
        if len(syllabus_text) < 50:
            st.warning("Please provide a more detailed syllabus for best results.")
        else:
            st.session_state.syllabus = syllabus_text
            st.session_state.test_stage = 'mcq_generating'
            st.rerun()

def render_generating_questions(q_type, next_stage, q_count):
    type_map = {'mcq': 'Multiple Choice', 'vsa': 'Very Short Answer', 'sa': 'Short Answer', 'la': 'Long Answer'}
    with st.spinner(f"Generating {type_map[q_type]} questions..."):
        questions_json = generate_questions_from_syllabus(st.session_state.syllabus, q_type, q_count)
        if questions_json and "questions" in questions_json:
            st.session_state.questions[q_type] = questions_json["questions"]
            st.session_state.test_stage = next_stage
            st.rerun()
        else:
            st.error(f"Failed to generate {type_map[q_type]} questions. Please try again with a different syllabus.")
            st.session_state.test_stage = 'start'
            st.rerun()

def render_mcq_test():
    st.subheader("Stage 1: Multiple Choice Questions")
    st.write("Pass Mark: 70%")
    mcq_questions = st.session_state.questions.get('mcq', [])
    with st.form("mcq_form"):
        user_answers = {}
        for i, q in enumerate(mcq_questions):
            st.markdown(f"**{i+1}. {q['question_text']}**")
            options = sorted(q['options'].items())
            selected_option = st.radio("Select your answer:", [opt[1] for opt in options], key=q['question_id'], label_visibility="collapsed")
            if selected_option:
                user_answers[q['question_id']] = next(key for key, value in options if value == selected_option)
            st.divider()
        if st.form_submit_button("Submit Answers"):
            st.session_state.user_answers['mcq'] = user_answers
            score = sum(1 for q in mcq_questions if user_answers.get(q['question_id']) == q['answer'])
            st.session_state.score['mcq'] = score
            st.session_state.test_stage = 'mcq_results'
            st.rerun()

def render_mcq_results(pass_mark_percent, next_stage):
    score = st.session_state.score.get('mcq', 0)
    total = len(st.session_state.questions.get('mcq', []))
    st.subheader(f"MCQ Results: You scored {score} / {total}")

    if 'feedback' not in st.session_state or 'mcq' not in st.session_state.feedback:
        with st.spinner("Analyzing your performance..."):
            feedback_text = generate_feedback_on_performance(score, total, st.session_state.questions.get('mcq', []), st.session_state.user_answers.get('mcq', {}), st.session_state.syllabus)
            st.session_state.feedback['mcq'] = feedback_text
    
    with st.container(border=True):
        st.subheader("💡 Performance Feedback")
        st.write(st.session_state.feedback.get('mcq', "No feedback generated."))
        
    if (score / total * 100) >= pass_mark_percent:
        st.success("Congratulations! You've passed this stage.")
        if st.button("Proceed to Very Short Answers", type="primary"):
            st.session_state.test_stage = next_stage
            st.rerun()
    else:
        st.error(f"You need to score at least {pass_mark_percent}% to proceed. Please review the material and try again.")
        if st.button("Restart Test"):
            for key in ['test_stage', 'syllabus', 'questions', 'user_answers', 'score', 'feedback']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

def render_subjective_test(q_type, next_stage):
    type_map = {'vsa': 'Very Short Answer', 'sa': 'Short Answer', 'la': 'Long Answer'}
    pass_map = {'vsa': '75%', 'sa': '80%', 'la': '90%'}
    st.subheader(f"Stage: {type_map[q_type]} Questions")
    st.write(f"Pass Mark: {pass_map[q_type]}")
    
    questions = st.session_state.questions.get(q_type, [])
    with st.form(f"{q_type}_form"):
        user_answers = {}
        for i, q in enumerate(questions):
            st.markdown(f"**{i+1}. {q['question_text']}**")
            answer = st.text_area("Your Answer:", key=f"{q_type}_answer_{q['question_id']}", height=150 if q_type != 'la' else 300)
            is_approach = st.checkbox("This is an outline of my approach", key=f"{q_type}_approach_{q['question_id']}")
            user_answers[q['question_id']] = {"answer": answer, "is_approach": is_approach}
            st.divider()
        if st.form_submit_button("Submit Answers"):
            st.session_state.user_answers[q_type] = user_answers
            st.session_state.test_stage = next_stage
            st.rerun()

def render_subjective_results(q_type, pass_mark_percent, next_stage):
    type_map = {'vsa': 'Very Short Answer', 'sa': 'Short Answer', 'la': 'Long Answer'}
    next_stage_map = {'sa_generating': 'Short Answers', 'la_generating': 'Long Answers', 'final_results': 'Final Results'}
    
    with st.spinner(f"Grading your {type_map[q_type]} answers... This may take a moment."):
        if 'score' not in st.session_state or q_type not in st.session_state.score:
            grading_result = grade_subjective_answers(
                q_type,
                st.session_state.questions.get(q_type, []),
                st.session_state.user_answers.get(q_type, {})
            )
            if grading_result:
                st.session_state.score[q_type] = grading_result['total_score']
                st.session_state.feedback[q_type] = grading_result
            else:
                st.error("Could not grade answers. Please try again.")
                st.session_state.score[q_type] = 0
                st.session_state.feedback[q_type] = {}

    score = st.session_state.score.get(q_type, 0)
    total = len(st.session_state.questions.get(q_type, []))
    feedback = st.session_state.feedback.get(q_type, {})

    st.subheader(f"{type_map[q_type]} Results: You scored {score} / {total}")

    with st.container(border=True):
        st.subheader("💡 Detailed Feedback")
        for fb in feedback.get('feedback_per_question', []):
            q_text = next((q['question_text'] for q in st.session_state.questions.get(q_type, []) if q['question_id'] == fb['question_id']), "Unknown Question")
            st.markdown(f"**Question:** {q_text}")
            st.markdown(f"**Score:** {fb['score_awarded']}/{fb['max_score']}")
            st.markdown(f"**Reasoning:** {fb['reasoning']}")
            st.divider()

    if (score / total * 100) >= pass_mark_percent:
        st.success("Congratulations! You've passed this stage.")
        if st.button(f"Proceed to {next_stage_map[next_stage]}", type="primary"):
            st.session_state.test_stage = next_stage
            st.rerun()
    else:
        st.error(f"You need to score at least {pass_mark_percent}% to proceed. Please review the feedback and try again.")
        if st.button(f"Restart {type_map[q_type]} Test"):
            st.session_state.test_stage = f"{q_type}_generating"
            # Clear previous answers for this stage
            if q_type in st.session_state.user_answers: del st.session_state.user_answers[q_type]
            if q_type in st.session_state.score: del st.session_state.score[q_type]
            if q_type in st.session_state.feedback: del st.session_state.feedback[q_type]
            st.rerun()

def render_final_results():
    st.balloons()
    st.header("🎉 Congratulations! You have completed the test! 🎉")
    st.markdown("You have demonstrated a strong understanding of the material across multiple cognitive levels.")

    st.subheader("Final Score Summary")
    # Display scores for all stages
    
    if st.button("Take a New Test", type="primary"):
        for key in ['test_stage', 'syllabus', 'questions', 'user_answers', 'score', 'feedback']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()


# --- AI & Utility Functions for Mock Test ---
def get_bloom_level_name(level):
    if level is None: return "N/A"
    levels = {1: "Remembering", 2: "Understanding", 3: "Applying", 4: "Analyzing", 5: "Evaluating"}
    return levels.get(level, "Unknown")

@gemini_api_call_with_retry
def generate_questions_from_syllabus(syllabus_text, question_type, question_count):
    model = genai.GenerativeModel('gemini-2.-flash-lite')
    
    type_instructions = {
        'mcq': """Generate {question_count} Multiple Choice Questions (MCQs). Each question object must have: `question_id`, `taxonomy_level`, `question_text`, `options` (an object with A,B,C,D), and `answer` (the correct key).""",
        'vsa': """Generate {question_count} Very Short Answer questions (1-2 sentences). Each question object must have: `question_id`, `taxonomy_level`, `question_text`, and a `grading_rubric` string detailing the key points for a correct answer.""",
        'sa': """Generate {question_count} Short Answer questions (1-2 paragraphs). Each question object must have: `question_id`, `taxonomy_level`, `question_text`, and a `grading_rubric` string explaining the concepts and structure of a good answer.""",
        'la': """Generate {question_count} Long Answer questions (multiple paragraphs). Each question object must have: `question_id`, `taxonomy_level`, `question_text`, and a `grading_rubric` string providing a detailed breakdown of marks for structure, arguments, and conclusion."""
    }

    prompt = f"""
    You are an expert Question Paper Setter. Your task is to create a high-quality assessment based STRICTLY on the provided syllabus.
    {type_instructions[question_type].format(question_count=question_count)}

    **Syllabus:**
    ---
    {syllabus_text}
    ---

    **General Instructions:**
    1.  **Strict Syllabus Adherence:** Do NOT include questions on topics outside the syllabus.
    2.  **Bloom's Taxonomy:** Distribute questions across cognitive levels (1-5).
    3.  **Output Format:** Your entire output must be a single, valid JSON object with a root key "questions", which is a list of question objects.

    Generate the JSON now.
    """
    response = model.generate_content(prompt)
    return resilient_json_parser(response.text)

@gemini_api_call_with_retry
def grade_subjective_answers(q_type, questions, user_answers):
    model = genai.GenerativeModel('gemini-2.-flash-lite')
    
    prompt = f"""
    You are a strict but fair AI examiner. Your task is to grade a student's answers based on a provided rubric. Some students may provide a full answer, while others may outline their approach; you must grade both fairly. An outlined approach that hits all the key points in the rubric should receive full marks.

    **Instructions:**
    1.  For each question, compare the student's answer to the `grading_rubric`.
    2.  If the student's response is an 'approach', evaluate if the outlined steps logically lead to the correct answer as per the rubric.
    3.  Award a score of 1 for a correct answer/approach, and 0 for an incorrect one. No partial marks.
    4.  Provide a concise, one-sentence reasoning for your grading decision.
    5.  Output a single valid JSON object with a root key "total_score" (integer) and "feedback_per_question" (a list of objects).
    6.  Each object in the feedback list must have: `question_id`, `score_awarded` (0 or 1), `max_score` (always 1), and `reasoning` (string).

    **Student's Test Paper:**
    ---
    {json.dumps({"questions": questions, "answers": user_answers}, indent=2)}
    ---

    Grade the paper and generate the JSON output now.
    """
    response = model.generate_content(prompt)
    return resilient_json_parser(response.text)

@gemini_api_call_with_retry
def generate_feedback_on_performance(score, total, questions, user_answers, syllabus):
    model = genai.GenerativeModel('gemini-2.-flash-lite')
    incorrect_questions = []
    for q in questions:
        q_id = q['question_id']
        if user_answers.get(q_id) != q['answer']:
            incorrect_questions.append({ "question": q['question_text'], "your_answer": user_answers.get(q_id), "correct_answer": q['answer'] })
    
    prompt = f"""
    You are an encouraging academic coach. A student scored {score}/{total} on a test covering: {syllabus}.
    Their incorrect answers were: {json.dumps(incorrect_questions, indent=2)}.
    Provide concise, actionable feedback in bullet points, identifying patterns and suggesting specific areas for improvement.
    """
    response = model.generate_content(prompt)
    return response.text

# --- AI & Utility Functions for Mastery Engine ---
@st.cache_resource
def get_google_search_service():
    """Initializes and returns the Google Custom Search API service, cached for performance."""
    try:
        api_key = st.secrets["google_search"]["api_key"]
        return build('customsearch', 'v1', developerKey=api_key)
    except KeyError:
        st.error("Google Search API key ('api_key') not found in st.secrets.toml. Please add it.")
        return None
    except Exception as e:
        st.error(f"Failed to build Google Search service: {e}")
        return None

def generate_allele_from_query(user_topic, context_chunks=None):
    model = genai.GenerativeModel('gemini-2.-flash-lite')
    gene_name_response = model.generate_content(f"Provide a concise, 3-5 word conceptual name for the topic: '{user_topic}'. Output only the name.")
    gene_name = gene_name_response.text.strip().replace('"', '') if gene_name_response.text else user_topic.title()
    gene_id = f"USER_{hashlib.md5(user_topic.encode()).hexdigest()[:8].upper()}"
    
    service = get_google_search_service();
    if not service: return None
    try: cse_id = st.secrets["google_search"]["cse_id"]
    except KeyError: st.error("Google Search CSE ID ('cse_id') not found in st.secrets.toml."); return None

    search_queries, synthesis_prompt_template = [], ""
    if context_chunks:
        st.info("Context detected. Generating contextual search queries...")
        full_context_text = " ".join([chunk.get('text', '') for chunk in context_chunks])
        summary_response = model.generate_content(f"Summarize the key topics from this material in 2-3 sentences: {full_context_text[:8000]}")
        context_summary = summary_response.text
        query_gen_prompt = f"""A student is studying: "{context_summary}". They now want to learn about: "{user_topic}". Generate 3 specific Google search queries to find information connecting their existing knowledge to this new topic. Output ONLY a valid JSON object with a key "queries" which is a list of 3 strings."""
        queries_response = model.generate_content(query_gen_prompt)
        queries_json = resilient_json_parser(queries_response.text)
        search_queries = queries_json['queries'] if queries_json and 'queries' in queries_json else [f"{user_topic} in context of {context_summary[:50]}"]
        synthesis_prompt_template = f"""You are an expert tutor. A student's study material is about: "{context_summary}". They want to understand "{user_topic}". Based ONLY on the provided web search results, synthesize a clear explanation of "{user_topic}", connecting it to their existing knowledge.\n\n**Web Search Results:**\n---\n{{search_snippets}}\n---"""
    else:
        search_queries = [f"{user_topic} explanation", f"youtube video tutorial {user_topic}"]
        synthesis_prompt_template = f"""Based ONLY on the web search results, provide a clear, comprehensive explanation of '{user_topic}'.\n\n**Web Search Results:**\n---\n{{search_snippets}}\n---"""

    text_content, video_url = [], None
    st.write("Performing targeted web search...")
    for query in search_queries:
        try:
            res = service.cse().list(q=query, cx=cse_id, num=3).execute()
            for item in res.get('items', []):
                if "youtube.com/watch" in item.get('link', '') and not video_url: video_url = item.get('link')
                if item.get('snippet'): text_content.append(item.get('snippet'))
        except HttpError as e: st.error(f"Google Search API error: {e}. Check key, CSE ID, and quota."); return None
        except Exception as e: st.warning(f"Search failed for query '{query}': {e}"); time.sleep(1)

    if not text_content: st.error(f"Could not find relevant text for '{user_topic}'."); return None
    
    with st.spinner("Synthesizing explanation from search results..."):
        synthesis_prompt = synthesis_prompt_template.format(search_snippets=" ".join(text_content))
        final_explanation = model.generate_content(synthesis_prompt).text

    content_alleles = []
    if final_explanation: content_alleles.append({"type": "text", "content": final_explanation})
    if video_url: content_alleles.append({"type": "video", "url": video_url})
    return {"gene_id": gene_id, "gene_name": gene_name, "difficulty": 0, "content_alleles": content_alleles}

# --- UI STATE FUNCTIONS for MASTERY ENGINE ---
def show_mastery_engine():
    st.header("🏆 Mastery Engine")
    if 'mastery_stage' not in st.session_state: st.session_state.mastery_stage = 'course_selection'
    if 'user_progress' not in st.session_state: st.session_state.user_progress = {}
    if 'current_genome' not in st.session_state: st.session_state.current_genome = None

    stage = st.session_state.mastery_stage
    stage_map = {'course_selection': render_course_selection, 'skill_tree': render_skill_tree, 'content_viewer': render_content_viewer, 'boss_battle': render_boss_battle}
    if stage in stage_map: stage_map[stage]()

def render_course_selection():
    st.subheader("Select Your Course or Create a New Concept")
    st.markdown("### Pre-built Courses")
    if st.button("Econ 101", use_container_width=True, type="primary"):
        st.session_state.current_genome = json.loads(json.dumps(ECON_101_GENOME)) # Deep copy
        progress = {}
        all_node_ids = {node['gene_id'] for node in st.session_state.current_genome['nodes']}
        destination_nodes = {edge['to'] for edge in st.session_state.current_genome['edges']}
        root_nodes = all_node_ids - destination_nodes
        for node_id in all_node_ids: progress[node_id] = 'unlocked' if node_id in root_nodes else 'locked'
        st.session_state.user_progress = progress
        st.session_state.mastery_stage = 'skill_tree'
        st.rerun()

    st.markdown("### Create Your Own Concept")
    user_interest = st.text_input("What concept are you interested in learning about?", key="user_allele_query")
    use_context = st.checkbox("Use context from 'Note & Lesson Engine' session", key="use_context", value=True)
    if st.button("Generate Concept Allele", use_container_width=True, disabled=not user_interest):
        context_chunks = st.session_state.get('all_chunks') if use_context else None
        if use_context and not context_chunks: st.warning("Contextual generation selected, but no files processed. Falling back to non-contextual mode.")
        if st.session_state.current_genome is None: st.session_state.current_genome = {"subject": "My Custom Concepts", "version": "1.0", "nodes": [], "edges": []}
        with st.spinner(f"Generating concept for '{user_interest}'..."):
            new_allele = generate_allele_from_query(user_interest, context_chunks=context_chunks)
            if new_allele:
                if new_allele['gene_id'] not in {n['gene_id'] for n in st.session_state.current_genome['nodes']}:
                    st.session_state.current_genome['nodes'].append(new_allele)
                    st.success(f"Concept '{new_allele['gene_name']}' generated and unlocked!")
                else: st.info(f"Concept '{new_allele['gene_name']}' already exists.")
                st.session_state.user_progress[new_allele['gene_id']] = 'unlocked'
                st.session_state.mastery_stage = 'skill_tree'
                st.rerun()

def render_skill_tree():
    st.subheader(f"Skill Tree: {st.session_state.current_genome['subject']}")
    nodes, progress = st.session_state.current_genome['nodes'], st.session_state.user_progress
    for node in nodes:
        node_id, node_name, status = node['gene_id'], node['gene_name'], progress.get(node_id, 'locked')
        if status == 'mastered': st.success(f"**{node_name}** - ✅ Mastered!", icon="✅")
        elif status == 'unlocked':
            if st.button(f"🧠 Learn: {node_name}", key=node_id, use_container_width=True, type="primary"):
                st.session_state.selected_node_id = node_id; st.session_state.mastery_stage = 'content_viewer'; st.rerun()
        else: st.info(f"**{node_name}** - 🔒 Locked", icon="🔒")
        st.markdown('<p style="text-align: center; margin: 0; padding: 0;">↓</p>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Back to Course Selection"): st.session_state.mastery_stage = 'course_selection'; st.rerun()

def render_content_viewer():
    node_id = st.session_state.selected_node_id
    node_data = next((n for n in st.session_state.current_genome['nodes'] if n['gene_id'] == node_id), None)
    if not node_data: st.error("Error: Could not load node data."); st.session_state.mastery_stage = 'skill_tree'; st.rerun(); return
    st.subheader(f"Learning: {node_data['gene_name']}")
    st.markdown("---")
    for allele in node_data['content_alleles']:
        if allele['type'] == 'text': st.markdown(allele['content'])
        elif allele['type'] == 'video': st.video(allele['url'])
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    if col1.button("Back to Skill Tree"): st.session_state.mastery_stage = 'skill_tree'; st.rerun()
    if col2.button(f"⚔️ Challenge Boss: {node_data['gene_name']}", type="primary"):
        st.session_state.mastery_stage = 'boss_battle'
        syllabus_parts = [a['content'] for a in node_data['content_alleles'] if a['type'] == 'text']
        st.session_state.syllabus = f"Topic: {node_data['gene_name']}. Content: {' '.join(syllabus_parts)}"
        st.session_state.test_stage = 'mcq_generating' # Start the test flow from MCQ
        for key in ['questions', 'user_answers', 'score', 'feedback']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

def render_boss_battle():
    node_id = st.session_state.selected_node_id
    node_data = next((n for n in st.session_state.current_genome['nodes'] if n['gene_id'] == node_id), None)
    st.subheader(f"Boss Battle: {node_data['gene_name']}")
    # The main show_mock_test_generator now handles the state machine for the battle
    show_mock_test_generator()


# --- MAIN APP ---
def main():
    if 'user_info' not in st.session_state: st.session_state.user_info = None
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
    except KeyError:
        st.error("Gemini API key ('api_key') not found in st.secrets.toml. Please add it."); st.stop()

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
        st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
        auth_url, _ = flow.authorization_url(prompt='consent')
        show_landing_page(auth_url)
        return

    st.sidebar.title("Vekkam Engine")
    user = st.session_state.user_info
    user_id = user.get('id') or user.get('email')
    st.sidebar.image(user['picture'], width=80)
    st.sidebar.subheader(f"Welcome, {user['given_name']}")
    if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()
    st.sidebar.divider()

    st.sidebar.subheader("Study Session History")
    user_data = load_user_data(user_id)
    if not user_data["sessions"]:
        st.sidebar.info("Your saved sessions will appear here.")
    else:
        for i, session in enumerate(list(user_data["sessions"])):
            with st.sidebar.expander(f"{session.get('timestamp', 'N/A')} - {session.get('title', 'Untitled')}"):
                is_editing = st.session_state.get('editing_session_id') == session.get('id')
                if is_editing:
                    new_title = st.text_input("Edit Title", value=session.get('title', ''), key=f"edit_title_{session.get('id')}", label_visibility="collapsed")
                    col1, col2 = st.columns(2)
                    if col1.button("Save", key=f"save_{session.get('id')}", type="primary", use_container_width=True):
                        user_data["sessions"][i]['title'] = new_title; save_user_data(user_id, user_data); st.session_state.editing_session_id = None; st.rerun()
                    if col2.button("Cancel", key=f"cancel_{session.get('id')}", use_container_width=True):
                        st.session_state.editing_session_id = None; st.rerun()
                else:
                    for note in session.get('notes', []): st.write(f"• {note.get('topic', 'No Topic')}")
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    if col1.button("👁️ View", key=f"view_{session.get('id')}", use_container_width=True):
                        reset_session("Note & Lesson Engine"); st.session_state.final_notes = session.get('notes', []); st.session_state.current_state = 'results'; st.session_state.messages = []; st.rerun()
                    if col2.button("✏️ Edit", key=f"edit_{session.get('id')}", use_container_width=True):
                        st.session_state.editing_session_id = session.get('id'); st.rerun()
                    if col3.button("🗑️ Delete", key=f"del_{session.get('id')}", type="secondary", use_container_width=True):
                        user_data["sessions"].pop(i); save_user_data(user_id, user_data); st.rerun()

    st.sidebar.divider()
    tool_choice = st.sidebar.radio("Select a Tool", ("Note & Lesson Engine", "Personal TA", "Mock Test Generator", "Mastery Engine"), key='tool_choice')
    
    if 'last_tool_choice' not in st.session_state: st.session_state.last_tool_choice = tool_choice
    if st.session_state.last_tool_choice != tool_choice:
        st.session_state.last_tool_choice = tool_choice
        reset_session(tool_choice); st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("API Status")
    st.sidebar.write(f"Gemini: **{check_gemini_api()}**")

    # --- Tool Routing ---
    if tool_choice == "Note & Lesson Engine":
        if 'current_state' not in st.session_state: st.session_state.current_state = 'upload'
        state_map = { 'upload': show_upload_state, 'processing': show_processing_state, 'workspace': show_workspace_state, 'synthesizing': show_synthesizing_state, 'results': show_results_state }
        state_map.get(st.session_state.current_state, show_upload_state)()
    elif tool_choice == "Personal TA": show_personal_ta_ui()
    elif tool_choice == "Mock Test Generator": show_mock_test_generator()
    elif tool_choice == "Mastery Engine": show_mastery_engine()

if __name__ == "__main__":
    main()
