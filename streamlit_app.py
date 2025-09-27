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
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    st.error("Required Google libraries not found! Please run: pip install google-auth-oauthlib google-api-python-client google-auth-httplib2")
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
GEMINI_MODEL = 'gemini-1.5-flash'

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
                wait_time = int(match.group(1)) + delay if match else delay * (2 ** retries)

                st.warning(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
            except Exception as e:
                st.error(f"An unexpected API error occurred in {func.__name__}: {e}")
                return None
        return None
    return wrapper

# --- GOOGLE API FUNCTIONS for Search & Sheets ---
def generate_alleles_from_search(topic, num_results=3):
    """Uses Google Custom Search API to find learning materials for a topic."""
    try:
        api_key = st.secrets["google_search"]["api_key"]
        cse_id = st.secrets["google_search"]["cse_id"]
        
        service = build("customsearch", "v1", developerKey=api_key)
        query = f"{topic} tutorial explained khan academy youtube"
        
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        
        alleles = []
        for item in res.get('items', []):
            source = item.get('displayLink', '').replace('www.', '')
            alleles.append({
                'type': 'video' if 'youtube.com' in source else 'text',
                'source': source,
                'title': item.get('title'),
                'url': item.get('link')
            })
        return alleles
    except HttpError as e:
        st.error(f"Google Search API Error: {e}. Check API key and CSE ID in secrets.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Search: {e}")
        return []

def append_to_google_sheet(credentials, spreadsheet_id, data_rows):
    """Appends rows of data to a specified Google Sheet."""
    try:
        service = build('sheets', 'v4', credentials=credentials)
        body = {'values': data_rows}
        
        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1",
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()
        return True
    except HttpError as e:
        st.error(f"Google Sheets API Error: {e}. Ensure API is enabled and spreadsheet ID is correct and shared with your service account email if applicable.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while writing to Google Sheets: {e}")
        return False

# --- PERSISTENT DATA STORAGE ---
def get_user_data_path(user_id):
    safe_filename = hashlib.md5(user_id.encode()).hexdigest() + ".json"
    return DATA_DIR / safe_filename

def load_user_data(user_id):
    filepath = get_user_data_path(user_id)
    if filepath.exists():
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"sessions": []}
    return {"sessions": []}

def save_user_data(user_id, data):
    filepath = get_user_data_path(user_id)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def save_session_to_history(user_id, final_notes):
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
        genai.get_model(GEMINI_MODEL)
        return "Valid"
    except Exception:
        st.sidebar.error("Gemini API Key in secrets is invalid.")
        return "Invalid"

def resilient_json_parser(json_string):
    try:
        # First, try to find a JSON block enclosed in markdown backticks
        match = re.search(r'```json\s*(\{.*?\})\s*```', json_string, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # If not found, try to find any valid JSON object in the string
        match = re.search(r'(\{.*?\})', json_string, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        st.error("Fatal Error: Could not parse a critical AI JSON response.")
        return None

def chunk_text(text, source_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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

# --- CONTENT PROCESSING ---
def process_source(file, source_type):
    try:
        source_id = f"{source_type}:{file.name}"
        model = genai.GenerativeModel(GEMINI_MODEL)
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
                response = model.generate_content(["Transcribe this audio recording.", audio_file])
                chunks = chunk_text(response.text, source_id)
                genai.delete_file(audio_file.name) # Clean up the uploaded file
                return {"status": "success", "source_id": source_id, "chunks": chunks}
            finally:
                os.unlink(tmp_path)
        elif source_type == 'image':
            image = Image.open(file)
            response = model.generate_content(["Extract all text from this image.", image])
            return {"status": "success", "source_id": source_id, "chunks": [{"chunk_id": f"{source_id}::chunk_0", "text": response.text}]}
        elif source_type == 'pdf':
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            chunks = chunk_text(text, source_id)
            return {"status": "success", "source_id": source_id, "chunks": chunks}
    except Exception as e:
        return {"status": "error", "source_id": f"{source_type}:{file.name}", "reason": str(e)}

# --- AGENTIC WORKFLOW FUNCTIONS ---
@gemini_api_call_with_retry
def generate_content_outline(all_chunks):
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt_chunks = [{"chunk_id": c['chunk_id'], "text_snippet": c['text'][:200] + "..."} for c in all_chunks if c.get('text') and len(c['text'].split()) > 10]
    
    if not prompt_chunks:
        st.error("Could not find enough content to generate an outline. Please check your uploaded files.")
        return None

    prompt = f"""
    You are a master curriculum designer. Analyze the content chunks and create a structured, logical topic outline.
    For each topic, you MUST list the `chunk_id`s that are most relevant. Base the outline STRICTLY on the provided content.
    Output ONLY a valid JSON object with a root key "outline", which is a list of objects. Each object must have keys "topic" (string) and "relevant_chunks" (a list of string chunk_ids).

    **Content Chunks:**
    ---
    {json.dumps(prompt_chunks, indent=2)}
    """
    response = model.generate_content(prompt)
    return resilient_json_parser(response.text)

@gemini_api_call_with_retry
def synthesize_note_block(topic, relevant_chunks_text, instructions):
    model = genai.GenerativeModel(GEMINI_MODEL)
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
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""
    You are a helpful study assistant. Answer the user question based strictly on the provided study material context.
    If the answer is not in the context, clearly state that the information is not available in the provided materials.

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
    """Initializes the Google OAuth flow with required scopes."""
    try:
        # The structure of st.secrets["google"] is expected to match the client_secret.json file.
        client_config = {"web": st.secrets["google"]}
        scopes = [
            "https://www.googleapis.com/auth/userinfo.profile", 
            "https://www.googleapis.com/auth/userinfo.email", 
            "openid",
            "https://www.googleapis.com/auth/spreadsheets" # Scope for Google Sheets API
        ]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets."); st.stop()

def reset_session(tool_choice):
    """Resets the session state, preserving user info and credentials."""
    user_info = st.session_state.get('user_info')
    credentials = st.session_state.get('credentials')
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.credentials = credentials
    st.session_state.tool_choice = tool_choice
    st.session_state.current_state = 'upload' # Default start state for Note Engine

# --- UI COMPONENTS & STATES ---

def show_landing_page(auth_url):
    """Displays the marketing/landing page for non-logged-in users."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            .main {
                background-color: #FFFFFF;
                font-family: 'Inter', sans-serif;
            }
            .hero-container {
                padding: 4rem 1rem;
                text-align: center;
            }
            .hero-title {
                font-size: 3.5rem;
                font-weight: 700;
                background: -webkit-linear-gradient(45deg, #004080, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
            }
            .hero-subtitle {
                font-size: 1.25rem;
                color: #555;
                max-width: 700px;
                margin: 0 auto 2rem auto;
                line-height: 1.6;
            }
            .how-it-works-card {
                padding: 1.5rem; text-align: center;
            }
            .how-it-works-card .step-number {
                display: inline-block; width: 40px; height: 40px; line-height: 40px;
                border-radius: 50%; background-color: #E6F2FF; color: #007BFF;
                font-weight: 700; font-size: 1.2rem; margin-bottom: 1rem;
            }
            .comparison-table-premium {
                width: 100%; border-collapse: separate; border-spacing: 0;
                border-radius: 12px; overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #E0E0E0;
            }
            .comparison-table-premium th, .comparison-table-premium td {
                padding: 1.2rem 1rem; text-align: left; border-bottom: 1px solid #E0E0E0;
            }
            .comparison-table-premium th {
                background-color: #F8F9FA; color: #333; font-weight: 600;
            }
            .comparison-table-premium tbody tr:last-child td { border-bottom: none; }
            .comparison-table-premium .check {
                color: #1E90FF; font-weight: bold; text-align: center; font-size: 1.2rem;
            }
            .comparison-table-premium .cross {
                color: #B0B0B0; font-weight: bold; text-align: center; font-size: 1.2rem;
            }
            .cta-button a {
                font-size: 1.1rem !important; font-weight: 600 !important;
                padding: 0.8rem 2rem !important; border-radius: 8px !important;
                background-image: linear-gradient(45deg, #007BFF, #0056b3) !important;
                border: none !important; transition: transform 0.2s, box-shadow 0.2s !important;
            }
            .cta-button a:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 15px rgba(0, 123, 255, 0.3);
            }
            .section-header {
                text-align: center; color: #004080; font-weight: 700;
                margin-top: 4rem; margin-bottom: 2rem;
            }
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
        st.markdown("""
            <div class="how-it-works-card">
                <div class="step-number">1</div>
                <h3>Aggregate Your Materials</h3>
                <p>Upload everything—audio lectures, textbook chapters, slide decks, and even whiteboard photos. Consolidate your entire syllabus in one place.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="how-it-works-card">
                <div class="step-number">2</div>
                <h3>Synthesize & Understand</h3>
                <p>Vekkam's AI analyzes and structures your content, creating a unified set of clear, editable notes. See the connections you never knew existed.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="how-it-works-card">
                <div class="step-number">3</div>
                <h3>Query, Test & Master</h3>
                <p>Chat with your personal AI tutor and generate mock tests directly from your notes. Turn passive knowledge into active, exam-ready expertise.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">A Knowledge Base That Grows With You</h2>', unsafe_allow_html=True)
    st.markdown("Vekkam isn't just for one-time cramming. Every session you create builds upon the last, creating a personal, searchable library of your entire academic career. Your Personal TA becomes more intelligent about your curriculum over time, making it an indispensable tool for finals, comprehensive exams, and lifelong learning.")

    st.markdown('<h2 class="section-header">The Unfair Advantage Over Other Tools</h2>', unsafe_allow_html=True)
    st.markdown("""
        <table class="comparison-table-premium">
            <thead>
                <tr>
                    <th>Capability</th>
                    <th>Vekkam</th>
                    <th>ChatGPT / Gemini</th>
                    <th>Turbolearn</th>
                    <th>Perplexity</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Multi-Modal Synthesis (Audio, PDF, IMG)</strong></td>
                    <td class="check">✔</td>
                    <td class="cross">Partial</td>
                    <td class="cross">YouTube Only</td>
                    <td class="cross">URL/Text Only</td>
                </tr>
                <tr>
                    <td><strong>Chat With <u>Your</u> Content Only</strong></td>
                    <td class="check">✔</td>
                    <td class="cross">✖ (General)</td>
                    <td class="check">✔</td>
                    <td class="cross">✖ (Web Search)</td>
                </tr>
                <tr>
                    <td><strong>Integrated Mock Test Generator</strong></td>
                    <td class="check">✔</td>
                    <td class="cross">✖</td>
                    <td class="cross">✖</td>
                    <td class="cross">✖</td>
                </tr>
                <tr>
                    <td><strong>Builds a Persistent Knowledge Base</strong></td>
                    <td class="check">✔</td>
                    <td class="cross">✖ (Chat History)</td>
                    <td class="check">✔</td>
                    <td class="cross">✖</td>
                </tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header" style="margin-top:2rem;">Ready to Stop Studying Harder and Start Studying Smarter?</h2>', unsafe_allow_html=True)
        st.link_button("Get Started for Free", auth_url, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

def show_upload_state():
    st.header("Note & Lesson Engine: Upload")
    uploaded_files = st.file_uploader("Select files", accept_multiple_files=True, type=['mp3', 'm4a', 'wav', 'png', 'jpg', 'pdf'])
    if st.button("Process Files", type="primary") and uploaded_files:
        st.session_state.initial_files = uploaded_files
        st.session_state.current_state = 'processing'
        st.rerun()

def show_processing_state():
    st.header("Initial Processing...")
    with st.spinner("Extracting content from all files... This may take a moment."):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_source, f, 'transcript' if f.type.startswith('audio/') else 'image' if f.type.startswith('image/') else 'pdf'): f for f in st.session_state.initial_files}
            for future in as_completed(futures):
                results.append(future.result())
        
        st.session_state.all_chunks = [c for r in results if r and r['status'] == 'success' for c in r['chunks']]
        st.session_state.extraction_failures = [r for r in results if r and r['status'] == 'error']
    st.session_state.current_state = 'workspace'
    st.rerun()

def show_workspace_state():
    st.header("Vekkam Workspace")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Controls & Outline")
        if st.button("Generate Outline & Find Resources", type="primary"):
            with st.spinner("AI is analyzing all content to create an outline..."):
                outline_json = generate_content_outline(st.session_state.all_chunks)
                
            if outline_json and "outline" in outline_json: 
                st.session_state.outline_data = outline_json["outline"]
                
                with st.spinner("Searching for learning resources and saving to Google Sheets..."):
                    try:
                        spreadsheet_id = st.secrets["google_sheets"]["spreadsheet_id"]
                        all_rows_to_append = [["Topic", "Type", "Source", "Title", "URL"]] # Header row
                        for item in st.session_state.outline_data:
                            topic = item['topic']
                            alleles = generate_alleles_from_search(topic)
                            for allele in alleles:
                                all_rows_to_append.append([topic, allele['type'], allele['source'], allele['title'], allele['url']])
                        
                        if len(all_rows_to_append) > 1:
                            credentials = st.session_state.get('credentials')
                            if credentials and append_to_google_sheet(credentials, spreadsheet_id, all_rows_to_append):
                                st.success(f"Appended {len(all_rows_to_append)-1} resources to your Google Sheet!")
                    except KeyError:
                        st.error("`spreadsheet_id` not found in st.secrets. Please add it to your secrets file.")
                    except Exception as e:
                        st.error(f"An error occurred during resource generation: {e}")
            else: 
                st.error("Failed to generate outline. The AI couldn't structure the provided content.")
        
        if st.session_state.get('outline_data'):
            initial_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
            st.session_state.editable_outline = st.text_area("Editable Outline:", value=initial_text, height=300)
            st.session_state.synthesis_instructions = st.text_area("Synthesis Instructions (Optional):", height=100, placeholder="e.g., 'Explain this like I'm 15' or 'Focus on key formulas'")
            if st.button("Synthesize Notes"):
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
    
    if st.session_state.get('user_info') and st.session_state.final_notes:
        user_id = st.session_state.user_info.get('id') or st.session_state.user_info.get('email')
        save_session_to_history(user_id, st.session_state.final_notes)

    st.session_state.current_state = 'results'
    st.rerun()

def show_results_state():
    st.header("Your Unified Notes")
    
    col_actions1, col_actions2, _ = st.columns([1, 1, 3])
    with col_actions1:
        if st.button("Go to Workspace"): 
            st.session_state.current_state = 'workspace'
            st.rerun()
    with col_actions2:
        if st.button("Start New Session"): 
            reset_session(st.session_state.tool_choice)
            st.rerun()

    st.divider()

    st.session_state.setdefault('selected_note_index', None)

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
                st.markdown("These are the raw text chunks from your source files that the AI used to generate the note.")
                st.code('\n\n'.join(selected_note['source_chunks']))
        else:
            st.info("👆 Select a topic from the left to view its details.")
    
    st.divider()
    st.subheader("Communicate with these Notes")
    
    st.session_state.setdefault("messages", [])
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the notes you just generated..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                current_notes_context = "\n\n".join([note['content'] for note in st.session_state.final_notes])
                response = answer_from_context(prompt, current_notes_context)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def show_personal_ta_ui():
    st.header("🎓 Your Personal TA")
    st.markdown("Ask questions and get answers based on the knowledge from all your past study sessions.")
    user_id = st.session_state.user_info.get('id') or st.session_state.user_info.get('email')
    user_data = load_user_data(user_id)

    if not user_data or not user_data["sessions"]:
        st.warning("You don't have any saved study sessions yet. Create some notes first to power up your TA!")
        return

    st.session_state.setdefault("ta_messages", [])

    for message in st.session_state.ta_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your Personal TA..."):
        st.session_state.ta_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting your past notes..."):
                full_context = []
                for session in user_data["sessions"]:
                    for note in session["notes"]:
                        full_context.append(f"Topic: {note['topic']}\nContent: {note['content']}")
                
                context_str = "\n\n---\n\n".join(full_context)
                response = answer_from_context(prompt, context_str)
                st.markdown(response)
        
        st.session_state.ta_messages.append({"role": "assistant", "content": response})

def show_mock_test_generator():
    """Main function to handle the multi-stage mock test generation and execution."""
    st.header("📝 Mock Test Generator")

    st.session_state.setdefault('test_stage', 'start')
    st.session_state.setdefault('syllabus', "")
    st.session_state.setdefault('questions', {})
    st.session_state.setdefault('user_answers', {})
    st.session_state.setdefault('score', {})
    st.session_state.setdefault('feedback', {})
    
    stage = st.session_state.test_stage
    if stage == 'start':
        render_syllabus_input()
    elif stage == 'generating':
        render_generating_questions()
    elif stage == 'mcq_test':
        render_mcq_test()
    elif stage == 'mcq_results':
        render_mcq_results()
    else:
        st.info("Further test stages (e.g., Fill-in-the-Blanks) are under development.")

def render_syllabus_input():
    """Renders the UI for the user to input their syllabus."""
    st.subheader("Step 1: Provide Your Syllabus")
    st.write("Paste the syllabus or topic outline you want to be tested on. The more detail you provide, the better the questions will be.")
    
    syllabus_text = st.text_area("Syllabus / Topics", height=250, key="syllabus_input_area")
    
    if st.button("Generate My Test", type="primary"):
        if len(syllabus_text) < 50:
            st.warning("Please provide a more detailed syllabus for best results.")
        else:
            st.session_state.syllabus = syllabus_text
            st.session_state.test_stage = 'generating'
            st.rerun()

def render_generating_questions():
    """Handles the background generation of questions for all stages."""
    with st.spinner("Building your test... The AI is analyzing the syllabus and crafting questions..."):
        questions_json = generate_questions_from_syllabus(st.session_state.syllabus, "MCQ", 10)
        
        if questions_json and "questions" in questions_json:
            st.session_state.questions['mcq'] = questions_json["questions"]
            st.session_state.test_stage = 'mcq_test'
            st.rerun()
        else:
            st.error("Failed to generate questions. The AI could not create a test from the provided syllabus. Please try again with a more detailed or different syllabus.")
            st.session_state.test_stage = 'start'
            st.rerun()

def render_mcq_test():
    """Renders the MCQ test form."""
    st.subheader("Stage 1: Multiple Choice Questions")
    st.write("Answer at least 7 out of 10 questions correctly to advance to the next stage.")
    mcq_questions = st.session_state.questions.get('mcq', [])
    
    if not mcq_questions:
        st.error("MCQ questions not found. Please restart the test.")
        if st.button("Restart"):
            st.session_state.test_stage = 'start'
            st.rerun()
        return

    with st.form("mcq_form"):
        user_answers = {}
        for i, q in enumerate(mcq_questions):
            st.markdown(f"**{i+1}. {q['question_text']}**")
            st.caption(f"Bloom's Taxonomy Level: {q.get('taxonomy_level', 'N/A')} ({get_bloom_level_name(q.get('taxonomy_level'))})")
            options_keys = sorted(q['options'].keys())
            options_values = [q['options'][key] for key in options_keys]
            
            selected_option_text = st.radio("Select your answer:", options_values, key=q['question_id'], label_visibility="collapsed")
            if selected_option_text:
                user_answers[q['question_id']] = options_keys[options_values.index(selected_option_text)]
            st.divider()

        submitted = st.form_submit_button("Submit Answers")
        if submitted:
            st.session_state.user_answers['mcq'] = user_answers
            score = sum(1 for q in mcq_questions if user_answers.get(q['question_id']) == q['answer'])
            st.session_state.score['mcq'] = score
            st.session_state.test_stage = 'mcq_results'
            st.rerun()

def render_mcq_results():
    """Displays the results of the MCQ test and provides feedback."""
    score = st.session_state.score.get('mcq', 0)
    total = len(st.session_state.questions.get('mcq', []))
    st.subheader(f"MCQ Results: You scored {score} / {total}")

    if 'mcq' not in st.session_state.feedback:
        with st.spinner("Analyzing your performance and generating feedback..."):
            all_questions = st.session_state.questions.get('mcq', [])
            user_answers = st.session_state.user_answers.get('mcq', {})
            feedback_text = generate_feedback_on_performance(score, total, all_questions, user_answers, st.session_state.syllabus)
            st.session_state.feedback['mcq'] = feedback_text
    
    with st.container(border=True):
        st.subheader("💡 Suggestions for Improvement")
        st.write(st.session_state.feedback.get('mcq', "No feedback generated."))
        
    if score >= 7:
        st.success("Congratulations! You've passed this stage.")
        if st.button("Proceed to Next Stage", type="primary"):
            st.info("Further stages are under development.")
    else:
        st.error("You need a score of 7/10 to proceed. Please review the material and try again.")
        if st.button("Restart Test"):
            for key in ['test_stage', 'syllabus', 'questions', 'user_answers', 'score', 'feedback']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

@gemini_api_call_with_retry
def generate_questions_from_syllabus(syllabus_text, question_type, question_count):
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""
    You are an expert Question Paper Setter and an authority on Bloom's Taxonomy. Generate {question_count} Multiple Choice Questions (MCQs) based STRICTLY on the provided syllabus.

    **Syllabus:**
    ---
    {syllabus_text}
    ---

    **Instructions:**
    1.  **Strict Syllabus Adherence:** Do NOT ask about topics outside the syllabus.
    2.  **Bloom's Taxonomy:** Distribute questions across cognitive skills (Remembering, Understanding, Applying, Analyzing).
    3.  **High-Quality Options:** Incorrect options (distractors) must be plausible but clearly wrong.
    4.  **Output Format:** Output a single, valid JSON object with a root key "questions", which is a list of question objects.
    5.  **Question Object Structure:** Each object must have: `question_id` (string), `taxonomy_level` (integer 1-5), `question_text` (string), `options` (object with keys "A", "B", "C", "D"), and `answer` (string key like "C").

    Generate the JSON now.
    """
    response = model.generate_content(prompt)
    return resilient_json_parser(response.text)

def get_bloom_level_name(level):
    """Maps a Bloom's Taxonomy level number to its name."""
    if level is None: return "N/A"
    levels = {1: "Remembering", 2: "Understanding", 3: "Applying", 4: "Analyzing", 5: "Evaluating"}
    return levels.get(level, "Unknown")

@gemini_api_call_with_retry
def generate_feedback_on_performance(score, total, questions, user_answers, syllabus):
    model = genai.GenerativeModel(GEMINI_MODEL)
    incorrect_questions = []
    for q in questions:
        q_id = q['question_id']
        if user_answers.get(q_id) != q['answer']:
            incorrect_questions.append({
                "question": q['question_text'],
                "correct_answer": q['options'][q['answer']],
                "user_answer": q['options'].get(user_answers.get(q_id), "Not Answered"),
                "taxonomy_level": q['taxonomy_level']
            })

    prompt = f"""
    You are an encouraging academic coach. A student scored {score}/{total} on a test for this syllabus: {syllabus}.
    Here are the questions they got wrong: {json.dumps(incorrect_questions, indent=2)}

    **Your Task:**
    1.  Start with encouragement.
    2.  Identify patterns in their mistakes (specific topics or cognitive skills like 'Applying' vs 'Remembering').
    3.  Provide specific, actionable advice for improvement.
    4.  Maintain a positive, motivating tone. Be concise and use bullet points.

    Write the feedback now.
    """
    response = model.generate_content(prompt)
    return response.text

def show_mastery_engine():
    """Renders the entire Genesis Module feature."""
    st.header("🏆 Mastery Engine")

    st.session_state.setdefault('mastery_stage', 'course_selection')
    st.session_state.setdefault('user_progress', {})
    
    stage = st.session_state.mastery_stage
    if stage == 'course_selection':
        render_course_selection()
    elif stage == 'skill_tree':
        render_skill_tree()
    elif stage == 'content_viewer':
        render_content_viewer()
    elif stage == 'boss_battle':
        render_boss_battle()

def render_course_selection():
    """Allows the user to select a course to begin."""
    st.subheader("Select Your Course")
    if st.button("Econ 101", use_container_width=True, type="primary"):
        st.session_state.current_genome = ECON_101_GENOME
        
        progress = {}
        genome_nodes = st.session_state.current_genome['nodes']
        all_node_ids = {node['gene_id'] for node in genome_nodes}
        destination_nodes = {edge['to'] for edge in st.session_state.current_genome['edges']}
        root_nodes = all_node_ids - destination_nodes

        for node_id in all_node_ids:
            progress[node_id] = 'unlocked' if node_id in root_nodes else 'locked'
        
        st.session_state.user_progress = progress
        st.session_state.mastery_stage = 'skill_tree'
        st.rerun()

def render_skill_tree():
    """Renders the interactive skill tree UI based on user progress."""
    st.subheader(f"Skill Tree: {st.session_state.current_genome['subject']}")

    nodes = st.session_state.current_genome['nodes']
    progress = st.session_state.user_progress

    for node in nodes:
        node_id, node_name = node['gene_id'], node['gene_name']
        status = progress.get(node_id, 'locked')
        
        if status == 'mastered':
            st.success(f"**{node_name}** - ✅ Mastered!", icon="✅")
        elif status == 'unlocked':
            if st.button(f"🧠 Learn: {node_name}", key=node_id, use_container_width=True, type="primary"):
                st.session_state.selected_node_id = node_id
                st.session_state.mastery_stage = 'content_viewer'
                st.rerun()
        else: # locked
            st.info(f"**{node_name}** - 🔒 Locked", icon="🔒")
        
        if node_id != nodes[-1]['gene_id']:
             st.markdown('<p style="text-align: center; margin: 0; padding: 0;">↓</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("Back to Course Selection"):
        st.session_state.mastery_stage = 'course_selection'
        st.rerun()

def render_content_viewer():
    """Displays the learning content for a selected node."""
    node_id = st.session_state.selected_node_id
    node_data = next((n for n in st.session_state.current_genome['nodes'] if n['gene_id'] == node_id), None)

    if not node_data:
        st.error("Error: Could not load node data."); st.session_state.mastery_stage = 'skill_tree'; st.rerun(); return

    st.subheader(f"Learning: {node_data['gene_name']}")
    st.divider()
    
    for allele in node_data['content_alleles']:
        if allele['type'] == 'text': st.markdown(allele['content'])
        elif allele['type'] == 'video': st.video(allele['url'])
    
    st.divider()
    col1, col2 = st.columns([1, 1])
    if col1.button("Back to Skill Tree"):
        st.session_state.mastery_stage = 'skill_tree'
        st.rerun()
    if col2.button(f"⚔️ Challenge Boss: {node_data['gene_name']}", type="primary"):
        st.session_state.mastery_stage = 'boss_battle'
        st.session_state.syllabus = f"Topic: {node_data['gene_name']}. Content: {next(a['content'] for a in node_data['content_alleles'] if a['type'] == 'text')}"
        st.session_state.test_stage = 'generating'
        for key in ['questions', 'user_answers', 'score', 'feedback']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

def render_boss_battle():
    """Repurposes the mock test generator as a boss battle."""
    node_id = st.session_state.selected_node_id
    node_data = next((n for n in st.session_state.current_genome['nodes'] if n['gene_id'] == node_id), None)
    
    st.subheader(f"Boss Battle: {node_data['gene_name']}")
    
    stage = st.session_state.get('test_stage', 'generating')
    
    if stage == 'generating': render_generating_questions()
    elif stage == 'mcq_test': render_mcq_test()
    elif stage == 'mcq_results':
        score = st.session_state.score.get('mcq', 0)
        total = len(st.session_state.questions.get('mcq', []))
        st.subheader(f"Battle Results: You scored {score} / {total}")
        
        if score >= 7:
            st.balloons()
            st.success("Victory! You have mastered this concept.")
            st.session_state.user_progress[node_id] = 'mastered'
            
            for edge in st.session_state.current_genome['edges']:
                if edge['from'] == node_id:
                    st.session_state.user_progress[edge['to']] = 'unlocked'
            
            if st.button("Return to Skill Tree", type="primary"):
                st.session_state.mastery_stage = 'skill_tree'
                st.rerun()
        else:
            st.error("Defeated. The concept is not yet mastered. Review the material and try again.")
            if st.button("Return to Learning"):
                st.session_state.mastery_stage = 'content_viewer'
                st.rerun()

# --- MAIN APPLICATION ---
def main():
    """Main function to run the Streamlit application."""
    st.session_state.setdefault('user_info', None)
    st.session_state.setdefault('credentials', None)

    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
    except (KeyError, FileNotFoundError):
        st.error("Gemini API key not configured in st.secrets."); st.stop()

    flow = get_google_flow()
    auth_code = st.query_params.get("code")

    if auth_code and not st.session_state.user_info:
        try:
            flow.fetch_token(code=auth_code)
            creds = flow.credentials
            st.session_state.credentials = creds
            
            user_info_service = build('oauth2', 'v2', credentials=creds)
            user_info = user_info_service.userinfo().get().execute()
            st.session_state.user_info = user_info
            
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.session_state.user_info = None
            st.session_state.credentials = None
    
    if not st.session_state.user_info:
        st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
        auth_url, _ = flow.authorization_url(prompt='consent')
        show_landing_page(auth_url)
        return

    # --- Post-Login Application UI ---
    st.sidebar.title("Vekkam Engine")
    user = st.session_state.user_info
    user_id = user.get('id') or user.get('email')
    st.sidebar.image(user['picture'], width=80)
    st.sidebar.subheader(f"Welcome, {user['given_name']}")
    if st.sidebar.button("Logout"): 
        st.session_state.clear()
        st.rerun()
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
                    new_title = st.text_input("Edit Title", value=session['title'], key=f"edit_title_{session['id']}", label_visibility="collapsed")
                    col1, col2 = st.columns(2)
                    if col1.button("Save", key=f"save_{session['id']}", type="primary", use_container_width=True):
                        user_data["sessions"][i]['title'] = new_title
                        save_user_data(user_id, user_data)
                        st.session_state.editing_session_id = None
                        st.rerun()
                    if col2.button("Cancel", key=f"cancel_{session['id']}", use_container_width=True):
                        st.session_state.editing_session_id = None
                        st.rerun()
                else:
                    for note in session.get('notes', []):
                        st.write(f"• {note['topic']}")
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    if col1.button("👁️ View", key=f"view_{session['id']}", use_container_width=True):
                        reset_session("Note & Lesson Engine")
                        st.session_state.final_notes = session.get('notes', [])
                        st.session_state.current_state = 'results'
                        st.session_state.messages = []
                        st.rerun()
                    if col2.button("✏️ Edit", key=f"edit_{session['id']}", use_container_width=True):
                        st.session_state.editing_session_id = session['id']
                        st.rerun()
                    if col3.button("🗑️ Delete", key=f"del_{session['id']}", type="secondary", use_container_width=True):
                        user_data["sessions"].pop(i)
                        save_user_data(user_id, user_data)
                        st.rerun()
    st.sidebar.divider()

    tool_options = ("Note & Lesson Engine", "Personal TA", "Mock Test Generator", "Mastery Engine")
    st.session_state.setdefault('tool_choice', tool_options[0])
    
    tool_choice_from_radio = st.sidebar.radio("Select a Tool", tool_options, key='tool_choice_radio', index=tool_options.index(st.session_state.tool_choice))

    if st.session_state.tool_choice != tool_choice_from_radio:
        reset_session(tool_choice_from_radio)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("API Status")
    st.sidebar.write(f"Gemini: **{check_gemini_api()}**")

    # --- Tool Routing ---
    if st.session_state.tool_choice == "Note & Lesson Engine":
        st.session_state.setdefault('current_state', 'upload')
        state_map = { 
            'upload': show_upload_state, 'processing': show_processing_state, 
            'workspace': show_workspace_state, 'synthesizing': show_synthesizing_state, 
            'results': show_results_state 
        }
        state_map[st.session_state.current_state]()
    elif st.session_state.tool_choice == "Personal TA":
        show_personal_ta_ui()
    elif st.session_state.tool_choice == "Mock Test Generator":
        show_mock_test_generator()
    elif st.session_state.tool_choice == "Mastery Engine":
        show_mastery_engine()

if __name__ == "__main__":
    main()
