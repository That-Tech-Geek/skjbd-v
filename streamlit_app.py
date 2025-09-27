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
        genai.get_model('models/gemini-pro')
        return "Valid"
    except Exception as e:
        st.sidebar.error(f"Gemini API Key in secrets is invalid: {e}")
        return "Invalid"

def resilient_json_parser(json_string):
    try:
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
        model_name = 'gemini-pro-vision' if source_type in ['image', 'pdf'] else 'gemini-pro'
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
    model = genai.GenerativeModel('gemini-pro')
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
    model = genai.GenerativeModel('gemini-pro')
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
    model = genai.GenerativeModel('gemini-pro')
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
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.tool_choice = tool_choice
    st.session_state.current_state = 'upload'
    st.session_state.all_chunks = []
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []

# --- LANDING PAGE ---
def show_landing_page(auth_url):
    """Displays the AARRR-framework-based landing page."""
    # (Landing page HTML/CSS is unchanged)
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Truncated for brevity
    # --- [A]cquisition & [A]ctivation: Hero Section ---
    with st.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">From Classroom Chaos to Concept Clarity</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Stop drowning in disorganized notes and endless lectures. Vekkam transforms your course materials into a powerful, interactive knowledge base that helps you study smarter, not harder.</p>', unsafe_allow_html=True)
        st.link_button("Activate Your Smart Study Hub", auth_url, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    # (Rest of landing page content is unchanged and truncated for brevity)

# --- UI STATE FUNCTIONS for NOTE & LESSON ENGINE ---
def show_upload_state():
    st.header("Note & Lesson Engine: Upload")
    uploaded_files = st.file_uploader("Select files", accept_multiple_files=True, type=['mp3', 'm4a', 'wav', 'png', 'jpg', 'pdf', 'pptx'])
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
    # (This function is unchanged)
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
    # (This function is unchanged)
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

    if 'selected_note_index' not in st.session_state:
        st.session_state.selected_note_index = None

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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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

# --- UI STATE FUNCTION for PERSONAL TA ---
def show_personal_ta_ui():
    # (This function is unchanged)
    st.header("🎓 Your Personal TA")
    st.markdown("Ask questions and get answers based on the knowledge from all your past study sessions.")
    user_id = st.session_state.user_info.get('id') or st.session_state.user_info.get('email')
    user_data = load_user_data(user_id)

    if not user_data or not user_data["sessions"]:
        st.warning("You don't have any saved study sessions yet. Create some notes first to power up your TA!")
        return

    if "ta_messages" not in st.session_state:
        st.session_state.ta_messages = []

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

# --- UI STATE FUNCTIONS for MOCK TEST GENERATOR ---
def show_mock_test_generator():
    # (This function and its helpers are unchanged and truncated for brevity)
    st.header("📝 Mock Test Generator")
    if 'test_stage' not in st.session_state: st.session_state.test_stage = 'start'
    # ... rest of function

def render_syllabus_input():
    # (This function is unchanged)
    st.subheader("Step 1: Provide Your Syllabus")
    # ... rest of function

def render_generating_questions():
    # (This function is unchanged)
    # ... rest of function
    
def render_mcq_test():
    # (This function is unchanged)
    # ... rest of function

def render_mcq_results():
    # (This function is unchanged)
    # ... rest of function

# --- AI & Utility Functions for Mock Test ---
def get_bloom_level_name(level):
    # (This function is unchanged)
    if level is None: return "N/A"
    levels = {1: "Remembering", 2: "Understanding", 3: "Applying", 4: "Analyzing", 5: "Evaluating"}
    return levels.get(level, "Unknown")

@gemini_api_call_with_retry
def generate_questions_from_syllabus(syllabus_text, question_type, question_count):
    # (This function is unchanged)
    # ... rest of function
    pass

@gemini_api_call_with_retry
def generate_feedback_on_performance(score, total, questions, user_answers, syllabus):
    # (This function is unchanged)
    # ... rest of function
    pass

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
    """Uses Google Search to find content and constructs a new allele dictionary, with optional context."""
    model = genai.GenerativeModel('gemini-pro')
    
    gene_name_response = model.generate_content(f"Provide a concise, 3-5 word conceptual name for the topic: '{user_topic}'. Output only the name.")
    gene_name = gene_name_response.text.strip().replace('"', '') if gene_name_response.text else user_topic.title()
    gene_id = f"USER_{hashlib.md5(user_topic.encode()).hexdigest()[:8].upper()}"
    
    service = get_google_search_service()
    if not service: return None

    try:
        cse_id = st.secrets["google_search"]["cse_id"]
    except KeyError:
        st.error("Google Search CSE ID ('cse_id') not found in st.secrets.toml. Please add it.")
        return None

    search_queries = []
    synthesis_prompt_template = ""

    if context_chunks:
        st.info("Context detected. Generating contextual search queries...")
        full_context_text = " ".join([chunk.get('text', '') for chunk in context_chunks])
        # Truncate to avoid exceeding API limits for the summary prompt
        context_summary_prompt = f"Summarize the key topics and concepts from this course material in 2-3 sentences: {full_context_text[:8000]}"
        summary_response = model.generate_content(context_summary_prompt)
        context_summary = summary_response.text

        query_gen_prompt = f"""A student is studying material summarized as: "{context_summary}". They now want to learn about: "{user_topic}". Generate 3 specific Google search queries to find supplementary information that connects their existing knowledge to this new topic. Output ONLY a valid JSON object with a single key "queries" which is a list of 3 strings."""
        queries_response = model.generate_content(query_gen_prompt)
        queries_json = resilient_json_parser(queries_response.text)
        
        if queries_json and 'queries' in queries_json:
            search_queries = queries_json['queries']
        else: # Fallback if JSON parsing fails
            search_queries = [f"{user_topic} explained in context of {context_summary[:50]}", f"how does {user_topic} relate to concepts in my notes"]
        
        synthesis_prompt_template = f"""You are an expert tutor. A student's current study material is about: "{context_summary}". They now want to understand: "{user_topic}". Based ONLY on the provided web search results, synthesize a clear explanation of "{user_topic}". Crucially, you must connect the new topic to the student's existing knowledge. Bridge the concepts and explain it as if you were adding a new, relevant chapter to their study notes. Start by acknowledging the connection (e.g., "Building on what you know about [concept from summary]...").\n\n**Web Search Results to Synthesize:**\n---\n{{search_snippets}}\n---"""
    else:
        search_queries = [f"{user_topic} explanation", f"{user_topic} simple definition", f"youtube video tutorial {user_topic}"]
        synthesis_prompt_template = f"""Based ONLY on the provided web search results, provide a clear, concise, and comprehensive explanation of '{user_topic}'.\n\n**Web Search Results to Synthesize:**\n---\n{{search_snippets}}\n---"""

    text_content, video_url = [], None
    st.write("Performing targeted web search...")
    for query in search_queries:
        try:
            res = service.cse().list(q=query, cx=cse_id, num=3).execute()
            items = res.get('items', [])
            for item in items:
                if "youtube.com/watch" in item.get('link', '') and not video_url:
                    video_url = item.get('link')
                if item.get('snippet'):
                    text_content.append(item.get('snippet'))
        except HttpError as e:
            st.error(f"Google Search API error: {e}. This could be due to an invalid API key, incorrect CSE ID, or exceeding your daily quota.")
            return None
        except Exception as e:
            st.warning(f"Google Search failed for query '{query}': {e}")
            time.sleep(1)

    if not text_content:
        st.error(f"Could not find any relevant text content for '{user_topic}'. The topic might be too obscure or the query too broad. Please try again.")
        return None

    full_text = " ".join(text_content)
    
    with st.spinner("Synthesizing explanation from search results..."):
        synthesis_prompt = synthesis_prompt_template.format(search_snippets=full_text)
        final_explanation_response = model.generate_content(synthesis_prompt)
        final_explanation = final_explanation_response.text

    content_alleles = []
    if final_explanation:
        content_alleles.append({"type": "text", "content": final_explanation})
    if video_url:
        content_alleles.append({"type": "video", "url": video_url})

    return {"gene_id": gene_id, "gene_name": gene_name, "difficulty": 0, "content_alleles": content_alleles}

# --- UI STATE FUNCTIONS for MASTERY ENGINE (GENESIS MODULE) ---
def show_mastery_engine():
    """Renders the entire Genesis Module feature."""
    st.header("🏆 Mastery Engine")
    # (Initialization logic is unchanged)
    if 'mastery_stage' not in st.session_state: st.session_state.mastery_stage = 'course_selection'
    if 'user_progress' not in st.session_state: st.session_state.user_progress = {}
    if 'current_genome' not in st.session_state: st.session_state.current_genome = None

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
    """Allows the user to select a course or create a new concept."""
    st.subheader("Select Your Course or Create a New Concept")
    
    st.markdown("### Pre-built Courses")
    if st.button("Econ 101", use_container_width=True, type="primary"):
        st.session_state.current_genome = json.loads(json.dumps(ECON_101_GENOME)) # Deep copy
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

    st.markdown("### Create Your Own Concept")
    user_interest = st.text_input("What concept are you interested in learning about?", key="user_allele_query")
    
    use_context = st.checkbox("Use context from 'Note & Lesson Engine' session", key="use_context", value=True)
    
    if st.button("Generate Concept Allele", use_container_width=True, disabled=not user_interest):
        context_chunks = st.session_state.get('all_chunks', [])
        if use_context and not context_chunks:
            st.warning("Contextual generation selected, but no files have been processed in the 'Note & Lesson Engine' yet. Falling back to non-contextual generation.")
            context_chunks = None # Force non-contextual

        if st.session_state.current_genome is None:
            st.session_state.current_genome = {"subject": "My Custom Concepts", "version": "1.0", "nodes": [], "edges": []}
        
        with st.spinner(f"Generating concept for '{user_interest}'... This might take a moment."):
            new_allele = generate_allele_from_query(user_interest, context_chunks=context_chunks)
            
            if new_allele:
                existing_node_ids = {node['gene_id'] for node in st.session_state.current_genome['nodes']}
                if new_allele['gene_id'] not in existing_node_ids:
                    st.session_state.current_genome['nodes'].append(new_allele)
                    st.session_state.user_progress[new_allele['gene_id']] = 'unlocked'
                    st.success(f"Concept '{new_allele['gene_name']}' generated and unlocked!")
                else:
                    st.info(f"Concept '{new_allele['gene_name']}' already exists.")
                    st.session_state.user_progress[new_allele['gene_id']] = 'unlocked'
                
                st.session_state.mastery_stage = 'skill_tree'
                st.rerun()

def render_skill_tree():
    # (This function is unchanged)
    st.subheader(f"Skill Tree: {st.session_state.current_genome['subject']}")
    # ... rest of function

def render_content_viewer():
    # (This function is unchanged)
    # ... rest of function

def render_boss_battle():
    # (This function is unchanged)
    # ... rest of function

# --- MAIN APP ---
def main():
    # (This function is mostly unchanged, just added a try/except for Gemini key)
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

    # --- Post-Login App ---
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
                is_editing = st.session_state.get('editing_session_id') == session['id']
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

    tool_choice = st.sidebar.radio("Select a Tool", ("Note & Lesson Engine", "Personal TA", "Mock Test Generator", "Mastery Engine"), key='tool_choice')
    
    if 'last_tool_choice' not in st.session_state: st.session_state.last_tool_choice = tool_choice
    if st.session_state.last_tool_choice != tool_choice:
        reset_session(tool_choice)
        st.session_state.last_tool_choice = tool_choice
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("API Status")
    st.sidebar.write(f"Gemini: **{check_gemini_api()}**")

    # --- Tool Routing ---
    if tool_choice == "Note & Lesson Engine":
        if 'current_state' not in st.session_state: reset_session(tool_choice)
        state_map = { 'upload': show_upload_state, 'processing': show_processing_state, 'workspace': show_workspace_state, 'synthesizing': show_synthesizing_state, 'results': show_results_state }
        state_function = state_map.get(st.session_state.current_state, show_upload_state)
        state_function()
    elif tool_choice == "Personal TA":
        show_personal_ta_ui()
    elif tool_choice == "Mock Test Generator":
        show_mock_test_generator()
    elif tool_choice == "Mastery Engine":
        show_mastery_engine()

if __name__ == "__main__":
    main()
