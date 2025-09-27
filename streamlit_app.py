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
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Truncated
    with st.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">From Classroom Chaos to Concept Clarity</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Stop drowning in disorganized notes and endless lectures. Vekkam transforms your course materials into a powerful, interactive knowledge base that helps you study smarter, not harder.</p>', unsafe_allow_html=True)
        st.link_button("Activate Your Smart Study Hub", auth_url, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    # Rest of landing page is truncated for brevity

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
        # Reset chunks before extending
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
    st.header("🎓 Your Personal TA")
    # ... code is unchanged

# --- UI STATE FUNCTIONS for MOCK TEST GENERATOR ---
def show_mock_test_generator():
    st.header("📝 Mock Test Generator")
    if 'test_stage' not in st.session_state:
        st.session_state.test_stage = 'start'
    if 'syllabus' not in st.session_state:
        st.session_state.syllabus = ""
    if 'questions' not in st.session_state:
        st.session_state.questions = {}
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'score' not in st.session_state:
        st.session_state.score = {}
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}

    stage = st.session_state.test_stage
    if stage == 'start':
        render_syllabus_input()
    elif stage == 'generating':
        render_generating_questions()
    elif stage == 'mcq_test':
        render_mcq_test()
    elif stage == 'mcq_results':
        render_mcq_results()
    elif stage in ('fib_test', 'short_answer_test', 'long_answer_test'):
        st.info(f"{stage.replace('_', ' ').title()} Stage - To be implemented")

def render_syllabus_input():
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
    if 'questions' not in st.session_state:
        st.session_state.questions = {}
    with st.spinner("Building your test... The AI is analyzing the syllabus and crafting questions..."):
        questions_json = generate_questions_from_syllabus(st.session_state.syllabus, "MCQ", 10)
        if questions_json and "questions" in questions_json:
            st.session_state.questions['mcq'] = questions_json["questions"]
            st.session_state.test_stage = 'mcq_test'
            st.rerun()
        else:
            st.error("Failed to generate questions from the provided syllabus. Please try again.")
            st.session_state.test_stage = 'start'
            st.rerun()

def render_mcq_test():
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
            score = 0
            for q in mcq_questions:
                if user_answers.get(q['question_id']) == q['answer']:
                    score += 1
            st.session_state.score['mcq'] = score
            st.session_state.test_stage = 'mcq_results'
            st.rerun()

def render_mcq_results():
    score = st.session_state.score.get('mcq', 0)
    total = len(st.session_state.questions.get('mcq', []))
    st.subheader(f"MCQ Results: You scored {score} / {total}")
    if 'feedback' not in st.session_state or 'mcq' not in st.session_state.feedback:
        with st.spinner("Analyzing your performance and generating feedback..."):
            feedback_text = generate_feedback_on_performance(
                score, total, st.session_state.questions.get('mcq', []),
                st.session_state.user_answers.get('mcq', {}), st.session_state.syllabus
            )
            st.session_state.feedback['mcq'] = feedback_text
    with st.container(border=True):
        st.subheader("💡 Suggestions for Improvement")
        st.write(st.session_state.feedback.get('mcq', "No feedback generated."))
    if score >= 7:
        st.success("Congratulations! You've passed this stage.")
        if st.button("Proceed to Next Stage", type="primary"):
            st.session_state.test_stage = 'fib_test'
            st.rerun()
    else:
        st.error("You need a score of 7/10 to proceed. Review the feedback and try again.")
        if st.button("Restart Test"):
            for key in ['test_stage', 'syllabus', 'questions', 'user_answers', 'score', 'feedback']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# --- AI & Utility Functions for Mock Test ---
def get_bloom_level_name(level):
    if level is None: return "N/A"
    levels = {1: "Remembering", 2: "Understanding", 3: "Applying", 4: "Analyzing", 5: "Evaluating"}
    return levels.get(level, "Unknown")

@gemini_api_call_with_retry
def generate_questions_from_syllabus(syllabus_text, question_type, question_count):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""You are an expert Question Paper Setter...""" # Truncated for brevity
    response = model.generate_content(prompt)
    return resilient_json_parser(response.text)

@gemini_api_call_with_retry
def generate_feedback_on_performance(score, total, questions, user_answers, syllabus):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""You are an encouraging academic coach...""" # Truncated for brevity
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
    # (This function is unchanged from the previous correct version)
    model = genai.GenerativeModel('gemini-pro')
    # ... code is unchanged

# --- UI STATE FUNCTIONS for MASTERY ENGINE (GENESIS MODULE) ---
def show_mastery_engine():
    # (This function is unchanged)
    st.header("🏆 Mastery Engine")
    # ... code is unchanged

def render_course_selection():
    # (This function is unchanged from the previous correct version)
    st.subheader("Select Your Course or Create a New Concept")
    # ... code is unchanged

def render_skill_tree():
    # (This function is unchanged)
    st.subheader(f"Skill Tree: {st.session_state.current_genome['subject']}")
    # ... code is unchanged

def render_content_viewer():
    # (This function is unchanged)
    st.subheader(f"Learning: {node_data['gene_name']}")
    # ... code is unchanged

def render_boss_battle():
    # (This function is unchanged)
    st.subheader(f"Boss Battle: {node_data['gene_name']}")
    # ... code is unchanged

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
                is_editing = st.session_state.get('editing_session_id') == session.get('id')
                if is_editing:
                    new_title = st.text_input("Edit Title", value=session.get('title', ''), key=f"edit_title_{session.get('id')}", label_visibility="collapsed")
                    col1, col2 = st.columns(2)
                    if col1.button("Save", key=f"save_{session.get('id')}", type="primary", use_container_width=True):
                        user_data["sessions"][i]['title'] = new_title
                        save_user_data(user_id, user_data)
                        st.session_state.editing_session_id = None
                        st.rerun()
                    if col2.button("Cancel", key=f"cancel_{session.get('id')}", use_container_width=True):
                        st.session_state.editing_session_id = None
                        st.rerun()
                else:
                    for note in session.get('notes', []):
                        st.write(f"• {note.get('topic', 'No Topic')}")
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    if col1.button("👁️ View", key=f"view_{session.get('id')}", use_container_width=True):
                        reset_session("Note & Lesson Engine")
                        st.session_state.final_notes = session.get('notes', [])
                        st.session_state.current_state = 'results'
                        st.session_state.messages = []
                        st.rerun()
                    if col2.button("✏️ Edit", key=f"edit_{session.get('id')}", use_container_width=True):
                        st.session_state.editing_session_id = session.get('id')
                        st.rerun()
                    if col3.button("🗑️ Delete", key=f"del_{session.get('id')}", type="secondary", use_container_width=True):
                        user_data["sessions"].pop(i)
                        save_user_data(user_id, user_data)
                        st.rerun()

    st.sidebar.divider()
    tool_choice = st.sidebar.radio("Select a Tool", ("Note & Lesson Engine", "Personal TA", "Mock Test Generator", "Mastery Engine"), key='tool_choice')
    
    if 'last_tool_choice' not in st.session_state:
        st.session_state.last_tool_choice = tool_choice
    
    if st.session_state.last_tool_choice != tool_choice:
        st.session_state.last_tool_choice = tool_choice
        reset_session(tool_choice) # This will preserve all_chunks
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("API Status")
    st.sidebar.write(f"Gemini: **{check_gemini_api()}**")

    # --- Tool Routing ---
    if tool_choice == "Note & Lesson Engine":
        if 'current_state' not in st.session_state:
            st.session_state.current_state = 'upload'
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
