# app.py

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
        genai.get_model('models/gemini-1.5-flash')
        return "Valid"
    except Exception:
        st.sidebar.error("Gemini API Key in secrets is invalid.")
        return "Invalid"

def resilient_json_parser(json_string):
    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```', json_string, re.DOTALL)
        if match:
            return json.loads(match.group(1))
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
                response = model.generate_content(["Transcribe this audio recording.", audio_file])
                chunks = chunk_text(response.text, source_id)
                genai.delete_file(audio_file.name)
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
    # ... (code is unchanged)
    
@gemini_api_call_with_retry
def synthesize_note_block(topic, relevant_chunks_text, instructions):
    # ... (code is unchanged)
    
@gemini_api_call_with_retry
def answer_from_context(query, context):
    # ... (code is unchanged)
    
# --- AUTHENTICATION & SESSION MANAGEMENT ---
def get_google_flow():
    try:
        client_config = {"web": st.secrets["google"]}
        scopes = [
            "https://www.googleapis.com/auth/userinfo.profile", 
            "https://www.googleapis.com/auth/userinfo.email", 
            "openid",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets."); st.stop()

def reset_session(tool_choice):
    user_info = st.session_state.get('user_info')
    credentials = st.session_state.get('credentials')
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.credentials = credentials
    st.session_state.tool_choice = tool_choice
    st.session_state.current_state = 'upload'

# --- LANDING PAGE ---
def show_landing_page(auth_url):
    # ... (code is unchanged)
    
# --- UI STATE FUNCTIONS for NOTE & LESSON ENGINE ---
def show_upload_state():
    # ... (code is unchanged)
    
def show_processing_state():
    # ... (code is unchanged)
    
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
                
                with st.spinner("Searching for learning alleles and saving to Google Sheets..."):
                    try:
                        spreadsheet_id = st.secrets["google_sheets"]["spreadsheet_id"]
                        all_rows_to_append = [["Topic", "Type", "Source", "Title", "URL"]]
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
                        st.error("`spreadsheet_id` not found in st.secrets.")
                    except Exception as e:
                        st.error(f"An error occurred during allele generation: {e}")
            else: 
                st.error("Failed to generate outline. The AI couldn't structure the provided content.")
        
        if st.session_state.get('outline_data'):
            # ... (rest of the function is unchanged)

def show_synthesizing_state():
    # ... (code is unchanged)
    
def show_results_state():
    # ... (code is unchanged)

# --- UI STATE FUNCTION for PERSONAL TA ---
def show_personal_ta_ui():
    # ... (code is unchanged)
    
# --- UI STATE FUNCTIONS for MOCK TEST GENERATOR ---
def show_mock_test_generator():
    # ... (code is unchanged)
    
# --- Helper Functions for Mock Test Stages ---
def render_syllabus_input():
    # ... (code is unchanged)
    
def render_generating_questions():
    # ... (code is unchanged)
    
def render_mcq_test():
    # ... (code is unchanged)
    
def render_mcq_results():
    # ... (code is unchanged)
    
# --- AI & Utility Functions for Mock Test ---
def get_bloom_level_name(level):
    # ... (code is unchanged)
    
@gemini_api_call_with_retry
def generate_questions_from_syllabus(syllabus_text, question_type, question_count):
    # ... (code is unchanged)
    
@gemini_api_call_with_retry
def generate_feedback_on_performance(score, total, questions, user_answers, syllabus):
    # ... (code is unchanged)
    
# --- UI STATE FUNCTIONS for MASTERY ENGINE (GENESIS MODULE) ---
def show_mastery_engine():
    # ... (code is unchanged)
    
def render_course_selection():
    # ... (code is unchanged)
    
def render_skill_tree():
    # ... (code is unchanged)
    
def render_content_viewer():
    # ... (code is unchanged)

def render_boss_battle():
    # ... (code is unchanged)
    
# --- MAIN APP ---
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
                    # ... (code is unchanged)
                else:
                    # ... (code is unchanged)
    st.sidebar.divider()

    tool_options = ("Note & Lesson Engine", "Personal TA", "Mock Test Generator", "Mastery Engine")
    st.session_state.setdefault('tool_choice', tool_options[0])
    
    tool_choice = st.sidebar.radio("Select a Tool", tool_options, key='tool_choice_radio')

    if st.session_state.tool_choice != tool_choice:
        st.session_state.tool_choice = tool_choice
        reset_session(tool_choice)
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
