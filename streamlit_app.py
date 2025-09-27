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
        {"type": "text", "content": "Scarcity refers to the basic economic problem..."},
        {"type": "video", "url": "https://www.youtube.com/watch?v=yoVc_S_gd_0"}
      ]
    },
    {
      "gene_id": "ECON101_OPPCOST",
      "gene_name": "Opportunity Cost",
      "difficulty": 2,
      "content_alleles": [
          {"type": "text", "content": "Opportunity cost is the potential forgone profit..."},
          {"type": "video", "url": "https://www.youtube.com/watch?v=PSU-SA-Fv_M"}
      ]
    },
    {
      "gene_id": "ECON101_SND",
      "gene_name": "Supply and Demand",
      "difficulty": 3,
      "content_alleles": [
          {"type": "text", "content": "Supply and demand is a model of microeconomics..."},
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

# --- [NEW] GOOGLE API FUNCTIONS for Search & Sheets ---
def generate_alleles_from_search(topic, num_results=3):
    """Uses Google Custom Search API to find learning materials for a topic."""
    try:
        api_key = st.secrets["google_search"]["api_key"]
        cse_id = st.secrets["google_search"]["cse_id"]
        
        service = build("customsearch", "v1", developerKey=api_key)
        # Add educational keywords to the query for better results
        query = f"{topic} tutorial explained khan academy youtube"
        
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        
        alleles = []
        if 'items' in res:
            for item in res.get('items', []):
                link = item.get('link')
                source = item.get('displayLink', '').replace('www.', '')
                allele_type = 'video' if 'youtube.com' in source else 'text'
                alleles.append({
                    'type': allele_type,
                    'source': source,
                    'title': item.get('title'),
                    'url': link
                })
        return alleles
    except HttpError as e:
        st.error(f"Google Search API Error: {e}. Check your API key and CSE ID in secrets.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Search: {e}")
        return []

def append_to_google_sheet(credentials, spreadsheet_id, data_rows):
    """Appends rows of data to a specified Google Sheet."""
    try:
        service = build('sheets', 'v4', credentials=credentials)
        body = {'values': data_rows}
        
        # Check if sheet is empty to add headers
        sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheets = sheet_metadata.get('sheets', '')
        # Simplification: assuming we write to the first sheet
        first_sheet_title = sheets[0].get('properties', {}).get('title', 'Sheet1')
        
        # Append data to the sheet
        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"{first_sheet_title}!A1",
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()
        return result
    except HttpError as e:
        st.error(f"Google Sheets API Error: {e}. Ensure API is enabled and spreadsheet ID is correct.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while writing to Google Sheets: {e}")
        return None


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
            # ... (no changes here)
        elif source_type == 'image':
            # ... (no changes here)
        elif source_type == 'pdf':
            # ... (no changes here)
    except Exception as e:
        return {"status": "error", "source_id": f"{source_type}:{file.name}", "reason": str(e)}

# --- AGENTIC WORKFLOW FUNCTIONS ---
# ... (All existing agentic functions like generate_content_outline, etc., remain unchanged) ...

# --- AUTHENTICATION & SESSION MANAGEMENT ---
def get_google_flow():
    """Initializes the Google OAuth flow with required scopes."""
    try:
        client_config = {
            "web": { "client_id": st.secrets["google"]["client_id"], "client_secret": st.secrets["google"]["client_secret"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x_509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [st.secrets["google"]["redirect_uri"]],
            }}
        # --- [MODIFIED] Added Google Sheets scope ---
        scopes = [
            "https://www.googleapis.com/auth/userinfo.profile", 
            "https://www.googleapis.com/auth/userinfo.email", 
            "openid",
            "https://www.googleapis.com/auth/spreadsheets" # <-- NEW SCOPE
        ]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets."); st.stop()

def reset_session(tool_choice):
    # --- [MODIFIED] Keep credentials on reset ---
    user_info = st.session_state.get('user_info')
    credentials = st.session_state.get('credentials')
    st.session_state.clear()
    st.session_state.user_info = user_info
    st.session_state.credentials = credentials
    st.session_state.tool_choice = tool_choice
    st.session_state.current_state = 'upload'
    st.session_state.all_chunks = []
    st.session_state.extraction_failures = []
    st.session_state.outline_data = []
    st.session_state.final_notes = []

# --- LANDING PAGE ---
# ... (show_landing_page function remains unchanged) ...

# --- UI STATE FUNCTIONS for NOTE & LESSON ENGINE ---
def show_upload_state():
    # ... (no changes here)

def show_processing_state():
    # ... (no changes here)

def show_workspace_state():
    st.header("Vekkam Workspace")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Controls & Outline")
        if st.button("Generate / Regenerate Full Outline"):
            with st.spinner("AI is analyzing all content to create an outline..."):
                outline_json = generate_content_outline(st.session_state.all_chunks)
                
            if outline_json and "outline" in outline_json: 
                st.session_state.outline_data = outline_json["outline"]
                
                # --- [NEW] Allele Generation and Sheets Integration ---
                with st.spinner("Searching for learning alleles and saving to Google Sheets..."):
                    try:
                        spreadsheet_id = st.secrets["google_sheets"]["spreadsheet_id"]
                        all_rows_to_append = []
                        # Add headers if we want
                        # all_rows_to_append.append(["Topic", "Type", "Source", "Title", "URL"])

                        for item in st.session_state.outline_data:
                            topic = item['topic']
                            alleles = generate_alleles_from_search(topic)
                            for allele in alleles:
                                row = [topic, allele['type'], allele['source'], allele['title'], allele['url']]
                                all_rows_to_append.append(row)
                        
                        if all_rows_to_append:
                            credentials = st.session_state.get('credentials')
                            if credentials:
                                append_to_google_sheet(credentials, spreadsheet_id, all_rows_to_append)
                                st.success(f"Appended {len(all_rows_to_append)} alleles to your Google Sheet!")
                            else:
                                st.error("Authentication credentials not found. Cannot write to Google Sheets.")

                    except KeyError:
                        st.error("`spreadsheet_id` not found in st.secrets. Please add it to your secrets file.")
                    except Exception as e:
                        st.error(f"An error occurred during allele generation: {e}")
                # --- End of New Feature ---

            else: 
                st.error("Failed to generate outline. The AI couldn't structure the provided content. Try adding more context-rich files.")

        if 'outline_data' in st.session_state and st.session_state.outline_data:
            initial_text = "\n".join([item.get('topic', '') for item in st.session_state.outline_data])
            st.session_state.editable_outline = st.text_area("Editable Outline:", value=initial_text, height=300)
            st.session_state.synthesis_instructions = st.text_area("Synthesis Instructions (Optional):", height=100, placeholder="e.g., 'Explain this like I'm 15' or 'Focus on key formulas'")
            if st.button("Synthesize Notes", type="primary"):
                st.session_state.current_state = 'synthesizing'
                st.rerun()
    with col2:
        st.subheader("Source Explorer")
        # ... (no changes here)

# ... (All other UI and helper functions remain largely unchanged) ...

# --- MAIN APP ---
def main():
    if 'user_info' not in st.session_state: st.session_state.user_info = None
    if 'credentials' not in st.session_state: st.session_state.credentials = None # Initialize credentials

    try: genai.configure(api_key=st.secrets["gemini"]["api_key"])
    except (KeyError, FileNotFoundError): st.error("Gemini API key not configured in st.secrets."); st.stop()

    flow = get_google_flow()
    auth_code = st.query_params.get("code")

    if auth_code and not st.session_state.user_info:
        try:
            flow.fetch_token(code=auth_code)
            
            # --- [MODIFIED] Store credentials in session state ---
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
    # ... (Rest of the main function remains unchanged, including the tool router) ...
    # The new functionality is now part of the 'Note & Lesson Engine'
    
    tool_choice = st.sidebar.radio("Select a Tool", ("Note & Lesson Engine", "Personal TA", "Mock Test Generator", "Mastery Engine"), key='tool_choice')
    
    if 'last_tool_choice' not in st.session_state: st.session_state.last_tool_choice = tool_choice
    if st.session_state.last_tool_choice != tool_choice:
        reset_session(tool_choice)
        st.session_state.last_tool_choice = tool_choice
        st.rerun()

    # ... (Tool routing logic remains the same) ...

if __name__ == "__main__":
    main()
