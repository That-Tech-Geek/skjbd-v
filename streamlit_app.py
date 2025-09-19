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
import requests

# --- GOOGLE OAUTH LIBRARIES ---
# This app uses the official Google libraries for authentication.
# If you see an error below, please install them from your terminal:
# pip install google-auth-oauthlib google-api-python-client
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

        The app cannot continue without these dependencies. Please install them and refresh the page.
    """)
    st.stop()


# --- CONFIGURATION & CONSTANTS ---
MAX_FILES = 20
MAX_TOTAL_SIZE_MB = 150
MAX_AUDIO_SIZE_MB = 25
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
    # ... (This function remains unchanged)
    pass

def synthesize_note_block(topic, relevant_chunks_text, instructions, api_key):
    # ... (This function remains unchanged)
    pass

# --- AUTHENTICATION SETUP ---
def get_google_flow():
    try:
        client_config = {
            "web": {
                "client_id": st.secrets["google_oauth"]["client_id"],
                "client_secret": st.secrets["google_oauth"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [st.secrets["google_oauth"]["redirect_uri"]],
            }
        }
        scopes = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"]
        return Flow.from_client_config(client_config, scopes=scopes, redirect_uri=st.secrets["google_oauth"]["redirect_uri"])
    except (KeyError, FileNotFoundError):
        st.error("OAuth credentials are not configured correctly in st.secrets.")
        st.stop()

# --- SESSION STATE & RESET ---
def reset_session():
    # ... (This function remains unchanged)
    pass

# --- MAIN APP LOGIC ---
def main():
    st.sidebar.title("Vekkam Engine")

    # Initialize session state
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    
    flow = get_google_flow()

    # Check for authorization code in query params
    query_params = st.query_params
    auth_code = query_params.get("code")

    if auth_code and not st.session_state.user_info:
        try:
            flow.fetch_token(code=auth_code)
            credentials = flow.credentials
            
            user_info_service = build('oauth2', 'v2', credentials=credentials)
            user_info = user_info_service.userinfo().get().execute()
            
            st.session_state.user_info = user_info
            st.query_params.clear() # Clean up URL
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
    
    # API Key Management
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", key="gemini_api_key_input")

    # ... (Rest of the app logic, which is gated by authentication, goes here)
    st.header("Upload Your Sources")
    st.write("You are logged in. The workspace is ready.")
    # (Here you would call show_upload_state, show_workspace_state etc.)

if __name__ == "__main__":
    main()

