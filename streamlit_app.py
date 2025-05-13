import streamlit as st
import requests
import tempfile
from urllib.parse import urlencode
from PyPDF2 import PdfReader
from io import StringIO
from PIL import Image
import pytesseract
import json
import igraph as ig
import plotly.graph_objects as go
import re
import sqlite3
from contextlib import closing
from streamlit_lottie import st_lottie
import requests as reqs
import contextlib
import csv
from fpdf import FPDF
import webbrowser
import fitz
import docx
from pptx import Presentation
import streamlit.components.v1 as components
import time
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from gtts import gTTS  # Import Google Text-to-Speech library
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Define missing variables
all_summaries = []
all_flashcards = []

# --- Error Reporting ---
def send_error_report(error_message):
    """Send an error report to the support team."""
    sender_email = "your_email@example.com"  # Replace with your email
    receiver_email = "sambit1912@gmail.com"
    password = "your_email_password"  # Replace with your email password

    subject = "Vekkam Error Report"
    body = f"An error occurred:\n\n{error_message}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
    except Exception as e:
        print(f"Failed to send error report: {str(e)}")

# --- Gemini Call ---
def call_gemini(prompt, temperature=0.7, max_tokens=2048):
    lang = st.session_state.get("language", "en")
    lang_name = [k for k, v in languages.items() if v == lang][0]
    prompt = f"Please answer in {lang_name}.\n" + prompt
    
    if not GEMINI_API_KEY:
        st.write("There is an error, we've notified the support team.")
        send_error_report("Gemini API key is not configured. Please check your secrets.toml file.")
        return "API key not configured"
        
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
    }
    try:
        with show_lottie_loading(t("Thinking with Gemini AI...")):
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as e:
        error_message = f"API Error: {str(e)}"
        send_error_report(error_message)
        st.error("There is an error, we've notified the support team.")
        return "Error occurred while calling Gemini API"
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        send_error_report(error_message)
        st.error("There is an error, we've notified the support team.")
        return "Error occurred while processing your request"

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(file).paragraphs)
    if name.endswith(".pptx"):
        return "\n".join(shape.text for slide in Presentation(file).slides for shape in slide.shapes if hasattr(shape, 'text'))
    if name.endswith(".txt"):
        return StringIO(file.getvalue().decode('utf-8')).read()
    if name.endswith((".jpg",".jpeg",".png")):
        return pytesseract.image_to_string(Image.open(file))
    return ""


# --- Visual/Equation/Code Understanding Helper (MUST BE DEFINED BEFORE USAGE) ---
def add_to_google_calendar(deadline):
    # Opens a Google Calendar event creation link in the browser
    import urllib.parse
    base = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    params = {
        "text": deadline['description'],
        "dates": f"{deadline['date'].replace('-', '')}/{deadline['date'].replace('-', '')}",
    }
    url = base + "&" + urllib.parse.urlencode(params)
    webbrowser.open_new_tab(url)

def extract_visuals_and_code(text, file=None):
    visuals = []
    # Detect LaTeX/math equations (simple regex for $...$ or \[...\])
    import re
    equations = re.findall(r'(\$[^$]+\$|\\\[[^\]]+\\\])', text)
    for eq in equations:
        visuals.append(("Equation", eq))
    # Detect code blocks (triple backticks or indented)
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    for code in code_blocks:
        visuals.append(("Code", code))
    # Detect possible diagrams/images in file (if image or PDF page)
    if file and hasattr(file, 'name') and file.name.lower().endswith((".jpg", ".jpeg", ".png")):
        visuals.append(("Diagram", "[Image uploaded]"))
    # For PDFs, could add more advanced image extraction if needed
    return visuals

# --- Calendar Integration Helper (MUST BE DEFINED BEFORE USAGE) ---
def detect_deadlines(text):
    prompt = (
        "Extract all assignment or exam deadlines (with date and description) from the following text. "
        "Return a JSON list of objects with 'date' and 'description'.\n\n" + text[:5000]
    )
    import json
    try:
        deadlines_json = call_gemini(prompt)
        deadlines = json.loads(deadlines_json)
        if isinstance(deadlines, dict):
            deadlines = list(deadlines.values())
        return deadlines
    except Exception:
        return []

# --- Lottie Loading Helper (MUST BE DEFINED BEFORE USAGE) ---
def load_lottieurl(url):
    r = reqs.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@contextlib.contextmanager
def show_lottie_loading(message="Loading..."):
    # Create a container for the loading animation
    container = st.empty()
    try:
        # Show the loading animation
        with container:
            st.spinner(message)
        yield
    finally:
        # Remove the entire container and its contents
        container.empty()
        # Force a rerun to ensure the UI updates
        

# --- Configuration from st.secrets ---
raw_uri       = st.secrets.get("google", {}).get("redirect_uri", "")
REDIRECT_URI  = raw_uri.rstrip("/") + "/" if raw_uri else ""
CLIENT_ID     = st.secrets.get("google", {}).get("client_id", "")
CLIENT_SECRET = st.secrets.get("google", {}).get("client_secret", "")
SCOPES        = ["openid", "email", "profile"]
GEMINI_API_KEY = st.secrets.get("gemini", {}).get("api_key", "")
CSE_API_KEY    = st.secrets.get("google_search", {}).get("api_key", "")
CSE_ID         = st.secrets.get("google_search", {}).get("cse_id", "")
CACHE_TTL      = 3600

# --- Session State ---
for key in ("token", "user"):
    if key not in st.session_state:
        st.session_state[key] = None

# --- SQLite DB for Learning Style ---
DB_PATH = "learning_style.db"

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS learning_style (
                    email TEXT PRIMARY KEY,
                    sensing_intuitive INTEGER,
                    visual_verbal INTEGER,
                    active_reflective INTEGER,
                    sequential_global INTEGER
                )
            ''')



def save_learning_style(email, scores):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute('''
                INSERT INTO learning_style (email, sensing_intuitive, visual_verbal, active_reflective, sequential_global)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    sensing_intuitive=excluded.sensing_intuitive,
                    visual_verbal=excluded.visual_verbal,
                    active_reflective=excluded.active_reflective,
                    sequential_global=excluded.sequential_global
            ''', (email, scores['Sensing/Intuitive'], scores['Visual/Verbal'], scores['Active/Reflective'], scores['Sequential/Global']))

def get_learning_style(email):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute('SELECT sensing_intuitive, visual_verbal, active_reflective, sequential_global FROM learning_style WHERE email=?', (email,))
        row = cur.fetchone()
        if row:
            return {
                'Sensing/Intuitive': row[0],
                'Visual/Verbal': row[1],
                'Active/Reflective': row[2],
                'Sequential/Global': row[3],
            }
        return None

# --- Export Helpers ---
def export_flashcards_to_anki(flashcards, filename="flashcards.csv"):
    with open(filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Front", "Back"])
        for q, a in flashcards:
            writer.writerow([q, a])
    return filename

init_db()

# --- SQLite DB for Memorization Tracking ---
def init_mem_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memorization (
                    email TEXT,
                    card_id TEXT,
                    question TEXT,
                    answer TEXT,
                    last_reviewed DATE,
                    next_due DATE,
                    correct_count INTEGER DEFAULT 0,
                    incorrect_count INTEGER DEFAULT 0,
                    PRIMARY KEY (email, card_id)
                )
            ''')

def get_due_cards(email, today):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute('''
            SELECT card_id, question, answer, last_reviewed, next_due, correct_count, incorrect_count
            FROM memorization
            WHERE email=? AND (next_due IS NULL OR next_due<=?)
            ORDER BY next_due ASC
            LIMIT 10
        ''', (email, today))
        return cur.fetchall()

def update_card_review(email, card_id, correct, today):
    # Simple spaced repetition: if correct, next_due += 3 days, else next_due = tomorrow
    import datetime
    next_due = (datetime.datetime.strptime(today, "%Y-%m-%d") + (datetime.timedelta(days=3) if correct else datetime.timedelta(days=1))).strftime("%Y-%m-%d")
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            if correct:
                conn.execute('''
                    UPDATE memorization SET last_reviewed=?, next_due=?, correct_count=correct_count+1 WHERE email=? AND card_id=?
                ''', (today, next_due, email, card_id))
            else:
                conn.execute('''
                    UPDATE memorization SET last_reviewed=?, next_due=?, incorrect_count=incorrect_count+1 WHERE email=? AND card_id=?
                ''', (today, next_due, email, card_id))

def add_memorization_card(email, card_id, question, answer):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute('''
                INSERT OR IGNORE INTO memorization (email, card_id, question, answer) VALUES (?, ?, ?, ?)
            ''', (email, card_id, question, answer))

init_mem_db()

# --- SQLite DB for Content Structure & Progress Tracking ---
def init_structure_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS content_structure (
                    email TEXT,
                    doc_id TEXT,
                    section TEXT,
                    progress REAL DEFAULT 0.0,
                    PRIMARY KEY (email, doc_id, section)
                )
            ''')

def save_content_structure(email, doc_id, sections):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            for section in sections:
                conn.execute('''
                    INSERT OR IGNORE INTO content_structure (email, doc_id, section) VALUES (?, ?, ?)
                ''', (email, doc_id, section))

def update_section_progress(email, doc_id, section, progress):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute('''
                UPDATE content_structure SET progress=? WHERE email=? AND doc_id=? AND section=?
            ''', (progress, email, doc_id, section))

def get_section_progress(email, doc_id):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute('''
            SELECT section, progress FROM content_structure WHERE email=? AND doc_id=?
        ''', (email, doc_id))
        return dict(cur.fetchall())
        
LOGO_URL = "https://github.com/rekfdjkzbdfvkgjerkdfnfcbvgewhs/Vekkam/blob/main/logo.png"  # <-- Replace with your actual raw GitHub URL

init_structure_db()

# --- OAuth Flow using st.query_params ---
def ensure_logged_in():
    params = st.query_params
    code = params.get("code")  # returns a str or None

    # Exchange code for token
    if code and not st.session_state.token:
        res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code"
            }
        )
        if res.status_code != 200:
            st.error(f"Token exchange failed ({res.status_code}): {res.text}")
            st.stop()
        st.session_state.token = res.json()

        # Clear code from URL
        st.query_params.clear()

        # Fetch user info
        ui = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {st.session_state.token['access_token']}"}
        )
        if ui.status_code != 200:
            st.error("Failed to fetch user info.")
            st.stop()
        st.session_state.user = ui.json()

    # If not logged in, show landing page
    if not st.session_state.token:
        # Landing page layout
        st.markdown("""
            <style>
            .main {
                padding: 2rem;
            }
            .feature-box {
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
            }
            .cta-button {
                background-color: #FF4B4B;
                color: white !important;
                padding: 15px 30px;
                border-radius: 5px;
                text-decoration: none !important;
                font-weight: bold;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                transition: background-color 0.3s ease;
                font-size: 1.2rem;
            }
            .cta-button:hover {
                background-color: #FF3333;
                text-decoration: none !important;
            }
            .google-icon {
                width: 24px;
                height: 24px;
                margin-right: 5px;
            }
            .hero-section {
                text-align: center;
                padding: 2rem 0;
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
                border-radius: 15px;
                margin-bottom: 2rem;
            }
            .feature-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            .testimonial-box {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stats-box {
                text-align: center;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
                margin: 10px 0;
            }
            .stats-number {
                font-size: 2rem;
                font-weight: bold;
                color: #FF4B4B;
            }
            </style>
        """, unsafe_allow_html=True)

        # Hero Section with CTA
        st.markdown('<div class="hero-section">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(LOGO_URL, width=200)
        with col2:
            st.markdown("<h1 style='font-size: 3rem; margin-bottom: 1rem;'>Welcome to Vekkam 📚</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='font-size: 1.5rem; color: #666; margin-bottom: 2rem;'>Your AI-powered study companion</h2>", unsafe_allow_html=True)
            
            # CTA Button
            auth_url = (
                "https://accounts.google.com/o/oauth2/v2/auth?"
                + urlencode({
                    "client_id": CLIENT_ID,
                    "redirect_uri": REDIRECT_URI,
                    "response_type": "code",
                    "scope": " ".join(SCOPES),
                    "access_type": "offline",
                    "prompt": "consent"
                })
            )
            st.markdown(
                f'<a href="{auth_url}" class="cta-button">'
                f'<img src="https://www.google.com/favicon.ico" class="google-icon" alt="Google icon">'
                f'Start Learning Now</a>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Stats Section
        st.markdown("## 📊 Trusted by Students Worldwide")
        stats_cols = st.columns(4)
        stats = [
        ]
        for i, (number, label) in enumerate(stats):
            with stats_cols[i]:
                st.markdown(f'<div class="stats-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="stats-number">{number}</div>', unsafe_allow_html=True)
                st.markdown(f'<div>{label}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Features Section
        st.markdown("## ✨ Powerful Features for Better Learning")
        features = [
            {
                "icon": "📖",
                "title": "Guide Book Chat",
                "description": "Search and chat with textbooks. Get instant explanations of any concept!",
                "details": [
                    "Smart search across multiple textbooks",
                    "Real-time concept explanations",
                    "Interactive Q&A with AI tutor",
                    "Save and review important concepts"
                ]
            },
            {
                "icon": "📝",
                "title": "Document Q&A",
                "description": "Upload your notes or books and get instant learning aids personalized for you.",
                "details": [
                    "Support for PDF, images, and text files",
                    "Automatic summary generation",
                    "Key points extraction",
                    "Custom flashcards creation"
                ]
            },
            {
                "icon": "📚",
                "title": "Paper Solver",
                "description": "Upload exam papers and get model answers with detailed explanations.",
                "details": [
                    "Step-by-step solutions",
                    "Exam tips and strategies",
                    "Common mistakes analysis",
                    "Practice questions generation"
                ]
            },
            {
                "icon": "🧠",
                "title": "Personalized Learning",
                "description": "AI-powered learning style assessment and personalized study recommendations.",
                "details": [
                    "Learning style analysis",
                    "Custom study plans",
                    "Progress tracking",
                    "Adaptive difficulty levels"
                ]
            }
        ]

        for feature in features:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown(f"### {feature.get('icon', '')} {feature.get('title', '')}")
            st.markdown(f"**{feature.get('description', '')}**")
            st.markdown("")
            for detail in feature.get('details', []):
                st.markdown(f"- {detail}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Benefits Section
        st.markdown("## 🎯 Why Choose Vekkam?")
        benefits = [
            {
                "icon": "🤖",
                "title": "AI-Powered Learning",
                "description": "Advanced AI technology adapts to your learning style and pace"
            },
            {
                "icon": "📊",
                "title": "Personalized Experience",
                "description": "Get customized study recommendations based on your learning style"
            },
            {
                "icon": "📱",
                "title": "Access Anywhere",
                "description": "Study on any device, anytime, with seamless synchronization"
            },
            {
                "icon": "🌐",
                "title": "Multi-Language Support",
                "description": "Learn in your preferred language with accurate translations"
            },
            {
                "icon": "📈",
                "title": "Track Progress",
                "description": "Monitor your learning journey with detailed analytics"
            },
            {
                "icon": "🎯",
                "title": "Exam Success",
                "description": "Boost your confidence with comprehensive exam preparation"
            }
        ]
        
        cols = st.columns(3)
        for i, benefit in enumerate(benefits):
            with cols[i % 3]:
                st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                st.markdown(f"### {benefit['icon']} {benefit['title']}")
                st.markdown(benefit['description'])
                st.markdown('</div>', unsafe_allow_html=True)

        # Testimonials Section
        st.markdown("## 💬 What Students Say")
        testimonials = [
            {
                "name": "Sarah K.",
                "role": "Medical Student",
                "text": "Vekkam helped me understand complex medical concepts through its interactive Q&A feature. The personalized learning approach made studying much more effective!"
            },
            {
                "name": "Michael R.",
                "role": "Engineering Student",
                "text": "The paper solver feature is amazing! It not only gives me the answers but also explains the concepts thoroughly. My grades have improved significantly."
            },
            {
                "name": "Priya M.",
                "role": "High School Student",
                "text": "I love how Vekkam adapts to my learning style. The flashcards and summaries are perfect for quick revision before exams."
            }
        ]

        for testimonial in testimonials:
            st.markdown('<div class="testimonial-box">', unsafe_allow_html=True)
            st.markdown(f"**{testimonial['name']}** - {testimonial['role']}")
            st.markdown(f"_{testimonial['text']}_")
            st.markdown('</div>', unsafe_allow_html=True)

        # Final CTA
        st.markdown("## 🚀 Ready to Transform Your Learning?")
        st.markdown("Join thousands of students who are already learning smarter with Vekkam!")
        st.markdown(
            f'<a href="{auth_url}" class="cta-button">'
            f'<img src="https://www.google.com/favicon.ico" class="google-icon" alt="Google icon">'
            f'Start Learning Now</a>',
            unsafe_allow_html=True
        )
        
        st.stop()

# Run OAuth check at startup
ensure_logged_in()

# --- After authentication UI ---
user = st.session_state.user

# Check for learning style in DB
learning_style = get_learning_style(user.get("email", ""))
if learning_style is None:
    st.title(f"Welcome, {user.get('name', '')}!")
    st.header("Learning Style Test")
    st.write("Answer the following questions to determine your learning style. This will help us personalize your experience.")
    likert = [
        "Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", "Somewhat Agree", "Agree", "Strongly Agree"
    ]
    questions = {
        "Sensing/Intuitive": [
            ("I am more interested in what is actual than what is possible.", "Sensing"),
            ("I often focus on the big picture rather than the details.", "Intuitive"),
            ("I trust my gut feelings over concrete evidence.", "Intuitive"),
            ("I enjoy tasks that require attention to detail.", "Sensing"),
            ("I prefer practical solutions over theoretical ideas.", "Sensing"),
            ("I am drawn to abstract concepts and patterns.", "Intuitive"),
            ("I notice details that others might miss.", "Sensing"),
            ("I like to imagine possibilities and what could be.", "Intuitive"),
            ("I rely on past experiences to guide me.", "Sensing"),
            ("I am energized by exploring new ideas.", "Intuitive"),
        ],
        "Visual/Verbal": [
            ("I remember best what I see (pictures, diagrams, charts).", "Visual"),
            ("I remember best what I hear or read.", "Verbal"),
            ("I prefer to learn through images and spatial understanding.", "Visual"),
            ("I prefer to learn through words and explanations.", "Verbal"),
        ],
        "Active/Reflective": [
            ("I learn best by doing and trying things out.", "Active"),
            ("I learn best by thinking and reflecting.", "Reflective"),
            ("I prefer group work and discussions.", "Active"),
            ("I prefer to work alone and think things through.", "Reflective"),
        ],
        "Sequential/Global": [
            ("I learn best in a step-by-step, logical order.", "Sequential"),
            ("I like to see the big picture before the details.", "Global"),
            ("I prefer to follow clear, linear instructions.", "Sequential"),
            ("I often make connections between ideas in a holistic way.", "Global"),
        ],
    }
    if "learning_style_answers" not in st.session_state:
        st.session_state.learning_style_answers = {}
    for dichotomy, qs in questions.items():
        st.subheader(dichotomy)
        for i, (q, side) in enumerate(qs):
            key = f"{dichotomy}_{i}"
            st.session_state.learning_style_answers[key] = st.radio(
                q,
                ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", "Somewhat Agree", "Agree", "Strongly Agree"],
                key=key
            )
    if st.button("Submit"):
        # Scoring: Strongly Disagree=0, ..., Neutral=50, ..., Strongly Agree=100 (for positive phrasing)
        # For each question, if side matches dichotomy, score as is; if not, reverse
        score_map = {0: 0, 1: 17, 2: 33, 3: 50, 4: 67, 5: 83, 6: 100}
        scores = {}
        for dichotomy, qs in questions.items():
            total = 0
            for i, (q, side) in enumerate(qs):
                key = f"{dichotomy}_{i}"
                val = st.session_state.learning_style_answers[key]
                idx = likert.index(val)
                # If the question is for the first side, score as is; if for the opposite, reverse
                if side == dichotomy.split("/")[0]:
                    score = score_map[idx]
                else:
                    score = score_map[6 - idx]
                total += score
            scores[dichotomy] = int(total / len(qs))
        with show_lottie_loading("Saving your learning style and personalizing your experience..."):
            save_learning_style(user.get("email", ""), scores)
            st.session_state.learning_style_answers = {}
        st.success("Learning style saved! Reloading...")
        st.balloons()
        
        
    st.stop()

st.sidebar.image(user.get("picture", ""), width=48)
st.sidebar.write(user.get("email", ""))

# --- Personalized for you box ---
def learning_style_description(scores):
    desc = []
    if scores['Sensing/Intuitive'] >= 60:
        desc.append("Prefers concepts, patterns, and big-picture thinking.")
    elif scores['Sensing/Intuitive'] <= 40:
        desc.append("Prefers facts, details, and practical examples.")
    if scores['Visual/Verbal'] >= 60:
        desc.append("Learns best with visuals, diagrams, and mind maps.")
    elif scores['Visual/Verbal'] <= 40:
        desc.append("Learns best with text, explanations, and reading.")
    if scores['Active/Reflective'] >= 60:
        desc.append("Enjoys interactive, hands-on, and group activities.")
    elif scores['Active/Reflective'] <= 40:
        desc.append("Prefers reflection, summaries, and solo study.")
    if scores['Sequential/Global'] >= 60:
        desc.append("Prefers holistic overviews and big-picture connections.")
    elif scores['Sequential/Global'] <= 40:
        desc.append("Prefers step-by-step, structured learning.")
    return desc

if learning_style:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Personalized for you")
    st.sidebar.write({k: f"{v}/100" for k, v in learning_style.items()})
    for d in learning_style_description(learning_style):
        st.sidebar.info(d)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    

# --- PDF/Text Extraction ---
def extract_pages_from_url(pdf_url):
    with show_lottie_loading("Extracting PDF from URL..."):
        r = requests.get(pdf_url)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(r.content); tmp.flush()
        reader = PdfReader(tmp.name)
        return {i+1: reader.pages[i].extract_text() for i in range(len(reader.pages))}

def extract_pages_from_file(file):
    with show_lottie_loading("Extracting PDF from file..."):
        reader = PdfReader(file)
        return {i+1: reader.pages[i].extract_text() for i in range(len(reader.pages))}

def extract_text(file):
    ext = file.name.lower().split('.')[-1]
    if ext == "pdf":
        return "\n".join(extract_pages_from_file(file).values())
    if ext in ("jpg","jpeg","png"):
        with show_lottie_loading("Extracting text from image..."):
            return pytesseract.image_to_string(Image.open(file))
    with show_lottie_loading("Extracting text from file..."):
        return StringIO(file.getvalue().decode()).read()

# --- Guide Book Search & Concept Q&A ---
def fetch_pdf_url(title, author, edition):
    q = " ".join(filter(None, [title, author, edition]))
    params = {"key": CSE_API_KEY, "cx": CSE_ID, "q": q, "fileType": "pdf", "num": 1}
    with show_lottie_loading("Searching for PDF guide book..."):
        items = requests.get("https://www.googleapis.com/customsearch/v1", params=params).json().get("items", [])
    return items[0]["link"] if items else None

def find_concept_pages(pages, concept):
    cl = concept.lower()
    return {p: t for p, t in pages.items() if cl in (t or "").lower()}

def ask_concept(pages, concept):
    found = find_concept_pages(pages, concept)
    if not found:
        return f"Couldn't find '{concept}'."
    combined = "\n---\n".join(f"Page {p}: {t}" for p, t in found.items())
    return call_gemini(f"Concept: '{concept}'. Sections:\n{combined}\nExplain with context and examples.")

# --- AI Learning Aids Functions ---
def generate_summary(text): 
    return call_gemini(
        f"Summarize this for an exam and separately list any formulae that are mentioned in the text. "
        f"If there aren't any, skip this section. Output only.:\n\n{text}",
        temperature=0.5
    )

def generate_questions(text): 
    return call_gemini(
        f"Generate 15 quiz questions for an exam (ignore authors, ISSN, etc.). Output only.:\n\n{text}",
        temperature=0.7, max_tokens=8192
    )

def generate_flashcards(text): 
    return call_gemini(
        f"Create flashcards (Q&A). Output only.:\n\n{text}",
        temperature=0.7, max_tokens=8192
    )

def generate_mnemonics(text): 
    return call_gemini(
        f"Generate mnemonics. Output only.:\n\n{text}",
        temperature=0.7, max_tokens=8192
    )

def generate_key_terms(text): 
    return call_gemini(
        f"List all the key terms in the doc, with definitions. Output only.:\n\n{text}",
        temperature=0.7, max_tokens=8192
    )

def generate_cheatsheet(text): 
    return call_gemini(
        f"Create a cheat sheet. Output only.:\n\n{text}",
        temperature=0.7, max_tokens=8192
    )

def generate_highlights(text): 
    return call_gemini(
        f"List key facts and highlights. Output only.:\n\n{text}",
        temperature=0.7, max_tokens=8192
    )

def generate_critical_points(text):
    return call_gemini(
        f"I haven't studied for the exam, so run me over the doc in detail but concise, "
        f"so that I'm ready for the exam. Output only.:\n\n{text}",
        temperature=0.7, max_tokens=8192
    )

# --- Explainer Podcast Feature ---
def generate_podcast(text, filename="explainer_podcast.mp3"):
    """
    Generate an explainer podcast from the given text.
    """
    # Step 1: Generate podcast script using call_gemini
    prompt = (
        "You are an expert educator. Create a conversational podcast script that explains the following document "
        "in a clear, engaging, and easy-to-understand manner. Use examples, analogies, and a friendly tone. "
        "Structure the explanation into sections with transitions between topics.\n\n"
        f"Document Content:\n{text}"
    )
    try:
        podcast_script = call_gemini(prompt)
    except Exception as e:
        st.error(f"Failed to generate podcast script: {str(e)}")
        return None

    # Step 2: Convert script to audio using gTTS
    try:
        tts = gTTS(podcast_script, lang="en")
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Failed to generate audio: {str(e)}")
        return None

# --- Helper to render each section in Streamlit ---
def render_section(title, content):
    st.subheader(title)
    if content.strip().startswith("<"):
        # raw HTML content
        components.html(content, height=600, scrolling=True)
    else:
        st.markdown(content, unsafe_allow_html=True)

def plot_mind_map(json_text):
    try:
        mind_map = json.loads(json_text)
    except json.JSONDecodeError:
        st.error("Mind map JSON invalid.")
        return
    nodes, edges, counter = [], [], 0
    def add_node(node, parent=None):
        nonlocal counter
        nid = counter; counter += 1
        label = node.get("title") or node.get("label") or "Node"
        nodes.append((nid, label))
        if parent is not None:
            edges.append((parent, nid))
        for child in node.get("children", []):
            add_node(child, nid)
    add_node(mind_map)
    g = ig.Graph(directed=True)
    g.add_vertices([str(n[0]) for n in nodes])
    g.vs["label"] = [n[1] for n in nodes]
    g.add_edges([(str(u),str(v)) for u,v in edges])
    layout = g.layout("tree")
    x,y = zip(*layout.coords)
    edge_x,edge_y = [],[]
    for u,v in edges:
        edge_x += [x[u],x[v],None]
        edge_y += [y[u],y[v],None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none")
    node_trace = go.Scatter(x=x, y=y, text=g.vs["label"], mode="markers+text", textposition="top center",
                            marker=dict(size=20, line=dict(width=2)))
    fig = go.Figure([edge_trace, node_trace],
        layout=go.Layout(margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False)))
    st.plotly_chart(fig, use_container_width=True)

# --- Multilingual Support ---
languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr"
}

ui_translations = {
    "en": {
        "Guide Book Chat": "Guide Book Chat",
        "Document Q&A": "Document Q&A",
        "Learning Style Test": "Learning Style Test",
        "Paper Solver/Exam Guide": "Paper Solver/Exam Guide",
        "Upload your exam paper (PDF or image). The AI will extract questions and show you how to answer for full marks!": "Upload your exam paper (PDF or image). The AI will extract questions and show you how to answer for full marks!",
        "Upload Exam Paper (PDF/Image)": "Upload Exam Paper (PDF/Image)",
        "Found {n} questions:": "Found {n} questions:",
        "Select questions to solve (default: all)": "Select questions to solve (default: all)",
        "Solve Selected Questions": "Solve Selected Questions",
        "Model Answers & Exam Tips": "Model Answers & Exam Tips",
        "Welcome, {name}!": "Welcome, {name}!",
        "Feature": "Feature",
        "Logout": "Logout",
        "Learning Aids": "Learning Aids",
        "Pick a function": "Pick a function",
        "Run": "Run",
        "Recommended for you:": "Recommended for you:",
        "Personalized for you": "Personalized for you",
        "Answer the following questions to determine your learning style.": "Answer the following questions to determine your learning style.",
        "Submit Learning Style Test": "Submit Learning Style Test",
        "Saving your learning style and personalizing your experience...": "Saving your learning style and personalizing your experience...",
        "Learning style saved! Reloading...": "Learning style saved! Reloading...",
        "Extracting PDF from URL...": "Extracting PDF from URL...",
        "Extracting PDF from file...": "Extracting PDF from file...",
        "Extracting text from image...": "Extracting text from image...",
        "Extracting text from file...": "Extracting text from file...",
        "Thinking with Gemini AI...": "Thinking with Gemini AI...",
        "Searching for PDF guide book...": "Searching for PDF guide book...",
        "Extracting questions from PDF...": "Extracting questions from PDF...",
        "Extracting questions from image...": "Extracting questions from image...",
        "Solving Q{n}...": "Solving Q{n}..."
    },
    "hi": {
        "Guide Book Chat": "गाइड बुक चैट",
        "Document Q&A": "दस्तावेज़ प्रश्नोत्तर",
        "Learning Style Test": "अधिगम शैली परीक्षण",
        "Paper Solver/Exam Guide": "पेपर सॉल्वर/परीक्षा गाइड",
        "Upload your exam paper (PDF or image). The AI will extract questions and show you how to answer for full marks!": "अपना परीक्षा पत्र (PDF या छवि) अपलोड करें। एआई प्रश्नों को निकालेगा और आपको पूर्ण अंक पाने के लिए उत्तर कैसे देना है, यह बताएगा!",
        "Upload Exam Paper (PDF/Image)": "परीक्षा पत्र अपलोड करें (PDF/छवि)",
        "Found {n} questions:": "{n} प्रश्न मिले:",
        "Select questions to solve (default: all)": "हल करने के लिए प्रश्न चुनें (डिफ़ॉल्ट: सभी)",
        "Solve Selected Questions": "चयनित प्रश्न हल करें",
        "Model Answers & Exam Tips": "मॉडल उत्तर और परीक्षा टिप्स",
        "Welcome, {name}!": "स्वागत है, {name}!",
        "Feature": "विशेषता",
        "Logout": "लॉगआउट",
        "Learning Aids": "अधिगम सहायक",
        "Pick a function": "एक फ़ंक्शन चुनें",
        "Run": "चलाएँ",
        "Recommended for you:": "आपके लिए अनुशंसित:",
        "Personalized for you": "आपके लिए वैयक्तिकृत",
        "Answer the following questions to determine your learning style.": "अपनी अधिगम शैली निर्धारित करने के लिए निम्नलिखित प्रश्नों का उत्तर दें।",
        "Submit Learning Style Test": "अधिगम शैली परीक्षण सबमिट करें",
        "Saving your learning style and personalizing your experience...": "आपकी अधिगम शैली सहेजी जा रही है और आपके अनुभव को वैयक्तिकृत किया जा रहा है...",
        "Learning style saved! Reloading...": "अधिगम शैली सहेजी गई! पुनः लोड हो रहा है...",
        "Extracting PDF from URL...": "URL से PDF निकाला जा रहा है...",
        "Extracting PDF from file...": "फ़ाइल से PDF निकाला जा रहा है...",
        "Extracting text from image...": "छवि से पाठ निकाला जा रहा है...",
        "Extracting text from file...": "फ़ाइल से पाठ निकाला जा रहा है...",
        "Thinking with Gemini AI...": "Gemini AI के साथ सोच रहे हैं...",
        "Searching for PDF guide book...": "PDF गाइड बुक खोजी जा रही है...",
        "Extracting questions from PDF...": "PDF से प्रश्न निकाले जा रहे हैं...",
        "Extracting questions from image...": "छवि से प्रश्न निकाले जा रहे हैं...",
        "Solving Q{n}...": "Q{n} हल किया जा रहा है..."
    },
    # Add more languages as needed
}

def t(key, **kwargs):
    lang = st.session_state.get("language", "en")
    txt = ui_translations.get(lang, ui_translations["en"]).get(key, key)
    return txt.format(**kwargs)

def export_summary_to_pdf(summary, filename="summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary.split('\n'):
        pdf.cell(200, 10, txt=line, ln=1, align='L')
    pdf.output(filename)
    return filename

# Language selector in sidebar
if "language" not in st.session_state:
    st.session_state["language"] = "en"
lang_choice = st.sidebar.selectbox("🌐 Language", list(languages.keys()), index=0)
st.session_state["language"] = languages[lang_choice]

# --- App Branding ---
st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    .sidebar-content {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)
col1, col2 = st.columns([1, 8])
with col1:
    st.image(LOGO_URL, width=180)
with col2:
    st.markdown("<h1 style='margin-bottom:0;'>Vekkam 📚</h1>", unsafe_allow_html=True)
    st.caption("Your AI-powered study companion")

# --- Sidebar Onboarding/Help ---
st.sidebar.markdown("---")
with st.sidebar.expander("❓ How to use this app", expanded=False):
    st.markdown("""
    - **Choose your language** from the sidebar.
    - **Take the Learning Style Test** (first login) for personalized recommendations.
    - **Guide Book Chat**: Search and chat with textbooks.
    - **Document Q&A**: Upload notes or books for instant learning aids.
    - **Paper Solver/Exam Guide**: Upload an exam paper and get model answers.
    - All features are personalized for you!
    """)

# --- Main UI ---
quiz_tabs = [t("Guide Book Chat"), t("Document Q&A"), t("Learning Style Test"), t("Paper Solver/Exam Guide"), "🗓️ Daily Quiz", "⚡ 6-Hour Battle Plan", "🎯 Discipline Hub"]
tab = st.sidebar.selectbox(t("Feature"), quiz_tabs)

# Add this after the existing imports
def search_educational_resources(query, num_results=5):
    """Search for educational resources using Google Programmable Search Engine."""
    if not CSE_API_KEY or not CSE_ID:
        return []
    
    try:
        # Construct the search URL
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": CSE_API_KEY,
            "cx": CSE_ID,
            "q": query,
            "num": num_results,
            "safe": "active",
            "lr": "lang_en",  # English language results
            "as_sitesearch": "edu",  # Prioritize educational sites
            "as_filetype": "pdf"  # Include PDF resources
        }
        
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        
        # Format the results
        formatted_results = []
        for item in results:
            formatted_results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "file_type": item.get("fileFormat", ""),
                "source": item.get("displayLink", "")
            })
        
        return formatted_results
    except Exception as e:
        st.error(f"Error searching for resources: {str(e)}")
        return []

# Main tab selection
if tab == t("Guide Book Chat"):
    st.header("❓ Ask Your Questions")
    st.info("Ask any question or upload an image of your question. Our AI will help you understand and solve it!")

    # Create two columns for text input and image upload
    col1, col2 = st.columns(2)
    
    with col1:
        question = st.text_area("Type your question here:", height=150, 
            placeholder="Example: Can you explain how photosynthesis works?")
    
    with col2:
        uploaded_image = st.file_uploader("Or upload an image of your question:", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of your question or problem")

    # Process the question (either from text or image)
    if st.button("Get Answer"):
        with show_lottie_loading("Analyzing your question..."):
            if uploaded_image:
                # Extract text from image
                image_text = pytesseract.image_to_string(Image.open(uploaded_image))
                if not image_text.strip():
                    st.error("Could not read text from the image. Please try uploading a clearer image.")
                    st.stop()
                question = image_text

            if not question.strip():
                st.warning("Please either type a question or upload an image.")
                st.stop()

            # Search for relevant resources
            with show_lottie_loading("Searching for relevant resources..."):
                search_results = search_educational_resources(question)
            
            # Generate a comprehensive answer
            prompt = (
                f"You are an expert tutor. Please help the student understand and solve this question. "
                f"Provide a clear, step-by-step explanation that includes:\n"
                f"1. Key concepts involved\n"
                f"2. Step-by-step solution or explanation\n"
                f"3. Examples or analogies to help understanding\n"
                f"4. Common mistakes to avoid\n"
                f"5. Related concepts to explore\n\n"
                f"Question: {question}"
            )
            
            answer = call_gemini(prompt)
            
            # Display the answer in a nicely formatted way
            st.markdown("### 📝 Answer")
            st.markdown(answer)
            
            # Display relevant resources if found
            if search_results:
                st.markdown("---")
                st.markdown("### 📚 Relevant Resources")
                for i, result in enumerate(search_results, 1):
                    with st.expander(f"{i}. {result['title']}"):
                        st.markdown(f"**Source:** {result['source']}")
                        st.markdown(f"**Description:** {result['snippet']}")
                        st.markdown(f"[Open Resource]({result['link']})")
                        if result['file_type']:
                            st.markdown(f"**Type:** {result['file_type']}")
            
            # Add a section for follow-up questions
            st.markdown("---")
            st.markdown("### 💭 Follow-up Questions")
            follow_up_prompt = (
                f"Based on the student's question and the answer provided, suggest 3 follow-up questions "
                f"that would help deepen their understanding of the topic. Make them specific and thought-provoking.\n\n"
                f"Original Question: {question}\n"
                f"Answer: {answer}"
            )
            follow_ups = call_gemini(follow_up_prompt)
            st.markdown(follow_ups)

            # Add a section for practice problems
            st.markdown("---")
            st.markdown("### 📚 Practice Problems")
            practice_prompt = (
                f"Create 2 practice problems related to the concepts in the question. "
                f"For each problem, provide:\n"
                f"1. The problem statement\n"
                f"2. A hint\n"
                f"3. The solution\n\n"
                f"Original Question: {question}\n"
                f"Answer: {answer}"
            )
            practice_problems = call_gemini(practice_prompt)
            st.markdown(practice_problems)

            # Add a section for additional resources
            st.markdown("---")
            st.markdown("### 🔍 Additional Resources")
            resources_prompt = (
                f"Suggest 3-4 additional resources (videos, articles, interactive tools) that would help "
                f"the student better understand the topic. Include brief descriptions of each resource.\n\n"
                f"Original Question: {question}\n"
                f"Answer: {answer}"
            )
            resources = call_gemini(resources_prompt)
            st.markdown(resources)

elif tab == t("Document Q&A"):
    st.header("\U0001F4A1 " + t("Document Q&A"))
    st.info("Upload one or more documents and get instant learning aids, personalized for your style. The AI can now synthesize across multiple files!")
    uploaded_files = st.file_uploader("Upload PDF/Image/TXT (multiple allowed)", type=["pdf","jpg","png","txt"], help="Upload your notes, textbook, or image.", accept_multiple_files=True)
    texts = []
    file_names = []
    if uploaded_files:
        for uploaded in uploaded_files:
            # Extract text from file
            ext = uploaded.name.lower().split('.')[-1]
            if ext == "pdf":
                with show_lottie_loading("Extracting PDF from file..."):
                    reader = PdfReader(uploaded)
                    text = "\n".join([page.extract_text() for page in reader.pages])
            elif ext in ("jpg", "jpeg", "png"):
                with show_lottie_loading("Extracting text from image..."):
                    text = pytesseract.image_to_string(Image.open(uploaded))
            else:
                with show_lottie_loading("Extracting text from file..."):
                    text = StringIO(uploaded.getvalue().decode()).read()
            texts.append(text)
            file_names.append(uploaded.name)

        # Combine all extracted text
        combined_text = "\n".join(texts)

        # Generate Podcast
        st.subheader("🎙️ Generate Explainer Podcast")
        if st.button("Create Podcast"):
            with show_lottie_loading("Generating podcast..."):
                podcast_file = generate_podcast(combined_text)
                if podcast_file:
                    st.success("Podcast generated successfully!")
                    st.audio(podcast_file, format="audio/mp3")
                    st.download_button("Download Podcast", data=open(podcast_file, "rb"), file_name="explainer_podcast.mp3", mime="audio/mp3")

        # --- Generate learning aids for each file ---
        for idx, (text, fname) in enumerate(zip(texts, file_names)):
            st.subheader(f"Learning Aids for {fname}")
            
            # Generate and display all learning aids
            with show_lottie_loading("Generating summary..."):
                render_section("📌 Summary", generate_summary(text))
            with show_lottie_loading("Generating quiz questions..."):
                render_section("📝 Quiz Questions", generate_questions(text))

            with st.expander("📚 Flashcards"):
                with show_lottie_loading("Generating flashcards..."):
                    render_section("Flashcards", generate_flashcards(text))

            with st.expander("🧠 Mnemonics"):
                with show_lottie_loading("Generating mnemonics..."):
                    render_section("Mnemonics", generate_mnemonics(text))

            with st.expander("🔑 Key Terms"):
                with show_lottie_loading("Generating key terms..."):
                    render_section("Key Terms", generate_key_terms(text))

            with st.expander("📋 Cheat Sheet"):
                with show_lottie_loading("Generating cheat sheet..."):
                    render_section("Cheat Sheet", generate_cheatsheet(text))

            with st.expander("⭐ Highlights"):
                with show_lottie_loading("Generating highlights..."):
                    render_section("Highlights", generate_highlights(text))

            with st.expander("📌 Critical Points"):
                with show_lottie_loading("Generating critical points..."):
                    render_section("Critical Points", generate_critical_points(text))

            # Store for batch export
            all_summaries.append(generate_summary(text))
            flashcards_raw = generate_flashcards(text)
            try:
                flashcards_json = json.loads(flashcards_raw)
                flashcards = [(fc['question'], fc['answer']) for fc in flashcards_json]
            except Exception:
                # fallback: try to split by Q/A
                flashcards = []
                for line in flashcards_raw.split('\n'):
                    if ':' in line:
                        q, a = line.split(':', 1)
                        flashcards.append((q.strip(), a.strip()))
            all_flashcards.extend(flashcards)

        # --- Batch Export ---
        if all_flashcards:
            st.info("Export all generated flashcards as an Anki-compatible CSV file.")
            if st.button("Export All Flashcards to Anki CSV"):
                fname = export_flashcards_to_anki(all_flashcards)
                st.success(f"Flashcards exported: {fname}")
                st.toast("Flashcards exported!")
        if all_summaries:
            st.info("Export all generated summaries as a PDF file.")
            if st.button("Export All Summaries to PDF"):
                combined_summary = "\n\n".join(all_summaries)
                fname = export_summary_to_pdf(combined_summary)
                st.success(f"Summary exported: {fname}")
                st.toast("Summary exported!")

# --- Product Hunt API Integration ---
PRODUCT_HUNT_TOKEN = st.secrets.get("producthunt", {}).get("api_token", "")
PRODUCT_HUNT_ID = st.secrets.get("producthunt", {}).get("product_id", "")  # e.g., "vekkam"

import time
@st.cache_data(ttl=300)
def get_ph_stats():
    if not PRODUCT_HUNT_TOKEN or not PRODUCT_HUNT_ID:
        return {"votes": 0, "comments": []}
    headers = {"Authorization": f"Bearer {PRODUCT_HUNT_TOKEN}"}
    # Get upvotes
    votes_url = f"https://api.producthunt.com/v2/api/graphql"
    votes_query = {
        "query": f"""
        query {{
          post(slug: \"{PRODUCT_HUNT_ID}\") {{
            votesCount
            comments(first: 5) {{
              edges {{
                node {{
                  id
                  body
                  user {{ name profileImage }}
                }}
              }}
            }}
          }}
        }}
        """
    }
    try:
        r = requests.post(votes_url, headers=headers, json=votes_query)
        data = r.json()
        post = data['data']['post']
        votes = post['votesCount']
        comments = [
            {
                "body": edge['node']['body'],
                "user": edge['node']['user']['name'],
                "avatar": edge['node']['user']['profileImage']
            }
            for edge in post['comments']['edges']
        ]
        return {"votes": votes, "comments": comments}
    except Exception:
        return {"votes": 0, "comments": []}

# --- Footer: Product Hunt Upvote Button & Live Stats ---
ph_stats = get_ph_stats()

# --- Duolingo-Style Onboarding ---
if 'onboarding_complete' not in st.session_state:
    st.session_state['onboarding_complete'] = False
if 'onboarding_step' not in st.session_state:
    st.session_state['onboarding_step'] = 0

ONBOARDING_STEPS = [
    'Welcome',
    'Language',
    'Goal',
    'LearningStyle',
    'Finish'
]

if not st.session_state['onboarding_complete']:
    step = st.session_state['onboarding_step']
    st.markdown("""
        <style>
        .onboard-center {text-align:center; margin-top:2em;}
        </style>
    """, unsafe_allow_html=True)
    st.progress((step+1)/len(ONBOARDING_STEPS), text=f"Step {step+1} of {len(ONBOARDING_STEPS)}")
    if ONBOARDING_STEPS[step] == 'Welcome':
        st.markdown('<div class="onboard-center"><img src="https://github.com/rekfdjkzbdfvkgjerkdfnfcbvgewhs/Vekkam/blob/main/logo.png" width="120"/><h2>Welcome to Vekkam!</h2><p>Your AI-powered study companion.</p></div>', unsafe_allow_html=True)
        if st.button("Let's get started!"):
            st.session_state['onboarding_step'] += 1
            
    elif ONBOARDING_STEPS[step] == 'Language':
        st.markdown('<div class="onboard-center"><h3>🌐 Choose your language</h3></div>', unsafe_allow_html=True)
        lang_choice = st.selectbox("Language", list(languages.keys()), index=0)
        st.session_state["language"] = languages[lang_choice]
        if st.button("Next"):
            st.session_state['onboarding_step'] += 1
            
    elif ONBOARDING_STEPS[step] == 'Goal':
        st.markdown('<div class="onboard-center"><h3>🎯 What is your main study goal?</h3></div>', unsafe_allow_html=True)
        goal = st.radio("Choose a goal:", ["Exam Prep", "Daily Review", "Master a Subject", "Ace Assignments"], key="onboard_goal")
        st.session_state['study_goal'] = goal
        if st.button("Next"):
            st.session_state['onboarding_step'] += 1
            
    elif ONBOARDING_STEPS[step] == 'LearningStyle':
        st.markdown('<div class="onboard-center"><h3>🧠 Discover your learning style</h3><p>This helps us personalize your experience!</p></div>', unsafe_allow_html=True)
        # Use the same learning style test as before
        likert = [
            "Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", "Somewhat Agree", "Agree", "Strongly Agree"
        ]
        questions = {
            "Sensing/Intuitive": [
                ("I am more interested in what is actual than what is possible.", "Sensing"),
                ("I often focus on the big picture rather than the details.", "Intuitive"),
                ("I trust my gut feelings over concrete evidence.", "Intuitive"),
                ("I enjoy tasks that require attention to detail.", "Sensing"),
                ("I prefer practical solutions over theoretical ideas.", "Sensing"),
                ("I am drawn to abstract concepts and patterns.", "Intuitive"),
                ("I notice details that others might miss.", "Sensing"),
                ("I like to imagine possibilities and what could be.", "Intuitive"),
                ("I rely on past experiences to guide me.", "Sensing"),
                ("I am energized by exploring new ideas.", "Intuitive"),
            ],
            "Visual/Verbal": [
                ("I remember best what I see (pictures, diagrams, charts).", "Visual"),
                ("I find it easier to follow spoken instructions than written ones.", "Verbal"),
                ("I prefer to learn through images and spatial understanding.", "Visual"),
                ("I often take notes to help me remember.", "Verbal"),
                ("I visualize information in my mind.", "Visual"),
                ("I prefer reading to watching videos.", "Verbal"),
                ("I use color and layout to organize my notes.", "Visual"),
                ("I find it easier to express myself in writing.", "Verbal"),
                ("I am drawn to infographics and visual summaries.", "Visual"),
                ("I enjoy listening to lectures or podcasts.", "Verbal"),
            ],
            "Active/Reflective": [
                ("I learn best by doing and trying things out.", "Active"),
                ("I prefer to think things through before acting.", "Reflective"),
                ("I enjoy group work and discussions.", "Active"),
                ("I need time alone to process new information.", "Reflective"),
                ("I like to experiment and take risks in learning.", "Active"),
                ("I often review my notes quietly after class.", "Reflective"),
                ("I am energized by interacting with others.", "Active"),
                ("I prefer to observe before participating.", "Reflective"),
                ("I learn by teaching others or explaining concepts aloud.", "Active"),
                ("I keep a journal or log to reflect on my learning.", "Reflective"),
            ],
            "Sequential/Global": [
                ("I learn best in a step-by-step, logical order.", "Sequential"),
                ("I like to see the big picture before the details.", "Global"),
                ("I prefer to follow clear, linear instructions.", "Sequential"),
                ("I often make connections between ideas in a holistic way.", "Global"),
                ("I am comfortable breaking tasks into smaller parts.", "Sequential"),
                ("I sometimes jump to conclusions without all the steps.", "Global"),
                ("I like outlines and structured notes.", "Sequential"),
                ("I understand concepts better when I see how they fit together.", "Global"),
                ("I prefer to finish one thing before starting another.", "Sequential"),
                ("I enjoy brainstorming and exploring many ideas at once.", "Global"),
            ],
        }
        if "learning_style_answers" not in st.session_state:
            st.session_state.learning_style_answers = {}
        for dichotomy, qs in questions.items():
            st.subheader(dichotomy)
            for i, (q, side) in enumerate(qs):
                key = f"{dichotomy}_{i}"
                st.session_state.learning_style_answers[key] = st.radio(
                    q,
                    ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", "Somewhat Agree", "Agree", "Strongly Agree"],
                    key=key
                )
        if st.button("Finish Onboarding"):
            # Scoring: Strongly Disagree=0, ..., Neutral=50, ..., Strongly Agree=100 (for positive phrasing)
            scores = {}
            for dichotomy, qs in questions.items():
                total = 0
                for i, (q, side) in enumerate(qs):
                    key = f"{dichotomy}_{i}"
                    val = st.session_state.learning_style_answers[key]
                    # Direct mapping of responses to scores
                    if val == "Strongly Disagree":
                        score = 0
                    elif val == "Disagree":
                        score = 17
                    elif val == "Somewhat Disagree":
                        score = 33
                    elif val == "Neutral":
                        score = 50
                    elif val == "Somewhat Agree":
                        score = 67
                    elif val == "Agree":
                        score = 83
                    else:  # Strongly Agree
                        score = 100
                    
                    # If the question is for the opposite side, reverse the score
                    if side != dichotomy.split("/")[0]:
                        score = 100 - score
                    
                    total += score
                scores[dichotomy] = int(total / len(qs))
            save_learning_style(user.get("email", ""), scores)
            st.session_state.learning_style_answers = {}
            st.session_state['onboarding_step'] += 1
            
    elif ONBOARDING_STEPS[step] == 'Finish':
        st.markdown('<div class="onboard-center"><h2>🎉 You are all set!</h2><p>Your experience is now personalized. Let\'s start learning!</p></div>', unsafe_allow_html=True)
        st.balloons()
        if st.button("Go to Dashboard"):
            st.session_state['onboarding_complete'] = True
            
    st.stop()

elif tab == "⚡ 6-Hour Battle Plan":
    st.header("⚡ 6-Hour Battle Plan")
    st.info("Upload your syllabus, guide books, and study materials. We'll create a focused 6-hour study plan using Vekkam's features to help you ace your exam!")

    # File upload section
    st.subheader("📚 Upload Your Materials")
    uploaded_files = st.file_uploader(
        "Upload your syllabus, guide books, and study materials (PDF/Image/TXT)",
        type=["pdf", "jpg", "jpeg", "png", "txt"],
        accept_multiple_files=True,
        help="Upload all relevant study materials. The more you provide, the better the plan will be!"
    )

    # Additional information
    st.subheader("📝 Additional Information")
    exam_date = st.date_input("When is your exam?", help="This helps us prioritize topics")
    exam_duration = st.number_input("Exam duration (in hours)", min_value=1, max_value=6, value=3, help="How long is your exam?")
    weak_topics = st.text_area("Topics you find challenging (optional)", help="List topics you find difficult, separated by commas")
    strong_topics = st.text_area("Topics you're confident in (optional)", help="List topics you're good at, separated by commas")

    if st.button("Generate Battle Plan"):
        if not uploaded_files:
            st.warning("Please upload at least one study material.")
            st.stop()

        with show_lottie_loading("Analyzing your materials and creating a battle plan..."):
            # Extract text from all files
            all_text = []
            for file in uploaded_files:
                ext = file.name.lower().split('.')[-1]
                if ext == "pdf":
                    reader = PdfReader(file)
                    text = "\n".join([page.extract_text() for page in reader.pages])
                elif ext in ("jpg", "jpeg", "png"):
                    text = pytesseract.image_to_string(Image.open(file))
                else:
                    text = StringIO(file.getvalue().decode()).read()
                all_text.append(text)

            combined_text = "\n---\n".join(all_text)

            # First, analyze the content and create a topic breakdown
            content_analysis_prompt = (
                f"Analyze the following study materials and create a structured breakdown of topics. "
                f"For each topic, identify:\n"
                f"1. Key concepts and formulas\n"
                f"2. Difficulty level (Easy/Medium/Hard)\n"
                f"3. Estimated time needed for mastery\n"
                f"4. Dependencies on other topics\n\n"
                f"Materials: {combined_text}"
            )
            content_analysis = call_gemini(content_analysis_prompt)

            # Generate the battle plan
            battle_plan_prompt = (
                f"Create a detailed 6-hour study plan using Vekkam's features. "
                f"Consider:\n"
                f"1. Exam date: {exam_date}\n"
                f"2. Exam duration: {exam_duration} hours\n"
                f"3. Challenging topics: {weak_topics}\n"
                f"4. Strong topics: {strong_topics}\n\n"
                f"Content Analysis: {content_analysis}\n\n"
                f"The plan should include:\n"
                f"1. Hour-by-hour breakdown:\n"
                f"   - Topic to cover\n"
                f"   - Vekkam feature to use (Guide Book Chat, Document Q&A, etc.)\n"
                f"   - Specific tasks and goals\n"
                f"   - Break times\n"
                f"2. For each topic:\n"
                f"   - Quick concept review using Guide Book Chat\n"
                f"   - Practice questions using Paper Solver\n"
                f"   - Flashcards for key points\n"
                f"   - Summary generation\n"
                f"3. Progress tracking and checkpoints\n"
                f"4. Energy management tips\n\n"
                f"Study Materials: {combined_text}"
            )

            battle_plan = call_gemini(battle_plan_prompt)

            # Display the battle plan in a structured way
            st.markdown("## 📋 Your 6-Hour Battle Plan")
            st.markdown(battle_plan)

            # Generate topic-specific resources
            st.markdown("---")
            st.markdown("## 📚 Topic-Specific Resources")
            resources_prompt = (
                f"Based on the content analysis and battle plan, create a list of resources for each topic:\n"
                f"1. Key formulas to memorize\n"
                f"2. Practice questions to attempt\n"
                f"3. Flashcards to create\n"
                f"4. Summary points\n\n"
                f"Content Analysis: {content_analysis}\n"
                f"Battle Plan: {battle_plan}"
            )
            resources = call_gemini(resources_prompt)
            st.markdown(resources)

            # Generate a quick reference guide
            st.markdown("---")
            st.markdown("## 📝 Quick Reference Guide")
            reference_prompt = (
                f"Create a quick reference guide that includes:\n"
                f"1. All key formulas and concepts\n"
                f"2. Common mistakes to avoid\n"
                f"3. Last-minute tips for each topic\n"
                f"4. Time management strategies\n\n"
                f"Content Analysis: {content_analysis}\n"
                f"Battle Plan: {battle_plan}"
            )
            reference_guide = call_gemini(reference_prompt)
            st.markdown(reference_guide)

            # Add a section for mental preparation
            st.markdown("---")
            st.markdown("## 🧠 Mental Preparation")
            mental_prompt = (
                f"Provide advice on:\n"
                f"1. How to stay calm during the exam\n"
                f"2. Time management strategies\n"
                f"3. How to handle difficult questions\n"
                f"4. Post-exam review tips\n\n"
                f"Based on the exam duration of {exam_duration} hours and the topics covered."
            )
            mental_prep = call_gemini(mental_prompt)
            st.markdown(mental_prep)

            # Add export options
            st.markdown("---")
            st.markdown("## 📤 Export Options")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📱 Export to PDF"):
                    pdf_content = f"""
                    BATTLE PLAN
                    ==========
                    
                    {battle_plan}
                    
                    TOPIC-SPECIFIC RESOURCES
                    =======================
                    {resources}
                    
                    QUICK REFERENCE GUIDE
                    ====================
                    {reference_guide}
                    
                    MENTAL PREPARATION
                    =================
                    {mental_prep}
                    """
                    filename = export_summary_to_pdf(pdf_content, "battle_plan.pdf")
                    st.success(f"Battle plan exported to {filename}")
            
            with col2:
                if st.button("📅 Add to Calendar"):
                    # Create calendar event for study session
                    event_title = "6-Hour Study Battle Plan"
                    event_desc = f"""
                    Battle Plan:
                    {battle_plan}
                    
                    Quick Reference:
                    {reference_guide}
                    """
                    add_to_google_calendar({
                        "date": exam_date.strftime("%Y-%m-%d"),
                        "description": event_title
                    })
                    st.success("Added to your calendar!")

# Keep only this section in the sidebar:
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Support Vekkam")
st.sidebar.markdown(
    f'''
    <div style="text-align:center;">
        <a href="https://www.producthunt.com/products/vekkam" target="_blank" id="ph-upvote-link">
            <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=456789&theme=light" alt="Upvote Vekkam on Product Hunt" style="width: 150px; margin-bottom: 8px;"/>
        </a><br>
        <span style="font-size:1em; font-weight:bold; color:#da552f;">🔥 {ph_stats['votes']} upvotes</span><br>
        <a href="https://www.producthunt.com/products/vekkam" target="_blank" style="font-size:0.9em; font-weight:bold; color:#da552f; text-decoration:none;">👉 Upvote & Comment!</a>
    </div>
    ''', unsafe_allow_html=True
)

# Add upvote nudge to sidebar
if 'ph_upvoted' not in st.session_state:
    st.session_state['ph_upvoted'] = False
if not st.session_state['ph_upvoted']:
    if st.sidebar.button("👍 I upvoted Vekkam!"):
        st.session_state['ph_upvoted'] = True
        st.sidebar.success("Thank you for supporting us! 🎉")
else:
    st.sidebar.info("Thanks for your upvote! 🧡")

# Add recent comments to sidebar if available
if ph_stats['comments']:
    st.sidebar.markdown("### 💬 Recent Comments")
    for c in ph_stats['comments']:
        st.sidebar.markdown(
            f'<div style="margin-bottom:0.5em; font-size:0.9em;"><img src="{c["avatar"]}" width="24" style="vertical-align:middle;border-radius:50%;margin-right:4px;"/> <b>{c["user"]}</b><br><span style="font-size:0.85em;">{c["body"]}</span></div>',
            unsafe_allow_html=True
        )

def generate_mind_map(text):
    """
    Generate a mind map from text using Gemini AI and render it using igraph and plotly.
    """
    prompt = (
        "You are a mind map generator. Create a detailed mind map from the following text. "
        "Return ONLY a JSON object with the following structure:\n"
        "{\n"
        '  "title": "Main topic",\n'
        '  "children": [\n'
        '    {\n'
        '      "title": "Subtopic 1",\n'
        '      "children": [\n'
        '        {"title": "Detail 1"},\n'
        '        {"title": "Detail 2"}\n'
        '      ]\n'
        '    },\n'
        '    {\n'
        '      "title": "Subtopic 2",\n'
        '      "children": [\n'
        '        {"title": "Detail 3"},\n'
        '        {"title": "Detail 4"}\n'
        '      ]\n'
        '    }\n'
        "}\n\n"
        "Rules:\n"
        "1. The JSON must be valid and properly formatted\n"
        "2. Each node must have a 'title' field\n"
        "3. Parent nodes can have a 'children' array\n"
        "4. Leaf nodes should not have a 'children' field\n"
        "5. Keep titles concise but descriptive\n"
        "6. Organize information hierarchically\n"
        "7. Include all important concepts from the text\n"
        "8. Maximum depth should be 3 levels\n\n"
        f"Text to analyze:\n{text}"
    )
    
    # Get mind map data from Gemini
    mind_map_json = call_gemini(prompt)
    
    try:
        # Parse the JSON response
        mind_map = json.loads(mind_map_json)
        
        # Create igraph graph
        g = ig.Graph(directed=True)
        
        # Add nodes and edges recursively
        def add_node(node, parent=None):
            nonlocal counter
            nid = counter
            counter += 1
            label = node.get("title", "Node")
            g.add_vertex(name=str(nid), label=label)
            if parent is not None:
                g.add_edge(parent, str(nid))
            if "children" in node:
                for child in node["children"]:
                    add_node(child, str(nid))
        
        counter = 0
        add_node(mind_map)
        
        # Calculate layout
        layout = g.layout("tree")
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in g.es:
            x0, y0 = layout[g.vs[edge.source].index]
            x1, y1 = layout[g.vs[edge.target].index]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        for node in g.vs:
            x, y = layout[node.index]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node["label"])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=mind_map["title"],
                    font=dict(size=16),
                    x=0.5,
                    y=0.95
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        # Add interactivity
        fig.update_layout(
            dragmode='pan',
            modebar_add=['zoom', 'pan', 'reset', 'zoomIn', 'zoomOut'],
            modebar_remove=['lasso', 'select'],
            modebar_activecolor='#FF4B4B'
        )
        
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        return mind_map
        
    except json.JSONDecodeError:
        st.error("Failed to parse mind map JSON from Gemini AI")
        return None
    except Exception as e:
        st.error(f"Error generating mind map: {str(e)}")
        return None

if tab == "🎯 Discipline Hub": # Corrected from elif to if
    st.header("🎯 Discipline Hub")
    st.info("Build strong study habits and stay accountable with our discipline features!")

    # Create tabs for different discipline features
    discipline_tabs = st.tabs([
        "📊 Study Streak", 
        "👥 Accountability", 
        "⏱️ Focus Mode", 
        "📅 Smart Schedule",
        "📈 Study Analytics",
        "🏆 Rewards",
        "🎯 Study Environment",
        "🚫 Distraction Blocker"
    ])
    
    with discipline_tabs[0]:  # Study Streak
        st.subheader("📊 Study Streak")
        
        # Initialize streak in session state if not exists
        if 'study_streak' not in st.session_state:
            st.session_state.study_streak = {
                'current_streak': 0,
                'longest_streak': 0,
                'last_study_date': None,
                'free_passes_remaining': 1,
                'total_study_time': 0
            }
        
        # Display streak stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Streak", f"{st.session_state.study_streak['current_streak']} days")
        with col2:
            st.metric("Longest Streak", f"{st.session_state.study_streak['longest_streak']} days")
        with col3:
            st.metric("Total Study Time", f"{st.session_state.study_streak['total_study_time']} hours")
        
        # Daily study goal
        st.subheader("🎯 Today's Study Goal")
        goal = st.number_input("Set your study goal (hours)", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
        
        if st.button("Start Study Session"):
            st.session_state['study_session_start'] = datetime.datetime.now()
            st.session_state['study_goal'] = goal
            st.success("Study session started! Keep going! 💪")
        
        # Study session tracking
        if 'study_session_start' in st.session_state:
            elapsed = (datetime.datetime.now() - st.session_state['study_session_start']).total_seconds() / 3600
            progress = min(elapsed / st.session_state['study_goal'], 1.0)
            st.progress(progress)
            st.write(f"Time studied: {elapsed:.1f} hours")
            
            if st.button("End Study Session"):
                # Update streak
                today = datetime.datetime.now().date()
                last_study = st.session_state.study_streak['last_study_date']
                
                if last_study is None or (today - last_study).days == 1:
                    st.session_state.study_streak['current_streak'] += 1
                elif (today - last_study).days > 1:
                    if st.session_state.study_streak['free_passes_remaining'] > 0:
                        st.session_state.study_streak['free_passes_remaining'] -= 1
                        st.warning("Used a free pass to maintain your streak!")
                    else:
                        st.session_state.study_streak['current_streak'] = 1
                
                # Update longest streak
                if st.session_state.study_streak['current_streak'] > st.session_state.study_streak['longest_streak']:
                    st.session_state.study_streak['longest_streak'] = st.session_state.study_streak['current_streak']
                
                # Update total study time
                st.session_state.study_streak['total_study_time'] += elapsed
                st.session_state.study_streak['last_study_date'] = today
                
                del st.session_state['study_session_start']
                st.success("Great job! Study session completed! 🎉")
    
    with discipline_tabs[1]:  # Accountability
        st.subheader("👥 Accountability Partners")
        
        # Study buddy matching
        st.write("Find a study buddy with similar goals!")
        if st.button("Find Study Buddy"):
            # TODO: Implement study buddy matching algorithm
            st.info("Matching you with a study buddy...")
        
        # Group study sessions
        st.subheader("Group Study Sessions")
        if st.button("Create Study Group"):
            st.info("Creating a new study group...")
        
        # Progress sharing
        st.subheader("Share Progress")
        if st.button("Share Today's Progress"):
            st.info("Sharing your progress with accountability partners...")
    
    with discipline_tabs[2]:  # Focus Mode
        st.subheader("⏱️ Focus Mode")
        
        # Pomodoro timer
        st.write("Set up a Pomodoro study session")
        work_time = st.number_input("Work duration (minutes)", min_value=5, max_value=60, value=25)
        break_time = st.number_input("Break duration (minutes)", min_value=1, max_value=30, value=5)
        
        if st.button("Start Pomodoro"):
            st.session_state['pomodoro_start'] = datetime.datetime.now()
            st.session_state['pomodoro_work_time'] = work_time
            st.session_state['pomodoro_break_time'] = break_time
            st.success("Pomodoro session started! Stay focused! 🎯")
        
        # Display timer if active
        if 'pomodoro_start' in st.session_state:
            elapsed = (datetime.datetime.now() - st.session_state['pomodoro_start']).total_seconds() / 60
            if elapsed < st.session_state['pomodoro_work_time']:
                progress = elapsed / st.session_state['pomodoro_work_time']
                st.progress(progress)
                st.write(f"Work time remaining: {st.session_state['pomodoro_work_time'] - elapsed:.1f} minutes")
            else:
                st.info("Time for a break! 🎉")
                if st.button("End Break"):
                    del st.session_state['pomodoro_start']
    
    with discipline_tabs[3]:  # Smart Schedule
        st.subheader("📅 Smart Schedule")
        
        # Schedule optimization
        st.write("Let AI optimize your study schedule")
        if st.button("Generate Smart Schedule"):
            # TODO: Implement AI schedule optimization
            st.info("Generating your optimized study schedule...")
        
        # Time blocking
        st.subheader("Time Blocking")
        if st.button("Create Time Blocks"):
            st.info("Creating time blocks for focused study...")
        
        # Priority management
        st.subheader("Priority Management")
        if st.button("Set Study Priorities"):
            st.info("Setting up your study priorities...")
    
    with discipline_tabs[4]:  # Study Analytics
        st.subheader("📈 Study Analytics")
        
        # Initialize analytics in session state if not exists
        if 'study_analytics' not in st.session_state:
            st.session_state.study_analytics = {
                'daily_hours': [],
                'topics': {},
                'productivity_score': 0,
                'focus_sessions': 0
            }
        
        # Time distribution chart
        st.write("### 📊 Study Time Distribution")
        if st.session_state.study_analytics['daily_hours']:
            chart_data = pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=len(st.session_state.study_analytics['daily_hours'])),
                'Hours': st.session_state.study_analytics['daily_hours']
            })
            st.line_chart(chart_data.set_index('Date'))
        
        # Topic breakdown
        st.write("### 📚 Topic Breakdown")
        if st.session_state.study_analytics['topics']:
            topic_data = pd.DataFrame({
                'Topic': list(st.session_state.study_analytics['topics'].keys()),
                'Hours': list(st.session_state.study_analytics['topics'].values())
            })
            st.bar_chart(topic_data.set_index('Topic'))
        
        # Productivity insights
        st.write("### 💡 Productivity Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Productivity Score", f"{st.session_state.study_analytics['productivity_score']}/100")
        with col2:
            st.metric("Focus Sessions", f"{st.session_state.study_analytics['focus_sessions']}")
        
        # Recommendations
        st.write("### 🎯 Recommendations")
        if st.session_state.study_analytics['productivity_score'] < 70:
            st.info("Try increasing your focus sessions and maintaining a consistent study schedule.")
        else:
            st.success("Great job! Keep up the good work!")
    
    with discipline_tabs[5]:  # Rewards
        st.subheader("🏆 Rewards & Achievements")
        
        # Initialize rewards in session state if not exists
        if 'rewards' not in st.session_state:
            st.session_state.rewards = {
                'points': 0,
                'level': 1,
                'achievements': [],
                'streak_bonus': 0
            }
        
        # Display current status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Study Points", f"{st.session_state.rewards['points']}")
        with col2:
            st.metric("Level", f"{st.session_state.rewards['level']}")
        
        # Achievements
        st.write("### 🏅 Achievements")
        achievements = [
            {"name": "Early Bird", "description": "Study before 9 AM", "points": 50},
            {"name": "Night Owl", "description": "Study after 8 PM", "points": 50},
            {"name": "Streak Master", "description": "Maintain a 7-day streak", "points": 100},
            {"name": "Focus Champion", "description": "Complete 5 focus sessions", "points": 75}
        ]
        
        for achievement in achievements:
            if achievement["name"] in st.session_state.rewards['achievements']:
                st.success(f"🏆 {achievement['name']} - {achievement['description']}")
            else:
                st.info(f"🔒 {achievement['name']} - {achievement['description']}")
        
        # Rewards shop
        st.write("### 🎁 Rewards Shop")
        rewards = [
            {"name": "Free Pass", "description": "Skip a day without breaking streak", "cost": 100},
            {"name": "Study Music", "description": "Unlock premium study music", "cost": 200},
            {"name": "Custom Theme", "description": "Unlock custom app theme", "cost": 300}
        ]
        
        for reward in rewards:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{reward['name']}** - {reward['description']}")
            with col2:
                if st.button(f"Buy ({reward['cost']} pts)", key=f"buy_{reward['name']}"):
                    if st.session_state.rewards['points'] >= reward['cost']:
                        st.session_state.rewards['points'] -= reward['cost']
                        st.success(f"Purchased {reward['name']}!")
                    else:
                        st.error("Not enough points!")
    
    with discipline_tabs[6]:  # Study Environment
        st.subheader("🎯 Study Environment")
        
        # Environment checklist
        st.write("### ✅ Environment Checklist")
        checklist_items = [
            "Quiet, well-lit space",
            "Comfortable seating",
            "All materials ready",
            "Water and snacks nearby",
            "Phone on silent",
            "Notifications disabled",
            "Music/white noise if preferred"
        ]
        
        for item in checklist_items:
            st.checkbox(item)
        
        # Study space tips
        st.write("### 💡 Study Space Tips")
        tips = [
            "Keep your study area clean and organized",
            "Use natural lighting when possible",
            "Maintain a comfortable temperature",
            "Have a dedicated space for studying",
            "Keep distractions out of sight"
        ]
        
        for tip in tips:
            st.info(tip)
        
        # Ambient sounds
        st.write("### 🎵 Ambient Sounds")
        sound_options = ["Rain", "Cafe", "White Noise", "Nature", "Classical Music"]
        selected_sound = st.selectbox("Choose ambient sound", sound_options)
        if st.button("Play Sound"):
            st.info(f"Playing {selected_sound}...")
    
    with discipline_tabs[7]:  # Distraction Blocker
        st.subheader("🚫 Distraction Blocker")
        
        # Block distracting websites
        st.write("### 🌐 Block Distracting Websites")
        distracting_sites = st.multiselect(
            "Select websites to block",
            ["Facebook", "Instagram", "Twitter", "YouTube", "TikTok", "Reddit"],
            default=["Facebook", "Instagram", "Twitter"]
        )
        
        if st.button("Start Blocking"):
            st.success(f"Blocking {len(distracting_sites)} websites during study sessions")
        
        # Focus mode settings
        st.write("### ⚙️ Focus Mode Settings")
        col1, col2 = st.columns(2)
        with col1:
            block_duration = st.number_input("Block duration (minutes)", min_value=15, max_value=240, value=60)
        with col2:
            break_duration = st.number_input("Break duration (minutes)", min_value=5, max_value=30, value=10)
        
        # Notification settings
        st.write("### 🔔 Notification Settings")
        notification_options = {
            "Block all notifications": True,
            "Allow important calls": True,
            "Allow study-related notifications": True,
            "Block social media notifications": True
        }
        
        for option, default in notification_options.items():
            st.checkbox(option, value=default)
        
        if st.button("Apply Settings"):
            st.success("Focus mode settings applied!")
