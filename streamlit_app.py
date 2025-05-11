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

# --- Configuration from st.secrets ---
raw_uri       = st.secrets["google"]["redirect_uri"]
REDIRECT_URI  = raw_uri.rstrip("/") + "/"
CLIENT_ID     = st.secrets["google"]["client_id"]
CLIENT_SECRET = st.secrets["google"]["client_secret"]
SCOPES        = ["openid", "email", "profile"]
GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
CSE_API_KEY    = st.secrets["google_search"]["api_key"]
CSE_ID         = st.secrets["google_search"]["cse_id"]
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

init_db()

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

    # If still not logged in, show Login link
    if not st.session_state.token:
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
        st.markdown(f"[**Login with Google**]({auth_url})")
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
                likert,
                key=key
            )
    if st.button("Submit Learning Style Test"):
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
        st.experimental_rerun()
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
    st.experimental_rerun()

# --- Lottie Loading Helper ---
def load_lottieurl(url):
    r = reqs.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@contextlib.contextmanager
def show_lottie_loading(message="Loading..."):
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_kyu7xb1v.json"  # Book animation
    lottie_json = load_lottieurl(lottie_url)
    lottie_placeholder = st.empty()
    msg_placeholder = st.empty()
    lottie_placeholder_lottie = lottie_placeholder.lottie(lottie_json, height=200, key="global_lottie")
    msg_placeholder.info(message)
    try:
        yield
    finally:
        lottie_placeholder.empty()
        msg_placeholder.empty()

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

# --- Learning Aids & Mind Map ---
def generate_summary(text):         return call_gemini(f"Summarize for exam, list formulae:\n{text}")
def generate_questions(text):       return call_gemini(f"Generate 15 quiz questions:\n{text}")
def generate_flashcards(text):      return call_gemini(f"Create flashcards (Q&A):\n{text}")
def generate_mnemonics(text):       return call_gemini(f"Generate mnemonics:\n{text}")
def generate_key_terms(text):       return call_gemini(f"List key terms with definitions:\n{text}")
def generate_cheatsheet(text):      return call_gemini(f"Create a cheat sheet:\n{text}")
def generate_highlights(text):      return call_gemini(f"List key facts and highlights:\n{text}")
def generate_critical_points(text): return call_gemini(f"Detailed but concise run-through:\n{text}")

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

# Language selector in sidebar
if "language" not in st.session_state:
    st.session_state["language"] = "en"
lang_choice = st.sidebar.selectbox("🌐 Language", list(languages.keys()), index=0)
st.session_state["language"] = languages[lang_choice]

# --- Main UI ---
tab = st.sidebar.selectbox(t("Feature"), [t("Guide Book Chat"), t("Document Q&A"), t("Learning Style Test"), t("Paper Solver/Exam Guide")])

if tab == "Guide Book Chat":
    st.header("Guide Book Chat")
    title   = st.text_input("Title")
    author  = st.text_input("Author")
    edition = st.text_input("Edition")
    concept = st.text_input("Ask about concept:")
    if st.button("Chat") and concept:
        url = fetch_pdf_url(title, author, edition)
        if not url:
            st.error("PDF not found")
        else:
            pages = extract_pages_from_url(url)
            st.write(ask_concept(pages, concept))

elif tab == "Learning Style Test":
    st.header("Learning Style Test")
    st.write("Answer the following questions to determine your learning style.")
    
    # Questions for each dichotomy
    questions = {
        "Sensing/Intuitive": [
            ("I prefer learning facts and concrete details.", "Sensing"),
            ("I enjoy exploring abstract concepts and theories.", "Intuitive"),
            ("I trust experience more than words and symbols.", "Sensing"),
            ("I like to imagine possibilities and what could be.", "Intuitive"),
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
    
    # Store answers in session state
    if "learning_style_answers" not in st.session_state:
        st.session_state.learning_style_answers = {}
    
    for dichotomy, qs in questions.items():
        st.subheader(dichotomy)
        for i, (q, side) in enumerate(qs):
            key = f"{dichotomy}_{i}"
            st.session_state.learning_style_answers[key] = st.radio(
                q,
                [f"Strongly {side}", f"Somewhat {side}", "Neutral", f"Somewhat Opposite", f"Strongly Opposite"],
                key=key
            )
    
    st.button("Submit")

elif tab == "Paper Solver/Exam Guide":
    st.header(t("Paper Solver/Exam Guide"))
    st.write(t("Upload your exam paper (PDF or image). The AI will extract questions and show you how to answer for full marks!"))
    exam_file = st.file_uploader(t("Upload Exam Paper (PDF/Image)"), type=["pdf", "jpg", "jpeg", "png"])
    if exam_file:
        # Extract text from file (multi-page supported)
        ext = exam_file.name.lower().split('.')[-1]
        if ext == "pdf":
            with show_lottie_loading("Extracting questions from PDF..."):
                pages = extract_pages_from_file(exam_file)
                text = "\n".join(pages.values())
        else:
            with show_lottie_loading("Extracting questions from image..."):
                text = pytesseract.image_to_string(Image.open(exam_file))
        # Improved question splitting: Q1, Q.1, 1., 1), Q2, etc.
        question_regex = r"(?:\n|^)(?:Q\.?\s*\d+|Q\s*\d+|\d+\.|\d+\)|Q\.|Q\s)(?=\s)"
        split_points = [m.start() for m in re.finditer(question_regex, text)]
        questions = []
        if split_points:
            for i, start in enumerate(split_points):
                end = split_points[i+1] if i+1 < len(split_points) else len(text)
                q = text[start:end].strip()
                if len(q) > 10:
                    questions.append(q)
        else:
            # fallback: split by lines with Q or numbers
            questions = re.split(r"\n\s*(?:Q\.?|\d+\.|\d+\))", text)
            questions = [q.strip() for q in questions if len(q.strip()) > 10]
        st.subheader(t("Found {n} questions:", n=len(questions)))
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}:** {q}")
        # Multiselect for which questions to solve
        selected = st.multiselect(
            t("Select questions to solve (default: all)"),
            options=[f"Q{i+1}" for i in range(len(questions))],
            default=[f"Q{i+1}" for i in range(len(questions))]
        )
        selected_indices = [int(s[1:]) - 1 for s in selected]
        if st.button(t("Solve Selected Questions")) and selected_indices:
            answers = []
            for idx in selected_indices:
                q = questions[idx]
                with show_lottie_loading(t("Solving Q{n}...", n=idx+1)):
                    prompt = (
                        f"You are an expert exam coach and math teacher. "
                        f"Given the following exam question, provide a model answer that would get full marks. "
                        f"If it is a math question, show all steps, calculations, and reasoning. "
                        f"If it is a theory question, answer as a top student would, using structure, keywords, and examples. "
                        f"Also, give tips on how to express the answer for maximum marks.\n\n"
                        f"Question: {q}"
                    )
                    answer = call_gemini(prompt)
                    answers.append((q, answer))
            st.header(t("Model Answers & Exam Tips"))
            for i, (q, a) in enumerate(answers, 1):
                st.markdown(f"**Q{i}:** {q}")
                st.write(a)

else:
    st.header("Document Q&A")
    uploaded = st.file_uploader("Upload PDF/Image/TXT", type=["pdf","jpg","png","txt"])
    if uploaded:
        text = extract_text(uploaded)
        st.subheader("Learning Aids")
        # Personalization: recommend/preselect based on learning style
        aid_options = [
            "Summary","Questions","Flashcards","Mnemonics",
            "Key Terms","Cheat Sheet","Highlights",
            "Critical Points","Concept Chat","Mind Map"
        ]
        recommended = []
        if learning_style:
            if learning_style['Visual/Verbal'] >= 60:
                recommended += ["Mind Map", "Highlights", "Summary"]
            if learning_style['Visual/Verbal'] <= 40:
                recommended += ["Summary", "Questions", "Flashcards", "Cheat Sheet"]
            if learning_style['Active/Reflective'] >= 60:
                recommended += ["Questions", "Flashcards"]
            if learning_style['Active/Reflective'] <= 40:
                recommended += ["Summary", "Critical Points"]
            if learning_style['Sequential/Global'] >= 60:
                recommended += ["Mind Map", "Summary"]
            if learning_style['Sequential/Global'] <= 40:
                recommended += ["Summary", "Cheat Sheet", "Key Terms"]
            if learning_style['Sensing/Intuitive'] >= 60:
                recommended += ["Mnemonics", "Mind Map"]
            if learning_style['Sensing/Intuitive'] <= 40:
                recommended += ["Highlights", "Key Terms"]
        # Remove duplicates, keep order
        recommended = [x for i, x in enumerate(recommended) if x not in recommended[:i]]
        default_idx = aid_options.index(recommended[0]) if recommended else 0
        st.markdown(
            f"**Recommended for you:** {', '.join(recommended) if recommended else aid_options[0]}"
        )
        choice = st.selectbox("Pick a function", aid_options, index=default_idx)
        if st.button("Run"):
            if choice == "Summary":       st.write(generate_summary(text))
            elif choice == "Questions":   st.write(generate_questions(text))
            elif choice == "Flashcards":  st.write(generate_flashcards(text))
            elif choice == "Mnemonics":   st.write(generate_mnemonics(text))
            elif choice == "Key Terms":   st.write(generate_key_terms(text))
            elif choice == "Cheat Sheet": st.write(generate_cheatsheet(text))
            elif choice == "Highlights":  st.write(generate_highlights(text))
            elif choice == "Critical Points": st.write(generate_critical_points(text))
            elif choice == "Concept Chat":
                cc = st.text_input("Concept to explain:")
                if cc:
                    pages = extract_pages_from_file(uploaded)
                    st.write(ask_concept(pages, cc))
                else:
                    st.info("Enter a concept first.")
            elif choice == "Mind Map":
                jm = call_gemini(f"Create JSON mind map from text:\n{text}")
                plot_mind_map(jm)

# --- Gemini Call ---
def call_gemini(prompt, temp=0.7, max_tokens=2048):
    lang = st.session_state.get("language", "en")
    lang_name = [k for k, v in languages.items() if v == lang][0]
    prompt = f"Please answer in {lang_name}.\n" + prompt
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temp, "maxOutputTokens": max_tokens}
    }
    with show_lottie_loading(t("Thinking with Gemini AI...")):
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
