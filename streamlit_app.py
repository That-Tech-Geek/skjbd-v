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
st.sidebar.image(user.get("picture", ""), width=48)
st.sidebar.write(user.get("email", ""))
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Gemini Call ---
def call_gemini(prompt, temp=0.7, max_tokens=2048):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temp, "maxOutputTokens": max_tokens}
    }
    return requests.post(url, json=payload).json()["candidates"][0]["content"]["parts"][0]["text"]

# --- PDF/Text Extraction ---
def extract_pages_from_url(pdf_url):
    r = requests.get(pdf_url)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(r.content); tmp.flush()
    reader = PdfReader(tmp.name)
    return {i+1: reader.pages[i].extract_text() for i in range(len(reader.pages))}

def extract_pages_from_file(file):
    reader = PdfReader(file)
    return {i+1: reader.pages[i].extract_text() for i in range(len(reader.pages))}

def extract_text(file):
    ext = file.name.lower().split('.')[-1]
    if ext == "pdf":
        return "\n".join(extract_pages_from_file(file).values())
    if ext in ("jpg","jpeg","png"):
        return pytesseract.image_to_string(Image.open(file))
    return StringIO(file.getvalue().decode()).read()

# --- Guide Book Search & Concept Q&A ---
def fetch_pdf_url(title, author, edition):
    q = " ".join(filter(None, [title, author, edition]))
    params = {"key": CSE_API_KEY, "cx": CSE_ID, "q": q, "fileType": "pdf", "num": 1}
    items = requests.get("https://www.googleapis.com/customsearch/v1", params=params).json().get("items", [])
    return items[0]["link"] if items else None

def find_concept_pages(pages, concept):
    cl = concept.lower()
    return {p: t for p, t in pages.items() if cl in (t or "").lower()}

def ask_concept(pages, concept):
    found = find_concept_pages(pages, concept)
    if not found:
        return f"Couldn’t find '{concept}'."
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

# --- Main UI ---
st.title(f"Welcome, {user.get('name', '')}!")
tab = st.sidebar.selectbox("Feature", ["Guide Book Chat", "Document Q&A"])

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

else:
    st.header("Document Q&A")
    uploaded = st.file_uploader("Upload PDF/Image/TXT", type=["pdf","jpg","png","txt"])
    if uploaded:
        text = extract_text(uploaded)
        st.subheader("Learning Aids")
        choice = st.selectbox("Pick a function", [
            "Summary","Questions","Flashcards","Mnemonics",
            "Key Terms","Cheat Sheet","Highlights",
            "Critical Points","Concept Chat","Mind Map"
        ])
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
