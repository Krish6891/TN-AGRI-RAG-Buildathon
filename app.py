import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

DB_DIR = "vectorstore/faiss_index"
BG_IMAGE = Path("assets/TN.png")
TN_LOGO = Path("assets/TN Logo.png")


def to_b64(path: Path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


bg_b64 = to_b64(BG_IMAGE)
logo_b64 = to_b64(TN_LOGO)

st.set_page_config(
    page_title="AgriScheme AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

if bg_b64:
    bg_css = (
        "background-image: linear-gradient(rgba(238,247,242,0.62), rgba(238,247,242,0.74)), "
        f"url('data:image/jpg;base64,{bg_b64}');"
    )
else:
    bg_css = "background: radial-gradient(circle at 20% 20%, #eaf6f1 0%, #f5fbf8 45%, #eef4ff 100%);"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Manrope', sans-serif;
    font-size: 20px;
}}

.stApp {{
    {bg_css}
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background: rgba(242, 245, 249, 0.90);
    backdrop-filter: blur(10px);
}}

.main-card {{
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 20px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 10px 30px rgba(7, 56, 70, 0.10);
    margin-bottom: 0.8rem;
}}

.hero-title {{
    text-align: center;
    font-size: 3.25rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.15;
}}

.hero-sub {{
    text-align: center;
    margin: 0.6rem 0 0 0;
    color: #3e5163;
    font-size: 1.2rem;
}}

.badges {{
    text-align: center;
    margin-top: 0.8rem;
}}

.badge {{
    display: inline-block;
    background: #dff4ec;
    color: #0e5f46;
    border: 1px solid #bfe6d8;
    border-radius: 999px;
    padding: 0.30rem 0.95rem;
    font-size: 0.95rem;
    margin-right: 0.5rem;
    margin-bottom: 0.35rem;
}}

.kpi {{
    background: #ffffff;
    border: 1px solid #e8edf2;
    border-radius: 14px;
    padding: 0.9rem;
    text-align: center;
    margin-bottom: 0.7rem;
    font-size: 1.05rem;
}}

.sidebar-logo-center {{
    margin: 1.1rem 0 0.8rem 0;
    display: flex;
    justify-content: center;
}}

.sidebar-logo-box {{
    width: 180px;
    height: 180px;
    border-radius: 18px;
    border: 1px solid #d8e4de;
    background: radial-gradient(circle at center, rgba(255,255,255,0.92), rgba(233,244,238,0.88));
    backdrop-filter: blur(6px);
    box-shadow: 0 8px 20px rgba(18, 66, 48, 0.10);
    background-size: 78%;
    background-repeat: no-repeat;
    background-position: center;
}}

.chat-shell {{
    background: rgba(255,255,255,0.80);
    backdrop-filter: blur(9px);
    border: 1px solid #e6edf3;
    border-radius: 16px;
    padding: 0.9rem;
    margin-bottom: 0.7rem;
}}

.credit-block {{
    text-align: center;
    font-size: 1.38rem;
    font-weight: 800;
    margin-top: 3rem;
    margin-bottom: 0.8rem;
    color: #233446;
    line-height: 1.35;
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return retriever, llm


retriever, llm = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


def build_history_text(messages, max_turns=4):
    recent = messages[-(max_turns * 2):]
    return "\n".join(
        [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent]
    )


def unique_sources(docs):
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in out:
            out.append(src)
    return out


with st.sidebar:
    st.markdown("## AgriScheme AI")
    st.caption("SaaS assistant for TN Government Schemes")

    lang = st.selectbox("Response language", ["English", "Tamil"])
    k = st.slider("Context depth (top-k chunks)", 2, 8, 5)
    st.caption("Purpose: Higher value searches more chunks (broader answers). Lower value gives tighter focus.")

    st.markdown("---")
    st.markdown(
        f'<div class="kpi"><b>Chats</b><br>{len(st.session_state.messages)//2}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="kpi"><b>Latest Sources</b><br>{len(st.session_state.last_sources)}</div>',
        unsafe_allow_html=True,
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    if logo_b64:
        st.markdown(
            f"""
            <div class="sidebar-logo-center">
                <div class="sidebar-logo-box"
                     style="background-image:url('data:image/png;base64,{logo_b64}');">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-logo-center"><div class="sidebar-logo-box"></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

st.markdown(
    """
<div class="main-card">
  <h1 class="hero-title">🌾 AgriScheme AI Assistant</h1>
  <p class="hero-sub">Ask about eligibility, subsidy, documents, benefits, and scheme comparisons.</p>
  <div class="badges">
    <span class="badge">RAG + FAISS</span>
    <span class="badge">Grounded Answers</span>
    <span class="badge">Source Citations</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Chat input moved upward just below hero
st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
with st.form("ask_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question", placeholder="Type your question about TN schemes...")
    asked = st.form_submit_button("Ask")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Quick Keywords")
kw_cols = st.columns(4)
keywords = [
    "Eligibility conditions",
    "Subsidy amount",
    "Required documents",
    "Women farmer schemes",
    "Drip irrigation support",
    "Loan / credit schemes",
    "SC/ST farmer benefits",
    "How to apply",
]
clicked_keyword = ""
for i, kw in enumerate(keywords):
    if kw_cols[i % 4].button(kw, use_container_width=True):
        clicked_keyword = kw

query = ""
if asked and user_input.strip():
    query = user_input.strip()
elif clicked_keyword:
    query = clicked_keyword

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    retriever.search_kwargs = {"k": k}
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    history = build_history_text(st.session_state.messages[:-1], max_turns=4)

    prompt = f"""
You are a TN government scheme assistant.
Rules:
1) Answer ONLY from the provided context.
2) If not found, say: "I could not find this in the provided scheme documents."
3) Keep answers simple and practical.
4) Use bullet points.
5) If comparison is asked, provide table-like bullet comparison.
6) Respond in: {lang}

Conversation history:
{history}

Question:
{query}

Context:
{context}
"""

    answer = llm.invoke(prompt).content
    sources = unique_sources(docs)
    st.session_state.last_sources = sources
    st.session_state.messages.append({"role": "assistant", "content": answer})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if st.session_state.last_sources:
    with st.expander("View latest sources"):
        for s in st.session_state.last_sources:
            st.markdown(f"- {s}")

st.markdown(
    """
<div class="credit-block">
    Created by N Gopalakrishnan<br>
    Batch 9
</div>
""",
    unsafe_allow_html=True,
)