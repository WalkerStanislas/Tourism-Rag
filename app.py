from __future__ import annotations
import streamlit as st
import asyncio
import datetime
from src.rag_pipeline import load_models, TourismeAgent


# ===================== I. CONFIGURATION =====================

st.set_page_config(
    page_title="Assistant Touristique du Burkina Faso",
    page_icon="🌍",
    layout="centered"
)

# ---------- CSS ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    .main {
        background-color: #F6F6F6;
        color: #222;
        font-family: 'Poppins', sans-serif;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #00704A;
        margin-top: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }

    .stChatMessage {
        border-radius: 20px !important;
        padding: 12px !important;
        margin: 10px 0 !important;
    }

    [data-testid="stChatMessageUser"] {
        background-color: #DCF8C6 !important;
        border: 1px solid #A9DAB6;
        color: #222;
    }

    [data-testid="stChatMessageAssistant"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)


# ---------- TITRE ----------
st.markdown('<div class="title">🇧🇫 Assistant Raogo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Votre guide intelligent pour explorer le Burkina Faso 🌍</div>', unsafe_allow_html=True)



# ===================== II. INITIALISATION =====================

qdrant_url = st.secrets["QDRANT_URL"]
qdrant_key = st.secrets["QDRANT_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

@st.cache_resource
def load_agent():
    embedding_model, qdrant, llm = load_models(qdrant_url, qdrant_key, gemini_api_key)
    return TourismeAgent(qdrant, embedding_model, llm)

agent = load_agent()


# ===================== III. HISTORIQUE =====================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Bonjour ! Je suis votre guide virtuel du Burkina Faso. Que souhaitez-vous découvrir ?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===================== IV. INTERACTION =====================

user_input = st.chat_input("Posez une question sur le tourisme burkinabè...")

async def process_user_message(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Raogo réfléchit... 🤔"):
            response = await agent.answer(prompt)
        placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if user_input:
    asyncio.run(process_user_message(user_input))


# ===================== V. FOOTER =====================

st.markdown(
    f"""
    <div class="footer">
        🕓 {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}<br>
        Développé dans le cadre du <b>Hackathon 2025 – MTDPCE</b>
    </div>
    """,
    unsafe_allow_html=True
)
