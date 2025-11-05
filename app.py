from __future__ import annotations
import streamlit as st
import asyncio
import datetime
from src.rag_pipeline import load_models, TourismeAgent


# ===================== I. CONFIGURATION DE L’APPLICATION =====================

st.set_page_config(
    page_title="Assistant Touristique du Burkina Faso",
    page_icon="🌍",
    layout="centered"
)

st.markdown('<div class="title">Assistant Raogo</div>', unsafe_allow_html=True)
st.caption("Propulsé par Gemini + Qdrant + SentenceTransformers — 100% open source 💡")


# ===================== II. INITIALISATION DE L’AGENT =====================

qdrant_url = st.secrets["QDRANT_URL"]
qdrant_key = st.secrets["QDRANT_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

@st.cache_resource
def load_agent():
    embedding_model, qdrant, llm = load_models(qdrant_url, qdrant_key, gemini_api_key)
    return TourismeAgent(qdrant, embedding_model, llm)

agent = load_agent()


# ===================== III. HISTORIQUE DE CONVERSATION =====================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Bonjour ! Je suis votre guide virtuel du Burkina Faso. Que souhaitez-vous découvrir ?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===================== IV. ENTRÉE UTILISATEUR =====================

user_input = st.chat_input("Posez une question sur le tourisme burkinabè...")

async def process_user_message(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Recherche en cours..."):
            response = await agent.answer(prompt)
        placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if user_input:
    asyncio.run(process_user_message(user_input))


# ===================== V. FOOTER =====================

st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; color: black; }
    .title { text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.caption(
    f"🕓 {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')} • "
    "Développé dans le cadre du Hackathon 2025 – MTDPCE"
)
