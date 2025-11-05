from __future__ import annotations
import streamlit as st
import asyncio
import datetime
from src.rag_pipeline import load_models, TourismeAgent


# I. CONFIGURATION DE L'APPLICATION

st.set_page_config(
    page_title="Assistant Touristique du Burkina Faso",
    page_icon="🌍",
    layout="centered"
)



st.markdown('<div class="title">Assistant Raogo</div>', unsafe_allow_html=True)
st.caption("Propulsé par Gemma 3 (1B) + Qdrant + SentenceTransformers — 100% open source 💡")


# II. INITIALISATION DE L'AGENT ET DES DONNÉES


# Charger les secrets depuis Streamlit
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_key = st.secrets["QDRANT_KEY"]
H_TOKEN = st.secrets["H_TOKEN"]

@st.cache_resource
def load_agent():
    embedding_model, qdrant, tokenizer, model = load_models(qdrant_url, qdrant_key, H_TOKEN)
    return TourismeAgent(qdrant, embedding_model, tokenizer, model)

agent = load_agent()


# III. HISTORIQUE DE CONVERSATION

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Bonjour ! Je suis votre guide virtuel du Burkina Faso. Que souhaitez-vous découvrir aujourd’hui ?"}
    ]

# Afficher les messages précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# IV. ENTRÉE UTILISATEUR

user_input = st.chat_input("Posez une question sur le tourisme burkinabè...")

async def process_user_message(prompt: str):
    """Gérer une nouvelle question et générer la réponse avec streaming."""
    # Affiche le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération de la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        partial_text = ""
        with st.spinner("Recherche en cours..."):
            response = await agent.answer(prompt)
        partial_text += response
        message_placeholder.markdown(partial_text)

    # Ajoute le message complet à l'historique
    st.session_state.messages.append({"role": "assistant", "content": partial_text})



# V. LOGIQUE PRINCIPALE

if user_input:
    asyncio.run(process_user_message(user_input))


# VI. FOOTER

# CSS personnalisé pour l'interface
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; color: black; }
    .stTextArea { border-radius: 10px; width: 100%; }
    .stButton>button { border-radius: 8px; background-color: #4285F4; color: white; font-size: 16px; width: 200px; }
    .stSelectbox [disabled] {
        background-color: #e9ecef;
        color: #6c757d;
        pointer-events: none;
        cursor: not-allowed;
    }
    .title { text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 20px; }
    .center-btn { display: flex; justify-content: center; }
    </style>
    """,
    unsafe_allow_html=True
)

st.caption(
    f"🕓 {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')} • "
    "Développé dans le cadre du Hackathon 2025 – MTDPCE"
)
