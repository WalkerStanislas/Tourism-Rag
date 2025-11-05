from __future__ import annotations
import streamlit as st
import asyncio
import datetime

from src.rag_pipeline_ import init_db, add_documents, TourismeAgent


# I. CONFIGURATION DE L'APPLICATION

st.set_page_config(
    page_title="Assistant Touristique du Burkina Faso",
    page_icon="🌍",
    layout="centered"
)


st.markdown('<div class="title">Assistant Raogo</div>', unsafe_allow_html=True)
st.caption("Propulsé par Gemma 3 (1B) + Qdrant + SentenceTransformers — 100% open source 💡")


# II. INITIALISATION DE L'AGENT ET DES DONNÉES

@st.cache_resource
def load_agent():
    docs = [
        {"text": "La Cascade de Banfora est située à environ 12 km de la ville de Banfora, dans la région des Cascades."},
        {"text": "Le pic de Sindou offre un panorama spectaculaire sur les formations rocheuses du sud-ouest du Burkina Faso."},
        {"text": "Le lac de Tengréla est connu pour ses hippopotames et son cadre naturel paisible près de Banfora."},
        {"text": "La mosquée de Bobo-Dioulasso, construite au XIXe siècle, est un symbole de l’architecture soudanaise."},
        {"text": "Le parc national d’Arly abrite une faune riche incluant éléphants, lions et antilopes, situé à l’est du Burkina Faso."},
    ]
    #init_db()
    #add_documents(docs)
    return TourismeAgent()

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
