from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login



# Charger les variables d'environnement
load_dotenv()

# Lecture des secrets
qdrant_url = os.getenv('QDRANT_URL')
qdrant_key = os.getenv('QDRANT_KEY')
H_TOKEN = os.getenv('H_TOKEN')

COLLECTION_NAME = "tourisme_burkina"
VECTOR_SIZE = 384

# Connexion huggingFace pour avoir accès au modèle
login(token=H_TOKEN)

# ===================== II. INITIALISATION DES MODÈLES =====================

def load_models():
    """Charger les modèles une seule fois."""
    # Embeddings
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Connexion Qdrant
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    
    # Charger le modèle Gemma
    model_name = "google/gemma-3-1b-it" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    return embedding_model, qdrant, tokenizer, model

embedding_model, qdrant, tokenizer, model = load_models()

# ===================== III. BASE VECTORIELLE =====================
def init_db():
    """Créer ou réinitialiser la collection Qdrant."""
    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation : {e}")
        return False

def add_documents(documents: List[dict]):
    """Ajouter des documents à la base vectorielle."""
    try:
        points = []
        for i, doc in enumerate(documents):
            vector = embedding_model.encode(doc["text"]).tolist()
            points.append(PointStruct(id=i, vector=vector, payload=doc))
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'ajout de documents : {e}")
        return False

# ===================== IV. RECHERCHE CONTEXTUELLE =====================
def get_embedding(text: str) -> List[float]:
    """Générer l'embedding d'un texte."""
    return embedding_model.encode(text).tolist()

def retrieve_relevant_chunks(user_query: str, top_k: int = 5) -> List[str]:
    """Récupérer les documents pertinents."""
    try:
        query_embedding = get_embedding(user_query)
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
        )
        if not results:
            return ["Aucun document pertinent trouvé."]
        return [p.payload["text"] for p in results]
    except Exception as e:
        st.error(f"Erreur recherche : {e}")
        return [f"Erreur : {e}"]

# ===================== V. GÉNÉRATION DE RÉPONSE =====================
def generate_answer(prompt: str, max_tokens: int = 512) -> str:
    """Générer une réponse avec Gemma."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Déplacer sur le bon device
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Génération
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Décoder la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la réponse générée (après le prompt)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response
    except Exception as e:
        return f"Erreur lors de la génération : {e}"

# ===================== VI. AGENT TOURISTIQUE =====================
@dataclass
class TourismeAgent:
    system_prompt: str = (
        "Tu es un guide touristique virtuel expert du Burkina Faso. "
        "Tu aides les utilisateurs à découvrir les sites naturels, culturels et historiques du pays. "
        "Réponds toujours en français clair et cite les sources quand elles sont disponibles."
    )

    def answer(self, question: str) -> str:
        """Générer une réponse à la question."""
        # Recherche contextuelle
        relevant_docs = retrieve_relevant_chunks(question, top_k=3)
        context = "\n\n".join(relevant_docs)
        
        # Construction du prompt
        prompt = f"""<start_of_turn>user
{self.system_prompt}

Contexte :
{context}

Question : {question}
<end_of_turn>
<start_of_turn>model
Réponse : """
        
        return generate_answer(prompt)
