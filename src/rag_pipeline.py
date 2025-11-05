from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os
import asyncio
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch


# ===================== I. CHARGEMENT DES VARIABLES D’ENVIRONNEMENT =====================

load_dotenv()

COLLECTION_NAME = "tourisme_burkina"
VECTOR_SIZE = 384


# ===================== II. INITIALISATION DES MODÈLES =====================

def load_models(qdrant_url: str, qdrant_key: str, hf_token: str):
    """Charger les modèles et initialiser les connexions nécessaires."""
    # Connexion HuggingFace
    if hf_token:
        login(token=hf_token)

    # Embeddings
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Qdrant
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Modèle Gemma
    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return embedding_model, qdrant, tokenizer, model


# ===================== III. BASE VECTORIELLE =====================

def init_db(qdrant: QdrantClient):
    """Créer ou réinitialiser la collection Qdrant."""
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def add_documents(qdrant: QdrantClient, embedding_model: SentenceTransformer, documents: List[dict]):
    """Ajouter des documents à la base vectorielle."""
    points = []
    for i, doc in enumerate(documents):
        vector = embedding_model.encode(doc["text"]).tolist()
        points.append(PointStruct(id=i, vector=vector, payload=doc))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# ===================== IV. RECHERCHE CONTEXTUELLE =====================

def retrieve_relevant_chunks(
    qdrant: QdrantClient,
    embedding_model: SentenceTransformer,
    user_query: str,
    top_k: int = 5
) -> List[str]:
    """Récupérer les documents pertinents."""
    query_embedding = embedding_model.encode(user_query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )
    if not results:
        return ["Aucun document pertinent trouvé."]
    return [p.payload["text"] for p in results]


# ===================== V. GÉNÉRATION DE RÉPONSE =====================

def generate_answer(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    max_tokens: int = 512
) -> str:
    """Générer une réponse avec le modèle de langage."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.split(prompt)[-1].strip()

    return response


# ===================== VI. AGENT TOURISTIQUE =====================

@dataclass
class TourismeAgent:
    qdrant: QdrantClient
    embedding_model: SentenceTransformer
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    system_prompt: str = (
        "Tu es un guide touristique virtuel expert du Burkina Faso. "
        "Tu aides les utilisateurs à découvrir les sites naturels, culturels et historiques du pays. "
        "Réponds toujours en français clair et cite les sources quand elles sont disponibles."
    )

    async def answer(self, question: str) -> str:
        """Générer une réponse complète à une question utilisateur."""
        relevant_docs = retrieve_relevant_chunks(self.qdrant, self.embedding_model, question, top_k=3)
        context = "\n\n".join(relevant_docs)

        prompt = f"""<start_of_turn>user
{self.system_prompt}

Contexte :
{context}

Question : {question}
<end_of_turn>
<start_of_turn>model
Réponse : """

        return generate_answer(self.tokenizer, self.model, prompt)


# ===================== VII. TEST LOCAL =====================

if __name__ == "__main__":
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    hf_token = os.getenv("H_TOKEN")

    embedding_model, qdrant, tokenizer, model = load_models(qdrant_url, qdrant_key, hf_token)

    docs = [
        {"text": "La Cascade de Banfora est située à environ 12 km de la ville de Banfora, dans la région des Cascades."},
        {"text": "Le pic de Sindou offre un panorama spectaculaire sur les formations rocheuses du sud-ouest du Burkina Faso."},
        {"text": "Le lac de Tengréla est connu pour ses hippopotames et son cadre naturel paisible près de Banfora."},
    ]

    init_db(qdrant)
    add_documents(qdrant, embedding_model, docs)

    agent = TourismeAgent(qdrant, embedding_model, tokenizer, model)

    async def test_agent():
        response = await agent.answer("Quels sites touristiques peut-on visiter à Banfora ?")
        print("\n=== RÉPONSE ===\n")
        print(response)

    asyncio.run(test_agent())
