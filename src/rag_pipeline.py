from __future__ import annotations
from dataclasses import dataclass
from typing import List
import asyncio
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_google_genai import ChatGoogleGenerativeAI


# ===================== I. CONFIGURATION =====================

load_dotenv()

COLLECTION_NAME = "tourisme_burkina"
VECTOR_SIZE = 384


def load_models(qdrant_url: str, qdrant_key: str, gemini_api_key: str):
    """Initialise les composants du pipeline RAG."""
    # Embedding model
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Qdrant
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.7,
    )

    return embedding_model, qdrant, llm


# ===================== II. BASE VECTORIELLE =====================

def init_db(qdrant: QdrantClient):
    """Créer ou réinitialiser la collection Qdrant."""
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def add_documents(qdrant: QdrantClient, embedding_model: SentenceTransformer, documents: List[dict]):
    """Ajouter des documents (textes touristiques) à la base vectorielle."""
    points = []
    for i, doc in enumerate(documents):
        vector = embedding_model.encode(doc["text"]).tolist()
        points.append(PointStruct(id=i, vector=vector, payload=doc))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# ===================== III. RECHERCHE CONTEXTUELLE =====================

async def retrieve_relevant_chunks(
    qdrant: QdrantClient,
    embedding_model: SentenceTransformer,
    user_query: str,
    top_k: int = 5
) -> List[str]:
    """Recherche les passages les plus pertinents."""
    try:
        query_embedding = embedding_model.encode(user_query).tolist()
        results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=top_k)
        if not results:
            return ["Aucun document pertinent trouvé."]
        return [r.payload["text"] for r in results]
    except Exception as e:
        print(f"Erreur recherche : {e}")
        return [f"Erreur : {e}"]


# ===================== IV. AGENT TOURISTIQUE =====================

@dataclass
class TourismeAgent:
    qdrant: QdrantClient
    embedding_model: SentenceTransformer
    llm: ChatGoogleGenerativeAI

    system_prompt: str = (
        "Tu es un guide touristique virtuel expert du Burkina Faso. "
        "Tu aides les utilisateurs à découvrir les sites naturels, culturels et historiques du pays. "
        "Réponds toujours en français clair, structuré et cite les sources quand elles sont disponibles."
    )

    async def answer(self, question: str) -> str:
        """Générer une réponse complète et contextuelle."""
        relevant_docs = await retrieve_relevant_chunks(self.qdrant, self.embedding_model, question, top_k=3)
        context = "\n\n".join(relevant_docs)

        prompt = f"""
{self.system_prompt}

Contexte :
{context}

Question :
{question}

Réponse :
"""

        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            return response.content
        except Exception as e:
            return f"Erreur génération de réponse : {e}"


# ===================== V. TEST LOCAL =====================

if __name__ == "__main__":
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    embedding_model, qdrant, llm = load_models(qdrant_url, qdrant_key, gemini_api_key)

    docs = [
        {"text": "La Cascade de Banfora est située à environ 12 km de la ville de Banfora, dans la région des Cascades."},
        {"text": "Le pic de Sindou offre un panorama spectaculaire sur les formations rocheuses du sud-ouest du Burkina Faso."},
        {"text": "Le lac de Tengréla est connu pour ses hippopotames et son cadre naturel paisible près de Banfora."},
    ]

    init_db(qdrant)
    add_documents(qdrant, embedding_model, docs)

    agent = TourismeAgent(qdrant, embedding_model, llm)

    async def test_agent():
        question = "Quels sites touristiques peut-on visiter à Banfora ?"
        response = await agent.answer(question)
        print("\n=== RÉPONSE ===\n")
        print(response)

    asyncio.run(test_agent())
