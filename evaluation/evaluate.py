import json
import time
import sys
import os
import asyncio

# Ajouter le dossier parent de src au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import load_models, TourismeAgent

# ===================== I. CHARGEMENT DES SECRETS ET MODÈLES =====================
gemini_api_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_KEY")

embedding_model, qdrant, llm = load_models(qdrant_url, qdrant_key, gemini_api_key)
agent = TourismeAgent(qdrant, embedding_model, llm)

# ===================== II. FONCTION D'ÉVALUATION =====================
def evaluate():
    # Charger les questions depuis un fichier JSON
    questions_file = os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(questions_file, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    results = []
    total_time = 0

    for item in test_questions:
        start = time.time()
        # Appel asynchrone de l'agent
        response = asyncio.run(agent.answer(item["question"]))
        duration = time.time() - start
        total_time += duration

        correct = item["expected_answer"].lower() in response.lower()
        results.append({
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "response": response,
            "correct": correct,
            "time": duration
        })

        print(f"Q: {item['question']}")
        print(f"A: {response}")
        print(f"Correct: {correct}\n")

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_time = total_time / len(results)

    print(f"=== ÉVALUATION GLOBALE ===")
    print(f"Précision: {accuracy*100:.2f}%")
    print(f"Temps moyen par question: {avg_time:.2f} sec")

    # Sauvegarder les résultats dans evaluation/results.json
    os.makedirs(os.path.join(os.path.dirname(__file__)), exist_ok=True)
    results_file = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "avg_time": avg_time, "details": results}, f, indent=2, ensure_ascii=False)

# ===================== III. EXECUTION =====================
if __name__ == "__main__":
    evaluate()
