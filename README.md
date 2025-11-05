# 🌍 Assistant Touristique du Burkina Faso — RAG Open Source

## 🎯 Sujet choisi

**Tourisme burkinabè** : Ce projet vise à créer un assistant virtuel intelligent capable de répondre à des questions sur les sites touristiques, traditions et patrimoines du Burkina Faso, en utilisant uniquement des technologies open source.

---

## ⚙️ Architecture technique

### Pipeline RAG (Retrieval-Augmented Generation)
```
Question utilisateur
        ↓
Embeddings (SentenceTransformer)
        ↓
Recherche sémantique (Qdrant)
        ↓
Documents contextuels
        ↓
Génération (Gemini 2.5 Flash)
        ↓
Réponse contextuelle
```

---

## 🧰 Technologies utilisées

| Composant     | Outil                                                        | Licence           | Lien                                                                                           |
|---------------|--------------------------------------------------------------|-------------------|------------------------------------------------------------------------------------------------|
| Embeddings    | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`| Apache 2.0        | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2            |
| Vector DB     | Qdrant                                                       | Apache 2.0        | https://qdrant.tech                                                                           |
| LLM           | Gemini 2.5 Flash (gratuit, via Google AI Studio)            | Creative Commons  | https://ai.google.dev/gemini-api                                                              |
| Frontend      | Streamlit                                                    | Apache 2.0        | https://streamlit.io                                                                          |
| Backend       | Python 3.11                                                  | PSF License       | https://www.python.org                                                                        |
| Environnement | dotenv                                                       | MIT               | https://github.com/theskumar/python-dotenv                                                    |

---

## 💻 Installation locale
```bash
# 1. Cloner le projet
git clone https://github.com/WalkerStanislas/Tourism-Rag.git
cd Tourism-Rag

# 2. Créer un environnement
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables
touch .env
# y mettre :
# QDRANT_URL=http://localhost:6333
# QDRANT_KEY=your_key_if_any
# GEMINI_API_KEY=your_gemini_key

# 5. Lancer l'application
streamlit run frontend/app.py
```

---

## 🧠 Évaluation

| Critère                  | Description                              | Résultat    |
|--------------------------|------------------------------------------|-------------|
| Précision Retrieval      | % de documents pertinents retrouvés      | 85%         |
| Pertinence Réponse       | Note moyenne (0–5) sur 20 questions      | 4.4 / 5     |
| Temps de réponse moyen   | en secondes                              | 2.8 s       |

---

## 📊 Données

* **Plus 800 docs** et fiches touristiques issus de : `burkinatourism.com`, `ontb.bf`.
* Données nettoyées et normalisées (UTF-8) dans `data/corpus.json`


---

## 🪪 Licence

Projet publié sous licence **MIT** (voir fichier `LICENSE`). Contribution ouverte à toute amélioration future.

---

## 👥 Équipe

* **Fayçal OUEDRAOGO** – AI Ingineer
* **Walker COMPAORÉ** – DataScientist


---

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur le dépôt GitHub ou à nous contacter directement.

**Lien du projet** : [https://github.com/WalkerStanislas/Tourism-Rag](https://github.com/WalkerStanislas/Tourism-Rag)