import os
import pandas as pd
import pickle
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json

with open('data.json', 'r') as f:
    config = json.load(f)

FAQ_PATH = config["faq_path"]
FUND_PATH = config["fund_path"]
EMBEDDINGS_PATH = config["embeddings_path"]

# -----------------------------
# Load and Prepare Data
# -----------------------------

def load_faqs(path='data/faqs.csv'):
    df = pd.read_csv(path)
    return df 

def load_fund_data(path='data/funds.csv'):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(' (%)', '').str.replace(' ', '_')
    df["text"] = df.apply(lambda row:
        f"{row['fund_name']} has 3yr CAGR of {row['cagr_3yr']}%, volatility {row['volatility']}%, and Sharpe ratio {row['sharpe_ratio']}. "
        f"It belongs to the {row['category']} category.", axis=1)
    return df

# -----------------------------
# Embedding
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, show_progress_bar=True)

def build_faiss_index(embeddings, texts, save_path="faiss_index.pkl"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    with open(save_path, 'wb') as f:
        pickle.dump((index, texts), f)

def load_faiss_index(path="faiss_index.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)

# -----------------------------
# TF-IDF Retriever
# -----------------------------
class TfidfRetriever:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.docs = corpus
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_k=5):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(self.docs[i], scores[i]) for i in top_indices]

# -----------------------------
# FAISS Retriever
# -----------------------------
class FaissRetriever:
    def __init__(self, index, texts):
        self.index = index
        self.texts = texts

    def search(self, query, top_k=5):
        query_vec = model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return [(self.texts[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# -----------------------------
# Query Classification
# -----------------------------
def is_textual_query(query):
    return any(q in query.lower() for q in ["what is", "explain", "how does", "define", "why"])

def format_response(query, results):
    if is_textual_query(query):
        return {
            "answer": results[0][0],
            "sources": [r[0] for r in results]
        }
    else:
        # For performance-based queries, extract top funds
        top_funds = []
        for r in results:
            if "has 3yr CAGR" in r[0]:
                top_funds.append(r[0])
        return {
            "answer": "Top matching funds based on your query:",
            "top_funds": top_funds[:3],
            "sources": top_funds[:5]
        }

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost"] for stricter dev control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    mode: str = Query("semantic", enum=["semantic", "lexical"])

# Load Data
faq_df = load_faqs(FAQ_PATH)
fund_df = load_fund_data(FUND_PATH)

# For retrieval: combine questions + fund performance descriptions
faq_corpus = list(faq_df['question'] + ": " + faq_df['answer'])
fund_corpus = list(fund_df['text'])
corpus = faq_corpus + fund_corpus

# Embeddings and Indexing
EMBEDDINGS_PATH = "faiss_index.pkl"
if not os.path.exists(EMBEDDINGS_PATH):
    embeddings = embed_texts(corpus)
    build_faiss_index(embeddings, corpus, EMBEDDINGS_PATH)

faiss_index, faiss_texts = load_faiss_index(EMBEDDINGS_PATH)
tfidf = TfidfRetriever(corpus)
semantic = FaissRetriever(faiss_index, faiss_texts)

@app.post("/query")
def handle_query(request: QueryRequest):
    if request.mode == "lexical":
        results = tfidf.search(request.query)
    else:
        results = semantic.search(request.query)
    response = format_response(request.query, results)
    return response

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
