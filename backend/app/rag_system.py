# backend/app/rag_system.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict

# Load embedding and generation models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text-generation", model="gpt2", max_length=150)

class RAGSystem:
    def __init__(self):
        # Initialize FAISS index
        self.index = None
        self.data = []

    def add_data(self, documents: List[Dict[str, str]]):
        # Convert documents to embeddings
        texts = [doc["content"] for doc in documents]
        embeddings = embedder.encode(texts, convert_to_tensor=True).cpu().numpy()
        
        # Create a FAISS index if it doesn't exist
        if self.index is None:
            d = embeddings.shape[1]  # Dimensionality of embeddings
            self.index = faiss.IndexFlatL2(d)
        
        # Add embeddings and store documents
        self.index.add(embeddings)
        self.data.extend(documents)

    def retrieve_and_generate(self, query: str):
        # Generate embedding for the query
        query_embedding = embedder.encode([query]).astype(np.float32)

        # Retrieve closest documents from FAISS index
        _, indices = self.index.search(query_embedding, k=1)
        retrieved_text = self.data[indices[0][0]]["content"]
        
        # Generate a response based on retrieved text and query
        prompt = f"Context: {retrieved_text}\nQuestion: {query}"
        response = generator(prompt)[0]["generated_text"]
        return response

# Initialize RAG system instance
rag_system = RAGSystem()
