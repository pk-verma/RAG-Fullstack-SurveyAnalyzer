# backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_system import get_insight
from app.data_processing import load_data

app = FastAPI()

# Load and process datasets
dataset_1 = load_data("path_to_sustainability.xlsx")
dataset_2 = load_data("path_to_christmas.xlsx")

# Add datasets to RAG system
rag_system.add_data(dataset_1)
rag_system.add_data(dataset_2)

class QueryRequest(BaseModel):
    query: str
    dataset_id: int

@app.post("/query")
async def query_data(request: QueryRequest):
    try:
        result = rag_system.retrieve_and_generate(request.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
