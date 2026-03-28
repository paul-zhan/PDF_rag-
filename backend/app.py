from fastapi import FastAPI
from fastapi import HTTPException
from services.generation_pipeline import GenerationPipeline

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

app = FastAPI()
pipeline = GenerationPipeline()

@app.get("/health")
def health_check():
    return {"status": "200", "message": "OK"}

@app.post("/chatbot_answer")
def bot_answer (request: QueryRequest):
    try:
        response = pipeline.bot_answer(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
