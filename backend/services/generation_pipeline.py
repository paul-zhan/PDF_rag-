from datetime import datetime, timezone
import json
import os
import time
import uuid
from pathlib import Path

import requests
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

class GenerationPipeline():

    def __init__(self):
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.trace_log_path = Path(
            os.getenv("EVAL_TRACE_PATH", "/app/evaluation_logs/rag_traces.jsonl")
        )
        self.prompt_version = os.getenv("PROMPT_VERSION", "v1")
        self.llm = Ollama(
            model=self.ollama_model,
            base_url=self.ollama_base_url
        )
        self.client = QdrantClient(url=self.qdrant_url)
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def ensure_model_ready(self):
        response = requests.post(
            f"{self.ollama_base_url}/api/show",
            json={"name": self.ollama_model},
            timeout=10
        )
        response.raise_for_status()

    def ensure_qdrant_ready(self, attempts=10, delay=1):
        ready_url = f"{self.qdrant_url}/readyz"
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(ready_url, timeout=3)
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                if attempt == attempts:
                    raise RuntimeError(f"Qdrant unavailable: {exc}") from exc
                time.sleep(delay)

    def retrieval(self, query, collection_name="pdf_embeddings", top_k=5):
        try:
            query_vector = self.model.encode(
                query,
                normalize_embeddings=True
            ).tolist()

            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k
            )

            retrieved_chunks = []
            for point in results.points:
                retrieved_chunks.append({
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "source": point.payload.get("source", ""),
                    "page": point.payload.get("page", 0),
                })

            return retrieved_chunks

        except Exception as e:
            print(f"Qdrant retrieval error: {e}")
            return []

    def augmentation(self, query, retrieved_context):
        augmented_prompt=f"""

        Given the context below, answer the question.

        Question: {query} 

        Context : {retrieved_context}

        Remember to answer only based on the context provided and not from any other source. 

        If the question cannot be answered based on the provided context, say I don't know.

        """
        return augmented_prompt

    def generation(self):
        query = "Tell me about llm"
        retrieval_results = self.retrieval(query)
        print(retrieval_results)
        augmented_prompt = self.augmentation(query, retrieval_results)

        response = self.llm.invoke(augmented_prompt)

        print(f"Answer: {response}")

    def _build_trace(self, query, answer, retrieval_results):
        contexts = [chunk.get("text", "") for chunk in retrieval_results if chunk.get("text")]
        sources = [
            {
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "score": chunk.get("score"),
            }
            for chunk in retrieval_results
        ]
        return {
            "trace_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "ollama_model": self.ollama_model,
            "prompt_version": self.prompt_version,
        }

    def _write_trace(self, trace):
        self.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(trace, ensure_ascii=True) + "\n")

    def bot_answer(self, query: str):
        self.ensure_qdrant_ready()
        self.ensure_model_ready()

        retrieval_results = self.retrieval(query)
        augmented_prompt = self.augmentation(query, retrieval_results)
        response = self.llm.invoke(augmented_prompt)
        
        trace = self._build_trace(query, answer=response, retrieval_results=retrieval_results)
        self._write_trace(trace)
        return {
            "answer": response,
            "trace_id": trace["trace_id"],
        }

    def main(self):
        self.bot_answer("Tell me about llm")
