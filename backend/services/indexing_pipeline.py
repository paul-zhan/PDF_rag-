from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from tqdm import tqdm
from pathlib import Path
import os
import time
import torch
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter#
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class IndexingPipeline():

    def __init__(self):    
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.client = QdrantClient(url=self.qdrant_url)

    def wait_for_qdrant(self, attempts=30, delay=2):
        ready_url = f"{self.qdrant_url}/readyz"
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(ready_url, timeout=3)
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                if attempt == attempts:
                    raise RuntimeError(f"Qdrant did not become ready: {exc}") from exc
                print(f"Waiting for Qdrant ({attempt}/{attempts}): {exc}")
                time.sleep(delay)

    def data_loading(self):

        folder = Path("documents")
        pdf_paths = list(folder.rglob("*.pdf"))
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        loaded_documents = []

        for pdf_path in tqdm(pdf_paths, desc="Loading PDFs"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                loaded_documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")

        print(f"Loaded {len(loaded_documents)} documents from the directory.")

        return loaded_documents



    def data_chunking(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100, add_start_index=True
    )
        all_splits = text_splitter.split_documents(docs)
        all_splits = [chunk for chunk in all_splits if chunk.page_content.strip()]
        print(f"Total number of chunks created: {len(all_splits)}")
        if not all_splits:
            raise ValueError("Chunking produced no chunks.")
        return all_splits

    def embedding_generation(self, data_chunks):
        
        texts = [chunk.page_content for chunk in data_chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,  # Add batch processing (adjust based on GPU memory)
            show_progress_bar=True  # See progress
        )

        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings



    def vector_store_creation(self, pdf_embeddings, data_chunks):
        collection_name = "pdf_embeddings"
        if collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        # Prepare points for insertion
        points = []
        for i, (embedding, chunk) in enumerate(zip(pdf_embeddings, data_chunks)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),  # Convert numpy array to list
                payload={
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0)
                }
            )
            points.append(point)
        
        # Add points to the collection
        self.client.upsert(
            collection_name = collection_name,
            points=points
        )
        
        print(f"Added {len(points)} embeddings to Qdrant collection")
        return self.client



    def main(self):

        print("torch cuda:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        self.wait_for_qdrant()
        loaded_documents = self.data_loading()
        data_chunks = self.data_chunking(loaded_documents)
        embedding = self.embedding_generation(data_chunks)
        self.vector_store_creation(embedding, data_chunks)


if __name__ == "__main__":
    pipeline = IndexingPipeline()
    pipeline.main()
