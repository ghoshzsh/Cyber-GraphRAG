from graphiti_core.embedder import OpenAIEmbedderConfig
from graphiti_core.embedder import OpenAIEmbedder
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key         = os.getenv("NVIDIA_API_KEY", "ollama"),
            embedding_model = os.getenv("NIM_EMBED_MODEL", "qwen3-embedding:4b"),
            base_url        = os.getenv("NIM_BASE_URL",    "http://localhost:11434/v1"),
            embedding_dim   = 1024,   # nv-embedqa-e5-v5 outputs 1024-dim vectors
        )
    )

res = embedder.embed_documents(["hello world", "test"])
print(len(res))