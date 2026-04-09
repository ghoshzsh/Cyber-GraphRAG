from langchain_ollama import OllamaEmbeddings

embedder = OllamaEmbeddings(
    model="qwen3-embedding:4b"  # or any embedding model you pulled
)

res = embedder.embed_documents(["hello world", "test"])
print(len(res))