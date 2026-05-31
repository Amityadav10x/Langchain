from langchain_ollama import OllamaEmbeddings

print("--- Initializing Local Ollama Embedding Engine ---")

# 1. Initialize the embedding wrapper with our downloaded local model
# Unlike OpenAI or Gemini, this requires no API token!
embedding = OllamaEmbeddings(model="nomic-embed-text")

# 2. Define the unstructured document array list
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

print("Processing local vector matrix conversions via Ollama...")

# 3. Generate the high-dimensional embeddings matrix array list
result = embedding.embed_documents(documents)

# 4. Output the matrix representation string
print("\nSuccess! Vector output matrix generated:")
print(str(result))

# Quick verification check to see the generated dimensions
print(f"\nNumber of embedded documents: {len(result)}")
print(f"Dimensions per vector (Nomic text layout): {len(result[0])}")