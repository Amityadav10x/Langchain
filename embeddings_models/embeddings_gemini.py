import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load the environment variables from your local .env file
load_dotenv()

# 2. 💡 FIX: Use the active gemini-embedding-001 model with output dimensionality
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=32
)

# 3. Define your document corpus list
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

print("Processing vector conversions with Gemini...")

# 4. Generate the embeddings matrix
result = embedding.embed_documents(documents)

# 5. Output the matrix representation string
print(str(result))