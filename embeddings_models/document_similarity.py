import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the environment variables from your local .env file
load_dotenv()

# 💡 Swapped to Google's active embedding engine and specified 300 dimensions
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=300
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about Amit'

print("Processing vector transformations with Gemini...")
# Generate high-dimensional vector embeddings via Gemini
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate structural proximity using Cosine Similarity metrics
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Index map matching via enumeration and sorting logic
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("\n--- Semantic Search Results ---")
print("User Query:", query)
print("Matched Document:", documents[index])
print("Similarity Score is:", score)