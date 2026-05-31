import os
from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# 1. Load your environment variables cleanly
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
if not hf_token:
    raise RuntimeError("Hugging Face API token not found. Set HF_TOKEN in your .env file.")

print("--- Connecting to Hugging Face Shared Serverless Router ---")

# 2. Use a model that is natively supported by the Serverless Providers
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  #
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512,
    temperature=0.5
)

# 3. Create the chat wrapper
model = ChatHuggingFace(llm=llm)

# 4. Use the proper LangChain Message structure
messages = [
    HumanMessage(content="What is the capital of India?")
]

print("Sending request to serverless provider...")
# 5. Invoke the model
result = model.invoke(messages)

print("\nResponse:")
print(result.content)