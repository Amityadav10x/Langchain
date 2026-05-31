from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

print("--- 1. RUNNING AS A BASIC LLM ---")
# OllamaLLM expects a single string and returns a single string
llm = OllamaLLM(model="phi3")
llm_response = llm.invoke("What is the capital of France?")
print(f"LLM Output: {llm_response}\n")