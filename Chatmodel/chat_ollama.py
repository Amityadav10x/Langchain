from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the chat model
chat = ChatOllama(model="phi3", temperature=0.3)

# Structure the messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of India?")
]

print("Thinking...")
response = chat.invoke(messages)

print("\n--- Raw Chat Response With Metadata ---")
# 💡 FIX: Printing the object directly shows content AND response_metadata!
print(response)