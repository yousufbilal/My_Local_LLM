from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import ollama
import json
from vector import retriever
import os

model = OllamaLLM(model="llama3.2")

# Change this line: adding RAG to here with REVIEWS
prompt = ChatPromptTemplate.from_messages([  
    ("system", """You are a friendly and helpful AI assistant.
    
    1. If the user asks about Pokemon, use this data to answer accurately: {reviews}
    2. If the user is just chatting (saying hello, asking how you are), respond naturally like a friend.
    3. If the user asks a Pokemon question but the data above is empty, tell them you don't have the specific stats for that Pokemon yet."""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chain = prompt | model

# 4. The "Thread" List (This stays alive while the script runs)
chat_history = []

print("AI: Hello! How can I help you today?")

while True:
    user_input = input('You: ')

    if user_input == "quit":
        break

    docs = retriever.invoke(user_input)
    # reviews_data = "\n".join([doc.page_content for doc in docs])
    temp_list = []
    
    for doc in docs:
        temp_list.append(doc.page_content)
    
    
    if temp_list:
        reviews_data = "\n".join(temp_list)
    else:
        reviews_data = "No specific Pokemon data found for this message."

    response = chain.invoke({
        "reviews": reviews_data,      # The RAG data
        "chat_history": chat_history, # The Memory
        "input": user_input           # The Current Question
    })

    print(f"AI: {response}")

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

