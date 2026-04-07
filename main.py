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
    #This line here determines the role and behaviour of the model 
     ("system", """ONLY use the Pokemon data provided below to answer the question.
      If the answer is not found, say "I don't have enough information."
      Here is the relevant Pokemon data: {reviews}."""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chain = prompt | model

# 4. The "Thread" List (This stays alive while the script runs)
chat_history = []



while True:
    user_input = input('You: ')

    if user_input == "quit":
        break

    docs = retriever.invoke(user_input)
    # reviews_data = "\n".join([doc.page_content for doc in docs])
    temp_list = []
    
    for doc in docs:
        temp_list.append(doc.page_content)
        reviews = "\n".join(temp_list)

    response = chain.invoke({
        "reviews": reviews,      # The RAG data
        "chat_history": chat_history, # The Memory
        "input": user_input           # The Current Question
    })

    print(f"AI: {response}")

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

