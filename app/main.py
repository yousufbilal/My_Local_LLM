from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from app.vector import pokemon_retriever, movies_retriever

# Router-specific libraries
from typing import Literal
from langchain_core.output_parsers import StrOutputParser


model = OllamaLLM(model="llama3.2")

# Change this line: adding RAG to here with REVIEWS
prompt = ChatPromptTemplate.from_messages([  
    ("system", """You are a friendly and helpful AI assistant.
    
    1. If the user asks about Pokemon, Movies use this data to answer accurately: {reviews}
    2. If the user is just chatting (saying hello, asking how you are), respond naturally like a friend.
    3. If the user asks a Pokemon or a Movie question but the data above is empty, tell them you don't have the specific information for that Pokemon or movie yet."""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert intent classifier for a RAG system.
    Your job is to route the user's query to the correct database.

    CATEGORIES:
    - 'pokemon': Use this for Pokedex entries, stats, evolutions, or specific Pokemon names.
    - 'movies': Use this for film titles, directors, actors, or release dates found in a movie database.
    - 'general': Use this for greetings, philosophy, or topics clearly outside of Pokemon/Movies.

    RULES:
    1. Respond with information from the list above.
    2. If the query is ambiguous, prioritize the most likely intent.
    3. If the query matches NO categories, respond with 'general'.

    EXAMPLES:
    User: "Who directed Inception?" -> movies
    User: "What level does Charmander evolve?" -> pokemon
    User: "How is the weather?" -> general
    User: "Tell me about Avatar 3" -> movies
    """),
    ("human", "{input}")
])

router_chain = router_prompt | model | StrOutputParser()

RETRIEVER_REGISTRY = {
    "pokemon": pokemon_retriever,
    "movies": movies_retriever,
    # "books": books_retriever,
    # "coding": coding_retriever,
    # "finance": finance_retriever
}


def route_and_retrieve(user_input):
    """Classify the query and fetch docs from the right retriever."""

    decision = router_chain.invoke({"input": user_input}).strip().lower()
    print(f"[Router → '{decision}']")  # helpful for debugging

    selected_retriever = RETRIEVER_REGISTRY.get(decision)

    if selected_retriever:
        docs = selected_retriever.invoke(user_input)
        result = "\n".join(doc.page_content for doc in docs)

        return result
    
    return ""

chain = prompt | model

# 4. The "Thread" List (This stays alive while the script runs
chat_history = []

print("AI: Hello! How can I help you today?")

while True:
    temp_list = []

    user_input = input('You: ')

    if user_input == "quit":
        break

    # docs = pokemon_retriever.invoke(user_input)

    # for doc in docs:
    #     temp_list.append(doc.page_content)
    
    # if temp_list:
    #     reviews_data = "\n".join(temp_list)
    # else:
    #     reviews_data = "No specific Pokemon data found for this message."

# --- THE SCALABLE PART ---
    # We call our function which handles ALL retrievers automatically
    reviews_data = route_and_retrieve(user_input)

    response = chain.invoke({
        "reviews": reviews_data,      # The RAG data
        "chat_history": chat_history, # The Memory
        "input": user_input           # The Current Question
    })

    print(f"AI: {response}")

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
