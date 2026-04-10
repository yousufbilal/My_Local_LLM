from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from app.vector import vector_store, movies_vector_store # Using the stores directly

model = OllamaLLM(model="llama3.2", temperature=0)

prompt = ChatPromptTemplate.from_messages([  
    ("system", """You are a helpful assistant. 
    Use the provided DATA to answer the user. 
    If the DATA is empty, it means the search found no relevant records in the Pokemon or Movie databases; in that case, answer naturally or say you don't know.
    
    DATA:
    {reviews}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

def ensemble_retrieve(query):
    """
    Explicitly retrieves and organizes data based on mathematical similarity.
    """
    # 1. Fetch raw results with scores (0.0 to 1.0)
    # k=5 gives us enough variety to handle typos or ambiguous names
    pk_results = vector_store.similarity_search_with_relevance_scores(query, k=5)
    mv_results = movies_vector_store.similarity_search_with_relevance_scores(query, k=5)
    
    formatted_context = []

    # 2. Process Pokemon matches with full logic
    for doc, score in pk_results:
        # Instead of a silent threshold, we label the confidence for the LLM
        if score > 0.4:  # Only ignore complete noise
            entry = (
                f"[SOURCE: POKEMON DATABASE]\n"
                f"[CONFIDENCE: {score:.2f} %]\n"
                f"CONTENT: {doc.page_content}\n"
                "---"
            )
            formatted_context.append(entry)

    # 3. Process Movie matches with full logic
    for doc, score in mv_results:
        if score > 0.4:
            entry = (
                f"[SOURCE: MOVIE DATABASE]\n"
                f"[CONFIDENCE: {score:.2f} %]\n"
                f"CONTENT: {doc.page_content}\n"
                "---"
            )
            formatted_context.append(entry)

    return "\n".join(formatted_context)

# --- CHAT LOOP ---
chat_history = []
chain = prompt | model

while True:
    user_input = input('-------USER-----: ')
    if user_input.lower() in ["quit", "exit"]: break

    # The AI doesn't guess anymore. It searches.
    # If "Mirrorborn" is in the movie DB, it will have a score near 0.9.
    # The Pokemon DB will return scores near 0.1, which are automatically deleted.
    
    reviews_data = ensemble_retrieve(user_input)

    response = chain.invoke({
        "reviews": reviews_data,
        "chat_history": chat_history,
        "input": user_input
    })

    print(f"******AI RESPOSE*********: : {response}")
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))