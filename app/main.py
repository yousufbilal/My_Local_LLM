from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from app.vector import pokemon_retriever, movies_retriever

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

chain = prompt | model

# 4. The "Thread" List (This stays alive while the script runs
chat_history = []

print("AI: Hello! How can I help you today?")

while True:
    temp_list = []

    user_input = input('You: ')

    if user_input == "quit":
        break

    docs = pokemon_retriever.invoke(user_input)

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





#**********************LangChain Query Routing  The Scalable "Tool-Based" Fix **********************

# from langchain_core.tools import create_retriever_tool

# # 1. Turn your retrievers into "Tools"
# # This is where it scales: just keep adding tools to this list
# pokemon_tool = create_retriever_tool(
#     pokemon_retriever,
#     "pokemon_search",
#     "Search for info about Pokemon stats, types, and lore."
# )

# movie_tool = create_retriever_tool(
#     movies_retriever,
#     "movie_search",
#     "Search for info about movie plots, actors, and reviews."
# )

# tools = [pokemon_tool, movie_tool]

# # 2. Bind the tools to your model
# # This gives Llama 3.2 the ability to "pick" a tool
# model_with_tools = model.bind_tools(tools)

# # 3. Your Updated Loop
# while True:
#     user_input = input('You: ')
#     if user_input == "quit": break

#     # Ask the model which tool to use
#     ai_msg = model_with_tools.invoke(user_input)
    
#     reviews_data = ""
    
#     # Check if the AI decided to use a tool
#     if ai_msg.tool_calls:
#         for tool_call in ai_msg.tool_calls:
#             # Match the tool call to the actual retriever
#             selected_tool = next(t for t in tools if t.name == tool_call["name"])
#             tool_output = selected_tool.invoke(tool_call["args"])
#             reviews_data += str(tool_output) + "\n"
#     else:
#         reviews_data = "No specific data found."

#     # YOUR EXACT CHAIN INVOKE UNCHANGED
#     response = chain.invoke({
#         "reviews": reviews_data,      
#         "chat_history": chat_history, 
#         "input": user_input           
#     })

#     print(f"AI: {response}")
#     chat_history.append(HumanMessage(content=user_input))
#     chat_history.append(AIMessage(content=response))
