from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import ollama
import json
from vector import retriever
import os

model = OllamaLLM(model="llama3.2")

#it is important when using RAG to restrict the prompt otherwise the model will halucinate 
# template = """
# ONLY use the Pokemon data provided below to answer the question.
# If the answer is not found, say "I don't have enough information."

# Here is the relevant Pokemon data: {reviews}

# Question: {questions}
# """

# prompt = ChatPromptTemplate.format_messages([
#     ("system", "You are a helpful assistant."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
# ])

# Change this line:
prompt = ChatPromptTemplate.from_messages([  
    #This line here determines the role and behaviour of the model 
    # ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

# 4. The "Thread" List (This stays alive while the script runs)
chat_history = []

# result = chain.invoke({"reviews":[],"questions": ""})
# print(result)

while True:

    user_input = input('You: ')

    if user_input == "quit":
        break

    # chain = prompt | model

    # reviews = retriever.invoke(user_input)
    # result = chain.invoke({"reviews":reviews,"questions": user_input})
    # print(result)

    response = chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    print(f"AI: {response}")

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))



#implement LangChain's built-in conversational RAG stack so the agent can have context of the conversation


# DB_FILE = "chat_history.json"
# # Your specialized Cybersecurity System Prompt
# SYSTEM_PROMPT = {
#     'role': 'system',
#     'content': (
#         "You are a Senior Cybersecurity Audit Assistant. Your expertise is in "
#         "IAM (Identity and Access Management) misuse and Multi-Agent Dispute resolution. "
#         "Be technical, concise, and focus on identifying 'Trust and Safety Gaps'."
#     )
# }


# def load_hisotry():
#     if os.path.exists(DB_FILE):
#         with open(DB_FILE,'r') as f:
#             return json.load(f)
#     # save_history([SYSTEM_PROMPT])
#     return [SYSTEM_PROMPT]

# def save_history(messages):
#     with open(DB_FILE,'w') as f:
#         json.dump(messages,f,indent=4)

# # messages = [
# #     {
# #         'role': 'system',
# #         'content': """You are a pirate. 
# #         Be concise and and funny and sarcastic."""
# #     }
# # ]

# def main():

#     messages=load_hisotry()

#     while True:
#         user_input = input('You: ')

#         if user_input.lower() in ['quit','exit','bye']: 
#             print ("sessiion saved goodbye")
#             break
    
#         if user_input.lower() in ['clear']: 
#             messages=[SYSTEM_PROMPT]
#             save_history(messages)
#             print("History cleared.\n")
#             continue

#         messages.append({'role': 'user', 'content': user_input})

#         try:
#             response = ollama.chat(model='llama3.2', messages=messages)

#             model_response = response['message']['content']
    
#             print(f"\nLlama: {model_response}\n")
#             messages.append({'role': 'assistant', 'content': model_response})
#             save_history(messages)

#         except Exception as e:
#             print(f"Error connecting to Ollama: {e}")    

# if __name__ == "__main__":
#     main()