from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ollama
import json
from vector import retriever
import os

model = OllamaLLM(model="llama3.2")

#it is important when using RAG to restrict the prompt otherwise the model will halucinate 
template = """
ONLY use the Pokemon data provided below to answer the question.
If the answer is not found, say "I don't have enough information."

Here is the relevant Pokemon data: {reviews}

Question: {questions}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

result = chain.invoke({"reviews":[],"questions": ""})

print(result)

while True:

    user_input = input('You: ')

    if user_input == "quit":
        break

    chain = prompt | model

    reviews = retriever.invoke(user_input)
    result = chain.invoke({"reviews":reviews,"questions": user_input})

    print(result)

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