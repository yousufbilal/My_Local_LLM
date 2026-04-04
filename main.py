import ollama

messages=[]

while True:
    user_input = input('You:')

    if user_input.lower() in ['quit','exit','bye']:
        print ("goodbye")
        break
    messages.append({'role': 'user', 'content': user_input})

    response = ollama.chat(model='llama3.2', messages=messages)

    model_response = response['message']['content']
    
    print(model_response)