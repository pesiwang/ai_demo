import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def chat_complete_create(prompt_str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_str}],
        temperature=0.6,
    )
    return response

if __name__ == '__main__':
    prompt = "who are you?"
    response = chat_complete_create(prompt)
    print(response)
    
