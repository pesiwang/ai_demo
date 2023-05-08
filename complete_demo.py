import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def complete_create(prompt_str):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_str,
        temperature=0.6,
    )
    return response


def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )

if __name__ == '__main__':
    prompt = generate_prompt('Horse')
    response = complete_create(prompt)
    print(response)
    
