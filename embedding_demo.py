import openai


def get_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


embedding = get_embedding("Your text goes here", model="text-embedding-ada-002")
embedding2 = get_embedding("Your text goes here, long text is same len of the vector?The curling competitions of the 2022 Winter Olympics were held at the Beijing National Aquatics Centre, one of the Olympic Green venues. Curling competitions were scheduled for every day of the games, from February 2 to February 20.[1] This was the eighth time that curling was part of the Olympic program", model="text-embedding-ada-002")
#print(f"embdding: {embedding}")
print(f"len: {len(embedding)}")
print(f"len2: {len(embedding2)}")

