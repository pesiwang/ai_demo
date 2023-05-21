import openai

from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval
import weaviate

# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


#article_df = pd.read_csv('./data/vector_database_wikipedia_articles_embedded.csv')
#
#print(article_df.head())
#
#article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
#article_df['content_vector'] = article_df.content_vector.apply(literal_eval)
#
## Set vector_id to be a string
#article_df['vector_id'] = article_df['vector_id'].apply(str)
#
#article_df.info(show_counts=True)

# Option #2 - SaaS - (Weaviate Cloud Service)
client = weaviate.Client(
    url="https://my-sandbox-cluster-fbvreg5d.weaviate.network",
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

print(f"client is ready: {client.is_ready()}")

# Clear up the schema, so that we can recreate it
client.schema.delete_all()
print(client.schema.get())


class_obj = {
    "class": "Question",
    "vectorizer": "text2vec-openai"
}

client.schema.create_class(class_obj)

print(client.schema.get())

