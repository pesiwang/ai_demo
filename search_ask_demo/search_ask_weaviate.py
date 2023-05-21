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


article_df = pd.read_csv('./data/vector_database_wikipedia_articles_embedded.csv')

print(article_df.head())

article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

article_df.info(show_counts=True)

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
client.schema.get()

# Define the Schema object to use `text-embedding-ada-002` on `title` and `content`, but skip it for `url`
article_schema = {
    "class": "Article",
    "description": "A collection of articles",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
    },
    "properties": [{
        "name": "title",
        "description": "Title of the article",
        "dataType": ["string"]
    },
    {
        "name": "content",
        "description": "Contents of the article",
        "dataType": ["text"],
        "moduleConfig": { "text2vec-openai": { "skip": True } }
    }]
}

# add the Article schema
client.schema.create_class(article_schema)

# get the schema to make sure it worked
print(client.schema.get())


### Step 1 - configure Weaviate Batch, which optimizes CRUD operations in bulk
# - starting batch size of 100
# - dynamically increase/decrease based on performance
# - add timeout retries if something goes wrong
client.batch.configure(
    batch_size=100,
    dynamic=True,
    timeout_retries=3,
)

### Step 2 - import data

print("Uploading data with vectors to Article schema..")

counter = 0

with client.batch as batch:
    for k, v in article_df.iterrows():

        # print update message every 100 objects
        if (counter % 100 == 0):
            print(f"Import {counter} / {len(article_df)} ")

        properties = {
            "title": v["title"],
            "content": v["text"]
        }

        vector = v["title_vector"]

        batch.add_data_object(properties, "Article", None, vector)
        counter = counter + 1

print(f"Importing ({len(article_df)}) Articles complete")

# Test that all data has loaded – get object count
result = (
    client.query.aggregate("Article")
    .with_fields("meta { count }")
    .do()
)
print("Object count: ", result["data"]["Aggregate"]["Article"])

# Test one article has worked by checking one object
test_article = (
    client.query
    .get("Article", ["title", "content", "_additional {id}"])
    .with_limit(1)
    .do()
)["data"]["Get"]["Article"][0]

print(test_article["_additional"]["id"])
print(test_article["title"])
print(test_article["content"])


def query_weaviate(query, collection_name, top_k=3):
    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )["data"][0]['embedding']

    near_vector = {"vector": embedded_query}

    # Queries input schema with vectorised user query
    query_result = (
        client.query
        .get(collection_name, ["title", "content", "_additional {certainty distance}"])
        .with_near_vector(near_vector)
        .with_limit(top_k)
        .do()
    )

    return query_result

query_result = query_weaviate("modern art in Europe", "Article")
counter = 0
for article in query_result["data"]["Get"]["Article"]:
    counter += 1
    print(f"{counter}. { article['title']} (Certainty: {round(article['_additional']['certainty'],3) }) (Distance: {round(article['_additional']['distance'],3) })")


query_result = query_weaviate("Famous battles in Scottish history", "Article")
counter = 0
for article in query_result["data"]["Get"]["Article"]:
    counter += 1
    print(f"{counter}. {article['title']} (Score: {round(article['_additional']['certainty'],3) })")


def near_text_weaviate(query, collection_name):
    nearText = {
        "concepts": [query],
        "distance": 0.7,
    }

    properties = [
        "title", "content",
        "_additional {certainty distance}"
    ]

    query_result = (
        client.query
        .get(collection_name, properties)
        .with_near_text(nearText)
        .with_limit(10)
        .do()
    )["data"]["Get"][collection_name]

    print(f"Objects returned: {len(query_result)}")

    return query_result

query_result = near_text_weaviate("modern art in Europe","Article")
counter = 0
for article in query_result:
    counter += 1
    print(f"{counter}. { article['title']} (Certainty: {round(article['_additional']['certainty'],3) }) (Distance: {round(article['_additional']['distance'],3) })")

query_result = near_text_weaviate("Famous battles in Scottish history","Article")
counter = 0
for article in query_result:
    counter += 1
    print(f"{counter}. { article['title']} (Certainty: {round(article['_additional']['certainty'],3) }) (Distance: {round(article['_additional']['distance'],3) })")

