import openai
import time
from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval

# Chroma's client library for Python
import chromadb

# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

chroma_client = chromadb.Client() # Ephemeral. Comment out for the persistent version.

# Uncomment the following for the persistent version.
import chromadb.config.Settings
persist_directory = './data/chroma_persistence' # Directory to store persisted Chroma data.
client = chromadb.Client(
    Settings(
        persist_directory=persist_directory,
        chroma_db_impl="duckdb+parquet",
    )
)

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)

wikipedia_content_collection = chroma_client.get_collection(name='wikipedia_content', embedding_function=embedding_function)
wikipedia_title_collection = chroma_client.get_collection(name='wikipedia_titles', embedding_function=embedding_function)

def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
    df = pd.DataFrame({
        'id': results['ids'][0],
        'score': results['distances'][0],
        'title': dataframe[dataframe.vector_id.isin(results['ids'][0])]['title'],
        'content': dataframe[dataframe.vector_id.isin(results['ids'][0])]['text'],
    })

    return df

title_query_result = query_collection(
    collection=wikipedia_title_collection,
    query="modern art in Europe",
    max_results=5,
    dataframe=article_df
)

print(title_query_result.head(5))

content_query_result = query_collection(
    collection=wikipedia_content_collection,
    query="Famous battles in Scottish history",
    max_results=3,
    dataframe=article_df
)

print(content_query_result.head(3))

