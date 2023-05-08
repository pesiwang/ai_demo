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

# Ignore unclosed SSL socket warnings - optional in case you get these errors
#import warnings
#
#warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning)

#embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'

# The file is ~700 MB so this will take some time
#wget.download(embeddings_url)

#import zipfile
#with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
#    zip_ref.extractall("./data")
#
t0 = time.time()

article_df = pd.read_csv('./data/vector_database_wikipedia_articles_embedded.csv')
t1 = time.time()
print(f"pd.read_csv cost {t1-t0} seconds")

print(article_df.head())

article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
t2 = time.time()
print(f"title_vector literal_eval cost {t2-t1} seconds")

article_df['content_vector'] = article_df.content_vector.apply(literal_eval)
t3 = time.time()
print(f"content_vector literal_eval cost {t3-t2} seconds")

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

t4 = time.time()
print(f"vector_id literal_eval cost {t4-t3} seconds")
article_df.info(show_counts=True)


#chroma_client = chromadb.Client() # Ephemeral. Comment out for the persistent version.

# Uncomment the following for the persistent version.
from chromadb.config import Settings
persist_directory = './data/chroma_persistence' # Directory to store persisted Chroma data.
chroma_client = chromadb.Client(
    Settings(
        persist_directory=persist_directory,
        chroma_db_impl="duckdb+parquet",
    )
)

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)

chroma_client.delete_collection(name='wikipedia_content')
chroma_client.delete_collection(name='wikipedia_titles')

wikipedia_content_collection = chroma_client.create_collection(name='wikipedia_content', embedding_function=embedding_function)
wikipedia_title_collection = chroma_client.create_collection(name='wikipedia_titles', embedding_function=embedding_function)


# Add the content vectors
wikipedia_content_collection.add(
    ids=article_df.vector_id.tolist(),
    documents=article_df.text.tolist(),
    embeddings=article_df.content_vector.tolist(),
)

t5 = time.time()
print(f" Add the content vectors cost {t5-t4} seconds")

# Add the title vectors
wikipedia_title_collection.add(
    ids=article_df.vector_id.tolist(),
    documents=article_df.title.tolist(),
    embeddings=article_df.title_vector.tolist(),
)

t6 = time.time()
print(f" Add the title vectors cost {t6-t5} seconds")

def query_collection(collection, query, max_results):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances', 'documents'])
    df = pd.DataFrame({
        'id': results['ids'][0],
        'distance': results['distances'][0],
        'document': results['documents'][0],
    })

    return df

title_query_result = query_collection(
    collection=wikipedia_title_collection,
    query="modern art in Europe",
    max_results=5,
)

t7 = time.time()
print(title_query_result.head(5))

print(f" query title cost {t7-t6} seconds")

content_query_result = query_collection(
    collection=wikipedia_content_collection,
    query="Famous battles in Scottish history",
    max_results=3,
)

t8 = time.time()
print(f" query content cost {t8-t7} seconds")
print(content_query_result.head(3))

