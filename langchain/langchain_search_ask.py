from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.document_loaders import TextLoader
loader = TextLoader('./state_of_the_union.txt', encoding='utf8')

from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)

index = index_creator.from_loaders([loader])

query = "What did the president say about Ketanji Brown Jackson"
result = index.query(query)
print(result)

print("-" * 50)

result_with_source = index.query_with_sources(query)
print(result_with_source)
"""

print("=" * 50)

# steps inside vector store index creator
# step 1. splitting documents into chunks
documents = loader.load()
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# step 2. Creating embeddings for each document
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# step 3. Storing documents and embeddings in a vectorstore

from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)

# step 4. query top k 
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

query = "What did the president say about Ketanji Brown Jackson"
print(qa.run(query))

"""

