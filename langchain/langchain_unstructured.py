from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.document_loaders import UnstructuredFileLoader
"""
loader = UnstructuredFileLoader('./state_of_the_union.txt')

docs = loader.load()
print(f"docs len:{len(docs)}")

print(docs[0].page_content[0:400])
"""

#loader = UnstructuredFileLoader('./data/state_of_the_union.txt', mode="elements")
loader = UnstructuredFileLoader('./data/layout-parser-paper.pdf', mode="elements")
#loader = UnstructuredFileLoader('./data/layout-parser-paper.pdf', mode="elements", strategy="fast")
#loader = UnstructuredFileLoader('./data/layout-parser-paper.pdf', mode="elements", strategy="hi_res")

docs = loader.load()
print(f"docs len:{len(docs)}")

print(docs[:5])
#print(docs[26:45])

"""

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


#result = index.query(query)
#print(result)
#
#print("-" * 50)

#result_with_source = index.query_with_sources(query)
#print(result_with_source)
"""
