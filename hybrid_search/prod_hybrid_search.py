from openai.types import other_file_chunking_strategy_object
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


documents = [
    Document(
        page_content='Product SKU-7742X is our flagship router. It supports '
        'gitgabit speeds and advanced QoS features.',
        metadata={'type': 'product'}
    ),
    Document(
        page_content='For network connectivity issues, first check the'
        'ethernet cable and router status lights',
        metadata={'type': 'troubleshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server'
        'rejected the connection. Check firewall settings',
        metadata={'type': 'error'}
    ),
    Document(
        page_content='The authentication process requires valid credentials'
        'Use OAuth2 for secure API access.',
        metadata={'type': 'auth'}
    ),
    Document(
        page_content='Router configuration guide: Access the admin panel'
        'at 192.168.1.1 to modify settings',
        metadata={'type': 'config'}
    ),
    Document(
        page_content='WCAG 2.1 compliance requires all images to have'
        'at text and sufficient color contrast.',
        metadata={'type': 'compliance'}
    ),
]

print(f'Loaded {len(documents)} documents.')

# Create a vector store from documents
vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name='hybrid_test'
)


# Vector Retriever - Langchain Runnable
vector_retriever = vectorstore.as_retriever(
    search_kwargs={'k':3}
)
print('Vector retriever ready')



#BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)
print('BM25 retriever ready')   


#Combine with Ensemble Retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights = [0.5, 0.5] #equal weight to both  #if 0.7, 03 code have vice versa semantic heavy
)
print('Hybrid retriever ready')


'''Test query and show results'''
def test_query(query, name, retriever):
    results = retriever.invoke(query)
    print(f'\\n{name} - Query: \"{query}\"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f' {i+1}. {preview}')
    return results


test_queries = [
    'SKU-7742X specifications',
    'E_CONN_REFUSED error',
    'How do I authenticate?',
    'WCAG compliance',
    'router configuration',
    'What is the IP address for router settings?',
]

for query in test_queries:
    print('='*60)

    #Vector only
    vector_results = test_query(query, 'Vector', vector_retriever)

    #BM25 only
    bm25_results = test_query(query, 'BM25', bm25_retriever)

    #Hybrid
    ensemble_results = test_query(query, 'Ensemble', ensemble_retriever)




