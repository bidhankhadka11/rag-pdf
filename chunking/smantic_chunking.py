from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

document = '''
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials. Register your application in the developer dashboard to receive a client ID and client secret. Exchange these for an access token using the /oauth/token endpoint, then include the token as a Bearer header on every subsequent request. Access tokens expire after 1 hour, so use the refresh token to obtain a new one without requiring the user to log in again.

## Rate Limiting
All API endpoints are rate limited to protect the service from abuse. Free tier accounts are limited to 60 requests per minute, while paid tiers allow up to 1000 requests per minute. Each response includes X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset headers so clients can track their usage. Exceeding the limit returns a 429 Too Many Requests status code, and you should implement exponential backoff before retrying.

## Error Handling
The API returns standard HTTP status codes to indicate success or failure. A 4xx status indicates a client error, such as invalid parameters or missing authentication, while a 5xx status indicates a server error. Every error response includes a JSON body with an "error" object containing a machine-readable code and a human-readable message. Clients should log the request ID from the X-Request-ID header when reporting issues to support.

## Webhooks
Webhooks let your application receive real-time notifications when events occur, such as a payment succeeding or a user updating their profile. Configure a webhook endpoint URL in the dashboard and subscribe to the event types you care about. Each webhook payload is signed with an HMAC signature in the X-Signature header, which you should verify to confirm the request came from us. If your endpoint doesn't return a 200 response, the webhook will be retried with exponential backoff for up to 24 hours.
'''


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

recursive_chunks = recursive_splitter.split_text(document)
# print(f"Recursive chunks: {len(recursive_chunks)}")
# for i, chunk in enumerate(recursive_chunks):
#     print(f"\\n---Chunk {i+1} ({len(chunk)} chars) ---")
#     print(chunk[:100] + "..." if len(chunk)> 100 else chunk)



semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90 #split at 90th percentile of dissimilarity
)

semantic_chunks = semantic_chunker.split_text(document)
# print(f"\nSemantic chunks: {len(semantic_chunks)}")
# for i, chunk in enumerate(semantic_chunks):
#     print(f"\n---Chunk {i+1} ({len(chunk)} chars) ---")
#     print(chunk[:100] + "..." if len(chunk)> 100 else chunk)


"""Testing the chunks"""

#Create two vector store for each chunking method
recursive_vectorstore = Chroma.from_texts(
    recursive_chunks,
    embeddings,
    collection_name='recursive_chunks'
)

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks,
    embeddings,
    collection_name='semantic_chunks'
)


#test queries
test_queries = [
    'How do I authenticate with Oauth2?',
    'What happens if I exceed rate limits?',
    'How are errors handled in the API?',
    'Do you support webhooks for real-time notifications?'
]

def test_retrieval(query, vectorstore, name:str):
    results = vectorstore.similarity_search(query, k=3)
    print(f"\\n{name} -Query: \'{query}\'")
    print(f'Retrieved: {results[0].page_content[:150]}...')
    return results[0].page_content


print(f"\n{'=' * 60}")
print(".   Retrieval Tests")
print(f"{'='* 60}")

for query in test_queries:
    print('=' * 60)
    recursive_result = test_retrieval(query, recursive_vectorstore, 'Recursive')
    semantic_result = test_retrieval(query, semantic_vectorstore, 'Semantic')
    
    
"""My take: for this document, recursive wins because 
the content is already structured with markdown headers marking topic boundaries 
— once chunk_size is big enough to hold a full section 
(which we just fixed), the \n\n separator does exactly the 
right thing essentially for free, no embedding calls needed.

Semantic chunking's edge is on long, unstructured prose with no formatting cues, 
where you actually need embedding similarity to find where topics shift. 
Here there's nothing to find — the headers already tell you. 
On top of that, this document's sentences are stylistically homogeneous (
"the API does X, returns Y"), so the 90th-percentile dissimilarity 
breakpoint has a weak signal to work with and lands in arbitrary spots."""