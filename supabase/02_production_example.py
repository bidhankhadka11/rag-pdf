"""
Supabase RAG Production Example
--Creating a full production RAG using supabase pgvector as database
"""
from langchain_core.runnables import Runnable
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import Optional

load_dotenv()


#Configuration
@dataclass
class Config:
    #Database - use pooler URL in production
    database_url: str = os.getenv(
        "SUPABASE_DATABASE_URL",
        os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")   
    )
    collection_name: str = "production_documents"
    
    # Model settings
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    #Search settings
    default_k: int = 5
    min_similarity: float = 0.5

    def __post_init__(self):
        # langchain-postgres needs the psycopg3 driver; a plain postgresql://
        # URL defaults to psycopg2 (not installed). Force +psycopg.
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )


class RAGService:
    """Production-ready RAG service with pgvector"""

    def __init__(self, config: Optional[Config] = None):  #Config should either be an instance of Config or none
        self.config = config or Config()
        self._vectorstore = None  #_ means the data / object inside the method
        self._chain = None

    @property #Lets you access methods without () e.g. rag_service.vectorstore is valid
    def vectorstore(self) -> PGVector:
        """Lazy initialization of vectorstore""" #This will only build until someone asks
        if self._vectorstore is None:
            embeddings = OpenAIEmbeddings(
                model = self.config.embedding_model,
            )
            self._vectorstore = PGVector(
                embeddings = embeddings,
                collection_name = self.config.collection_name,
                connection=self.config.database_url,
                use_jsonb=True, 
            ) 
        return self._vectorstore 

    
    @property
    def chain(self):
        """lazy initialization of RAG Chain"""
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain


    def _create_chain(self):  #_ represents private only meant to be called from inside
        """Create the RAG chain"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs = {"k": self.config.default_k}
        )

        llm = ChatOpenAI(
            model = self.config.chat_model, temperature=self.config.temperature
        )

        prompt = ChatPromptTemplate.from_template(
            """
            You are helpful assistant. Answer the question based on the provided context.

            Context:
            {context}

            Question: {question}

            Answer concisely and accurately. If the context doesn't contain relevant info,
            say "I don't have enough information to answer that question"
            """
        )
        #Turns the retriever's list[Document] int one string for {context}
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser() 
        )


    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and return their IDs"""
        return self.vectorstore.add_documents(documents) #Using the pg vector method add_documents
    

    def search(
        self, query: str, k:Optional[int]= None, filter_dict: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Search with optional filtering"""
        search_kwargs = {"k": k or self.config.default_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore.similarity_search_with_score(query, **search_kwargs)

                
    def ask(self, question:str) -> str:
        """Ask a question using RAG"""
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str) -> str:
        """Ask using a question and return sources"""
        #Get relevant documents
        docs_with_scores = self.search(question)

        #Generate answer
        answer = self.ask(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": score,
                }
                for doc, score in docs_with_scores
            ],
        }



def main():
    #Initialize service
    print("Initializing RAG service...")
    service = RAGService()
    print("Service Ready!")

    #Add sample documents
    print("\nAdding sample documents...")
    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain_docs", "topic": "overview"},
        ),
        Document(
            page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            metadata={"source": "langgraph_docs", "topic": "overview"},
        ),
        Document(
            page_content="Vector stores are databases optimized for storing and searching embeddings.",
            metadata={"source": "vector_guide", "topic": "database"},
        ),
        Document(
            page_content="RAG combines retrieval with generation for more accurate LLM responses.",
            metadata={"source": "rag_guide", "topic": "architecture"},
        ),
        Document(
            page_content="Embeddings convert text into numerical vectors for semantic similarity.",
            metadata={"source": "embeddings_guide", "topic": "fundamentals"},
        ),
        Document(
            page_content="Chroma is an open-source embedding database for AI applications.",
            metadata={"source": "chroma_docs", "topic": "database"},
        ),
        Document(
            page_content="FAISS is a library for efficient similarity search developed by Facebook.",
            metadata={"source": "faiss_docs", "topic": "database"},
        ),
        Document(
            page_content="Pinecone is a managed vector database service for production workloads.",
            metadata={"source": "pinecone_docs", "topic": "database"},
        ),
    ]

    ids = service.add_documents(sample_docs)
    print(f"Added {len(ids)} documents")

    #Test search
    print("\nTesting Search...")
    results = service.search("What is rag?", k =2)
    for doc, score in results:
        print(f" Score: {score:.4f} - {doc.page_content[:50]}...")

    #Test Rag 
    print("\nTesting RAG...")
    question = "What is pinecone and how to use it?"
    response = service.ask_with_sources(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources: {len(response['sources'])}")

    print("\n\nProduction RAG service demo complete!")


if __name__ == "__main__":
    main()
    
    