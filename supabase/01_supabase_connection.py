"""
Connectiong to Supabase Databse for pg vector 
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

#for demo, get local if supabase not configured
DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

# langchain-postgres uses psycopg3, not psycopg2 - force that driver
# regardless of which URL (Supabase or local) ended up being used above
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

def connect_to_supabase():
    """Connect to Supabase PG Vector"""

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    vectorstore = PGVector(
        embeddings = embeddings,
        collection_name = "production_docs",
        connection = DATABASE_URL,
        use_jsonb = True,
    )

    return vectorstore

def verify_connection(vectorstore):
    """Verify the connection works"""
    from langchain_core.documents import Document

    #Add a test document
    test_doc = Document(
        page_content = "This is a test document for Supabase connection",
        metadata = {"test": True},
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Added test document: {ids[0]}")

        results = vectorstore.similarity_search("test document")
        if results:
            print(f"Search works: {results[0].page_content}")

        #clean up
        # vectorstore.delete(ids)
        # print(f"Deleted test document")

        return True

    except Exception as e:
        print(f"Connection failed: {e}")
        return False



if __name__ == "__main__":
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    if SUPABASE_URL:
        print(f"\n Connecting to Supabase...")
        print(f" Host: {SUPABASE_URL.split('@')[1].split('/')[0]}")
    else:
        print("\nSUPABASE_DATABASE_URL not found in .env")
        print("Using local PostgreSQL instead")
    
    vectorstore = connect_to_supabase()
    verify_connection(vectorstore)
 
    
    