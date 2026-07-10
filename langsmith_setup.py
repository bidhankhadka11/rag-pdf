'''
LangSmith setup and observability
Production monitoring for Langchain/Langgraph
'''

import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()

# Enable Langsmith tracing
os.environ["LANGSMITH_TRACING"] = "true"


@traceable(name="basic_tracing") 
def demo_basic_tracing():
    '''Basic Langsmith tracing'''
    
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence"
    )

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with LangSmith tracing enabled... ")

    result = chain.invoke({"topic": "Machine Learning"})

    print(f"Result: {result}")
    print("\nCheck LangSmith dashboard for trace details.")


@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """Name your runs for easier identification."""

    llm = ChatAnthropic(model = "claude-haiku-4-5", temperature=0.2)

    prompt = ChatPromptTemplate.from_template("Summarize: {text}")

    chain = prompt | llm | StrOutputParser()

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )
    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'.")


@traceable(name="metadata_demo", tags=["metadata","filtering"])
def demo_trace_with_metadata(user_id:str, request_type:str):
    """Add custom metadata to traces."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    result = llm.invoke(f"Hello from user {user_id}")

    return result.content



if __name__ == "__main__":
    demo_basic_tracing()
    demo_named_runs()
    demo_trace_with_metadata("user123", "metadata_demo")