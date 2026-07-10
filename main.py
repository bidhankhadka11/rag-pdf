from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_anthropic import ChatAnthropic

print(f"langchain-core verision: {core_version}")
print(f"langchain-graph verision: {lg_version}")

def main():
    llm = ChatAnthropic(model="claude-haiku-4-5")
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from Anthropic: {response}")

    print("Setup complete") 


if __name__ == "__main__":
    main()


