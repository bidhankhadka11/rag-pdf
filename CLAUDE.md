# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal learning project for RAG (retrieval-augmented generation) with LangChain. It is **not** a single application — it is a collection of standalone, self-contained tutorial scripts, each demonstrating one RAG concept in isolation (vector stores, text splitting, retrievers, hybrid search, cost optimization, observability, etc.). Files generally do not import from each other; each defines its own sample data inline and runs independently.

## Environment & commands

- Package/venv manager is **uv**. Python is pinned to **3.11** (`.python-version`).
- Run any script: `uv run python <file>.py` (e.g. `uv run python rag_pipeline.py`, `uv run python supabase/01_supabase_connection.py`).
- Add a dependency: `uv add <package>` (updates `pyproject.toml` + `uv.lock`).
- There is **no test suite, linter, or build step.** Each script is "run" directly and verified by reading its `print` output. Most files gate their demo behind `if __name__ == "__main__":`; a few (`supabase/02_production_example.py`, `chunking/smantic_chunking.py`, `hybrid_search/prod_hybrid_search.py`, `embeddings_deep.py`) run top-level code or are still being built out.

## Secrets & configuration

- All credentials come from `.env` via `load_dotenv()` at the top of every script (`.env` is gitignored). Keys used: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`, `SUPABASE_DATABASE_URL`.
- Never echo `.env` values back into output; redact when inspecting.

## Provider conventions (important, and inconsistent across files)

- **Embeddings** are always OpenAI `text-embedding-3-small`.
- **Chat model choice varies by file** and is deliberate — don't "normalize" it: `main.py` uses Anthropic (`claude-haiku-4-5`), `rag_pipeline.py` uses `init_chat_model("claude-haiku-4-5")`, and the OpenAI-centric files (`cost_optimization.py`, `langsmith_setup.py`, `supabase/02_production_example.py`) use `gpt-4o-mini` / `gpt-4o`.
- Model IDs matter: `claude-3-haiku-20240307` is retired and 404s — the current small Anthropic model is `claude-haiku-4-5`.
- `temperature`/`top_p`/`top_k` are fine on `claude-haiku-4-5` but are rejected (400) on Anthropic's newest tier (Sonnet 5, Opus 4.7/4.8, Fable 5). Keep this in mind before swapping models.

## Vector store backends

Two backends appear, and the setup differs:

- **Chroma** (`langchain_chroma`) — used by the standalone demo scripts, persisted to a throwaway `tempfile` dir. No external service needed.
- **Supabase pgvector** (`langchain_postgres.PGVector`) — used by the `supabase/` scripts. Two hard-won connection gotchas live here:
  - `langchain-postgres` requires the **psycopg3** driver, but a plain `postgresql://` URL defaults to psycopg2 (not installed). Rewrite the scheme to `postgresql+psycopg://` before passing it to `PGVector` (see `supabase/01_supabase_connection.py`).
  - The Supabase **direct** host (`db.<ref>.supabase.co`) is **IPv6-only**. Use the **session/transaction pooler** host (`aws-0-<region>.pooler.supabase.com`, user `postgres.<ref>`) for IPv4 networks. `SUPABASE_DATABASE_URL` in `.env` should be the pooler URL.

## LangChain import layout (v1.x)

This project is on LangChain **1.x**, where imports are spread across several packages — get the package right or the import fails:

- `langchain_core` — `Document`, prompts (`ChatPromptTemplate`), runnables (`RunnablePassthrough`), output parsers (`StrOutputParser`).
- `langchain_text_splitters` — `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`, etc.
- `langchain_experimental.text_splitter` — `SemanticChunker`.
- `langchain_classic.retrievers` — the "advanced" retrievers (`MultiQueryRetriever`, `EnsembleRetriever`, `ContextualCompressionRetriever`, `ParentDocumentRetriever`). Note: **`langchain_classic`**, not `langchain.retrievers`.
- `langchain_community.retrievers` — `BM25Retriever` (keyword search, backed by `rank-bm25`).
- `langchain_chroma`, `langchain_postgres`, `langchain_openai`, `langchain_anthropic` — provider integrations.

## Core RAG pattern (LCEL)

The canonical pipeline shape used throughout (clearest in `rag_pipeline.py`) is an LCEL chain built with the `|` operator, where every stage is a Runnable:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

`retriever` comes from `vector_store.as_retriever(...)`; `format_docs` joins the retrieved `Document.page_content` values into the string that fills the prompt's `{context}` slot. Prompts consistently include a grounding instruction ("answer based only on the context") plus an "if you don't know, say I don't know" hallucination guard.

## Observability

`langsmith_setup.py` and `cost_optimization.py` use LangSmith. Tracing is controlled by the `LANGSMITH_TRACING` env flag and the `@traceable` decorator. `LANGSMITH_TRACING` defaults to `false` in `.env`; `langsmith_setup.py` force-enables it at runtime.
