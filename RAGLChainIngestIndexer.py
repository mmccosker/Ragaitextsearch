"""
RAG Pipeline with FAISS (Meta) Vector Store + LangChain
=========================================================
Steps:
  1. Load a list of documents (txt/pdf/docx supported here)
  2. Split them into chunks using LangChain's TextSplitter
  3. Embed chunks and build a FAISS vector store (the "metastore")
  4. Wire up a Retrieval chain with your choice of LLM:
       - OpenAI (paid, needs OPENAI_API_KEY)
       - Ollama (free, runs locally, needs `ollama serve` running)

Install dependencies:
    pip install langchain langchain-community langchain-openai langchain-ollama \
                faiss-cpu pypdf python-docx tiktoken
"""

import os
from pathlib import Path

# ---- LangChain core imports ----
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # <-- Meta's FAISS lib via LangChain wrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---- Model provider imports (pick one at runtime) ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


# =========================================================
# 1. CONFIG — choose your model provider here
# =========================================================
MODEL_PROVIDER = os.getenv("RAG_PROVIDER", "ollama")  # "openai" or "ollama"

# Folder containing your source documents
# Project docs: C:\python3\Python\documents
DOCS_DIR = "./documents"

# Where to persist the FAISS index (the "metastore")
FAISS_INDEX_PATH = "./faiss_index"

# Chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Model names
OPENAI_LLM_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OLLAMA_LLM_MODEL =  "gemma3:1b"         # "llama3.1" run: ollama pull llama3.1
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # run: ollama pull nomic-embed-text


# =========================================================
# 2. LOAD DOCUMENTS — combine a list of files into one corpus
# =========================================================
def load_documents(docs_dir: str):
    """Load .txt, .pdf, and .docx files from a directory into a single list."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")

    all_documents = []
    loaders_by_ext = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
    }

    for file_path in docs_dir.rglob("*"):
        if file_path.suffix.lower() in loaders_by_ext:
            loader_cls = loaders_by_ext[file_path.suffix.lower()]
            try:
                loader = loader_cls(str(file_path))
                loaded = loader.load()
                all_documents.extend(loaded)
                print(f"Loaded {len(loaded)} doc(s) from {file_path.name}")
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

    if not all_documents:
        raise ValueError(f"No supported documents found in {docs_dir}")

    return all_documents


# =========================================================
# 3. SPLIT DOCUMENTS INTO CHUNKS
# =========================================================
def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


# =========================================================
# 4. GET EMBEDDINGS + LLM BASED ON PROVIDER CHOICE
# =========================================================
def get_embeddings_and_llm(provider: str):
    provider = provider.lower()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("Set OPENAI_API_KEY env variable to use OpenAI.")
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
        llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0)

    elif provider == "ollama":
        # Free, runs locally. Requires `ollama serve` running and models pulled:
        #   ollama pull llama3.1
        #   ollama pull nomic-embed-text
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)

    else:
        raise ValueError("MODEL_PROVIDER must be 'openai' or 'ollama'")

    return embeddings, llm


# =========================================================
# 5. BUILD OR LOAD THE FAISS METASTORE
# =========================================================
def build_or_load_vectorstore(chunks, embeddings, index_path=FAISS_INDEX_PATH):
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}")
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Building new FAISS index from chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_path)
        print(f"FAISS index saved to {index_path}")

    return vectorstore


# =========================================================
# 6. BUILD THE RETRIEVAL CHAIN
# =========================================================
def build_retrieval_chain(vectorstore, llm, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt_template = """Use the following context to answer the question.
If you don't know the answer from the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# =========================================================
# 7. MAIN PIPELINE
# =========================================================
def main():
    print(f"Using model provider: {MODEL_PROVIDER}")

    # Step 1: Load documents
    documents = load_documents(DOCS_DIR)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Get embeddings + LLM
    embeddings, llm = get_embeddings_and_llm(MODEL_PROVIDER)

    # Step 4: Build/load FAISS metastore
    vectorstore = build_or_load_vectorstore(chunks, embeddings)

    # Step 5: Build retrieval chain
    qa_chain = build_retrieval_chain(vectorstore, llm)

    # Step 6: Ask a question
    print("\nRAG pipeline ready. Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        result = qa_chain.invoke({"query": query})
        print("\nAnswer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "unknown")
            print(f"  - {source}")
        print()


if __name__ == "__main__":
    main()